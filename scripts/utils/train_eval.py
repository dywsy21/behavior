"""Periodic in-training evaluation, extracted from scripts/finetune.py.

Owns the eval dataloader iteration state and the rollout → metric-reduce →
snapshot pipeline, so the training loop only calls ``evaluator.evaluate(...)``.
"""

import logging
import time
from contextlib import contextmanager
from copy import deepcopy

import torch.distributed as dist

from utils.eval_snapshot import save_eval_snapshot
from utils.metric import (
    _batch_size,
    _select_batch_rows,
    reduce_payload,
    reduce_per_emb_metrics,
    rollout_and_calculate_metrics,
)

logger = logging.getLogger(__name__)


@contextmanager
def evaluation_state(model):
    """Disable training-only history dropout without contaminating train metrics.

    Teacher-forced validation also calls the training forward, which pushes
    accuracy counters. Give it private counters and restore all module modes,
    including deliberately eval-only children, even when evaluation raises.
    """
    modes = [(module, module.training) for module in model.modules()]
    accumulator = getattr(model, "_train_acc", None)
    accuracy_attrs = {
        name: getattr(model, name)
        for name in ("train_action_accuracy", "train_cot_accuracy")
        if hasattr(model, name)
    }
    try:
        if accumulator is not None:
            model._train_acc = deepcopy(accumulator)
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        if accumulator is not None:
            model._train_acc = accumulator
        for name in ("train_action_accuracy", "train_cot_accuracy"):
            if name in accuracy_attrs:
                setattr(model, name, accuracy_attrs[name])
            elif hasattr(model, name):
                delattr(model, name)


class PeriodicEvaluator:
    """Runs one eval batch periodically during training and returns log metrics."""

    def __init__(self, eval_dataloader, eval_sampler, eval_processor, parts_meta, output_dir):
        self.eval_dataloader = eval_dataloader
        self.eval_sampler = eval_sampler
        self.eval_processor = eval_processor
        self.parts_meta = parts_meta
        self.output_dir = output_dir
        self._iter = iter(eval_dataloader)

    def _next_batch(self):
        try:
            return next(self._iter)
        except StopIteration:
            self.eval_sampler.set_epoch(self.eval_sampler.epoch + 1)
            self._iter = iter(self.eval_dataloader)
            return next(self._iter)

    def evaluate(self, model, accelerator, step: int, eval_batch=None) -> dict:
        """Run rollout metrics on one eval batch (or a provided batch, e.g. overfit mode).

        Returns a flat dict ready to merge into the tracker ``log_dict``:
        ``eval/action/*`` for aggregate metrics, ``per_emb/*`` for per-embodiment ones,
        plus the eval wall time under ``_eval_time_sec`` (not logged by tracker prefix).
        """
        start_time = time.time()
        if eval_batch is None:
            eval_batch = self._next_batch()

        with evaluation_state(model):
            rollout_metrics, per_emb_raw, eval_preds = rollout_and_calculate_metrics(
                eval_batch,
                model,
                accelerator,
                processor=self.eval_processor,
                return_per_emb_raw=True,
                return_preds=True,
                parts_meta=self.parts_meta,
            )
        if dist.is_initialized():
            # All ranks must participate in the collective op unconditionally,
            # even when this rank's dict is empty, otherwise other ranks hang.
            rollout_metrics = reduce_payload(
                {
                    k: (v.item() if hasattr(v, "item") else float(v))
                    for k, v in rollout_metrics.items()
                }
            )
            per_emb_metrics = reduce_per_emb_metrics(per_emb_raw or {})
            rollout_metrics.update(per_emb_metrics)
        elif per_emb_raw:
            per_emb_metrics = reduce_per_emb_metrics(per_emb_raw)
            rollout_metrics.update(per_emb_metrics)

        log_dict = {}
        for k, v in rollout_metrics.items():
            if k.startswith("per_emb/"):
                log_dict[k] = v  # per_emb/ already includes the eval/ sub-prefix.
            else:
                log_dict[f"eval/action/{k}"] = v

        try:
            snapshot_batch = eval_batch
            if isinstance(eval_preds, dict) and eval_preds.get("_rollout_indices") is not None:
                snapshot_batch = _select_batch_rows(
                    eval_batch,
                    eval_preds["_rollout_indices"],
                    _batch_size(eval_batch),
                )
            save_eval_snapshot(
                snapshot_batch,
                eval_preds,
                self.output_dir,
                step,
                parts_meta=self.parts_meta,
            )
        except Exception as e:
            logger.warning(
                f"save_eval_snapshot failed at step {step}, skipping: {e}",
                exc_info=True,
            )

        eval_time = time.time() - start_time
        logger.info(f"Evaluation took {eval_time:.2f} seconds")
        log_dict["_eval_time_sec"] = eval_time
        return log_dict
