"""Regression coverage for deterministic, task-diverse periodic eval sampling."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# scripts/finetune.py imports this module as ``utils.train_eval``. Match that
# runtime import path so the test exercises the real PeriodicEvaluator class.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.train_eval import PeriodicEvaluator  # noqa: E402
from utils.metric import rollout_and_calculate_metrics  # noqa: E402


class _TaskOrderedDataset(Dataset):
    """Five contiguous task regions, matching task_stratified eval ordering."""

    def __init__(self, num_tasks: int = 5, samples_per_task: int = 1_000):
        self.num_tasks = num_tasks
        self.samples_per_task = samples_per_task

    def __len__(self) -> int:
        return self.num_tasks * self.samples_per_task

    def __getitem__(self, index: int):
        return {"index": index, "task": index // self.samples_per_task}


def _sampler_indices(dataset, *, seed: int, epoch: int, rank: int, num_replicas: int):
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
        shuffle=True,
    )
    sampler.set_epoch(epoch)
    return list(sampler)


def test_periodic_eval_shuffle_is_seed_reproducible_and_task_diverse():
    dataset = _TaskOrderedDataset()
    num_replicas = 2
    batch_size = 8
    streams_a = [
        _sampler_indices(dataset, seed=2026, epoch=0, rank=rank, num_replicas=num_replicas)
        for rank in range(num_replicas)
    ]
    streams_b = [
        _sampler_indices(dataset, seed=2026, epoch=0, rank=rank, num_replicas=num_replicas)
        for rank in range(num_replicas)
    ]

    assert streams_a == streams_b
    first_fifty_global_batches = []
    for batch_index in range(50):
        for rank_stream in streams_a:
            start = batch_index * batch_size
            first_fifty_global_batches.extend(rank_stream[start : start + batch_size])
    seen_tasks = {dataset[index]["task"] for index in first_fifty_global_batches}
    assert seen_tasks == set(range(dataset.num_tasks))
    assert seen_tasks != {0}


def test_distributed_sampler_rank_shards_cover_dataset_with_only_padding_duplicates():
    dataset = _TaskOrderedDataset(num_tasks=5, samples_per_task=5)  # 25 is not divisible by 3.
    num_replicas = 3
    streams = [
        _sampler_indices(dataset, seed=7, epoch=0, rank=rank, num_replicas=num_replicas)
        for rank in range(num_replicas)
    ]
    merged = [index for stream in streams for index in stream]

    assert len({len(stream) for stream in streams}) == 1
    assert set(range(len(dataset))) <= set(merged)
    assert len(merged) - len(set(merged)) == len(merged) - len(dataset)


def test_periodic_evaluator_wrap_advances_epoch_and_changes_order(tmp_path):
    dataset = _TaskOrderedDataset(num_tasks=5, samples_per_task=2)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=123, shuffle=True)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)
    evaluator = PeriodicEvaluator(
        eval_dataloader=loader,
        eval_sampler=sampler,
        eval_processor=None,
        parts_meta=None,
        output_dir=tmp_path,
    )
    epoch_zero = list(
        DistributedSampler(dataset, num_replicas=1, rank=0, seed=123, shuffle=True)
    )
    epoch_one_sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, seed=123, shuffle=True
    )
    epoch_one_sampler.set_epoch(1)
    epoch_one = list(epoch_one_sampler)

    for _ in range(len(loader)):
        evaluator._next_batch()
    wrapped_batch = evaluator._next_batch()

    assert sampler.epoch == 1
    assert epoch_one != epoch_zero
    assert wrapped_batch["index"].tolist() == epoch_one[:2]


def test_memlite_periodic_eval_teacher_forces_mixed_but_rolls_out_low_only():
    class _Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return nullcontext()

    class _Model:
        training = True
        train_action_accuracy = 0.5
        train_cot_accuracy = 0.25

        def __init__(self):
            self.teacher_branches = None
            self.rollout_branches = None

        def __call__(self, batch, inference_mode=False):
            branches = [sample["memlite_branch"] for sample in batch["samples"]]
            if not inference_mode:
                self.teacher_branches = branches
                assert set(branches) == {"high", "low"}
                return torch.tensor(0.0), {"ce_loss": torch.tensor(1.0), "fm_loss": torch.tensor(2.0)}
            self.rollout_branches = branches
            assert branches == ["low", "low"]
            return {"action": batch["action"].clone()}

    batch = {
        "samples": [
            {"memlite_branch": "high"},
            {"memlite_branch": "high"},
            {"memlite_branch": "low"},
            {"memlite_branch": "low"},
        ],
        "action": torch.zeros(4, 2, 1),
        "action_dim_is_pad": torch.zeros(4, 1, dtype=torch.bool),
        "action_is_pad": torch.zeros(4, 2, dtype=torch.bool),
        "embodiment": ["r1"] * 4,
    }
    model = _Model()
    metrics, preds = rollout_and_calculate_metrics(
        batch,
        model=model,
        accelerator=_Accelerator(),
        return_preds=True,
    )

    assert model.teacher_branches == ["high", "high", "low", "low"]
    assert model.rollout_branches == ["low", "low"]
    assert metrics["teacher_forcing/ar_action_ce_loss"] == 1.0
    assert preds["_rollout_indices"] == [2, 3]
    assert preds["action_gt"].shape[0] == 2
