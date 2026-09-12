# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""PolicyInferencer — reusable batched inference for G05 policies."""

from __future__ import annotations

import dataclasses
import logging
import time
from copy import deepcopy
from contextlib import nullcontext
from typing import Any

import torch

from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.utils.data.data_utils import collate_fn_pad_sequences, custom_collate_fn
from g05.utils.common.pytorch_utils import dict_apply

logger = logging.getLogger(__name__)


# ``MEMLiteMixedBuilder`` validates every branch's training fields in its
# regular ``build`` method.  Serving has no future high-level target, so the
# high branch needs non-empty values only to get through that *preprocessing*
# contract.  They are deliberately removed before the high-level AR prefill
# (see ``G05Policy._clone_memlite_samples``); they are never model context or
# supervision.
_MEMLITE_EMPTY_MEMORY_SENTINEL = "No prior memory."
_MEMLITE_RUNTIME_TARGET_PLACEHOLDER = "__memlite_runtime_target_placeholder__"


def _sync_if_cuda_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclasses.dataclass(slots=True)
class PreparedSample:
    """Preprocessed sample plus its sub_processor reference."""

    obs_dict: dict[str, Any]
    sample: dict[str, Any]
    sub_processor: Any
    raw_state_anchor: dict[str, torch.Tensor] | None = None


def resolve_processor(processor, data):
    """Return the matching sub-processor from MixtureProcessor, or single directly.

    Supports both raw_obs (key="embodiment_type") and obs_dict (key="embodiment") forms.
    """
    if isinstance(processor, MixtureProcessor):
        emb = data.get("embodiment_type") or data.get("embodiment")
        if emb is None and len(processor.processors) == 1:
            emb = next(iter(processor.processors))
        return processor[emb]
    return processor


class PolicyInferencer:
    """Unified inference wrapper: obs_dicts in → action dicts out."""

    def __init__(self, policy, processor, device: str = "cuda"):
        self.policy = policy
        self.processor = processor
        self.device = device

    # ──────── Public API ────────

    def infer(self, obs_dicts: list[dict]) -> list[dict]:
        """Full pipeline: raw obs_dicts → postprocessed action dicts."""
        results, _ = self.infer_with_timing(obs_dicts)
        return results

    def infer_with_timing(self, obs_dicts: list[dict]) -> tuple[list[dict], dict[str, float]]:
        """Same as :meth:`infer` plus per-stage timing (ms).

        Returns ``(results, {"preprocess_ms", "infer_ms", "postprocess_ms"})``.
        """
        if not obs_dicts:
            return [], {"preprocess_ms": 0.0, "infer_ms": 0.0, "postprocess_ms": 0.0}

        _sync_if_cuda_available()
        t0 = time.monotonic()
        prepared = [self._prepare(obs_dict) for obs_dict in obs_dicts]
        padding_input_id = prepared[0].sub_processor.pad_token_id
        batch = self._collate([p.sample for p in prepared], padding_input_id=padding_input_id)
        _sync_if_cuda_available()
        t1 = time.monotonic()
        batch = self.forward_batch(batch)
        _sync_if_cuda_available()
        t2 = time.monotonic()
        model_timing = batch.pop("_timing", {})
        cot_texts = batch.get("cot_text")
        results = []
        for i, p in enumerate(prepared):
            anchor = getattr(p, "raw_state_anchor", None)
            action = self._postprocess_single(batch, index=i, sub_processor=p.sub_processor,
                                             **({"raw_state_anchor": anchor} if anchor is not None else {}))
            if cot_texts is not None:
                action["_cot_text"] = cot_texts[i]
            results.append(action)
        _sync_if_cuda_available()
        t3 = time.monotonic()
        timing = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
        }
        if isinstance(model_timing, dict):
            timing.update(model_timing)
        return results, timing

    def infer_one(self, obs_dict: dict) -> dict:
        """Single obs convenience."""
        return self.infer([obs_dict])[0]

    def _prepare_branch_batch(self, obs_dicts: list[dict]) -> tuple[list[PreparedSample], dict]:
        """Prepare a batch for a MEM-Lite branch without invoking policy routing."""
        if not obs_dicts:
            return [], {}
        prepared = [self._prepare(obs_dict) for obs_dict in obs_dicts]
        padding_input_id = prepared[0].sub_processor.pad_token_id
        batch = self._collate([p.sample for p in prepared], padding_input_id=padding_input_id)
        return prepared, batch

    @staticmethod
    def _with_memlite_runtime_branch(
        obs_dicts: list[dict],
        *,
        branch: str,
        memory_texts: list[str] | None = None,
        intent_texts: list[str] | None = None,
    ) -> list[dict]:
        """Copy observations and satisfy the selected builder *before* preprocess.

        The configured ``MEMLiteMixedBuilder`` routes in
        ``GalaxeaCoTProcessor.preprocess``.  Consequently the branch marker
        and its live conditioning text must be present on the raw-data copy,
        not appended to the already-built RoboVQA sample.  Tensor processing
        mutates nested action / state mappings, so these are deep-copied too:
        the high branch must never contaminate the immediately following low
        branch's official observation.

        High-level intent / memory-update / status values are validation-only
        placeholders.  They occur after ``<EOC>`` and are stripped from the
        high prefill sample by the policy, so no future target is exposed to
        generation.  Low-level intent is the explicit current planner output;
        an empty value is rejected rather than replaced with a synthetic plan.
        """
        if branch not in {"high", "low"}:
            raise ValueError(f"Unknown MEM-Lite branch {branch!r}; expected high or low")
        if branch == "high":
            if memory_texts is None or len(memory_texts) != len(obs_dicts):
                raise ValueError("high-level memory_texts must align with obs_dicts")
        elif intent_texts is None or len(intent_texts) != len(obs_dicts):
            raise ValueError("low-level intent_texts must align with obs_dicts")

        copied = []
        for index, obs_dict in enumerate(obs_dicts):
            item = deepcopy(obs_dict)
            item["memlite_branch"] = branch
            if branch == "high":
                memory = str(memory_texts[index] or "")
                item["memory"] = memory if memory.strip() else _MEMLITE_EMPTY_MEMORY_SENTINEL
                item["intent"] = _MEMLITE_RUNTIME_TARGET_PLACEHOLDER
                item["atomic_task"] = _MEMLITE_RUNTIME_TARGET_PLACEHOLDER
                item["memory_update"] = _MEMLITE_RUNTIME_TARGET_PLACEHOLDER
                item["intent_status"] = "CONTINUE"
                item["status"] = "CONTINUE"
            else:
                intent = str(intent_texts[index] or "")
                if not intent.strip():
                    raise ValueError("MEM-Lite low-level inference requires a non-empty current intent")
                item["intent"] = intent
                item["atomic_task"] = intent
            copied.append(item)
        return copied

    def _branch_device(self, batch: dict) -> dict:
        """Move branch inputs to the configured device while preserving metadata."""
        return dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)

    def _branch_context(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def infer_high_level(
        self,
        obs_dicts: list[dict],
        memory_texts: list[str] | str | None = None,
        *,
        max_new_tokens: int = 160,
    ) -> list[dict]:
        """Run the independent MEM-Lite high-level branch."""
        if not hasattr(self.policy, "generate_high_level"):
            raise NotImplementedError("loaded policy has no MEM-Lite high-level API")
        if int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1 for MEM-Lite high-level inference")
        if isinstance(memory_texts, str):
            memories = [memory_texts] * len(obs_dicts)
        elif memory_texts is None:
            memories = [""] * len(obs_dicts)
        else:
            memories = list(memory_texts)
            if len(memories) != len(obs_dicts):
                raise ValueError(f"memory_texts batch size {len(memories)} != obs {len(obs_dicts)}")
            if memories and any(m != memories[0] for m in memories[1:]):
                raise ValueError("batched high-level inference requires identical memory_texts")
        branch_obs = self._with_memlite_runtime_branch(
            obs_dicts, branch="high", memory_texts=[str(memory or "") for memory in memories]
        )
        prepared, batch = self._prepare_branch_batch(branch_obs)
        if not prepared:
            return []
        if "samples" not in batch or "pixel_values" not in batch:
            raise ValueError("processor output lacks samples/pixel_values for high-level branch")
        memory_arg = str(memories[0] or "") if memories else ""
        branch_batch = self._branch_device(batch)
        with torch.no_grad(), self._branch_context():
            result = self.policy.generate_high_level(
                samples=branch_batch["samples"],
                pixel_values=branch_batch["pixel_values"],
                memory_text=memory_arg,
                max_new_tokens=int(max_new_tokens),
            )

        def item(key: str, index: int, default=""):
            value = result.get(key, default) if isinstance(result, dict) else default
            if isinstance(value, (list, tuple)):
                return value[index] if index < len(value) else default
            return value

        return [
            {
                "intent": str(item("intent", i, "") or ""),
                "memory": str(item("memory", i, "") or ""),
                "status": str(item("status", i, "CONTINUE") or "CONTINUE").upper(),
                "raw": str(item("high_level_text", i, "") or ""),
                "generation_metadata": item("high_level_generation_metadata", i, {}),
            }
            for i in range(len(obs_dicts))
        ]

    def infer_low_level_action(self, obs_dicts: list[dict], intent_texts: list[str] | str) -> list[dict]:
        """Run independent low-level AR action generation and postprocess output."""
        if not hasattr(self.policy, "generate_low_level_action"):
            raise NotImplementedError("loaded policy has no MEM-Lite low-level API")
        intents = [intent_texts] * len(obs_dicts) if isinstance(intent_texts, str) else list(intent_texts)
        if len(intents) != len(obs_dicts):
            raise ValueError(f"intent_texts batch size {len(intents)} != obs {len(obs_dicts)}")
        branch_obs = self._with_memlite_runtime_branch(
            obs_dicts, branch="low", intent_texts=[str(intent or "") for intent in intents]
        )
        prepared, batch = self._prepare_branch_batch(branch_obs)
        if not prepared:
            return []
        branch_batch = self._branch_device(batch)
        with torch.no_grad(), self._branch_context():
            result = self.policy.generate_low_level_action(
                samples=branch_batch["samples"], pixel_values=branch_batch["pixel_values"],
                intent_text=intents, action_dim_is_pad=branch_batch.get("action_dim_is_pad"),
                action_gt=branch_batch.get("action"),
            )
        if not isinstance(result, dict) or "action" not in result:
            raise ValueError("MEM-Lite low-level API returned no action")
        result_batch = dict_apply(result, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        result_batch.setdefault("proprio", branch_batch["proprio"].cpu())
        for key in ("action_dim_is_pad", "proprio_dim_is_pad"):
            if key in branch_batch and key not in result_batch:
                result_batch[key] = branch_batch[key].cpu() if isinstance(branch_batch[key], torch.Tensor) else branch_batch[key]
        return [self._postprocess_single(result_batch, i, sample.sub_processor,
                                        **({"raw_state_anchor": sample.raw_state_anchor}
                                           if getattr(sample, "raw_state_anchor", None) is not None else {}))
                for i, sample in enumerate(prepared)]

    def forward_batch(self, batch: dict) -> dict:
        """Core: to device → predict_action → to cpu."""
        batch = dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = self.policy.predict_action(batch)
        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        return batch

    # ──────── Private helpers ────────

    def _prepare(self, obs_dict: dict) -> PreparedSample:
        """Preprocess one request and annotate its batching key."""
        sub_processor = resolve_processor(self.processor, obs_dict)
        raw_state_anchor = None
        if any(getattr(transform, "supports_raw_state_anchor", False)
               for transform in (getattr(sub_processor, "action_state_transforms", None) or [])):
            # Preprocessing mutates nested state mappings and clips normalized
            # values. Preserve the true last observation BEFORE that mutation;
            # this private CPU metadata never enters the model or its KV cache.
            raw_state_anchor = {}
            for key, dim in (("left_arm", 7), ("right_arm", 7), ("trunk_qpos", 4)):
                value = torch.as_tensor(obs_dict["state"][key]).detach().cpu()
                if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != dim:
                    raise ValueError(f"Raw {key} anchor must be [time, {dim}], got {tuple(value.shape)}")
                if not torch.isfinite(value[-1]).all():
                    raise ValueError(f"Non-finite raw {key} action anchor")
                raw_state_anchor[key] = value[-1:].clone().unsqueeze(0)
        sample = sub_processor.preprocess(obs_dict)

        sample.pop("action", None)
        sample.pop("action_is_pad", None)
        sample.pop("gt_action", None)

        return PreparedSample(
            obs_dict=obs_dict,
            sample=sample,
            sub_processor=sub_processor,
            raw_state_anchor=raw_state_anchor,
        )

    @staticmethod
    def _collate(samples: list[dict], padding_input_id: int) -> dict:
        """Collate inference samples, padding text fields when present."""
        batch = [dict(sample) for sample in samples]
        first = batch[0]
        has_text_tokens = {"input_ids", "labels", "attention_mask"} <= set(first.keys())
        if has_text_tokens:
            return collate_fn_pad_sequences(batch, padding_input_id=padding_input_id)

        collated = custom_collate_fn(batch)
        if "samples" in first:
            collated["samples"] = [sample["samples"] for sample in batch]
        sample_meta_keys = ("idx", "task", "embodiment", "dataset_locator", "frequency")
        collated["sample_meta"] = [
            {key: sample.get(key) for key in sample_meta_keys if key in sample} for sample in batch
        ]
        return collated

    @staticmethod
    def _selected_action_source(batch: dict, index: int) -> str | None:
        """Read a scalar or per-sample action-source annotation from a model batch."""
        action_source = batch.get("selected_action_source")
        if isinstance(action_source, (list, tuple)):
            return action_source[index] if index < len(action_source) else None
        return action_source

    @staticmethod
    def _build_ar_presence_mask(
        action: torch.Tensor,
        action_dim_is_pad: torch.Tensor | None,
        absent_keys: set[str],
        sub_processor,
    ) -> torch.BoolTensor:
        """Map AR grouped-key presence into the postprocessor flat action mask.

        AR decoding reports absent *grouped* action keys (for example
        ``lower_body`` or ``left_control``). The postprocessor is the single
        source of truth for mapping those groups back to the embodiment raw
        wire keys, so send the group mask through that same inverse pipeline.
        """
        merger = getattr(sub_processor, "action_state_merger", None)
        parts_meta = getattr(merger, "max_action_shape_meta", None)
        if not isinstance(parts_meta, dict):
            raise ValueError(
                "AR-only inference needs action_state_merger.max_action_shape_meta "
                "to map absent action groups to wire keys."
            )

        dims = {key: int(dim) for key, dim in parts_meta.items()}
        action_dim = action.shape[-1]
        if sum(dims.values()) != action_dim:
            raise ValueError(
                "AR-only action dimension does not match action_state_merger parts metadata: "
                f"action_dim={action_dim}, metadata_dim={sum(dims.values())}."
            )

        # Accept either a grouped key (the normal tokenizer contract) or a raw
        # alternative key. The latter is resolved to its group where possible.
        aliases = {}
        for group in getattr(merger, "_action_layout", None) or ():
            aliases.update({raw_key: group.name for raw_key in group.part_names})
        grouped_absent = {aliases.get(key, key) for key in absent_keys}
        unknown = grouped_absent - set(dims)
        if unknown:
            raise ValueError(
                "AR decoder reported absent action keys not present in merger metadata: "
                f"{sorted(unknown)}."
            )

        mask = torch.ones((1, action_dim), dtype=torch.bool, device=action.device)
        offset = 0
        for key, dim in dims.items():
            if key in grouped_absent:
                mask[..., offset : offset + dim] = False
            offset += dim

        if action_dim_is_pad is not None:
            dim_is_pad = action_dim_is_pad.to(device=action.device, dtype=torch.bool)
            if dim_is_pad.ndim == 1:
                dim_is_pad = dim_is_pad.unsqueeze(0)
            mask &= ~dim_is_pad
        return mask

    @staticmethod
    def _raw_absent_keys(action_op_mask) -> set[str]:
        """Derive wire-key absence from the postprocessor raw mask dictionary."""
        if not isinstance(action_op_mask, dict):
            raise ValueError(
                "AR-only postprocess did not return a per-key action_op_mask; "
                "cannot safely map absent grouped actions to wire keys."
            )
        return {
            key
            for key, mask in action_op_mask.items()
            if isinstance(mask, torch.Tensor) and not mask.to(dtype=torch.bool).any()
        }

    @staticmethod
    def _postprocess_single(batch: dict, index: int, sub_processor, *, raw_state_anchor=None) -> dict:
        """Slice one sample from a batched model output and postprocess it."""
        item_batch = {
            "action": batch["action"][index : index + 1],
            "proprio": batch["proprio"][index : index + 1],
        }
        if raw_state_anchor is not None:
            item_batch["_raw_state_anchor"] = raw_state_anchor
        if "action_dim_is_pad" in batch:
            item_batch["action_dim_is_pad"] = batch["action_dim_is_pad"][index : index + 1]
        if "proprio_dim_is_pad" in batch:
            item_batch["proprio_dim_is_pad"] = batch["proprio_dim_is_pad"][index : index + 1]
        action_source = PolicyInferencer._selected_action_source(batch, index)
        ar_absent_keys = batch.get("ar_absent_keys") if action_source == "ar" else None
        ar_absent = (
            set(ar_absent_keys[index])
            if ar_absent_keys is not None and index < len(ar_absent_keys)
            else set()
        )
        if action_source == "ar":
            # Do not reuse a training/input mask here: AR absence describes the
            # generated grouped action, not the observation that seeded it.
            item_batch["action_op_mask"] = PolicyInferencer._build_ar_presence_mask(
                item_batch["action"],
                item_batch.get("action_dim_is_pad"),
                ar_absent,
                sub_processor,
            )
        elif "action_op_mask" in batch:
            item_batch["action_op_mask"] = batch["action_op_mask"][index : index + 1]
        postprocessed = sub_processor.postprocess(item_batch)
        action = postprocessed["action"]
        if postprocessed.get("_state_anchor_diagnostics"):
            action["_state_anchor_diagnostics"] = postprocessed["_state_anchor_diagnostics"]
        normalizer = getattr(sub_processor, "normalizer", None)
        diagnostics = {
            key: dict(norm.last_backward_diagnostics)
            for key, norm in getattr(normalizer, "normalizers", {}).get("action", {}).items()
            if getattr(norm, "last_backward_diagnostics", {})
        }
        if diagnostics:
            action["_normalization_diagnostics"] = diagnostics
        # ``ar_absent_keys`` only describes the discrete AR decode. In dual-head
        # mode the selected wire action is FM, so applying this list would erase
        # valid continuous controls (notably BEHAVIOR lower_body) from the FM
        # result. For AR-only output, derive names only *after* this inverse
        # transform so the wire protocol sees raw keys (base_qvel/trunk_qpos),
        # never an intermediate group name such as lower_body.
        absent = (
            PolicyInferencer._raw_absent_keys(postprocessed.get("action_op_mask"))
            if action_source == "ar"
            else set()
        )
        if absent:
            action["_absent_keys"] = absent
        return action
