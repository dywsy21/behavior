"""WebSocket policy server for MEM multi-frame inference with history buffers.

Based on serve_policy.py, with per-camera frame buffers (deque) for MEM video
encoders where obs_size > 1. Single-frame models are also supported with
deque maxlen=1, equivalent to the old expand behavior.

Usage:
    # Load config from the training run_dir.
    python scripts/serve_policy_mem.py \
        --ckpt_path /path/to/checkpoints/step_10000/model_state_dict.pt \
        --action_steps 15 \
        eval_embodiment=galaxea_r1lite

    # Manually specify a task config when the run_dir has no .hydra/config.yaml.
    python scripts/serve_policy_mem.py \
        --ckpt_path /path/to/model_state_dict.pt \
        --task_config configs/task/pretrain/10k_pretrain_full_AC2_2cb_qwen35_2b_mem.yaml \
        --action_steps 15 \
        eval_embodiment=galaxea_r1lite
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import functools
import logging
import time
from numbers import Integral

from g05.utils.websocket import packb, unpackb
import numpy as np
import torch
from omegaconf import OmegaConf

from pathlib import Path
from typing import Any

import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.data.normalizer import ActionNormalizationError
from g05.utils.memlite_protocol import MemLiteProtocolError, ActionDecodeError, parse_memlite_output
from g05.utils.common.pytorch_utils import dict_apply
from g05.data_processor.processor.mixture_processor import MixtureProcessor

from g05.utils.data.processor_utils import build_processors
from g05.utils.eval.eval_utils import filter_embodiment
from g05.utils.checkpoint.ckpt_utils import (
    find_run_dir,
    load_config_from_run_dir,
    load_config_from_task_yaml,
)
from g05.models.g05.inferencer import (
    PolicyInferencer,
    resolve_processor,
)

import websockets

logger = logging.getLogger(__name__)


def _resolve_stats_path(cfg) -> Path:
    """Resolve dataset stats path for both `last.pt` and `checkpoints/step_x.pt`."""
    run_dir = cfg.get("run_dir", None)
    if run_dir:
        candidate = Path(str(run_dir)) / "dataset_stats.json"
        if candidate.exists():
            return candidate

    ckpt_path = Path(cfg.ckpt_path)
    candidates = [
        ckpt_path.parent / "dataset_stats.json",
        ckpt_path.parent.parent / "dataset_stats.json",
    ]
    if cfg.get("datastatics_path", None):
        candidates.append(Path(str(cfg.datastatics_path)))

    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"dataset_stats.json not found (tried: {checked})")


# ──────── setup: model + processor initialization ────────


def _validate_predict_cot_templates(processor) -> None:
    """Validate only the high-level text prefix used by MEM-Lite inference.

    ``MEMLiteMixedBuilder`` deliberately has no universal template: its high
    branch emits text before ``<EOV>``, while its low branch must retain the
    action-only ``<EOV><EOC>`` order. Accessing the mixed builder's ``template``
    property raises by design, so validate its high branch directly.
    """
    processors = (
        list(processor.processors.values())
        if isinstance(processor, MixtureProcessor)
        else [processor]
    )
    invalid = []
    for sub_processor in processors:
        builder = getattr(sub_processor, "samples_builder", None)
        if type(builder).__name__ == "MEMLiteMixedBuilder":
            effective = getattr(builder, "_high", None)
            label = "MEMLiteMixedBuilder._high"
        else:
            eval_builder = getattr(builder, "_eval_builder", None)
            effective = eval_builder or builder
            label = type(effective).__name__ if effective is not None else "missing"
        try:
            template = getattr(effective, "template", "")
        except Exception as exc:
            invalid.append(f"{label} ({type(exc).__name__})")
            continue
        eoc_pos, eov_pos = template.find("<EOC>"), template.find("<EOV>")
        if eoc_pos < 0 or eov_pos < 0 or eoc_pos > eov_pos:
            invalid.append(label)
    if invalid:
        raise ValueError(
            "predict_cot=true requires a high-level text template whose <EOC> "
            "precedes <EOV>. Invalid builders: " + ", ".join(invalid)
        )


def _configure_action_execution_start_index(processor, cfg) -> int | None:
    """Apply a checkpoint's real action-history length to all active processors.

    Observation history and action history are independent.  Older saved
    configurations omit ``past_action_size`` and retain the historical
    postprocess convention by leaving the processor setting unset.  A config
    which declares action history, however, must use that exact value when
    choosing the first decoded row to execute.
    """
    data_cfg = cfg.get("data", None)
    raw_start = data_cfg.get("past_action_size", None) if data_cfg is not None else None
    processors = (
        list(processor.processors.values())
        if isinstance(processor, MixtureProcessor)
        else [processor]
    )
    if raw_start is None:
        for sub_processor in processors:
            sub_processor.set_action_execution_start_index(None)
        logger.info(
            "Action execution start unset: using legacy num_obs_steps - 1 convention"
        )
        return None
    if isinstance(raw_start, bool) or not isinstance(raw_start, Integral):
        raise TypeError(
            "data.past_action_size must be an integer when provided, "
            f"got {raw_start!r}"
        )
    start = int(raw_start)
    horizon = int(data_cfg.action_size)
    if not 0 <= start < horizon:
        raise ValueError(
            "data.past_action_size must select a decoded row inside action_size: "
            f"past_action_size={start}, action_size={horizon}"
        )
    for sub_processor in processors:
        sub_processor.set_action_execution_start_index(start)
    logger.info(
        "Action execution start configured from data.past_action_size=%d "
        "(decoded_horizon=%d)",
        start,
        horizon,
    )
    return start


def setup(cfg, device: str = "cuda"):
    """Load policy and processor, aligned with eval_open_loop.py:362-413."""
    model = load_model_from_checkpoint(
        cfg.model.model_arch,
        cfg.ckpt_path,
        device=device,
        extra_prefixes=["normalizer."],
        eval_mode=False,
    )

    if cfg.model.get("model_weights_to_bf16", True):
        model = model.to(torch.bfloat16)

    model.apply_fp32_params()

    policy = model.eval()
    if hasattr(policy, "action_tokenizer"):
        policy.action_tokenizer.to(device)

    if cfg.model.get("use_torch_compile", False):
        logger.info("Compiling model with torch.compile (first inference will be slow)...")
        policy = torch.compile(policy, mode="max-autotune")

    stats_path = _resolve_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(stats_path)

    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(dataset_stats)
    processor.eval()

    # A CoT/MEM inference prefix must place <EOC> before <EOV>.  The default
    # BaseSamplesBuilder places <EOV> before <EOC> because it is an action-only
    # template; using predict_cot=true with that template would make the text
    # decoder consume action tokens as if they were CoT.  Fail early instead of
    # returning silently corrupted AR actions.
    if bool(cfg.model.model_arch.get("predict_cot", False)):
        _validate_predict_cot_templates(processor)

    from g05.data_processor.transforms.action_filter import BaseActionFilter

    def _neutralize_action_filter(p):
        safe_filter = BaseActionFilter()
        safe_filter.set_shape_meta(p.shape_meta)
        p.action_filter = safe_filter

    if isinstance(processor, MixtureProcessor):
        for emb_name in processor.processors:
            _neutralize_action_filter(processor.processors[emb_name])
    else:
        _neutralize_action_filter(processor)

    action_horizon = int(cfg.data.action_size)
    if isinstance(processor, MixtureProcessor):
        for emb_name, sub_processor in processor.processors.items():
            sub_processor.action_horizon = action_horizon
    else:
        processor.action_horizon = action_horizon
    _configure_action_execution_start_index(processor, cfg)
    runtime = OmegaConf.to_container(cfg.memlite_runtime, resolve=True) if cfg.get("memlite_runtime") else {}
    processor.memlite_runtime = runtime
    if runtime.get("schema_version", 0) >= 5:
        stride = float(cfg.data.obs_stride_second)
        if abs(stride * 30 - int(runtime["action_steps"])) > 1e-6:
            raise ValueError("MEM-Lite runtime action cadence must match training history stride")
        children = list(processor.processors.values()) if isinstance(processor, MixtureProcessor) else [processor]
        if any(getattr(p, "image_history_mode", None) != "preserve_cameras" for p in children):
            raise ValueError("Causal MEM-Lite requires preserved camera histories")

    return policy, processor


# ──────── input validation ────────


def _parse_task_and_plan(raw_obs: dict) -> None:
    """Parse a [PLAN] marker from task and split it into task + plan."""
    task_str = raw_obs.get("task", "")
    if "[PLAN]" in task_str:
        parts = task_str.split("[PLAN]", 1)
        raw_obs["task"] = parts[0].strip()
        raw_obs["plan"] = parts[1].strip()


def _validate_obs(raw_obs: dict, processor) -> None:
    """Validate that client raw_obs satisfies the raw_shape protocol."""
    p = resolve_processor(processor, raw_obs)
    sm = p.shape_meta

    for required in ("images", "state", "task"):
        if required not in raw_obs:
            raise ValueError(f"raw_obs is missing required field '{required}'")

    if (
        isinstance(processor, MixtureProcessor)
        and len(processor.processors) > 1
        and "embodiment_type" not in raw_obs
    ):
        raise ValueError("Mixture mode requires raw_obs to contain an 'embodiment_type' field")

    if not isinstance(raw_obs["task"], str):
        raise ValueError(f"raw_obs['task'] must be str, got {type(raw_obs['task'])}")

    if "frequency" in raw_obs and not isinstance(raw_obs["frequency"], (int, float, np.number)):
        raise ValueError(
            f"raw_obs['frequency'] must be numeric, got {type(raw_obs['frequency'])}"
        )

    if "coarse_task" in raw_obs and not isinstance(raw_obs["coarse_task"], str):
        raise ValueError(
            f"raw_obs['coarse_task'] must be str, got {type(raw_obs['coarse_task'])}"
        )

    if "plan" in raw_obs and not isinstance(raw_obs["plan"], str):
        raise ValueError(f"raw_obs['plan'] must be str, got {type(raw_obs['plan'])}")

    expected_img_keys = {m["key"] for m in sm["images"]}
    actual_img_keys = set(raw_obs["images"].keys())
    missing = expected_img_keys - actual_img_keys
    if missing:
        raise ValueError(f"images is missing keys defined by shape_meta: {missing}")
    extra = actual_img_keys - expected_img_keys
    if extra:
        logger.warning(f"images contains keys not defined by shape_meta and they will be ignored: {extra}")

    for m in sm["images"]:
        img = raw_obs["images"][m["key"]]
        if not isinstance(img, np.ndarray):
            raise ValueError(f"images['{m['key']}'] must be np.ndarray, got {type(img)}")
        if img.ndim != 3:
            raise ValueError(f"images['{m['key']}'] must be [C,H,W] (3D), got shape={img.shape}")
        if img.dtype != np.uint8:
            logger.warning(f"images['{m['key']}'] dtype={img.dtype}, expected uint8")

    expected_state_keys = {m["key"] for m in sm["state"]}
    actual_state_keys = set(raw_obs["state"].keys())
    missing = expected_state_keys - actual_state_keys
    if missing:
        raise ValueError(f"state is missing keys defined by shape_meta: {missing}")
    extra = actual_state_keys - expected_state_keys
    if extra:
        logger.warning(f"state contains keys not defined by shape_meta and they will be ignored: {extra}")

    for m in sm["state"]:
        s = raw_obs["state"][m["key"]]
        if not isinstance(s, np.ndarray):
            raise ValueError(f"state['{m['key']}'] must be np.ndarray, got {type(s)}")
        if s.ndim != 1:
            raise ValueError(f"state['{m['key']}'] must be [D] (1D), got shape={s.shape}")
        expected_dim = m["raw_shape"]
        if s.shape[0] != expected_dim:
            raise ValueError(
                f"state['{m['key']}'] dim={s.shape[0]}, expected raw_shape {expected_dim}"
            )


# ──────── obs -> preprocess input construction ────────


def build_obs_dict(raw_obs: dict, processor, frame_buffers=None, state_buffers=None, mem_state=None, event=None) -> dict:
    """Build the processor.preprocess input from client raw observations.

    Data format is aligned with base_lerobot_dataset.py:304-323 __getitem__ output.
    Client input always uses raw_shape; shape appears only after processor-internal
    transforms.

    When frame_buffers/state_buffers are provided in multi-frame MEM mode, take
    K historical frames from the buffers. Otherwise, fall back to single-frame
    expand, preserving old obs_size=1 behavior.
    """
    _validate_obs(raw_obs, processor)

    p = resolve_processor(processor, raw_obs)
    num_obs_steps = p.num_obs_steps
    action_horizon = getattr(p, "action_horizon", num_obs_steps)

    expected_img_keys = {m["key"] for m in p.shape_meta["images"]}
    expected_state_keys = {m["key"] for m in p.shape_meta["state"]}

    if frame_buffers and state_buffers:
        images = {
            k: torch.stack(list(frame_buffers[k])) for k in expected_img_keys if k in frame_buffers
        }
        state = {
            k: torch.stack(list(state_buffers[k]))
            for k in expected_state_keys
            if k in state_buffers
        }
    else:
        images = {
            k: torch.from_numpy(np.array(v, copy=True))
            .unsqueeze(0)
            .expand(num_obs_steps, -1, -1, -1)
            for k, v in raw_obs["images"].items()
            if k in expected_img_keys
        }
        state = {
            k: torch.from_numpy(np.array(v, copy=True))
            .unsqueeze(0)
            .expand(num_obs_steps, -1)
            .float()
            for k, v in raw_obs["state"].items()
            if k in expected_state_keys
        }

    data = {
        "images": images,
        "state": state,
        "task": raw_obs["task"],
        "action": {
            m["key"]: torch.zeros(action_horizon, m["raw_shape"]) for m in p.shape_meta["action"]
        },
        "action_is_pad": torch.ones(action_horizon, dtype=torch.bool),
        "state_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
        "image_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
        "idx": 0,
    }
    if "coarse_task" in raw_obs:
        data["coarse_task"] = raw_obs["coarse_task"]
    if "plan" in raw_obs:
        data["plan"] = raw_obs["plan"]
    if mem_state is not None:
        data["memory"] = str(getattr(mem_state, "memory_text", "") or "")
        data["intent"] = str(getattr(mem_state, "intent_text", "") or "")
        data["previous_intent"] = data["intent"] or "None"
        data["intent_status"] = str(getattr(mem_state, "status", "INVALID") or "INVALID")
    elif "memory" in raw_obs or "intent" in raw_obs:
        data["memory"] = str(raw_obs.get("memory", "") or "")
        data["intent"] = str(raw_obs.get("intent", "") or "")
    if event:
        data["event"] = str(event)
    if "execution_feedback" in raw_obs:
        data["execution_feedback"] = str(raw_obs["execution_feedback"])
    frequency = raw_obs.get("frequency", 15)
    data["frequency"] = float(frequency)
    if isinstance(processor, MixtureProcessor):
        data["embodiment"] = raw_obs.get("embodiment_type") or next(iter(processor.processors))
    return data


# ──────── WebSocket handler ────────


@dataclasses.dataclass
class MemLiteEpisodeState:
    """Connection-local semantic state; never shared across episodes."""
    memory_text: str = ""
    intent_text: str = ""
    status: str = "INVALID"
    memory_initialized: bool = False
    task_id: int | None = None
    failure_count: int = 0
    high_level_generation: int = 0
    chunks_since_intent: int = 0
    last_memory_update: str = ""
    last_high_level_raw: str = ""
    def reset(self):
        self.memory_text = ""
        self.intent_text = ""
        self.status = "INVALID"
        self.memory_initialized = False
        self.task_id = None
        self.failure_count = 0
        self.high_level_generation = 0
        self.chunks_since_intent = 0
        self.last_memory_update = ""
        self.last_high_level_raw = ""


def _serialize_error(code: int, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _high_level_format_valid(raw: str) -> bool:
    """Telemetry uses exactly the same validator as proposal admission."""
    try:
        parse_memlite_output(raw)
    except MemLiteProtocolError:
        return False
    return True


class ActionAdmissionError(ValueError):
    """A decoded proposal is invalid; hold instead of executing it."""


class ChunkedPolicyWrapper:
    """Wraps model inference with chunk caching, step-by-step serving, and frame buffers.

    Maintains per-camera frame buffers (deque) for multi-frame models (MEM
    video encoder, obs_size > 1). When obs_size == 1, buffers hold exactly
    one frame, equivalent to the old expand() behavior.

    One instance per client connection.
    """

    # A high-level status marks the current intent as complete (or no longer
    # valid). We still execute the low-level branch once for the observation
    # that produced this status, then request a fresh high-level plan on the
    # next request. Keeping this set explicit avoids accidentally treating
    # arbitrary status strings as a replan trigger.
    _TERMINAL_HL_STATUSES = frozenset({"DONE", "FAILED", "REPLAN"})

    def __init__(
        self,
        inferencer,
        processor,
        *,
        action_steps: int = 16,
        strict_memlite: bool = False,
        memlite_replan_every_chunks: int = 0,
        memlite_high_level_max_new_tokens: int = 160,
        trace_hook=None,
    ):
        if memlite_replan_every_chunks < 0:
            raise ValueError("memlite_replan_every_chunks must be >= 0")
        if memlite_high_level_max_new_tokens < 1:
            raise ValueError("memlite_high_level_max_new_tokens must be >= 1")
        self.inferencer = inferencer
        self.processor = processor
        self.action_steps = action_steps
        self.memlite_replan_every_chunks = int(memlite_replan_every_chunks)
        self.memlite_high_level_max_new_tokens = int(memlite_high_level_max_new_tokens)
        self._trace_hook = trace_hook
        runtime = getattr(processor, "memlite_runtime", {})
        self._causal_runtime = int(runtime.get("schema_version", 0)) >= 5
        self._runtime = runtime
        from g05.utils.memlite_progress import MotionProgressMonitor
        self._progress_monitor = MotionProgressMonitor(runtime.get("progress_monitor"))
        self._progress_hold_pending = False
        self._terminal_hold_chunks = 0
        self._last_gripper_commands = {}
        self._chunk_is_hold = False
        if self._causal_runtime:
            if action_steps != int(runtime["action_steps"]):
                raise ValueError("action_steps disagrees with trained MEM-Lite history cadence")
            if not runtime.get("require_explicit_grippers", True):
                raise ValueError("Causal R1Pro MEM-Lite cannot disable explicit gripper admission")
            if int(runtime["replan_every_chunks"]) < 1 or int(runtime.get("terminal_recheck_every_chunks", 4)) < 1:
                raise ValueError("Causal MEM-Lite replan/recheck cadence must be positive")
            self.memlite_replan_every_chunks = int(runtime["replan_every_chunks"])
            self.memlite_high_level_max_new_tokens = int(runtime["high_level_max_new_tokens"])
        self._cached_chunk = None
        self._cached_action_horizon: int | None = None
        self._cached_action_execution_start_index: int | None = None
        self._chunk_step = 0
        self._chunk_index = 0
        self._request_count = 0
        self._served_action_count = 0
        self._cot_text = None
        self.mem_state = MemLiteEpisodeState()
        self._high_level_pending = True
        self._last_high_level_reason = "start"
        self._last_memlite_updated = False
        # Evaluation must never silently substitute the legacy one-stage route
        # for a failed high-/low-level MEM-Lite decision.
        self._strict_memlite = strict_memlite
        model = getattr(inferencer, "policy", None)
        model_cfg = getattr(model, "model_config", None)
        self._memlite_enabled = bool(
            hasattr(inferencer, "infer_high_level") and hasattr(inferencer, "infer_low_level_action")
            and (bool(getattr(model, "predict_cot", False)) or bool(getattr(model_cfg, "predict_cot", False)))
        )

        # Frame buffer for multi-frame MEM video encoder
        if isinstance(processor, MixtureProcessor):
            p = next(iter(processor.processors.values()))
        else:
            p = processor
        self._num_obs_steps = p.num_obs_steps
        self._cam_keys = [m["key"] for m in p.shape_meta["images"]]
        self._state_keys = [m["key"] for m in p.shape_meta["state"]]
        self._frame_buffers: dict[str, collections.deque] = {
            k: collections.deque(maxlen=self._num_obs_steps) for k in self._cam_keys
        }
        self._state_buffers: dict[str, collections.deque] = {
            k: collections.deque(maxlen=self._num_obs_steps) for k in self._state_keys
        }
        self._buffers_initialized = False
        if self._num_obs_steps > 1:
            logger.info(
                "Frame buffer enabled: obs_size=%d, cameras=%s",
                self._num_obs_steps,
                self._cam_keys,
            )

    @property
    def need_obs(self) -> bool:
        """Whether the next call to get_action needs a real observation."""
        # A terminal or periodic replan is deliberately deferred until the
        # current cached action chunk has been consumed. Advertising
        # ``need_obs`` immediately would make the evaluator discard the rest
        # of that chunk after its first action.
        return self._cached_chunk is None or self._chunk_step >= self.action_steps

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    @property
    def served_action_count(self) -> int:
        return self._served_action_count

    @property
    def action_execution_start_index(self) -> int | None:
        """Execution offset actually used for the currently cached chunk."""
        return self._cached_action_execution_start_index

    @property
    def decoded_action_horizon(self) -> int | None:
        """Postprocess action horizon observed before chunk serving."""
        return self._cached_action_horizon

    def _trace(self, event: str, **fields) -> None:
        """Emit diagnostics without allowing telemetry to alter policy behavior."""
        if self._trace_hook is None:
            return
        try:
            self._trace_hook(
                {
                    "source": "policy",
                    "event": event,
                    "request_count": self._request_count,
                    "chunk_index": self._chunk_index,
                    "chunk_position": self._chunk_step,
                    "served_action_count": self._served_action_count,
                    **fields,
                }
            )
        except Exception:
            logger.exception("MEM-Lite trace hook failed; preserving policy execution")

    def _invalidate_chunk(self, reason: str):
        self._cached_chunk = None
        self._chunk_step = 0
        self._high_level_pending = True
        self._last_high_level_reason = reason

    def _event_from_obs(self, raw_obs: dict):
        event = raw_obs.get("event") or raw_obs.get("intent_status")
        if raw_obs.get("replan"):
            event = "replan"
        if event is None:
            return None
        event = str(event).strip().lower()
        return {"complete": "done", "success": "done", "fail": "failed", "retry": "replan"}.get(event, event)

    def _handle_event(self, raw_obs: dict):
        event = self._event_from_obs(raw_obs)
        if event not in {"done", "failed", "timeout", "replan"}:
            return event
        if event == "failed":
            self.mem_state.failure_count += 1
        self.mem_state.status = event.upper()
        self._invalidate_chunk(event)
        return event

    def initialize_memlite_memory(
        self,
        *,
        task_id: int,
        canonical_memory: str,
        explicit_memory: str | None = None,
    ) -> bool:
        """Set an episode's initial memory exactly once before its first HL call.

        The BEHAVIOR bridge owns task identity and calls this method only after
        an official reset / connection start, or after it has reset a detected
        task switch.  A non-empty explicitly supplied memory wins over the
        canonical task prior.  Once initialized, even a later model-produced
        empty memory is preserved as state and never silently re-initialized.
        """
        if not self._memlite_enabled:
            return False
        normalized_task_id = int(task_id)
        if self.mem_state.memory_initialized:
            if self.mem_state.task_id != normalized_task_id:
                raise ValueError(
                    "MEM-Lite task identity changed without an episode reset: "
                    f"{self.mem_state.task_id} -> {normalized_task_id}"
                )
            return False
        existing = str(self.mem_state.memory_text or "").strip()
        supplied = str(explicit_memory or "").strip()
        canonical = str(canonical_memory or "").strip()
        if existing:
            chosen, source = existing, "existing_state"
        elif supplied:
            chosen, source = supplied, "explicit_input"
        elif canonical:
            chosen, source = canonical, "task_id_canonical"
        else:
            raise ValueError(
                "MEM-Lite initial memory is required before high-level inference"
            )
        self.mem_state.memory_text = chosen
        self.mem_state.memory_initialized = True
        self.mem_state.task_id = normalized_task_id
        self._trace(
            "memory_initialized",
            task_id=normalized_task_id,
            source=source,
            memory=chosen,
        )
        return True

    async def get_action(self, raw_obs: dict) -> tuple[dict, str | None]:
        self._request_count += 1
        event = self._handle_event(raw_obs)
        self._last_memlite_updated = False
        high_status = None
        # Replan only at the next chunk recomputation, never in the middle of
        # a cached chunk. ``chunks_since_intent`` counts low-level chunks
        # produced under the current high-level intent.
        if (
            self._memlite_enabled
            and self.need_obs
            and not self._high_level_pending
            and self.memlite_replan_every_chunks > 0
            and self.mem_state.chunks_since_intent >= self.memlite_replan_every_chunks
        ):
            self._high_level_pending = True
            self._last_high_level_reason = "periodic"
            self._trace(
                "periodic_replan_pending",
                every_chunks=self.memlite_replan_every_chunks,
                chunks_since_intent=self.mem_state.chunks_since_intent,
                trigger="periodic",
            )
        self._trace(
            "request",
            need_obs=self.need_obs,
            high_level_pending=self._high_level_pending,
            event_input=event,
            chunks_since_intent=self.mem_state.chunks_since_intent,
        )
        if self.need_obs:
            if not raw_obs:
                raise ValueError("need obs for recompute but got empty")
            t0=time.monotonic(); _parse_task_and_plan(raw_obs); t1=time.monotonic()
            self._update_buffers(raw_obs); t2=time.monotonic()
            if self._progress_monitor.cfg.enabled:
                progress_event = self._progress_monitor.observe(
                    executed_steps=self._served_action_count, velocity=raw_obs["state"]["base_qvel"],
                    head_rgb=raw_obs["images"]["head_rgb"], intent=self.mem_state.intent_text)
                raw_obs = {**raw_obs, "execution_feedback": self._progress_monitor.feedback}
                if progress_event is not None:
                    self._progress_hold_pending = True
                    self._high_level_pending = True
                    self._last_high_level_reason = "observed_repeated_rotation"
                    self._trace("no_progress_detected", **progress_event)
            obs_dict=build_obs_dict(raw_obs,self.processor,frame_buffers=self._frame_buffers,state_buffers=self._state_buffers,mem_state=self.mem_state if self._memlite_enabled else None,event=event); t3=time.monotonic()
            if self._memlite_enabled and self._causal_runtime:
                actions = await self._causal_actions(obs_dict, raw_obs)
            elif self._memlite_enabled:
                try:
                    if self._high_level_pending or not self.mem_state.intent_text:
                        trigger = self._last_high_level_reason
                        input_memory = self.mem_state.memory_text
                        high=await asyncio.to_thread(
                            self.inferencer.infer_high_level,
                            [obs_dict],
                            [input_memory],
                            max_new_tokens=self.memlite_high_level_max_new_tokens,
                        )
                        if high:
                            item=high[0]
                            self.mem_state.intent_text=str(item.get("intent","") or "")
                            self.mem_state.memory_text=str(item.get("memory","") or "")
                            self.mem_state.last_memory_update=str(
                                item.get("memory_update", self.mem_state.memory_text) or ""
                            )
                            self.mem_state.last_high_level_raw=str(item.get("raw", "") or "")
                            self.mem_state.status=str(item.get("status","CONTINUE") or "CONTINUE").upper()
                            high_status = self.mem_state.status
                            self.mem_state.high_level_generation+=1; self.mem_state.chunks_since_intent=0
                            self._high_level_pending=False; self._last_memlite_updated=True
                            self._trace(
                                "high_level",
                                input_memory=input_memory,
                                output_intent=self.mem_state.intent_text,
                                memory_update=self.mem_state.last_memory_update,
                                output_memory=self.mem_state.memory_text,
                                intent_status=self.mem_state.status,
                                raw=self.mem_state.last_high_level_raw,
                                format_valid=_high_level_format_valid(self.mem_state.last_high_level_raw),
                                generation_metadata=item.get("generation_metadata", {}),
                                trigger=trigger,
                                generation=self.mem_state.high_level_generation,
                            )
                        else:
                            raise ValueError("MEM-Lite high-level API returned no result")
                    actions=await asyncio.to_thread(self.inferencer.infer_low_level_action,[obs_dict],[self.mem_state.intent_text])
                except Exception as exc:
                    self._trace(
                        "policy_exception",
                        stage="memlite_high_or_low",
                        exception_type=type(exc).__name__,
                        message=str(exc),
                        strict_memlite=self._strict_memlite,
                    )
                    if self._strict_memlite:
                        raise RuntimeError(
                            "MEM-Lite high-/low-level inference failed in strict mode"
                        ) from exc
                    logger.warning("MEM-Lite unavailable; falling back to infer(): %s",exc)
                    self._memlite_enabled=False
                    actions=await asyncio.to_thread(self.inferencer.infer,[obs_dict])
            else:
                actions=await asyncio.to_thread(self.inferencer.infer,[obs_dict])
            t4=time.monotonic(); action=actions[0]; self._cot_text=action.pop("_cot_text",None)
            diagnostics = action.pop("_normalization_diagnostics", {})
            if diagnostics:
                self._trace("action_normalization", parts=diagnostics)
            state_anchor_diagnostics = action.pop("_state_anchor_diagnostics", {})
            if state_anchor_diagnostics:
                self._trace("state_anchor", parts=state_anchor_diagnostics)
            absent_keys=action.pop("_absent_keys",set())
            for key in absent_keys: action.pop(key,None)
            self._cached_chunk=dict_apply(action,lambda x:x[0].numpy() if isinstance(x,torch.Tensor) else x)
            cached_horizons = {}
            for part, values in self._cached_chunk.items():
                values = np.asarray(values)
                if values.ndim < 2:
                    raise ValueError(
                        "Postprocessed action part must be [horizon, dim] before chunk serving: "
                        f"{part} shape={values.shape}"
                    )
                cached_horizons[str(part)] = int(values.shape[0])
            if not cached_horizons:
                raise ValueError("Postprocess returned no action parts for chunk serving")
            if len(set(cached_horizons.values())) != 1:
                raise ValueError(
                    "Postprocessed action parts disagree on serving horizon: "
                    f"{cached_horizons}"
                )
            self._cached_action_horizon = next(iter(cached_horizons.values()))
            if self._cached_action_horizon < self.action_steps:
                raise ValueError(
                    "Postprocessed action horizon is shorter than requested action_steps: "
                    f"horizon={self._cached_action_horizon}, action_steps={self.action_steps}"
                )
            active_processor = resolve_processor(self.processor, obs_dict)
            configured_start = getattr(active_processor, "action_execution_start_index", None)
            self._cached_action_execution_start_index = (
                int(configured_start) if configured_start is not None else None
            )
            self._chunk_step=0
            self._chunk_index += 1
            if self._memlite_enabled:
                self.mem_state.chunks_since_intent+=1
                self._trace(
                    "hold_chunk" if self._chunk_is_hold else "low_level_chunk",
                    intent=self.mem_state.intent_text,
                    intent_status=self.mem_state.status,
                    chunks_since_intent=self.mem_state.chunks_since_intent,
                    action_keys=sorted(self._cached_chunk),
                    action_execution_start_index=self._cached_action_execution_start_index,
                    decoded_action_horizon=self._cached_action_horizon,
                    postprocess_action_horizon=self._cached_action_horizon,
                    executed_action_steps=self.action_steps,
                )
                # Keep the current low-level chunk even for terminal planner
                # statuses. Mark pending only after caching it, so there is no
                # same-request planning loop and the next request replans.
                if high_status in self._TERMINAL_HL_STATUSES:
                    self._high_level_pending = True
                    self._last_high_level_reason = f"status:{high_status.lower()}"
            logger.info("Recompute: %.1fms total | buffers=%.1fms build_obs=%.1fms infer=%.1fms post=%.1fms",(time.monotonic()-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000,(time.monotonic()-t4)*1000)
        single_step={}
        for part,arr in self._cached_chunk.items():
            single_step[part]=arr[self._chunk_step] if arr.ndim>=1 and self._chunk_step<arr.shape[0] else (arr[0] if arr.ndim>=1 else arr)
        self._chunk_step+=1
        self._served_action_count += 1
        if self._causal_runtime:
            for key in ("left_gripper", "right_gripper"):
                if key in single_step:
                    self._last_gripper_commands[key] = np.asarray(single_step[key]).copy()
        self._trace(
            "served_action",
            low_level_intent=self.mem_state.intent_text if self._memlite_enabled else None,
            action_keys=sorted(single_step),
            high_level_pending=self._high_level_pending,
            action_execution_start_index=self._cached_action_execution_start_index,
            decoded_action_horizon=self._cached_action_horizon,
            postprocess_action_horizon=self._cached_action_horizon,
            executed_action_steps=self.action_steps,
        )
        return single_step,self._cot_text

    def _safe_hold(self, raw_obs, reason):
        """R1Pro target-position hold; retain LAST SENT gripper command.

        State-width conversion is used only before any command was sent.
        There is no official success signal or hidden goal in this policy.
        """
        state = raw_obs["state"]
        output = {}
        for key in ("base_qvel", "trunk_qpos", "left_arm", "right_arm", "left_gripper", "right_gripper"):
            if key == "base_qvel":
                value = np.zeros(3, dtype=np.float32)
            elif key.endswith("gripper"):
                value = self._last_gripper_commands.get(key)
                if value is None:
                    value = np.array([np.clip(2.0 * float(np.asarray(state[key]).sum()) / 0.1 - 1.0, -1, 1)], dtype=np.float32)
            else:
                value = np.asarray(state[key], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError("Non-finite proprioception prevents a safe hold")
            output[key] = torch.as_tensor(np.asarray(value).copy()).reshape(1, 1, -1).repeat(1, self.action_steps, 1)
        self._chunk_is_hold = True
        self._trace("action_hold", reason=reason, intent_status=self.mem_state.status,
                    observed_state={key: np.asarray(value).tolist() for key, value in state.items()})
        return [output]

    def _admit_action(self, action):
        absent = set(action.get("_absent_keys", set()))
        expected = {"base_qvel": 3, "trunk_qpos": 4, "left_arm": 7,
                    "right_arm": 7, "left_gripper": 1, "right_gripper": 1}
        # v9 has no noop-dropout. Any absent group is a decode failure,
        # especially an absent constant gripper target.
        if absent or not expected.keys() <= action.keys():
            raise ActionAdmissionError(f"Missing explicit action groups: {sorted(absent | (expected.keys() - action.keys()))}")
        diagnostics = action.get("_normalization_diagnostics", {})
        self._trace("action_proposal", normalization=diagnostics)
        if any(float(d.get("max_normalized_excess", 0)) > 1.0 for d in diagnostics.values()):
            raise ActionAdmissionError("Decoded action exceeds train-only support by >1 normalized unit")
        for key, dim in expected.items():
            tensor = action[key]
            values = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
            if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < self.action_steps or values.shape[2] != dim:
                raise ActionAdmissionError(f"Invalid {key} shape {values.shape}")
            limit = 1.001 if key.endswith("gripper") else 6.5
            if not np.isfinite(values).all() or np.max(np.abs(values)) > limit:
                raise ActionAdmissionError(f"Non-finite or implausible absolute {key} target")

    async def _causal_actions(self, obs_dict, raw_obs):
        self._chunk_is_hold = False
        if not self.mem_state.memory_initialized:
            raise ValueError("Initialize canonical task memory before causal MEM-Lite inference")
        if self._progress_hold_pending:
            self._progress_hold_pending = False
            self._high_level_pending = True
            self._last_high_level_reason = "recover_after_rotation_pause"
            return self._safe_hold(raw_obs, "observed_repeated_rotation_pause")
        try:
            if self.mem_state.status == "DONE" and not self._high_level_pending:
                self._terminal_hold_chunks += 1
                if self._terminal_hold_chunks < int(self._runtime.get("terminal_recheck_every_chunks", 4)):
                    return self._safe_hold(raw_obs, "done_wait_for_recheck")
                self._high_level_pending = True
                self._last_high_level_reason = "done_recheck"
            if self._high_level_pending or not self.mem_state.intent_text:
                input_memory, previous_intent = self.mem_state.memory_text, self.mem_state.intent_text
                high = await asyncio.to_thread(self.inferencer.infer_high_level, [obs_dict], [input_memory],
                                               max_new_tokens=self.memlite_high_level_max_new_tokens)
                if not high or len(high) != 1:
                    raise MemLiteProtocolError("Expected one high-level proposal")
                item = high[0]
                proposal = parse_memlite_output(item.get("raw", ""))
                if item.get("generation_metadata", {}).get("budget_exhausted", False):
                    raise MemLiteProtocolError("High-level generation exhausted its budget")
                if self.mem_state.task_id is not None and not proposal["memory"].startswith(f"Task={self.mem_state.task_id};"):
                    raise MemLiteProtocolError("High-level memory changed the task identity")
                # Commit all fields only after the entire proposal passes.
                self.mem_state.intent_text = proposal["intent"]
                self.mem_state.memory_text = proposal["memory"]
                self.mem_state.status = proposal["status"]
                self.mem_state.last_memory_update = proposal["memory"]
                self.mem_state.last_high_level_raw = proposal["raw"]
                self.mem_state.high_level_generation += 1
                self.mem_state.chunks_since_intent = 0
                self._terminal_hold_chunks = 0
                self._last_memlite_updated = True
                self._high_level_pending = False
                if self._progress_monitor.acknowledge_recovery_exit(
                        previous_intent=previous_intent, next_intent=proposal["intent"], status=proposal["status"]):
                    # The high-level input still contained the observed alert.
                    # Only subsequent low/high calls see its acknowledged state.
                    obs_dict["execution_feedback"] = "none"
                    raw_obs["execution_feedback"] = "none"
                    self._trace("recovery_feedback_acknowledged", previous_intent=previous_intent,
                                next_intent=proposal["intent"], reason="planner_exit_and_observed_base_stop")
                self._trace("high_level", input_memory=input_memory, previous_intent=previous_intent,
                            output_intent=proposal["intent"], output_memory=proposal["memory"],
                            memory_update=proposal["memory"], intent_status=proposal["status"], raw=proposal["raw"],
                            format_valid=True, trigger=self._last_high_level_reason,
                            generation_metadata=item.get("generation_metadata", {}))
            if self.mem_state.status != "CONTINUE":
                if self.mem_state.status != "DONE":
                    self._high_level_pending = True
                    self._last_high_level_reason = f"status:{self.mem_state.status.lower()}"
                return self._safe_hold(raw_obs, f"non_executable_status:{self.mem_state.status}")
            actions = await asyncio.to_thread(self.inferencer.infer_low_level_action, [obs_dict], [self.mem_state.intent_text])
            if not actions or len(actions) != 1:
                raise ActionAdmissionError("Expected one low-level action proposal")
            self._admit_action(actions[0])
            return actions
        except (MemLiteProtocolError, ActionDecodeError, ActionNormalizationError, ActionAdmissionError) as exc:
            self._high_level_pending = True
            self._last_high_level_reason = f"rejected:{type(exc).__name__}"
            self.mem_state.failure_count += 1
            self._trace("proposal_rejected", exception_type=type(exc).__name__, message=str(exc),
                        raw=getattr(exc, "raw", None))
            return self._safe_hold(raw_obs, self._last_high_level_reason)

    def reset(self):
        """Invalidate cache and clear frame/semantic buffers (episode boundary)."""
        self._cached_chunk = None
        self._cached_action_horizon = None
        self._cached_action_execution_start_index = None
        self._chunk_step = 0
        self._chunk_index = 0
        self._request_count = 0
        self._served_action_count = 0
        self.mem_state.reset()
        self._progress_monitor.reset()
        self._progress_hold_pending = False
        self._terminal_hold_chunks = 0
        self._last_gripper_commands.clear()
        self._chunk_is_hold = False
        self._cot_text = None
        self._high_level_pending = True
        self._last_high_level_reason = "start"
        self._last_memlite_updated = False
        for buf in self._frame_buffers.values():
            buf.clear()
        for buf in self._state_buffers.values():
            buf.clear()
        self._buffers_initialized = False
        self._trace("reset", reason="episode_reset")

    def _update_buffers(self, raw_obs: dict) -> None:
        """Append current observation to frame/state buffers.

        On first call after reset, fills all slots with the first observation
        (clamp-to-start, matching training's episode boundary behavior).
        """
        p = resolve_processor(self.processor, raw_obs)
        expected_img_keys = {m["key"] for m in p.shape_meta["images"]}
        expected_state_keys = {m["key"] for m in p.shape_meta["state"]}

        img_tensors = {
            k: torch.from_numpy(np.array(v, copy=True))
            for k, v in raw_obs["images"].items()
            if k in expected_img_keys
        }
        state_tensors = {
            k: torch.from_numpy(np.array(v, copy=True)).float()
            for k, v in raw_obs["state"].items()
            if k in expected_state_keys
        }

        if not self._buffers_initialized:
            for _ in range(self._num_obs_steps):
                for k, t in img_tensors.items():
                    self._frame_buffers[k].append(t.clone())
                for k, t in state_tensors.items():
                    self._state_buffers[k].append(t.clone())
            self._buffers_initialized = True
        else:
            for k, t in img_tensors.items():
                self._frame_buffers[k].append(t)
            for k, t in state_tensors.items():
                self._state_buffers[k].append(t)


async def handler(ws, inferencer, processor, action_steps=16, visualize: bool = False):
    """Handle a single WebSocket connection."""
    client = ws.remote_address
    logger.info(f"Client connected: {client} (action_steps={action_steps})")

    await ws.send(packb({"action_steps": action_steps}))
    policy = ChunkedPolicyWrapper(inferencer, processor, action_steps=action_steps)

    visualizer = None
    if visualize:
        from scripts.utils.mem_live_viz import MemLiveVisualizer

        visualizer = MemLiveVisualizer(window_prefix=f"mem[{client[0]}:{client[1]}]")

    try:
        async for msg in ws:
            t0 = time.monotonic()
            try:
                raw_obs = unpackb(msg)
                if isinstance(raw_obs, dict) and raw_obs.get("__reset__"):
                    policy.reset()
                    if visualizer is not None:
                        visualizer.reset()
                    await ws.send(packb({"__reset__": True}))
                    logger.info("Episode reset from client %s", client)
                    continue
                was_fresh = policy.need_obs
                action, cot_text = await policy.get_action(raw_obs)
                resp = {"action": action, "need_obs": policy.need_obs}
                if policy._memlite_enabled:
                    resp["memlite"] = {
                        "intent": policy.mem_state.intent_text,
                        "memory": policy.mem_state.memory_text,
                        "intent_status": policy.mem_state.status,
                        "failure_count": policy.mem_state.failure_count,
                        "high_level_generation": policy.mem_state.high_level_generation,
                        "chunks_since_intent": policy.mem_state.chunks_since_intent,
                        "updated": policy._last_memlite_updated,
                        "reason": policy._last_high_level_reason if policy._last_memlite_updated else None,
                    }
                if cot_text is not None:
                    resp["cot_text"] = cot_text
                await ws.send(packb(resp))
                if visualizer is not None:
                    visualizer.update(
                        images=raw_obs.get("images") if isinstance(raw_obs, dict) else None,
                        cot_text=cot_text,
                        is_fresh=was_fresh,
                        chunk_step=policy._chunk_step,
                        action_steps=action_steps,
                    )
            except ValueError as exc:
                await ws.send(packb(_serialize_error(400, str(exc))))
            except Exception as exc:
                logger.exception("Inference request failed")
                await ws.send(packb(_serialize_error(500, str(exc))))
            dt = (time.monotonic() - t0) * 1000
            logger.debug(f"{'Infer' if policy._chunk_step == 1 else 'Cache'}: {dt:.1f}ms")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client}")
    finally:
        if visualizer is not None:
            visualizer.close()


async def serve(
    handler_fn,
    policy,
    processor,
    host: str,
    port: int,
    *,
    device: str = "cuda",
    action_steps: int = 16,
    visualize: bool = False,
):
    """Start the WebSocket server."""
    inferencer = PolicyInferencer(policy, processor, device=device)
    bound_handler = functools.partial(
        handler_fn,
        inferencer=inferencer,
        processor=processor,
        action_steps=action_steps,
        visualize=visualize,
    )
    async with websockets.serve(bound_handler, host, port, max_size=None):
        mode = "RTC" if action_steps == 1 else f"chunk({action_steps})"
        logger.info(
            "Policy server listening on ws://%s:%d (mode=%s, device=%s)",
            host,
            port,
            mode,
            device,
        )
        await asyncio.Future()


# ──────── main ────────


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket policy server — MEM multi-frame variant"
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--task_config",
        default=None,
        help="Override task config YAML (when run dir has no .hydra/config.yaml)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda", help="CUDA device for model inference")
    parser.add_argument(
        "--action_steps",
        type=int,
        default=16,
        help="Steps to serve per inference. >1=chunk mode (default: 16), 1=RTC",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open cv2 window showing head/wrist frames + COT/affordance overlay (needs DISPLAY; "
        "first connected client gets the window, others run normally).",
    )
    parser.add_argument(
        "--pure-ar",
        action="store_true",
        help=(
            "Force discrete action-token inference: discrete_action=true, "
            "continuous_action=false, return_continuous_action=false."
        ),
    )
    parser.add_argument(
        "--predict-cot",
        dest="predict_cot",
        action="store_true",
        default=None,
        help=(
            "Enable AR text generation before action tokens. Requires a checkpoint "
            "trained with a CoT-compatible template."
        ),
    )
    parser.add_argument(
        "--no-predict-cot",
        dest="predict_cot",
        action="store_false",
        help="Disable AR text generation before action tokens.",
    )
    args, remaining = parser.parse_known_args()

    overrides = [r for r in remaining if "=" in r]

    # --- Config loading. ---
    if args.task_config:
        cfg = load_config_from_task_yaml(args.task_config, args.ckpt_path, overrides)
        print(f"Loaded config from task YAML: {args.task_config}")
    else:
        run_dir = find_run_dir(args.ckpt_path)
        print(f"Found run dir: {run_dir}")
        cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

    # Apply explicit serving-mode switches after loading the run config.  This
    # keeps old dual-head checkpoints usable without changing their saved YAML,
    # while making the selected wire action unambiguous.
    if args.pure_ar:
        cfg.model.model_arch.discrete_action = True
        cfg.model.model_arch.continuous_action = False
        cfg.model.model_arch.return_continuous_action = False
        # The processor flag is usually an interpolation, but run-dir configs
        # are resolved before serving; set it explicitly as well.
        cfg.model.processor.discrete_action = True
        logger.info("Serving mode override: pure AR action-token inference")
    if args.predict_cot is not None:
        cfg.model.model_arch.predict_cot = bool(args.predict_cot)
        # EOV must be a supervised prefix boundary when CoT is generated.
        cfg.model.model_arch.input_preprocessor.pred_eov = bool(args.predict_cot)
        logger.info("Serving mode override: predict_cot=%s", bool(args.predict_cot))

    eval_embodiment = cfg.get("eval_embodiment", None)
    if eval_embodiment and "embodiment_datasets" in cfg.data:
        filter_embodiment(cfg, eval_embodiment)

    from g05.utils.logging.logging_config import setup_logging

    setup_logging(log_level=logging.INFO, is_main_process=True)

    policy, processor = setup(cfg, device=args.device)
    logger.info("Model and processor loaded on %s.", args.device)

    asyncio.run(
        serve(
            handler,
            policy,
            processor,
            args.host,
            args.port,
            device=args.device,
            action_steps=args.action_steps,
            visualize=args.visualize,
        )
    )


if __name__ == "__main__":
    main()
