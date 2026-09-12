"""Serve the trained FM intent adapter with an independent AR MEM-Lite planner."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import functools
import importlib.util
from pathlib import Path
import sys

import websockets
from omegaconf import OmegaConf

from serve_policy_mem import setup, ChunkedPolicyWrapper, _resolve_stats_path
from g05.models.g05.inferencer import PolicyInferencer
from g05.models.g05.memlite_fm_inferencer import SeparatedMEMLiteFMInferencer
from g05.utils.eval.eval_utils import filter_embodiment
from g05.utils.checkpoint import ckpt_utils
from g05.utils.data.normalizer import ActionNormalizationError


BRIDGE_DIR = Path("/mnt/sdc1/robodojo/behavior_bridge_staging")
TASKS_PATH = Path("/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos/meta/tasks.jsonl")


def project_fm_grippers(action):
    """Match official smooth gripper input saturation before strict admission.

    R1Pro's MultiFingerGripperController has default [-1,1] input limits and
    BaseController._preprocess_command clips to them. This projection therefore
    does not change the executed gripper command. Reject nonfinite or >1 unit
    excess instead of hiding gross model corruption. Arms/base are untouched.
    """
    import torch
    output = dict(action)
    diagnostics = dict(output.get("_normalization_diagnostics", {}))
    for key in ("left_gripper", "right_gripper"):
        if key not in output:
            raise ActionNormalizationError(f"FM did not return an explicit {key}")
        value = output[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != 1:
            raise ActionNormalizationError(f"FM {key} must have shape [B,H,1]")
        if not torch.isfinite(value).all():
            raise ActionNormalizationError(f"Non-finite FM {key}")
        excess = (value.abs()-1).clamp_min(0)
        if excess.max() > 1:
            raise ActionNormalizationError(f"FM {key} exceeds its [-1,1] command range by >1 unit")
        diagnostics[f"fm_command.{key}"] = {"count":value.numel(),
            "clipped_count":int((excess>0).sum()), "max_normalized_excess":float(excess.max()),
            "decoded_abs_max":float(value.abs().max())}
        output[key] = value.clamp(-1,1)
    output["_normalization_diagnostics"] = diagnostics
    return output


class FMControllerInferencer(PolicyInferencer):
    def infer_low_level_action(self, obs_dicts, intent_texts):
        if self.policy.discrete_action or not self.policy.continuous_action:
            raise ValueError("Continuous gripper projection is scoped to FM-only serving")
        return [project_fm_grippers(a) for a in super().infer_low_level_action(obs_dicts, intent_texts)]


def load_bridge(directory=BRIDGE_DIR):
    """Reuse the deployed official 23D wire/reset/initial-memory implementation."""
    path = Path(directory).resolve(strict=True) / "serve_behavior_policy_mem.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    name = "_memlite_fm_official_behavior_bridge"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_serving_config(checkpoint, overrides=(), *, bridge):
    """Resolve serving state without depending on old training-only env vars."""
    run = ckpt_utils.find_run_dir(checkpoint)
    cfg = OmegaConf.load(run / ".hydra/config.yaml")
    OmegaConf.set_struct(cfg, False)
    # These training-only entries are not consumed by setup. In old high-level
    # runs they reference environment variables that no longer exist.
    cfg.model.pretrained_ckpt = None
    cfg.resume_ckpt = None
    if cfg.data.get("memlite_branch_sampling"):
        cfg.data.memlite_branch_sampling.sampling_index_path = None
    if cfg.get("memlite_recovery"):
        cfg.memlite_recovery.enabled = False
        cfg.memlite_recovery.manifest = None
    cfg.run_dir = str(run)
    cfg.output_dir = str(run / f"eval_{Path(checkpoint).stem}")
    cfg.exp_name = run.name
    cfg.logger.task = "eval"
    cfg.logger.experiment_name = f"eval_{run.name}"
    cfg.logger.mode = "disabled"
    cfg.ckpt_path = str(Path(checkpoint).resolve(strict=True))
    ckpt_utils._apply_hf_processor_sidecar(cfg, run)
    bridge._safe_apply_action_tokenizer_sidecar(cfg, run)
    ckpt_utils._register_hydra_builtin_resolvers()
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    # FM high_checkpoint/controller_init MUST be explicit in new saved recipes;
    # do not silently pick today's default if checkpoint provenance is missing.
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def load_separated(low_checkpoint, *, device="cuda", high_device=None,
                   low_overrides=(), disable_intent=False, bridge=None):
    bridge = bridge or load_bridge()
    # The bridge's loader preserves OmegaConf tokenizer aliases when resolving
    # run-local sidecars. Generic ckpt_utils currently breaks these aliases.
    low_cfg = load_serving_config(low_checkpoint, low_overrides, bridge=bridge)
    spec = low_cfg.get("memlite_fm")
    if not spec or int(spec.schema_version) != 1 or not spec.high_frozen:
        raise ValueError("Checkpoint lacks the explicit separated MEM-Lite FM contract")
    high_checkpoint = str(spec.high_checkpoint)
    high_cfg = load_serving_config(high_checkpoint, bridge=bridge)
    for cfg in (low_cfg, high_cfg):
        filter_embodiment(cfg, "galaxea_r1pro")
        cfg.model.use_torch_compile = False
        cfg.model.model_arch.attn_implementation = "sdpa"
    if hashlib.sha256(_resolve_stats_path(low_cfg).read_bytes()).digest() != hashlib.sha256(_resolve_stats_path(high_cfg).read_bytes()).digest():
        raise ValueError("High/low normalization sidecars differ; revalidate before serving")
    if int(low_cfg.data.obs_size) != 1 or int(high_cfg.data.obs_size) != 6:
        raise ValueError("Separated controller expects current-frame FM and six-frame planner")
    low_policy, low_processor = setup(low_cfg, device=device)
    if disable_intent:
        low_policy.intent_enabled = False
    high_device = high_device or device
    high_policy, high_processor = setup(high_cfg, device=high_device)
    high_policy.requires_grad_(False)
    if int(high_processor.memlite_runtime.get("action_steps", 0)) != int(spec.action_steps):
        raise ValueError("Planner history cadence and FM execution window disagree")
    low_children = list(low_processor.processors.values())
    if any(p.num_obs_steps != 1 for p in low_children):
        raise ValueError("Low preprocessor must use the current observation only")
    return SeparatedMEMLiteFMInferencer(
        high=PolicyInferencer(high_policy, high_processor, device=high_device),
        low=FMControllerInferencer(low_policy, low_processor, device=device)), high_processor


def behavior_handler_for(inferencer, processor, task_instructions, *, bridge, trace_root=None):
    probe = ChunkedPolicyWrapper(inferencer, processor, action_steps=16, strict_memlite=True)
    if not probe._memlite_enabled or not probe._causal_runtime:
        raise ValueError("Official FM server requires the causal high/low MEM-Lite runtime")
    return functools.partial(bridge.behavior_handler, inferencer=inferencer, processor=processor,
        task_instructions=task_instructions, action_steps=16, require_memlite=True,
        memlite_replan_every_chunks=int(processor.memlite_runtime["replan_every_chunks"]),
        memlite_high_level_max_new_tokens=int(processor.memlite_runtime["high_level_max_new_tokens"]),
        trace_root=trace_root)


async def serve_official(handler, *, bridge, host, port):
    async with websockets.serve(handler, host, port, max_size=None, ping_interval=None,
                               process_request=bridge._health_check):
        logging.info("Separated MEM-Lite FM official BEHAVIOR server listening on ws://%s:%d", host, port)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--high-device")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bridge-dir", type=Path, default=BRIDGE_DIR)
    ap.add_argument("--tasks_path", type=Path, default=TASKS_PATH)
    ap.add_argument("--trace-root", type=Path)
    ap.add_argument("--disable-intent", action="store_true", help="Exact preserved FM route for diagnostics")
    args, overrides = ap.parse_known_args()
    if any("=" not in value for value in overrides):
        ap.error("Remaining arguments must be explicit Hydra key=value overrides")
    logging.basicConfig(level=logging.INFO)
    bridge = load_bridge(args.bridge_dir)
    inferencer, processor = load_separated(args.ckpt_path, device=args.device,
        high_device=args.high_device, low_overrides=overrides, disable_intent=args.disable_intent, bridge=bridge)
    handler = behavior_handler_for(inferencer, processor, bridge.load_task_instructions(args.tasks_path),
        bridge=bridge, trace_root=args.trace_root)
    asyncio.run(serve_official(handler, bridge=bridge, host=args.host, port=args.port))


if __name__ == "__main__":
    main()
