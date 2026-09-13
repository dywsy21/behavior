"""Step-explicit prefix-only FM diagnostic service.

Derived from the verified A3 aligned service. Neural source, history transport,
padding metadata and action clock are unchanged; the completed checkpoint step
and read-only runtime location are explicit instead of hard-coded to 5000.
The historical wire kind is retained for client protocol compatibility only.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_checkpoint_path(run, checkpoint, step):
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError('checkpoint-step must be a positive integer')
    if Path(checkpoint).resolve() != (Path(run) / f'checkpoints/step_{step}.pt').resolve():
        raise ValueError('Service checkpoint differs from the explicitly completed run/step')


def verify_model_compatibility(training_root, serving_root):
    """All shared neural implementation bytes must match except the reviewed
    SkillFM serving entry. Its parameter-building/training methods must match.
    """
    files = {}
    special = "src/g05/models/g05/g05_policy_memlite_skill_fm.py"
    for left in (training_root / "src/g05/models").rglob("*.py"):
        relative = str(left.relative_to(training_root))
        right = serving_root / relative
        if relative != special:
            assert right.is_file() and sha(left) == sha(right), relative
        files[relative] = {"training": sha(left), "serving": sha(right)}
    def methods(root):
        tree = ast.parse((root / special).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G05PolicyMEMLiteSkillFM")
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
    left, right = methods(training_root), methods(serving_root)
    for method in ("__init__", "train", "forward_train", "configure_coordination_trainability",
                   "post_checkpoint_load", "remap_checkpoint_state_dict"):
        assert left[method] == right[method], method
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-step", type=int, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--serving-source", type=Path)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--source-receipt", type=Path, required=True)
    p.add_argument("--checkpoint-sha", required=True)
    p.add_argument("--initial-action-count", type=int, required=True)
    p.add_argument("--prefix-window-sha", required=True)
    p.add_argument("--inference-alignment-receipt", type=Path, required=True)
    args = p.parse_args()
    validate_checkpoint_path(args.run, args.checkpoint, args.checkpoint_step)
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    padding_patch = json.loads(args.inference_alignment_receipt.read_text())
    if set(padding_patch['files']) != {'serve_formal_a_controller.py',
            'native_r1pro_action_padding_v1.py', 'native_r1pro_inference_alignment_v2.py'}:
        raise ValueError('Wrong padding-only runtime patch inventory')
    for name, expected in padding_patch['files'].items():
        if sha(runtime/name) != expected:
            raise ValueError('Padding-only runtime source changed: '+name)
    if (args.initial_action_count < 80 or len(args.prefix_window_sha) != 64
            or any(c not in '0123456789abcdef' for c in args.prefix_window_sha)):
        raise ValueError('Require exact actual prefix count and frozen diagnostic window SHA')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_receipt = json.loads((args.run / "coordination_run_receipt.json").read_text())
    assert run_receipt["state"] == "complete" and run_receipt["returncode"] == 0
    source = args.serving_source or args.adapter / "replayable_serving_source_v1"
    sys.path[:0] = [str(args.adapter), str(source / "src"), str(source)]
    from native_oracle_low_v1.replayable_source import (
        load_approved_official23_support,
        replayable_bridge_action_to_vector,
    )
    from native_oracle_low_v1.runtime import NativeLowOnlyService
    from native_ab_transport import serve_skill_fm_websocket
    from native_a2_history import LowHistoryIngress, history_admission_to_wire
    from native_a2_source_contract import validate_source_receipt, verify_full_loaded_checkpoint
    snapshot_path = args.snapshot or args.adapter / "native_fm_source_snapshot_v1.json"
    snapshot_receipt = validate_source_receipt(args.source_receipt,
        training_source=Path(run_receipt['source_root']),serving_source=source,
        snapshot=snapshot_path,config=args.config)
    compatibility = verify_model_compatibility(Path(run_receipt["source_root"]), source)
    assert sha(args.checkpoint) == args.checkpoint_sha
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.data.processor_utils import build_processors
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from g05.models.g05.memlite_native_fm_inferencer import NativeFMSnapshot, NativeFMPolicyInferencer, NativeFMActionChunk
    from native_r1pro_action_padding_v1 import derive_r1pro_action_dim_is_pad
    from native_r1pro_inference_alignment_v2 import make_aligned_native_inferencer
    AlignedNativeFMInferencer = make_aligned_native_inferencer(NativeFMPolicyInferencer,
        action_history_steps=padding_patch['action_history_steps'])
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.manual_seed(17)
    cfg = OmegaConf.load(args.config)
    policy, checkpoint = load_model_from_checkpoint(cfg.model.model_arch, str(args.checkpoint),
        device="cuda", extra_prefixes=["normalizer."], eval_mode=False,return_full_checkpoint=True)
    coverage = verify_full_loaded_checkpoint(policy,checkpoint,expected_step=args.checkpoint_step)
    del checkpoint
    gate=policy.configure_coordination_trainability()
    assert gate['num_obs_steps']==6 and gate['trainability_profile']=='low_ae_lora_history'
    policy.eval()
    assert policy.continuous_action and not policy.discrete_action and not policy.predict_cot
    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.datastatics_path))
    processor.eval()
    static_action_mask = derive_r1pro_action_dim_is_pad(processor.processors['galaxea_r1pro'],
        action_dim=int(policy.model_config.action_dim))
    snapshot = NativeFMSnapshot(**json.loads(snapshot_path.read_text()))
    evidence_root = args.adapter.parents[1] / "data/review_receipts"
    support = load_approved_official23_support(source_root=source,
        inverse_impulse_receipt=evidence_root / "r1pro_native_fm_inverse_impulse_candidate_20260909.json",
        constant_zero_receipt=evidence_root / "r1pro_native_fm_constant_zero_reference_support_limited_20260909.json")
    assert sha(cfg.datastatics_path)==support.normalizer_stats_sha256
    receipt = dict(kind="formal_a2_native_skill_fm_prefix_history6_service", checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha(args.checkpoint), config=str(args.config), config_sha256=sha(args.config),
        run_receipt_sha256=sha(args.run / "coordination_run_receipt.json"),
        service_entry_sha256=sha(__file__), runtime_root=str(runtime),
        checkpoint_load=coverage, model_compatibility=compatibility, snapshot=snapshot_receipt,
        model_components=["low"], subgoal_origin="frozen_oracle_diagnostic", not_success_rate=True,
        initial_action_count=args.initial_action_count, prefix_window_sha256=args.prefix_window_sha,
        history_context_kind='actual_demo_prefix_then_policy',
        inference_alignment_fix=dict(receipt_sha256=sha(args.inference_alignment_receipt), files=padding_patch['files'],
            source='robot_action_shape_metadata_only', padded_dimensions=static_action_mask.nonzero().flatten().tolist(),
            real_control_dimensions=int((~static_action_mask).sum()), expert_actions_used=False,
            action_history_steps=padding_patch['action_history_steps'], action_execution_start_index=0),
        num_obs_steps=6,trainability_profile='low_ae_lora_history',checkpoint_step=args.checkpoint_step,
        source_receipt_sha256=sha(args.source_receipt),normalizer_stats_sha256=sha(cfg.datastatics_path),
        history_wire_schema='string_anchor_keys_v1',
        history_runtime_files={name:sha(runtime/name) for name in
            ('native_a2_history.py','native_ab_transport.py','serve_formal_a_controller.py')})
    (args.output_dir / "load_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")

    class FormalALowService(NativeLowOnlyService):
        # Reuse the existing infer_chunk transport. This constructor binds
        # actual runtime objects directly after the checks above; it makes no
        # release/SR assertions about the training-instance diagnostic window.
        def __init__(self):
            self.low = {"num_obs_steps": 6}
            self.history_ingress=LowHistoryIngress(initial_action_count=args.initial_action_count)
            self.low_only_receipt = {"constructed_components": ["low"], "loaded_components": ["low"], "called_components": ["low"]}
            self.low_component_identity = {"checkpoint_sha256": receipt["checkpoint_sha256"], "config_sha256": receipt["config_sha256"]}
            self.inferencer = AlignedNativeFMInferencer(policy, processor, device="cuda", snapshot=snapshot)
            self._chunk_factory = NativeFMActionChunk
            self._snapshot = snapshot
            self._bridge = replayable_bridge_action_to_vector
            self._bridge_identity = {"file": str(args.adapter / "native_oracle_low_v1/replayable_source.py"),
                                     "sha256": sha(args.adapter / "native_oracle_low_v1/replayable_source.py")}
            self._official23_support = support
            self._allow_test_seam = False
            self.calls = 0
        def reset_runtime(self, seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.calls = 0
            self.history_ingress.reset()
        def infer_chunk(self, observation, installed_subgoal, *, execute_steps):
            begin = time.monotonic()
            prepared,admission=self.history_ingress.prepare(observation,execute_steps=execute_steps)
            response = super().infer_chunk(prepared, installed_subgoal, execute_steps=execute_steps)
            response['inference_alignment_admission'] = dict(self.inferencer.last_inference_alignment_admission)
            self.history_ingress.commit(admission)
            response['history_admission']=history_admission_to_wire(admission)
            self.calls += 1
            print(json.dumps({"call": self.calls, "seconds": time.monotonic()-begin,
                              "timing": response["timing"]}), flush=True)
            return response
    print(json.dumps({"ready": True, "port": args.port, "checkpoint_step": args.checkpoint_step}), flush=True)
    asyncio.run(serve_skill_fm_websocket(FormalALowService(), identity=receipt, host="127.0.0.1", port=args.port))


if __name__ == "__main__":
    main()
