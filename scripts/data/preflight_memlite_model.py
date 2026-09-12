"""Real checkpoint, conditioning, grouped CE and free AR contract; no optimizer."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from g05.data.memlite_recovery_dataset import recovery_raw_sample
from g05.models.g05.inferencer import PolicyInferencer
from g05.utils.common.pytorch_utils import dict_apply
from scripts.serve_policy_mem import setup
from serve_behavior_policy_mem import load_runtime_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0] > 40*1024**3
    # Match this recipe's FP32 master weights. The existing CE helper disables
    # autocast and casts its inputs to FP32, unlike deployment-only free AR.
    cfg = load_runtime_config(str(args.checkpoint), ["model.use_torch_compile=false", "model.model_weights_to_bf16=false"])
    OmegaConf.set_struct(cfg.model.model_arch, False)
    cfg.model.model_arch.memlite_conditioning = {"enabled": True, "velocity_dropout": .5, "inference_velocity": "masked",
                                              "velocity_indices": [24,25,26], "proprio_dim": 27}
    cfg.model.model_arch.memlite_action_group_loss = {"enabled": True, "lower_body_weight": 2.}
    policy, processors = setup(cfg, device="cuda:0")
    p = processors["galaxea_r1pro"]
    data = json.loads(args.manifest.read_text())
    sources = data["samples"]["train"]
    selected = [next(r for r in sources if r["memlite_branch"] == branch) for branch in ("high", "low")]
    def load(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    samples = []
    for i, row in enumerate(selected):
        row = {**row, "index": i}
        root = Path(row["trajectory"])
        s = p.preprocess(recovery_raw_sample(row, root, p, load(root / "trajectory.npz"), load))
        s["embodiment"] = s["embodiment_type"] = "galaxea_r1pro"
        samples.append(s)
    batch = PolicyInferencer._collate(deepcopy(samples), padding_input_id=p.pad_token_id)
    batch = dict_apply(batch, lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x)
    assert policy.memlite_conditioning.enabled
    helper = policy.model.ar_helper
    assert helper.action_group_loss.lower_body_weight == 2.
    losses = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, full in (("fused", False), ("eager", True)):
            helper._need_full_logits = full
            total, result = policy._forward_train_memlite(batch["samples"], batch["pixel_values"],
                actions=batch["action"], action_pad_masks=batch["action_is_pad"],
                action_dim_is_pad=batch["action_dim_is_pad"])
            losses[name] = {k: float(v) for k,v in result.items() if isinstance(v, (float,int)) or isinstance(v, torch.Tensor) and v.numel()==1}
            cache = helper._last_ce_cache
            assert (cache["token_weights"] == 2).any(), "No actual lower-body tokens received weight"
            assert torch.isfinite(result["ce_loss"]) and float(result["fm_loss"]) == 0
        low = PolicyInferencer._collate([deepcopy(samples[1])], padding_input_id=p.pad_token_id)
        low = dict_apply(low, lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x)
        result = policy.generate_low_level_action(samples=low["samples"], pixel_values=low["pixel_values"],
            intent_text=[selected[1]["intent"]], action_dim_is_pad=low["action_dim_is_pad"], action_gt=None)
        assert not result.get("ar_absent_keys", [set()])[0]
        assert result["action"].shape == (1,32,27) and torch.isfinite(result["action"]).all()
    if abs(losses["fused"]["ce_loss"] - losses["eager"]["ce_loss"]) > .05:
        raise ValueError("Fused/eager weighted CE diverged beyond bf16 tolerance")
    output = {"passed": True, "checkpoint": str(args.checkpoint), "trained": False, "optimizer_updates": 0,
        "manual_approval": False, "losses": losses, "real_lower_body_weight_observed": True,
        "free_ar_shape": list(result["action"].shape), "example_ids": [r["review_id"] for r in selected]}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
