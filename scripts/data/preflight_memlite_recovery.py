"""Real processor/conditioning parity on packaged samples; no model training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from g05.data.memlite_recovery_dataset import recovery_raw_sample
from g05.models.g05.helpers.memlite_conditioning import MEMLiteConditioning
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.data.data_utils import collate_fn_pad_sequences


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-index", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    register_default_resolvers()
    cfg = OmegaConf.load(args.source_run / ".hydra/config.yaml")
    processors = build_processors(cfg)
    processors.set_normalizer_from_stats(load_dataset_stats_from_json(args.source_run / "dataset_stats.json"))
    processors.eval()
    p = processors["galaxea_r1pro"]
    data = json.loads(args.manifest.read_text())
    conditioning = MEMLiteConditioning(enabled=True, velocity_dropout=1.)
    rows = []
    recovery_examples = []
    for split in ("train", "eval"):
        sources = data["samples"][split]
        if not sources:
            continue
        # Task/branch stratification keeps previews useful even before all
        # collection jobs have finished. Never issues manual approval.
        selected = []
        for task in range(5):
            for branch in ("high", "low"):
                candidates = [r for r in sources if r["task_id"] == task and r["memlite_branch"] == branch]
                if candidates:
                    selected.extend(candidates[i] for i in sorted(set([0, len(candidates)//2, len(candidates)-1])))
        for i, original in enumerate(selected):
            entry = {**original, "index": i}
            root = Path(entry["trajectory"])
            def load(path):
                with np.load(path, allow_pickle=False) as f:
                    return {k: f[k] for k in f.files}
            raw = recovery_raw_sample(entry, root, p, load(root / "trajectory.npz"), load)
            sample = p.preprocess(raw)
            sample["embodiment"] = sample["embodiment_type"] = "galaxea_r1pro"
            if len(recovery_examples) < 2:
                recovery_examples.append(deepcopy(sample))
            nested = sample["samples"]
            assert nested["execution_feedback"] == entry["execution_feedback"]
            assert nested["memlite_branch"] == entry["memlite_branch"]
            before = sample["proprio"].clone()
            action_before = sample["action"].clone()
            a = conditioning.prepare([nested], training=True)[0]
            b = conditioning.prepare([nested], training=False)[0]
            def prop(x):
                v = x["proprio"]
                return v["value"] if isinstance(v, dict) else v
            assert prop(a).shape == (6, 27)
            torch.testing.assert_close(prop(a), prop(b), rtol=0, atol=0)
            assert a["template"] == b["template"]
            torch.testing.assert_close(sample["proprio"], before, rtol=0, atol=0)
            torch.testing.assert_close(sample["action"], action_before, rtol=0, atol=0)
            if entry["memlite_branch"] == "low":
                assert not prop(a)[:, 24:].any()
                torch.testing.assert_close(prop(a)[:, :24], prop(nested)[:, :24], rtol=0, atol=0)
            else:
                torch.testing.assert_close(prop(a), prop(nested), rtol=0, atol=0)
            pixels = sample["pixel_values"]
            assert torch.isfinite(sample["action"]).all() and torch.isfinite(sample["proprio"]).all()
            rows.append({"review_id": entry["review_id"], "branch": entry["memlite_branch"], "split": split,
                "proprio_shape": list(sample["proprio"].shape), "action_shape": list(sample["action"].shape),
                "pixel_shapes": {k: list(v.shape) for k,v in pixels.items()} if isinstance(pixels, dict) else list(pixels.shape),
                "feedback": entry["execution_feedback"], "input_only_mask_parity": True})
    if not rows:
        raise ValueError("No real recovery samples to check")
    output = {"passed": True, "manual_approval": False, "trained": False, "rows": rows}
    if args.source_index:
        original = instantiate_dataset(cfg, is_training_set=True)
        original.set_processor(processors)
        processors.eval()
        index = json.loads(args.source_index.read_text())
        originals = [original[index["tasks"][0]["high"][0]],
                     original[index["tasks"][0]["low_strata"]["yaw_stop"][0][0]]]
        for sample in originals:
            sample.setdefault("execution_feedback", "none")
        for recovery in recovery_examples:
            for sample in originals:
                if sample.keys() != recovery.keys():
                    raise ValueError(f"Mixed source keys differ: original-only={sample.keys()-recovery.keys()}, recovery-only={recovery.keys()-sample.keys()}")
        mixed = collate_fn_pad_sequences(deepcopy(originals+recovery_examples))
        output["real_original_recovery_collation"] = {"passed": True, "batch_size": len(mixed["samples"]),
                                                     "keys": sorted(mixed)}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"passed": True, "examples": len(rows), "trained": False}), flush=True)


if __name__ == "__main__":
    main()
