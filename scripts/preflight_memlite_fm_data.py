"""Audit real FM source targets/sampling and materialize manually reviewable rows.

No optimizer, no simulator, no new semantic labels, no automatic manual approval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path
import textwrap

import numpy as np
import torch
from omegaconf import OmegaConf

from memlite_fm_v11_common import config, sha, TASK5_RUN, RECOVERY, STATS, COHORT, SIDECAR
from g05.data.memlite_recovery_dataset import RecoveryAugmentedDataset, recovery_raw_sample
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.common.task_event_sampler import build_memlite_train_sampler


def range_pick(ranges, n=3):
    total = sum(end-start for start, end in ranges)
    positions = sorted(set(np.linspace(0, total-1, n).astype(int).tolist()))
    output = []
    for position in positions:
        for start, end in ranges:
            if position < end-start:
                output.append(start+position)
                break
            position -= end-start
    return output


def physical_action(sample, processor):
    value = sample["action"].unsqueeze(0).float().clone()
    unpacked = processor.action_state_merger.backward({"action": value})
    return {k: v[0] for k, v in processor.normalizer.backward(unpacked)["action"].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg = config()
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    p_all = build_processors(cfg)
    p_all.set_normalizer_from_stats(load_dataset_stats_from_json(STATS))
    original = instantiate_dataset(cfg, is_training_set=True)
    original.set_processor(p_all)
    p_all.eval()  # deterministic image parity / manual inspection, not training transforms
    dataset = RecoveryAugmentedDataset(original, RECOVERY / "recovery_manifest.json", p_all)
    p = p_all["galaxea_r1pro"]
    index = json.loads((RECOVERY / "motion_recovery_index_v2.json").read_text())
    for child in original.datasets:
        def fail_resample():
            raise RuntimeError("No silent source substitution in FM data audit")
        child._resample_random_idx = fail_resample
    owned, counts, first = [], Counter(), []
    kind_by_index = np.full(len(dataset), -1, dtype=np.int8)
    task_by_index = np.full(len(dataset), -1, dtype=np.int8)
    kinds = ("yaw_stop", "yaw_start_reverse", "gripper", "recovery", "other")
    for task in index["tasks"]:
        for code, kind in enumerate(kinds):
            for start, end in task["low_strata"][kind]:
                assert (kind_by_index[start:end] == -1).all()
                kind_by_index[start:end], task_by_index[start:end] = code, int(task["task_id"])
    for rank in range(4):
        sampler = build_memlite_train_sampler(dataset,
            sampling_config=OmegaConf.to_container(cfg.data.memlite_branch_sampling, resolve=True),
            batch_size=8, num_replicas=4, rank=rank, seed=17, high_fraction=0.)
        batches = list(itertools.islice(sampler, args.steps))
        assert len(batches) == args.steps
        seen = set()
        for batch in batches:
            assert len(set(batch)) == 8 and all(kind_by_index[i] >= 0 for i in batch)
            assert Counter(kinds[kind_by_index[i]] for i in batch) == dict(cfg.data.memlite_branch_sampling.low_quotas)
            counts.update((int(task_by_index[i]), kinds[kind_by_index[i]]) for i in batch)
            seen.update(batch)
        owned.append(seen)
        first.append(batches[:5])
    assert all(owned[a].isdisjoint(owned[b]) for a in range(4) for b in range(a))
    if args.steps % 5 == 0:
        assert all(len({counts[t, kind] for t in range(5)}) == 1 for kind in kinds)

    old_cfg = OmegaConf.load(TASK5_RUN / ".hydra/config.yaml")
    old_processors = build_processors(old_cfg)
    old_processors.set_normalizer_from_stats(load_dataset_stats_from_json(STATS))
    old_processors.eval()
    old_p = old_processors["galaxea_r1pro"]
    captured = []
    preprocess = p.preprocess
    def capture(raw):
        captured.append(deepcopy(raw))
        return preprocess(raw)
    p.preprocess = capture
    rows, tensor_rows = [], []

    def add(sample, metadata, raw=None):
        assert sample["samples"]["memlite_branch"] == "low"
        assert "action" not in sample["samples"] and "<action_action" not in sample["samples"]["template"]
        assert sample["proprio"].shape == (1, 27) and sample["action"].shape == (32, 27)
        assert not sample["action_is_pad"].all()
        assert torch.isfinite(sample["proprio"]).all() and torch.isfinite(sample["action"]).all()
        assert all(value.shape == (1,3,256,256) for value in sample["pixel_values"].values())
        differences = {}
        if raw is not None:
            old = old_p.preprocess(deepcopy(raw))
            for key in ("proprio", "action", "gt_action", "action_dim_is_pad", "action_is_pad"):
                assert torch.equal(sample[key], old[key]), key
            for key, value in sample["pixel_values"].items():
                assert torch.equal(value, old["pixel_values"][key]), key
            assert sample["samples"]["command"] == old["samples"]["command"]
            assert sample["samples"]["template"].split("<EOC>")[0] == old["samples"]["template"].split("<EOC>")[0]
            differences["old_native_processor_exact"] = True
        actions = physical_action(sample, p)
        base = actions["lower_body"][:16, -3:].numpy()
        identifier = len(rows)
        filename = f"sample_{identifier:03d}.pt"
        torch.save({"sample": sample, "metadata": metadata}, args.output / filename)
        rows.append({"id": identifier, "file": filename, **metadata, **differences,
                     "command": sample["samples"]["command"], "intent": sample["samples"]["intent"],
                     "base_first16": base.tolist(), "valid_action_steps": int((~sample["action_is_pad"]).sum()),
                     "sample_sha256": sha(args.output / filename)})
        tensor_rows.append(sample)
        print(json.dumps({"review_id": identifier, **metadata}), flush=True)

    for task in index["tasks"]:
        for kind in kinds:
            for i in range_pick(task["low_strata"][kind]):
                captured.clear()
                sample = dataset[i]
                assert len(captured) == 1
                add(sample, {"split": "train", "task_id": int(task["task_id"]), "category": kind,
                             "source_index": i, "source": sample["memlite_source_kind"]}, captured[0])

    # Old, frozen heldout cohort is NEVER selected for the optimizer smoke.
    cohort = json.loads((COHORT / "manifest.json").read_text())
    selection_count = Counter()
    for entry in cohort["entries"]:
        if entry["branch"] != "low":
            continue
        key = entry["task_id"], entry["source"], entry["category"]
        if selection_count[key] >= 1:
            continue
        selection_count[key] += 1
        sample = torch.load(COHORT / entry["file"], weights_only=False, map_location="cpu")["sample"]
        sample = deepcopy(sample)
        for camera in sample["pixel_values"]:
            sample["pixel_values"][camera] = sample["pixel_values"][camera][-1:].clone()
        sample["proprio"] = sample["proprio"][-1:].clone()
        for key in ("proprio_is_pad", "image_is_pad"):
            if key in sample:
                sample[key] = sample[key][-1:].clone()
        slot = sample["samples"]
        slot["proprio"]["value"] = sample["proprio"]
        slot.pop("action", None)
        slot["template"] = p.samples_builder.template
        add(sample, {"split": "eval", "task_id": entry["task_id"], "category": entry["category"],
                     "source": entry["source"], "cohort_file": entry["file"]})

    # Images are generated directly from reviewed observations, not synthetic.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for start in range(0, len(rows), 6):
        picked = rows[start:start+6]
        fig, axes = plt.subplots(len(picked), 4, figsize=(16, 3*len(picked)), squeeze=False)
        for r, row in enumerate(picked):
            sample = tensor_rows[row["id"]]
            for c, (key, value) in enumerate(sample["pixel_values"].items()):
                rgb = ((value[0].float().permute(1,2,0).numpy()+1)/2).clip(0,1)
                axes[r,c].imshow(rgb)
                axes[r,c].set_title(f"#{row['id']} {key}", fontsize=8)
                axes[r,c].axis("off")
            axes[r,3].axis("off")
            base = np.asarray(row["base_first16"])
            description = f"#{row['id']} task{row['task_id']} {row['split']} {row['category']}\n" + textwrap.fill(row["intent"], 48)
            description += f"\nfirst16 mean base: {np.round(base.mean(0),4)}\nlast4 mean base: {np.round(base[-4:].mean(0),4)}\nvalid steps: {row['valid_action_steps']}"
            axes[r,3].text(0, .98, description, va="top", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.output / f"review_{start//6:02d}.jpg", dpi=100)
        plt.close(fig)
    summary = {"passed": True, "manual_approval": False, "steps_audited": args.steps,
               "source_rank_disjoint": True, "actual_view_length": len(dataset),
               "train_active_episodes": [list(map(int, d._active_episode_indices)) for d in original.datasets],
               "data_hashes": {"sidecar": sha(SIDECAR), "stats": sha(STATS),
                   "recovery_manifest": sha(RECOVERY / "recovery_manifest.json")},
               "counts": {f"task{t}/{kind}": count for (t,kind),count in sorted(counts.items())},
               "first_five_batches": first, "rows": rows}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"passed": True, "review_rows": len(rows), "manual_approval": False}), flush=True)


if __name__ == "__main__":
    main()
