"""Expand inverse action bounds using every eligible TRAIN window only.

The initialized checkpoint's means/stds, tail quantiles, gripper mapping and
state normalizer are byte-value preserved. No validation sample contributes.
This is not a refit of the policy's normalized coordinate system.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from g05.data.memlite_stage1_labels import anchor_records
from g05.utils.training.stage1_runtime import atomic_json, sha256

PARTS = {"left_arm": slice(0,7), "left_gripper": slice(7,8),
         "right_arm": slice(8,15), "right_gripper": slice(15,16), "lower_body": slice(16,23)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--original-stats", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pa.set_cpu_count(8)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads((args.release/"manifest.json").read_text())
    groups = defaultdict(list)
    with (args.release/"episodes.jsonl").open() as stream:
        for line in stream:
            episode = json.loads(line)
            if episode["split"] == "train" and episode["segments"]:
                r = episode["row"]
                groups[(r["data/chunk_index"],r["data/file_index"])].append(episode)
    lo, hi = np.full((32,23), np.inf), np.full((32,23), -np.inf)
    counts = np.zeros(32, np.int64)
    observed_tasks, observed_episodes = set(), []
    for ordinal, ((chunk, file), episodes) in enumerate(sorted(groups.items())):
        path = Path(manifest["raw_root"])/f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"
        table = pq.read_table(path, columns=["episode_index","frame_index","action","observation.state"]).combine_chunks()
        identities = table["episode_index"].to_numpy()
        clock = table["frame_index"].to_numpy()
        a = table["action"].chunk(0).values.to_numpy(zero_copy_only=False).reshape(-1,23)
        s = table["observation.state"].chunk(0).values.to_numpy(zero_copy_only=False).reshape(-1,61)
        for ep in episodes:
            row = ep["row"]
            selected = np.flatnonzero(identities == row["episode_index"])
            if len(selected) != row["length"] or not np.array_equal(clock[selected], np.arange(row["length"])):
                raise ValueError("TRAIN clock/identity changed")
            action, state = a[selected], s[selected]
            anchors = anchor_records(ep["segments"],ep["phase"])
            frames = np.array([x[0] for x in anchors])
            ends = np.array([ep["segments"][x[1]]["end"] for x in anchors])
            for h in range(32):
                good = frames+h < ends
                f = frames[good]
                if not len(f):
                    continue
                future = action[f+h]
                current = state[f]
                lower = np.concatenate((future[:,3:7],future[:,:3]),axis=1).copy()
                lower[:,:3] -= current[:,53:56]
                transformed = np.concatenate((future[:,7:14]-current[:,3:10], future[:,14:15],
                    future[:,15:22]-current[:,28:35], future[:,22:23], lower),axis=1)
                if not np.isfinite(transformed).all():
                    raise ValueError("Nonfinite TRAIN action")
                lo[h] = np.minimum(lo[h],transformed.min(0))
                hi[h] = np.maximum(hi[h],transformed.max(0))
                counts[h] += len(f)
            observed_tasks.add(row["task_index"])
            observed_episodes.append(row["episode_index"])
        if ordinal%100 == 0:
            atomic_json(args.output/"progress.json",dict(files=ordinal+1,total_files=len(groups),
                episodes=len(observed_episodes),anchors=int(counts[0]),seconds=time.monotonic()-started))
    expected = sum(x["train_anchors"] for x in manifest["task_counts"].values())
    if counts[0] != expected or observed_tasks != set(range(100)):
        raise ValueError("TRAIN normalization coverage mismatch")
    original = json.loads(args.original_stats.read_text())
    expanded = deepcopy(original)
    for name, sl in PARTS.items():
        stats = expanded["galaxea_r1pro"]["action"][name]
        for field, values, op in (("min",lo[:,sl],np.minimum),("max",hi[:,sl],np.maximum)):
            stats["stepwise_"+field] = op(np.asarray(stats["stepwise_"+field]),values).tolist()
            across = values.min(0) if field == "min" else values.max(0)
            stats["global_"+field] = op(np.asarray(stats["global_"+field]),across).tolist()
    atomic_json(args.output/"stats.json",expanded)
    # Explicit whitelist proof: only min/max action safety bounds may change.
    stripped = deepcopy(expanded)
    for name in PARTS:
        for field in ("stepwise_min","stepwise_max","global_min","global_max"):
            stripped["galaxea_r1pro"]["action"][name][field] = original["galaxea_r1pro"]["action"][name][field]
    if stripped != original:
        raise AssertionError("Normalized coordinates or state transform changed")
    atomic_json(args.output/"receipt.json",dict(status="TRAIN_BOUNDS_COMPLETE", manifest_sha256=sha256(args.release/"manifest.json"),
        original_stats_sha256=sha256(args.original_stats),stats_sha256=sha256(args.output/"stats.json"),
        normalized_coordinates_preserved=True,forward_action_clip=None,state_forward_clip_unchanged=True,
        tasks=sorted(observed_tasks),episodes=sorted(observed_episodes),windows_per_horizon=counts.tolist(),
        no_eval_contribution=True,seconds=time.monotonic()-started))
    print(json.dumps(dict(status="TRAIN_BOUNDS_COMPLETE",anchors=int(counts[0]),seconds=time.monotonic()-started)),flush=True)


if __name__ == "__main__":
    main()
