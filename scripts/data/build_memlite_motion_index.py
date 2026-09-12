"""Audit raw TRAIN motion transitions and partition the existing sampling view.

This produces a base index, not a trainable recovery index. Accepted recovery
records and manual review must be added by the recovery packager before use.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from g05.data.motion_events import classify_motion_windows
from g05.utils.common.task_event_sampler import branch_fingerprint


def ranges(mask, offset=0):
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[int(a + offset), int(b + offset)] for a, b in
            zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = json.loads(args.base_index.read_text())
    assert base["schema_version"] == 1
    identity = base["view_identity"]
    assert len(identity["dataset_roots"]) == len(identity["active_episodes"]) == 1
    root = Path(identity["dataset_roots"][0][0])
    sidecar = Path(identity["sidecar_paths"][0])
    episodes = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for r in pq.read_table(path).to_pylist():
            episodes[int(r["episode_index"])] = r
    train = identity["active_episodes"][0]
    assert train == [i for i in range(1000) if i % 200 < 190]
    lengths = np.asarray([int(episodes[i]["length"]) for i in range(1000)])
    offsets = np.full(1000, -1, np.int64)
    offsets[train] = np.cumsum([0] + [int(lengths[i]) for i in train[:-1]])
    size = sum(int(lengths[i]) for i in train)
    assert size == base["dataset_length"]
    high = [i for t in base["tasks"] for i in t["high"]]
    eligible = np.zeros(size, bool)
    for t in base["tasks"]:
        for a, b in t["critical_low_ranges"] + t["other_low_ranges"]:
            assert not eligible[a:b].any()
            eligible[a:b] = True
    assert branch_fingerprint({"high": high, "low_ranges": ranges(eligible)}, size) == base["branch_fingerprint"]
    full16 = np.zeros(size, bool)
    side = pq.ParquetFile(sidecar)
    boundaries = [k for k in ("action_horizon_end", "segment_end", "skill_end", "primitive_end") if k in side.schema_arrow.names]
    for batch in side.iter_batches(batch_size=65536, columns=["episode_index", "frame_index", "memlite_branch"] + boundaries):
        ep = batch.column("episode_index").to_numpy(zero_copy_only=False).astype(np.int64)
        f = batch.column("frame_index").to_numpy(zero_copy_only=False).astype(np.int64)
        branch = np.asarray(batch.column("memlite_branch").to_pylist())
        keep = (offsets[ep] >= 0) & (branch == "low")
        if not keep.any():
            continue
        ep, f = ep[keep], f[keep]
        end = lengths[ep].copy()
        for key in boundaries:
            val = pc.fill_null(batch.column(key), -1).to_numpy(zero_copy_only=False)[keep]
            end = np.minimum(end, np.where(val >= 0, val, lengths[ep]))
        logical = offsets[ep] + f
        full16[logical] = (end - f >= 16) & eligible[logical]
    category = np.zeros(size, np.uint8)
    files = defaultdict(list)
    for ep in train:
        r = episodes[ep]
        files[root / f'data/chunk-{r["data/chunk_index"]:03d}/file-{r["data/file_index"]:03d}.parquet'].append(ep)
    done = 0
    witnesses = defaultdict(list)
    for path, eps in sorted(files.items()):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action", "observation.state"],
                              filters=[("episode_index", "in", eps)])
        for ep in eps:
            data = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
            assert len(data) == lengths[ep]
            assert np.array_equal(data["frame_index"].to_numpy(), np.arange(len(data)))
            action = data["action"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1, 23)
            state = data["observation.state"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1, 61)
            c = classify_motion_windows(action, state)
            a, b = int(offsets[ep]), int(offsets[ep] + lengths[ep])
            category[a:b] = np.where(full16[a:b], c, 0)
            for code in (1, 2, 3):
                if len(witnesses[(ep // 200, code)]) < 10:
                    frames = np.flatnonzero(category[a:b] == code)
                    if len(frames):
                        f = int(frames[len(frames)//2])
                        witnesses[(ep // 200, code)].append({"episode": ep, "frame": f,
                            "instance_id": int(episodes[ep]["task_instance_id"]),
                            "state_yaw": float(state[f, 2]), "action_yaw16": action[f:f+16, 2].tolist(),
                            "gripper16": action[f:f+16, [14, 22]].tolist()})
            done += 1
            if done % 100 == 0:
                print(json.dumps({"train_episodes_audited": done}), flush=True)
    tasks = []
    names = {1: "yaw_stop", 2: "yaw_start_reverse", 3: "gripper", 0: "other"}
    for task in range(5):
        a, b = int(offsets[task * 200]), int(offsets[task * 200 + 189] + lengths[task * 200 + 189])
        strata = {name: ranges(eligible[a:b] & (category[a:b] == code), a) for code, name in names.items()}
        counts = {name: sum(y-x for x,y in rs) for name, rs in strata.items()}
        assert all(v >= 4 for v in counts.values())
        task_high = [i for i in high if a <= i < b]
        tasks.append({"task_id": task, "high": task_high, "high_original": task_high,
                      "high_recovery": [], "low_strata": strata, "counts": counts,
                      "witnesses": {names[c]: witnesses[(task, c)] for c in (1, 2, 3)}})
    assert done == 950
    output = {"schema_version": 2, "ready_for_training": False, "dataset_length": size,
        "branch_fingerprint": base["branch_fingerprint"], "view_identity": identity, "tasks": tasks,
        "low_quotas": {"yaw_stop": 1, "yaw_start_reverse": 1, "gripper": 1, "recovery": 1, "other": 3},
        "high_recovery_period": 4, "base_index_sha256": hashlib.sha256(args.base_index.read_bytes()).hexdigest(),
        "train_source_instances": {str(t): [int(episodes[ep]["task_instance_id"]) for ep in train if ep//200 == t] for t in range(5)},
        "eval_source_instances": {str(t): [int(episodes[ep]["task_instance_id"]) for ep in range(t*200+190, (t+1)*200)] for t in range(5)},
        "definitions": {"horizon": 16, "stable_steps": 4, "moving_abs_yaw": .05,
            "stopped_abs_yaw": .01, "priority": "yaw_stop > yaw_start_reverse > gripper > other",
            "partial_windows": "preserved in other; never classified from padded/future-primitive actions"}}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"complete": True, "counts": [t["counts"] for t in tasks],
                      "ready_for_training": False}), flush=True)


if __name__ == "__main__":
    main()
