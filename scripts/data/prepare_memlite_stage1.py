"""Build a compact, unreleased candidate manifest; never overwrite raw data.

A separate evidence receipt must pass clock/visual/loader/split checks before
the trainer will accept this directory. Fixed per-episode stride phases are
shared by high and low. Original five-task holdouts remain held out by source
identity, and newly chosen holdouts are grouped by task instance.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

from g05.data.memlite_stage1_labels import compile_episode, anchor_records, fixed_phase
from g05.utils.memlite_skill_protocol import canonical_json
from g05.utils.training.stage1_runtime import atomic_json, sha256


def split_episodes(rows, old):
    prior = {int(x["episode_index"]): x for x in old["episodes"]}
    prior_group = {}
    by_identity = {(int(r["task_index"]), int(r["raw_episode_id"]), int(r["task_instance_id"])): r for r in rows}
    protected_episodes = {}
    for split, indices in old["splits"].items():
        for episode in indices:
            row = prior[episode]
            identity = tuple(int(row[k]) for k in ("task_index", "raw_episode_id", "task_instance_id"))
            current = by_identity.get(identity)
            if current is None or int(current["length"]) != row["length"]:
                raise ValueError("Old source identity/clock is missing or changed")
            group = (identity[0], identity[2])
            if group in prior_group and prior_group[group] != split:
                raise ValueError("Old TRAIN and eval share a source instance")
            prior_group[group] = split
            protected_episodes[int(current["episode_index"])] = split
    groups = defaultdict(set)
    for row in rows:
        task, instance = int(row["task_index"]), int(row["task_instance_id"])
        if instance >= 301:
            raise ValueError("Public-test instance must not enter demonstration preparation")
        groups[task].add(instance)
    assignments = dict(prior_group)
    for task, instances in groups.items():
        target = max(1, round(len(instances) * .05))
        known_eval = sum(assignments.get((task, i)) == "eval" for i in instances)
        fresh = [i for i in instances if (task, i) not in assignments]
        fresh.sort(key=lambda i: hashlib.sha256(f"stage1-split-17:{task}:{i}".encode()).digest())
        if target - known_eval > len(fresh):
            raise ValueError("Cannot create holdout without moving old TRAIN into validation")
        chosen = set(fresh[:max(0, target - known_eval)])
        for i in fresh:
            assignments[(task, i)] = "eval" if i in chosen else "train"
    return assignments, protected_episodes


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--old-split", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.root / "meta/info.json").read_text())
    if info["fps"] != 30 or info["features"]["action"]["shape"] != [23]:
        raise ValueError("Unexpected embodiment or data clock")
    paths = sorted((args.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    rows = sorted([r for p in paths for r in pq.read_table(p).to_pylist()], key=lambda r: r["episode_index"])
    if len(rows) != 20000 or {r["task_index"] for r in rows} != set(range(100)):
        raise ValueError("Expected the fixed official 100-task corpus")
    old = json.loads(args.old_split.read_text())
    assignments, protected = split_episodes(rows, old)
    task_rows = pq.read_table(args.root / "meta/tasks.parquet").to_pylist()
    tasks = {str(r["task_index"]): r["task"].replace("_", " ") for r in task_rows}
    old_rows = {x["episode_index"]: x for x in old["episodes"]}
    # Entire source episodes are conservatively excluded: prior audits found
    # ambiguous recovery at ep27 and contradictory handover at ep821. This also
    # prevents their issued-command histories from leaking back downstream.
    quarantined_identities = {tuple(old_rows[i][k] for k in ("task_index", "raw_episode_id", "task_instance_id"))
                              for i in (27, 821)}
    offsets, candidates, task_ids = [], {"train": [], "eval": []}, {"train": [], "eval": []}
    exclusions, parent_warnings, summary = [], [], defaultdict(Counter)
    annotation_hash = hashlib.sha256()
    with (args.output / "episodes.jsonl").open("wb") as stream:
        for serial, row in enumerate(rows):
            if time.monotonic() - started > 3500:
                raise TimeoutError("CPU preparation budget reached; candidate remains unreleased")
            task = int(row["task_index"])
            identity = tuple(int(row[k]) for k in ("task_index", "raw_episode_id", "task_instance_id"))
            raw = (args.root / row["annotation_path"]).read_bytes()
            annotation_hash.update(row["annotation_path"].encode() + b"\0" + hashlib.sha256(raw).digest())
            annotation = json.loads(raw)
            split = assignments[(task, int(row["task_instance_id"]))]
            try:
                if identity in quarantined_identities:
                    raise ValueError("Preserved prior visual-audit quarantine (whole source episode)")
                issues = []
                segments = compile_episode(annotation, row, tasks[str(task)], issues=issues)
                if issues:
                    parent_warnings.append(dict(episode_index=row["episode_index"], issues=issues))
                for seg in segments:
                    handovers = defaultdict(set)
                    for skill in seg["skills"]:
                        if skill["verb"] == "HANDOVER":
                            handovers[skill["target"]].add(skill["arm"])
                    if any({"LEFT_TO_RIGHT", "RIGHT_TO_LEFT"} <= directions for directions in handovers.values()):
                        raise ValueError("Contradictory simultaneous handover directions")
                anchors = anchor_records(segments, fixed_phase(row))
                if not anchors:
                    raise ValueError("No eligible fixed-phase anchors")
            except (ValueError, TypeError, KeyError) as error:
                exclusions.append(dict(episode_index=row["episode_index"], identity=identity,
                                       reason=str(error), split=split))
                segments, anchors = [], []
            phase = fixed_phase(row)
            record = dict(row=row, task_name=tasks[str(task)], split=split, phase=phase,
                          annotation_sha256=hashlib.sha256(raw).hexdigest(), segments=segments)
            offsets.append(stream.tell())
            stream.write(canonical_json(record).encode() + b"\n")
            n = len(anchors)
            candidates[split].append(np.column_stack((np.full(n, serial, dtype=np.int32), np.arange(n, dtype=np.int32))))
            task_ids[split].append(np.full(n, task, dtype=np.int16))
            summary[str(task)].update({f"{split}_episodes": int(n > 0), f"{split}_anchors": n,
                                       "excluded_episodes": int(n == 0)})
            if serial % 2000 == 0:
                print(json.dumps(dict(episodes=serial, excluded=len(exclusions), seconds=time.monotonic()-started)), flush=True)
    np.save(args.output / "episode_offsets.npy", np.asarray(offsets, dtype=np.int64))
    fixed_eval = []
    for split in ("train", "eval"):
        array = np.concatenate(candidates[split])
        tasks_array = np.concatenate(task_ids[split])
        np.save(args.output / f"{split}_candidates.npy", array)
        np.save(args.output / f"{split}_tasks.npy", tasks_array)
        if split == "eval":
            for task in range(100):
                ids = np.flatnonzero(tasks_array == task)
                if len(ids) < 32:
                    raise ValueError(f"Insufficient held-out task {task}")
                # Spread over the entire held-out task, including all episodes.
                fixed_eval.extend(ids[np.linspace(0, len(ids)-1, 32, dtype=np.int64)].tolist())
    np.save(args.output / "fixed_eval_indices.npy", np.asarray(fixed_eval, dtype=np.int64))
    atomic_json(args.output / "exclusions.json", exclusions)
    atomic_json(args.output / "parent_warnings.json", parent_warnings)
    atomic_json(args.output / "split_provenance.json", dict(old=old,
        assignments=[dict(task=k[0], instance=k[1], split=v) for k, v in sorted(assignments.items())],
        protected_episodes=protected))
    files = {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    manifest = dict(format_version="memlite-stage1-data-v1", status="CANDIDATE_REQUIRES_ACCEPTANCE",
        raw_root=str(args.root), source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_revision="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2", seed=17, stride=16,
        task_names=tasks, task_counts=dict(summary), episodes=len(rows), excluded_episodes=len(exclusions),
        annotation_corpus_sha256=annotation_hash.hexdigest(),
        meta_sha256={str(p.relative_to(args.root)): sha256(p) for p in paths + [args.root/"meta/info.json", args.root/"meta/tasks.parquet"]},
        old_split_sha256=sha256(args.old_split), files=files,
        outcome_supervision=False, terminal_supervision=False, clock="published local frame_index; offset=0; overflow quarantined",
        normalizer="inherit each initialized checkpoint coordinate system; no evaluation refitting",
        seconds=time.monotonic()-started)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(dict(status=manifest["status"], excluded=len(exclusions), seconds=manifest["seconds"])), flush=True)


if __name__ == "__main__":
    main()
