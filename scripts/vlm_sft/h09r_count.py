"""TRAIN-only bounded feasibility counts; never creates training examples.

Run from a pinned Git blob with --reference-source pointing to the unchanged
H09 common.py/prepare.py source. Reads no validation/test examples or videos.
All extra episode selection is fixed before labels, outcomes or images.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-source", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--extra-per-task", type=int, default=12)
    ap.add_argument("--max-seconds", type=int, default=420)
    args = ap.parse_args()
    started = time.monotonic()
    sys.path.insert(0, str(args.reference_source / "scripts/vlm_sft"))
    from common import (GRIPS, JOINTS, POSITIONS, QUATERNIONS, TOKENS,
                        classify_window, sha, skill_text)
    from prepare import ROOT, LABELS, RELEASE

    tasks = (0, 1, 3)
    groups_path = args.data / "groups.json"
    groups = json.loads(groups_path.read_text())
    meta_paths = sorted((ROOT / "meta/episodes").rglob("*.parquet"))
    episodes = [e for p in meta_paths for e in pq.read_table(p).to_pylist()]
    protected = {(g["episode"]["task_index"], g["episode"]["task_instance_id"])
                 for g in groups.values() if g["split"] != "train"}
    development = {(0, 138), (3, 242)}
    original_eval = set()
    for task in tasks:
        rows = sorted((e for e in episodes if e["task_index"] == task),
                      key=lambda e: e["episode_index"])
        assert len(rows) == 200
        original_eval.update((task, e["task_instance_id"]) for e in rows[-10:])
    blocked = original_eval | protected | development
    eligible = [e for e in episodes if e["task_index"] in tasks and
                (e["task_index"], e["task_instance_id"]) not in blocked]
    old_train = {int(ep) for ep, g in groups.items() if g["split"] == "train"}
    old_instances = {(g["episode"]["task_index"], g["episode"]["task_instance_id"])
                     for g in groups.values() if g["split"] == "train"}
    selected = [("h09_train", e) for e in eligible if e["episode_index"] in old_train]
    assert len(selected) == len(old_train) == 36
    for task in tasks:
        pool = [e for e in eligible if e["task_index"] == task and
                (task, e["task_instance_id"]) not in old_instances]
        pool.sort(key=lambda e: hashlib.sha256(
            f'h09r:41:{task}:{e["task_instance_id"]}'.encode()).hexdigest())
        seen = set()
        for e in pool:
            if e["task_instance_id"] in seen:
                continue
            selected.append(("additional_train", e))
            seen.add(e["task_instance_id"])
            if len(seen) == args.extra_per_task:
                break
        assert len(seen) == args.extra_per_task
    selected.sort(key=lambda x: x[1]["episode_index"])
    quarantine = defaultdict(list)
    quarantine_path = RELEASE / "quarantine_ranges.parquet"
    for r in pq.read_table(quarantine_path).to_pylist():
        quarantine[int(r["episode_index"])].append((int(r["frame_start"]), int(r["frame_end"])))
    counts = defaultdict(Counter)
    source_episodes = defaultdict(set)
    rejects = defaultdict(Counter)
    token_counts = defaultdict(Counter)
    stratum_tokens = defaultdict(Counter)
    amount_ratios = defaultdict(list)
    sources = []

    def rest_flags(states, f):
        v = states[f, :3]
        low_base = np.linalg.norm(v[:2]) < .02 and abs(v[2]) < .03
        if f < 3 or not low_base:
            return bool(low_base), False
        past = states[f-3:f+1]
        for arm in POSITIONS:
            if np.linalg.norm(np.diff(past[:, POSITIONS[arm]], axis=0), axis=1).max()*30 >= .02:
                return True, False
            rot = Rotation.from_quat(past[:, QUATERNIONS[arm]])
            if np.linalg.norm((rot[1:]*rot[:-1].inv()).as_rotvec(), axis=1).max()*30 >= .05:
                return True, False
            if abs(np.diff(past[:, GRIPS[arm]].mean(axis=1))).max()*30 >= .005:
                return True, False
        if abs(np.diff(past[:, JOINTS["torso"]], axis=0)).max()*30 >= .03:
            return True, False
        return True, True

    for cohort, e in selected:
        if time.monotonic()-started > args.max_seconds:
            raise RuntimeError("Bounded scan incomplete; do not publish full counts")
        ep, task = e["episode_index"], e["task_index"]
        path = ROOT / f'data/chunk-{e["data/chunk_index"]:03d}/file-{e["data/file_index"]:03d}.parquet'
        raw = pq.read_table(path, filters=[("episode_index", "=", ep)],
                            columns=["frame_index", "action", "observation.state"]).sort_by("frame_index")
        states = np.asarray(raw["observation.state"].to_pylist(), dtype=float)
        actions = np.asarray(raw["action"].to_pylist(), dtype=float)
        assert np.array_equal(raw["frame_index"].to_numpy(), np.arange(e["length"]))
        cols = ["frame_index", "active_skills_semantic_json", "low_action_supervision_mask",
                "action_horizon_end", "segment_start", "segment_end", "memlite_branch", "source_kind"]
        label_rows = pq.read_table(LABELS, filters=[("episode_index", "=", ep)], columns=cols).to_pylist()
        labels = {r["frame_index"]: r for r in label_rows}
        assert all(r["source_kind"] == "original_demo" for r in label_rows)
        h = hashlib.sha256(states.tobytes()+actions.tobytes())
        h.update(json.dumps(label_rows, sort_keys=True, separators=(",", ":")).encode())
        sources.append({"cohort": cohort, "task": task, "episode": ep,
                        "instance": e["task_instance_id"], "frames": e["length"],
                        "parquet": str(path), "extracted_arrays_and_labels_sha256": h.hexdigest()})
        grid = set(range(0, len(states)-16, 16))
        # Add only the first released-low frame of each existing semantic phase.
        # These are count-only overlapping candidates, never causal history.
        first_low = {}
        for f, r in labels.items():
            if r["low_action_supervision_mask"] and r["memlite_branch"] == "low":
                start = r["segment_start"]
                first_low[start] = min(first_low.get(start, f), f)
        supplemental = set(first_low.values()) - grid
        for scheme, frames in (("stride16", sorted(grid)), ("phase_first_low_extra", sorted(supplemental))):
            for f in frames:
                if f+16 >= len(states):
                    continue
                r = labels.get(f)
                if r is None:
                    counts[f"{cohort}/t{task}/{scheme}/MISSING"]["windows"] += 1
                    continue
                skills = json.loads(r["active_skills_semantic_json"])
                verbs = "+".join(sorted({x.get("verb", "UNKNOWN") for x in skills})) or "UNKNOWN"
                key = f"{cohort}/t{task}/{scheme}/{verbs}"
                c = counts[key]; c["windows"] += 1
                source_episodes[key+"/present"].add(ep)
                low, rest = rest_flags(states, f)
                phase = 0 <= f-r["segment_start"] <= 16
                c["low_base_windows"] += low; c["near_rest_windows"] += rest
                c["phase_start_windows"] += phase
                released = (r["low_action_supervision_mask"] and r["memlite_branch"] == "low"
                            and min(r["action_horizon_end"], r["segment_end"]) > f+16
                            and not any(lo <= f+16 and hi >= f for lo, hi in quarantine[ep]))
                if not released:
                    rejects[key]["not_released_same_skill_window"] += 1
                    continue
                try:
                    skill_text(skills)
                except ValueError:
                    rejects[key]["unbound_instruction"] += 1
                    continue
                c["released_same_skill"] += 1
                c["released_near_rest"] += rest; c["released_phase_start"] += phase
                token, evidence = classify_window(states[f:f+17], actions[f:f+16], actions[f-1] if f else None)
                if token is not None and token not in TOKENS:
                    token = None; evidence["reject"] = "executor_contract_mismatch"
                if token is None:
                    rejects[key][evidence["reject"]] += 1
                    continue
                c["pure_direction_accepted"] += 1
                token_counts[key][token] += 1; source_episodes[key+"/accepted"].add(ep)
                for name, yes in (("low_base", low), ("near_rest", rest), ("phase_start", phase),
                                  ("near_rest_and_phase_start", rest and phase)):
                    if yes:
                        c["accepted_"+name] += 1
                        stratum_tokens[key+"/"+name][token] += 1
                        source_episodes[key+"/"+name].add(ep)
                if not token.startswith("BASE_") and token != "HOLD":
                    c["manipulation_direction"] += 1
                # Necessary endpoint-only amplitude check, NOT executor equivalence.
                if token.startswith(("LEFT_", "RIGHT_", "BOTH_")) and token.split("_",1)[1] in ("FORWARD","BACK","LEFT","RIGHT","UP","DOWN"):
                    from semantic_robot.v2.protocol import TRANSLATIONS
                    part, move = token.lower().split("_", 1)
                    active = ("left", "right") if part == "both" else (part,)
                    required = np.asarray(TRANSLATIONS[move])*.01
                    ok = all(np.linalg.norm(np.asarray(evidence["delta_eef_base_m"][a])-(required if a in active else 0)) <= .0025
                             for a in POSITIONS)
                    ok &= max(evidence["max_rotation_excursion_rad"].values()) <= np.deg2rad(1.5)
                    c["translation_endpoint_checked"] += 1
                    c["translation_endpoint_within_native_10mm_tolerance"] += bool(ok)
                    for a in active:
                        amount_ratios[key].append(float(np.linalg.norm(evidence["delta_eef_base_m"][a])/.01))
        print(json.dumps({"completed_episode": ep, "cohort": cohort}), file=sys.stderr, flush=True)
    # TRAIN examples only: confirm comparison refers to the actual old release.
    old_rows = [json.loads(x) for x in (args.data / "train.jsonl").read_text().splitlines()]
    assert all(r["episode_index"] in old_train and (r["task_id"], r["instance_id"]) not in blocked for r in old_rows)
    result = {"status": "complete_counts_not_training_data", "wall_seconds": time.monotonic()-started,
              "tasks": tasks, "reference_source": str(args.reference_source), "new_model_calls": 0,
              "new_controls": 0, "training_updates": 0, "horizon_frames": 16, "source_hz": 30,
              "additional_episode_selection": "sha256(h09r:41:task:instance), first 12 unique per task",
              "exclusions": {"original_5pct_all_instances": sorted(original_eval),
                             "h09_validation_test_all_instances": sorted(protected),
                             "development_instances": sorted(development)},
              "eligible_pool_episodes": dict(Counter(e["task_index"] for e in eligible)),
              "eligible_pool_instances": {t: len({e["task_instance_id"] for e in eligible if e["task_index"] == t}) for t in tasks},
              "old_train_rows": len(old_rows), "old_train_tokens": dict(Counter(r["target"] for r in old_rows)),
              "old_train_file_sha256": sha(args.data / "train.jsonl"),
              "source_identity": {"groups_sha256": sha(groups_path), "labels_sha256": sha(LABELS),
                                  "quarantine_sha256": sha(quarantine_path),
                                  "release_manifest_sha256": sha(RELEASE / "composite_release_manifest.json"),
                                  "common_sha256": sha(args.reference_source / "scripts/vlm_sft/common.py"),
                                  "episode_meta_sha256": {str(p): sha(p) for p in meta_paths}},
              "sources": sources, "counts": dict(counts), "rejections": dict(rejects),
              "tokens": dict(token_counts), "stratum_tokens": dict(stratum_tokens),
              "source_episodes_by_stratum": {k: sorted(v) for k, v in source_episodes.items()},
              "translation_amount_ratio_quantiles": {k: {"n": len(v), "min_q25_median_q75_max": np.quantile(v,[0,.25,.5,.75,1]).tolist()} for k,v in amount_ratios.items()},
              "limits": ["Only 72 predetermined train-source episodes receive raw-window inspection, not all eligible episodes.",
                         "No validation/test examples, images or outcomes read; groups metadata used only to exclude instances.",
                         "Phase-start candidates may overlap the stride grid: count separately, never sum as independent samples/history.",
                         "Pure direction and endpoint compatibility do not establish native executor trajectory, timing, contact or success.",
                         "Near-rest is a fixed past-three-frame kinematic screen, not a causal shortcut/vision ablation."]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
