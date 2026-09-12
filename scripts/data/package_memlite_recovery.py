"""Validate and label real recovery data; manual review gates trainable output."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from g05.data.memlite_sidecar import MemLiteSidecar
from g05.utils.common.task_event_sampler import branch_fingerprint, canonical_ranges


class RecoveryQualityRejected(ValueError):
    """A valid recording whose local expert execution is unsuitable to imitate."""


def validate_approach_dynamics(states, actions, stages):
    approach = np.asarray(stages) == "approach"
    # An approach begins only after measured yaw <.04 and small heading error.
    # A >.20 rad/s mismatch therefore is not normal braking lag; it indicates
    # uncommanded rotation/contact, even when endpoint distance later passes.
    if approach.any():
        error = np.abs(states[approach, 2] - actions[approach, 2])
        if float(error.max()) > .20:
            raise RecoveryQualityRejected(f"Uncommanded approach rotation; max yaw tracking error={error.max():.4f}")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def initial_intent(sidecar, episode, target):
    # An annotation's first primitive can start after frame 0 (e.g. frame 198
    # after an unlabeled lead-in). Read the first *annotated* initial primitive,
    # not an arbitrary first-16-frame cutoff or a later completed primitive.
    frames = sidecar._load_episode(episode)
    for frame in sorted(frames):
        for row in frames[frame]:
            if row.get("ignore") or row.get("memory_input_corruption", "none") != "none":
                continue
            if row.get("primitive_idx") != 0:
                continue
            intent = row.get("intent", "")
            if intent and target in intent:
                return intent, int(frame)
    raise ValueError(f"Missing initial primitive for episode={episode}, target={target}")


def check_trajectory(root, manifest, source_index):
    task, instance = int(manifest["task_id"]), int(manifest["instance_id"])
    split = manifest["source_split"]
    if split not in {"train", "eval"} or manifest["mode"] != "train":
        raise ValueError("Non-training simulator source or missing data split")
    if instance not in source_index[f"{split}_source_instances"][str(task)]:
        raise ValueError("Recovery source is not in original metadata split")
    if not manifest["accepted"] or manifest["abort"] is not None:
        raise ValueError("Unsuccessful recovery cannot be training supervision")
    source_path = Path(manifest["execution_source"])
    if digest(source_path) != manifest["script_sha256"]:
        raise ValueError("Collector execution source was not retained immutably")
    with np.load(root / "trajectory.npz", allow_pickle=False) as data:
        states, actions = data["states"], data["actions"]
    if states.shape != (len(actions), 61) or actions.shape[1:] != (23,):
        raise ValueError("Incorrect official raw action/state layout")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite recovery data")
    geometry = manifest["geometry_diagnostic_only"]
    if len(geometry) != len(actions) or any(r["step"] != i for i, r in enumerate(geometry)):
        raise ValueError("Missing, reordered or misaligned executed steps")
    if np.max(np.abs(actions[:, 2])) > .30001 or not set(np.unique(actions[:, [14,22]])) <= {-1., 1.}:
        raise ValueError("Teacher command exceeds original yaw/gripper support")
    start = int(manifest["recovery_start"])
    stages = np.asarray([r["stage"] for r in geometry])
    validate_approach_dynamics(states, actions, stages)
    if stages[start] != "brake" or abs(states[start+32, 2]) > .05:
        raise ValueError("Brake correction was not measured to work")
    yaw = np.unwrap([r["yaw"] for r in geometry[:start]])
    if abs(yaw[-1]-yaw[0]) <= 6.2:
        raise ValueError("No actual full-turn failure before recovery")
    settled = np.flatnonzero(stages == "settle")
    if len(settled) < 90 or np.max(np.abs(states[-32:, :3])) > .06:
        raise ValueError("Recovery did not end in a stable measured stop")
    if abs(geometry[-1]["heading_error"]) >= .10 or geometry[-1]["distance"] > manifest["quality"]["standoff"] + .03:
        raise ValueError("Target reorientation/approach quality gate failed")
    if manifest["quality"]["initial_distance"] > manifest["quality"]["standoff"] + .25:
        if manifest["quality"]["initial_distance"] - geometry[-1]["distance"] < .20:
            raise ValueError("No measured approach progress from a far starting point")
    hashes = {"manifest.json": digest(root / "manifest.json"), "trajectory.npz": digest(root / "trajectory.npz")}
    for frame in manifest["frames"]:
        f = int(frame["step"])
        path = root / frame["file"]
        with np.load(path, allow_pickle=False) as data:
            props = [data[k] for k in data.files if k.endswith("::proprio")]
            if len(props) != 1 or not np.array_equal(props[0], states[f]):
                raise ValueError("RGB anchor proprio differs from pre-action state[t]")
            if int(np.asarray(data["task_id"]).reshape(-1)[0]) != task:
                raise ValueError("Official observed task identity mismatch")
        hashes[frame["file"]] = digest(path)
    return states, actions, stages, int(settled[0]), hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-index", type=Path, required=True)
    ap.add_argument("--collections", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tasks-file", type=Path, default=Path("/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos/meta/tasks.jsonl"))
    ap.add_argument("--review", type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index = json.loads(args.source_index.read_text())
    samples = {"train": [], "eval": []}
    trajectories, counts, seen, rejected = [], Counter(), set(), []
    commands = {}
    for line in args.tasks_file.read_text().splitlines():
        r = json.loads(line)
        commands[int(r["task_index"])] = r["task"]
    dataset_root = Path(index["view_identity"]["dataset_roots"][0][0])
    episode_map = {}
    for path in sorted((dataset_root / "meta/episodes").rglob("*.parquet")):
        for r in pq.read_table(path).to_pylist():
            episode_map[int(r["task_index"]), int(r["task_instance_id"])] = int(r["episode_index"])
    side = MemLiteSidecar(index["view_identity"]["sidecar_paths"][0])
    for collection in args.collections:
        for path in sorted(collection.glob("task*_instance*/manifest.json")):
            manifest = json.loads(path.read_text())
            if not manifest.get("accepted"):
                rejected.append({"path": str(path), "reason": manifest.get("abort")})
                continue
            root = path.parent.resolve()
            try:
                states, actions, stages, settle, hashes = check_trajectory(root, manifest, index)
            except RecoveryQualityRejected as error:
                rejected.append({"path": str(path), "reason": str(error), "stage": "execution_dynamics_quality"})
                continue
            task, instance, split = manifest["task_id"], manifest["instance_id"], manifest["source_split"]
            identity = (task, instance)
            if identity in seen:
                raise ValueError("Duplicate source instance; do not count repeated rollouts as independent data")
            seen.add(identity)
            episode = episode_map[identity]
            target = manifest["target"]
            try:
                original_intent, intent_source_frame = initial_intent(side, episode, target)
            except ValueError as error:
                # Some task instances start with a different valid subtask.
                # A fixed-target geometric recovery must not be labeled as
                # recovering an unrelated first intent, even if it moved well.
                rejected.append({"path": str(path), "reason": str(error),
                                 "stage": "initial_intent_target_semantic_check"})
                continue
            intent = f"stop rotating, face [{target}], and approach [{target}]"
            start = manifest["recovery_start"]
            previous_memory = f"Task={task}; Completed=none.; Active={original_intent}."
            previous_intent = original_intent
            translation_path = 0.
            last_feedback_step = start
            recovery_exited = False
            for f in range(start, min(settle+64, len(actions)-31), 16):
                phase = stages[f]
                translation_path += float(np.linalg.norm(states[f, :2])) * (f-last_feedback_step) / 30.
                last_feedback_step = f
                feedback = manifest["feedback"] if translation_path <= .20 and not recovery_exited else "none"
                # Describe current evidence, not the next teacher action. At
                # the first brake/align frame that action has not run yet.
                observed = {"brake": "Repeated rotation observed; recovery pause requested.",
                    "align": "Target facing not yet established; reorientation required.",
                    "approach": "Target facing established; approach not yet finished.",
                    "settle": "Target within local approach range; checking base stop."}[phase]
                if phase == "brake" and f > start and abs(float(states[f, 2])) <= .05:
                    observed = "Base stopped after repeated rotation; target facing not yet confirmed."
                status = "REPLAN" if f == start else "CONTINUE"
                resume = phase == "settle" and f >= settle+32
                next_intent = original_intent if resume else intent
                if resume:
                    observed = "Local rotation recovery completed; original manipulation remains pending."
                current_memory = f"Task={task}; Completed=none.; Recovery={observed}"
                common = {"trajectory": str(root), "step": f, "mode": "train", "source_split": split,
                    "task_id": task, "instance_id": instance, "source_episode": episode,
                    "original_intent_source_frame": intent_source_frame,
                    "command": commands[task], "memory": previous_memory, "previous_intent": previous_intent,
                    "intent": next_intent, "memory_update": current_memory, "intent_status": status,
                    "execution_feedback": feedback}
                # Planner target is current observed state. Input memory is the
                # preceding summary, never the target or a future completion.
                samples[split].append({**common, "memlite_branch": "high",
                    "review_id": f"t{task}_i{instance}_f{f}_high"})
                if status == "CONTINUE" and not resume:
                    samples[split].append({**common, "memlite_branch": "low",
                        "memory": current_memory, "previous_intent": next_intent,
                        "review_id": f"t{task}_i{instance}_f{f}_low"})
                previous_memory, previous_intent = current_memory, next_intent
                if resume:
                    # The planner sees the alert on its exit decision; serving
                    # acknowledges it only AFTER that proposal and a real stop.
                    recovery_exited = True
            trajectories.append({"path": str(root), "task_id": task, "instance_id": instance, "split": split,
                                 "hashes": hashes, "quality": manifest["quality"],
                                 "execution_source": manifest["execution_source"],
                                 "execution_source_sha256": manifest["script_sha256"]})
            counts[split, task] += 1
    payload = {"schema_version": 1, "ready_for_training": False, "samples": samples,
        "trajectories": trajectories, "rejected": rejected,
        "source_index_sha256": digest(args.source_index),
        "scope": "Real train-instance local rotation recovery; no synthetic actions and no task-success claims."}
    # The review covers exact content excluding the later approval receipt.
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["review_payload_sha256"] = payload_hash
    if args.review:
        review = json.loads(args.review.read_text())
        if review.get("status") != "approved_by_primary_manual_inspection" or review.get("payload_sha256") != payload_hash:
            raise ValueError("Review is missing, not manual, or refers to different data")
        id_to_sample = {r["review_id"]: r for split in samples.values() for r in split}
        ids = set(id_to_sample)
        inspected = review.get("inspected_samples", [])
        if len(inspected) < 60 or any(r["review_id"] not in ids or len(r.get("notes", "")) < 20 for r in inspected):
            raise ValueError("At least 60 documented, actually inspected sample IDs are required")
        if len({r["review_id"] for r in inspected}) != len(inspected):
            raise ValueError("Duplicate manual-inspection IDs do not count")
        inspected_rows = [id_to_sample[r["review_id"]] for r in inspected]
        for task in range(5):
            local = [r for r in inspected_rows if r["task_id"] == task]
            if len(local) < 10 or {r["memlite_branch"] for r in local} != {"high", "low"}:
                raise ValueError("Manual review must cover both branches and >=10 examples for every task")
            if {r["source_split"] for r in local} != {"train", "eval"}:
                raise ValueError("Manual review must cover training and heldout recovery examples")
        for task in range(5):
            if counts["train", task] < 10 or counts["eval", task] < 2:
                raise ValueError("Need at least 10 accepted train and 2 independent heldout source instances per task")
        payload["ready_for_training"] = True
        payload["manual_review"] = review
    manifest_path = (args.output_dir / "recovery_manifest.json").resolve()
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    if payload["ready_for_training"]:
        augmented = deepcopy(index)
        augmented["ready_for_training"] = True
        offset = int(index["dataset_length"])
        for t in augmented["tasks"]:
            t["low_strata"]["recovery"] = []
        for i, row in enumerate(samples["train"]):
            task = augmented["tasks"][row["task_id"]]
            if row["memlite_branch"] == "high":
                task["high"].append(offset+i)
                task["high_recovery"].append(offset+i)
            else:
                task["low_strata"]["recovery"].append([offset+i, offset+i+1])
        augmented["dataset_length"] = offset + len(samples["train"])
        augmented["view_identity"]["recovery"] = {"manifest": str(manifest_path), "sha256": digest(manifest_path), "split": "train"}
        spec = {"high": [i for t in augmented["tasks"] for i in t["high"]],
                "low_ranges": [r for t in augmented["tasks"] for rs in t["low_strata"].values() for r in rs]}
        augmented["branch_fingerprint"] = branch_fingerprint(spec, augmented["dataset_length"])
        (args.output_dir / "motion_recovery_index_v2.json").write_text(json.dumps(augmented, indent=2) + "\n")
    print(json.dumps({"ready_for_training": payload["ready_for_training"], "review_payload_sha256": payload_hash,
        "counts": {f"{split}_task{task}": count for (split, task), count in counts.items()},
        "sample_counts": {k: len(v) for k,v in samples.items()}, "rejected": len(rejected)}), flush=True)


if __name__ == "__main__":
    main()
