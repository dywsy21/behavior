"""Prepare the registered additional TRAIN reference from immutable raw source.

No simulator import, reset, model or successful-label construction. Held-out
groups are rejected even though their original dataset mode was TRAIN.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from common import CAMERAS, sha, skill_text, write_json
from native_grasp_sources import reserve_groups, audit_source
from native_reference_profile import PROFILE, PURPOSE, ROOT as EXPERIMENT_ROOT, SOURCE, HELDOUT, BUDGET
from native_teacher_contract import SCHEMA as TEACHER_SCHEMA
from native_teacher_reference_prepare import SCHEMA, LABEL_COLUMNS, validate_segment

COUNTS_SHA = "d94850ebfeadeef8b28f618e543ebae7cb7a18bd127c5d79ba6f4c18c0f015e3"
AUDIT_SHA = "5cb0ed5a5c4ca48e5148dbeab4406c373242bcd3764c4ebb5a8d9864c7934278"
ROBOT_SHA = "a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93"


def select_training_source(counts, audit):
    if audit.get("schema") != "h09x-grasp-instance-split-v1" or audit.get("counts_sha256") != COUNTS_SHA:
        raise ValueError("Wrong source audit identity")
    expected = reserve_groups(counts)
    if len(audit.get("groups",[])) != len(expected): raise ValueError("Extra or missing reserved source group")
    for split, source in expected:
        rows = [r for r in audit["groups"] if r["source"] == source]
        if len(rows) != 1 or rows[0]["split"] != split:
            raise ValueError("Source split changed after reservation")
    matches = [r for r in audit["groups"] if [r["source"][k] for k in ("task", "episode", "instance")] == SOURCE]
    if len(matches) != 1 or matches[0]["split"] != "train": raise ValueError("Not the registered TRAIN source")
    row = matches[0]; selected = row["selected_earliest_eligible"]
    if (selected is None or selected["start"] != 596 or selected["end"] != 1186 or
            selected["hand"] != "right" or selected["skill"]["verb"] != "GRASP" or
            selected["near_prefix_controls"] != 993):
        raise ValueError("Registered complete GRASP reference changed")
    return row


def prepare(counts_path, audit_path, robot_template, output):
    import pyarrow.parquet as pq
    from prepare import ROOT, LABELS, RELEASE
    started = time.monotonic(); output = Path(output).resolve()
    if output != EXPERIMENT_ROOT/"reference_prepare_train114_v1" or output.exists():
        raise ValueError("Fresh registered H09Y preparation path required")
    if sha(counts_path) != COUNTS_SHA or sha(audit_path) != AUDIT_SHA:
        raise ValueError("Counts/source audit bytes changed")
    counts = json.loads(Path(counts_path).read_text()); audit = json.loads(Path(audit_path).read_text())
    registered = select_training_source(counts, audit); source = registered["source"]
    if sha(LABELS) != counts["source_identity"]["labels_sha256"] or sha(RELEASE/"quarantine_ranges.parquet") != counts["source_identity"]["quarantine_sha256"]:
        raise ValueError("Original annotation release identity changed")
    meta = []
    for path, expected in counts["source_identity"]["episode_meta_sha256"].items():
        if sha(path) != expected: raise ValueError("Episode metadata changed")
        meta.extend(pq.read_table(path).to_pylist())
    matches = [e for e in meta if e["episode_index"] == SOURCE[1]]
    if len(matches) != 1: raise ValueError("Ambiguous source episode")
    e = matches[0]
    actual_parquet = ROOT/f"data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet"
    if (e["task_index"] != SOURCE[0] or e["task_instance_id"] != SOURCE[2] or
            e["length"] != source["frames"] or str(actual_parquet) != source["parquet"] or e["tasks"] != ["picking_up_trash"]):
        raise ValueError("Actual task/instance/episode/parquet metadata mismatch")
    table = pq.read_table(actual_parquet, filters=[("episode_index", "=", SOURCE[1])],
                          columns=["frame_index", "action", "observation.state"]).sort_by("frame_index")
    labels = pq.read_table(LABELS, filters=[("episode_index", "=", SOURCE[1])], columns=LABEL_COLUMNS).to_pylist()
    quarantine = pq.read_table(RELEASE/"quarantine_ranges.parquet").to_pylist()
    fresh = audit_source(source, table, labels, quarantine)
    if fresh != {k: v for k, v in registered.items() if k != "split"}:
        raise ValueError("Fresh complete arrays/labels differ from fixed source audit")
    selected = fresh["selected_earliest_eligible"]; start, end = selected["start"], selected["end"]
    semantic = next(r["active_skills_semantic_json"] for r in labels if r["frame_index"] == start and
                    r["source_kind"] == "original_demo" and r["memlite_branch"] == "low")
    all_states = np.asarray(table["observation.state"].to_pylist(), float)
    all_actions = np.asarray(table["action"].to_pylist(), float)
    prefix, segment, states, selection = validate_segment(all_states, all_actions, table["frame_index"].to_numpy(),
        labels, source, start, end, semantic, quarantine)
    template = json.loads(Path(robot_template).read_text())
    robot_path = Path(template["robot_config_path"])
    if template["robot_config_sha256"] != ROBOT_SHA or sha(robot_path) != ROBOT_SHA:
        raise ValueError("Audited actual robot configuration changed")
    for mount in ("/mnt/sdc1", "/mnt/nvme_tmp"):
        if shutil.disk_usage(mount).free < 80*1024**3: raise RuntimeError("Disk reserve before preparation")
    folder = output/"task_1"; folder.mkdir(parents=True, exist_ok=False)
    for name, values in (("prefix", prefix), ("segment", segment), ("source_states", states)):
        np.save(folder/(name+".npy"), values, allow_pickle=False)
    pilot = {"task": 1, "episode": 264, "instance": 114, "frame": start, "verb": "GRASP"}
    label = next(r for r in labels if r["frame_index"] == start and r["source_kind"] == "original_demo" and r["memlite_branch"] == "low")
    reference = {"schema": TEACHER_SCHEMA, "source": source, "pilot": pilot,
        "active_instruction": skill_text(semantic), "private_original_semantic_json": semantic, "source_label": label,
        "source_states": all_states[start:start+17].tolist(), "source_actions": all_actions[start:start+16].tolist(),
        "prefix_controls": start, "proposals": {"training_eligible": False, "reason": "COMPLETE_EXPERT_REFERENCE_NOT_NATIVE_BC"},
        "source_review_media": {}, "training_eligible": False, "data_purpose": PURPOSE}
    for view, camera in CAMERAS.items():
        stem = "videos/observation.rgb."+camera
        reference["source_review_media"][view] = {"video": str(ROOT/f"{stem}/chunk-{e[stem+'/chunk_index']:03d}/file-{e[stem+'/file_index']:03d}.mp4"),
            "start_seconds": e[stem+"/from_timestamp"]+start/30, "frame_start": start, "frame_end": end-1}
    skill = json.loads(semantic)[0]
    spec = {"verb": "GRASP", "hand": "right", "support_hand": None, "target": skill["target"],
            "destination": skill["destination"], "payloads": [], "goal_frame": "target"}
    write_json(folder/"teacher_reference.json", reference); write_json(folder/"private_spec.json", spec)
    write_json(folder/"segment_labels.json", [r for r in labels if start <= r["frame_index"] < end])
    write_json(folder/"label_selection_audit.json", selection)
    window = {"kind": "native_oracle_low_window", "immutable": True, "window_id": "h09y-train-t1-e264-i114-full-reference",
        "task_name": e["tasks"][0], "official_mode": "train", "instance_id": 114, "seed": 0, "max_steps": end+14,
        "robot_config_path": str(robot_path), "robot_config_sha256": ROBOT_SHA, "max_chunks": 1, "execute_steps": 1,
        "prefix_actions_path": str(folder/"prefix.npy"), "prefix_actions_sha256": sha(folder/"prefix.npy"),
        "semantic_subgoal": {"parent_goal": e["tasks"][0], "active_skills_semantic_json": "[]", "active_skills_text": reference["active_instruction"]}}
    write_json(folder/"window.json", window)
    row = {"task": 1, "episode": 264, "instance": 114, "start": start, "end": end, "verb": "GRASP", "hand": "right",
        "support_hand": None, "payloads": [], "files_sha256": {p.name: sha(p) for p in folder.iterdir()},
        "prefix_controls": len(prefix), "segment_controls": len(segment), "controls_including_tail_and_final_hold": end+13,
        "source_arrays_and_labels_sha256": source["extracted_arrays_and_labels_sha256"]}
    manifest = {"schema": SCHEMA, "status": "PREPARED_NO_RESET_NO_OUTCOME_NO_SEED", "reference_profile": PROFILE,
        "purpose": PURPOSE, "held_out_instance_groups": HELDOUT, "sources": [row], "h09r_counts_sha256": COUNTS_SHA,
        "source_audit_sha256": AUDIT_SHA, "robot_template_sha256": sha(robot_template),
        "new_controls": 0, "new_resets": 0, "wall_seconds": time.monotonic()-started}
    manifest["bytes"] = sum(p.stat().st_size for p in folder.iterdir())
    if manifest["bytes"] > 2*1024**2: raise RuntimeError("Bounded source preparation exceeded 2MiB")
    write_json(output/"preparation.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--counts", required=True); p.add_argument("--source-audit", required=True)
    p.add_argument("--robot-template", required=True); p.add_argument("--output", required=True); a = p.parse_args()
    print(json.dumps(prepare(a.counts, a.source_audit, a.robot_template, a.output), indent=2))
