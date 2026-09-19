"""Bounded four-instance SOURCE feasibility audit, not labels or seed success.

Reserve groups before reading their actions/labels; no image/model/physics call.
All instances come from previously hash-counted additional TRAIN sources. The
two held-out groups are reserved for future evaluation, not training admission.
"""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from common import sha, write_json
from native_teacher_reference_prepare import LABEL_COLUMNS, validate_segment

SPLIT = "h09x-grasp-instance-split-v1"


def reserve_groups(counts):
    excluded = {tuple(g) for rows in counts["exclusions"].values() for g in rows}
    excluded |= {(0, 138), (3, 242)}
    pool = [r for r in counts["sources"] if r["task"] == 1 and r["cohort"] == "additional_train"
            and (r["task"], r["instance"]) not in excluded]
    if len({r["instance"] for r in pool}) != len(pool): raise ValueError("Ambiguous source instance")
    known = [r for r in pool if r["instance"] == 192]
    if len(known) != 1: raise ValueError("Existing developed training instance missing")
    fresh = sorted((r for r in pool if r["instance"] != 192), key=lambda r: hashlib.sha256(
        f"{SPLIT}:{r['task']}:{r['instance']}".encode()).hexdigest())
    if len(fresh) < 3: raise ValueError("Fewer than three untouched candidate instance groups")
    return [("train", known[0]), ("train", fresh[2]), ("heldout", fresh[0]), ("heldout", fresh[1])]


def npy_sha(value):
    f = BytesIO(); np.save(f, np.asarray(value, dtype=np.float32), allow_pickle=False)
    return hashlib.sha256(f.getvalue()).hexdigest()


def audit_source(source, data, label_rows, quarantine):
    states = np.asarray(data["observation.state"].to_pylist(), float)
    actions = np.asarray(data["action"].to_pylist(), float)
    frames = data["frame_index"].to_numpy()
    h = hashlib.sha256(states.tobytes()+actions.tobytes())
    h.update(json.dumps(label_rows, sort_keys=True, separators=(",", ":")).encode())
    if h.hexdigest() != source["extracted_arrays_and_labels_sha256"]:
        raise ValueError("Original complete episode arrays/label rows changed")
    low = [(i, r) for i, r in enumerate(label_rows)
           if r["source_kind"] == "original_demo" and r["memlite_branch"] == "low"]
    if len({r["frame_index"] for _, r in low}) != len(low):
        raise ValueError("Conflicting original low branch")
    segments = {}
    for _, row in low:
        skills = json.loads(row["active_skills_semantic_json"])
        if len(skills) == 1 and skills[0]["verb"] == "GRASP" and not skills[0].get("unbound_relation"):
            key = (row["segment_start"], row["segment_end"], row["active_skills_semantic_json"])
            segments[key] = skills[0]
    candidates = []
    for (start, end, semantic), skill in sorted(segments.items()):
        row = {"start": start, "end": end, "skill": skill, "status": "NOT_USABLE"}
        try:
            prefix, segment, _, selection = validate_segment(states, actions, frames, label_rows,
                source, start, end, semantic, quarantine)
            arms = ("left", "right") if skill["arm"] == "UNSPECIFIED" else (skill["arm"].lower(),)
            if any(arm not in ("left", "right") for arm in arms): raise ValueError("Unsupported source hand")
            close = [(f, arm) for f in range(max(start, 2), end-17) for arm in arms
                     if actions[f, 14 if arm == "left" else 22] < -.5 and
                     np.count_nonzero(actions[f, [14, 22]] < -.5) == 1 and
                     np.all((actions[f-1, [14, 22]] >= .999) & (actions[f-1, [14, 22]] <= 1)) and
                     np.all((actions[f-2, [14, 22]] >= .999) & (actions[f-2, [14, 22]] <= 1))]
            if not close: raise ValueError("No single-hand CLOSE after two actual full-open source commands")
            first = min(f for f, arm in close)
            at_first = [arm for f, arm in close if f == first]
            if len(at_first) != 1: raise ValueError("Simultaneous both-hand CLOSE is not this pilot")
            # Index f is the NEXT source action after f prefix controls. The
            # first CLOSE index396 is control397; H09V's paid prefix is396.
            near = first
            if not start < near < end-16: raise ValueError("No complete interior near-grasp reference")
            row.update(status="SOURCE_AVAILABLE_PHYSICAL_SEED_UNVERIFIED", hand=at_first[0],
                first_close_action_index_zero_based=first, first_close_control_one_based=first+1,
                near_prefix_controls=near,
                near_prefix_sha256=npy_sha(actions[:near]), source_prefix_sha256=npy_sha(prefix),
                source_segment_sha256=npy_sha(segment), reference_controls_with_tail_hold=end+13,
                selection=selection, native_success=False)
        except (ValueError, KeyError) as exc:
            row["reason"] = str(exc)
        candidates.append(row)
    eligible = [r for r in candidates if r["status"] == "SOURCE_AVAILABLE_PHYSICAL_SEED_UNVERIFIED"]
    return {"source": source, "branches": dict(Counter(str((r["source_kind"], r["memlite_branch"])) for r in label_rows)),
            "duplicate_frame_rows": len(label_rows)-len({r["frame_index"] for r in label_rows}),
            "full_arrays_all_rows_sha256": h.hexdigest(), "grasp_segments": candidates,
            "selected_earliest_eligible": eligible[0] if eligible else None,
            "training_eligible": False, "new_controls": 0}


def main():
    import pyarrow.parquet as pq
    from prepare import LABELS, RELEASE
    p = argparse.ArgumentParser(); p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.monotonic(); counts = json.loads(args.counts.read_text())
    if sha(LABELS) != counts["source_identity"]["labels_sha256"] or sha(RELEASE/"quarantine_ranges.parquet") != counts["source_identity"]["quarantine_sha256"]:
        raise ValueError("Source annotation/quarantine identity mismatch")
    groups = reserve_groups(counts)
    quarantine = pq.read_table(RELEASE/"quarantine_ranges.parquet").to_pylist()
    result = []
    for split, source in groups:
        if time.monotonic()-started > 300: raise RuntimeError("Bounded source scan timed out")
        data = pq.read_table(source["parquet"], filters=[("episode_index", "=", source["episode"])],
                             columns=["frame_index", "observation.state", "action"]).sort_by("frame_index")
        labels = pq.read_table(LABELS, filters=[("episode_index", "=", source["episode"])], columns=LABEL_COLUMNS).to_pylist()
        result.append({"split": split, **audit_source(source, data, labels, quarantine)})
    value = {"schema": SPLIT, "status": "SOURCE_ONLY_NO_NEW_SEED_OR_NATIVE_BC", "counts_sha256": sha(args.counts),
             "exclusions": counts["exclusions"], "groups": result, "wall_seconds": time.monotonic()-started,
             "new_controls": 0, "new_resets": 0, "model_calls": 0, "training_updates": 0,
             "split_selection": "keep developed192 TRAIN; first two hash-ordered fresh groups heldout; third TRAIN; before labels"}
    if len(json.dumps(value)) > 1024**2: raise RuntimeError("Lightweight source evidence exceeded 1MiB")
    write_json(args.output, value)
    print(json.dumps({"path": str(args.output), "sha256": sha(args.output), "seconds": value["wall_seconds"],
        "groups": [{"split": r["split"], "instance": r["source"]["instance"], "episode": r["source"]["episode"],
                    "selected": None if r["selected_earliest_eligible"] is None else {k: r["selected_earliest_eligible"][k]
                     for k in ("start", "end", "hand", "near_prefix_controls", "near_prefix_sha256")}}
                    for r in result]}, indent=2))


if __name__ == "__main__": main()
