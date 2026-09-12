"""All-row invariants and deterministic stratified samples for HUMAN review.

This script cannot certify that a human inspected the samples. Its output is
an audit packet, not permission to train or a human-review PASS marker.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import random
import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from g05.utils.memlite_protocol import parse_memlite_output, MemLiteProtocolError
from scripts.data.build_memlite_annotations import _as_skill_list, _fallback_metadata, _primitive_identity, make_memory


def source_primitive_intervals(episode):
    """Independently derive event times from source annotations, not labels.

    Only rendering/identity helpers are shared. No annotate_episode(),
    sidecar target, sidecar memory frame, or sidecar skill boundary is used
    to decide which events are complete at an observation.
    """
    raw = json.loads(Path(episode["annotation_path"]).read_text())
    duration = raw.get("valid_duration", raw.get("meta_data", {}).get("valid_duration"))
    if not isinstance(duration, list) or len(duration) != 2:
        raise ValueError("Source valid duration needs explicit independent review")
    start, end = map(int, duration)
    length = int(episode["length"])
    offset = -start if end - start == length else 0
    skills = {int(row.get("skill_idx", index)): row for index, row in enumerate(_as_skill_list(raw.get("skill_annotation", [])))}
    primitives = _as_skill_list(raw.get("primitive_annotation", []))
    if not primitives:
        raise ValueError("Primitive-free episode needs explicit independent review")
    result, seen = [], set()
    for index, primitive in enumerate(primitives):
        duration = primitive.get("frame_duration", primitive.get("duration"))
        if not isinstance(duration, list) or len(duration) != 2 or any(isinstance(x, (list, dict)) for x in duration):
            raise ValueError("Non-scalar primitive interval needs explicit independent review")
        left, right = map(int, duration)
        if left > right:
            continue
        left, right = max(start, left), min(end, right)
        left, right = max(0, left + offset), min(length, right + offset)
        if right <= left:
            continue
        ids = primitive.get("skill_idxes", primitive.get("skill_indices", primitive.get("skill_idx", primitive.get("skill_index"))))
        ids = list(ids) if isinstance(ids, list) else [ids]
        if not ids or any(value is None for value in ids):
            continue
        ids = [int(value) for value in ids]
        primitive_idx = int(primitive.get("primitive_idx", index))
        fallback = _fallback_metadata(primitive, primitive_idx, ids, skills)
        row = {**primitive, "_start": left, "_end": right, "_primitive_idx": primitive_idx,
               "_skill_idx": min(ids), "_skill": primitive,
               "_intent_fallback_descriptions": fallback[0] if fallback else None}
        identity = _primitive_identity(row)
        if identity not in seen:
            seen.add(identity)
            result.append(row)
    return sorted(result, key=lambda row: (row["_start"], row["_end"], row["_primitive_idx"], row["_skill_idx"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.sidecar_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == 5
    episodes = {row["episode_index"]: row for row in manifest["episodes"]}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    rng = random.Random(709)
    counts = Counter()
    by_task = defaultdict(Counter)
    categories_seen, reservoir = Counter(), defaultdict(list)
    maxima = defaultdict(lambda: (0, None))
    errors = []
    current_ep, lows, highs = None, {}, []

    def check(condition, message, row):
        if not condition and len(errors) < 100:
            errors.append({"message": message, "episode": row["episode_index"], "frame": row["frame_index"], "branch": row["memlite_branch"]})

    def finish_episode():
        if current_ep is None:
            return
        high_rows = sorted(highs, key=lambda row: row["frame_index"])
        source_events = source_primitive_intervals(episodes[current_ep])
        check(len({row["frame_index"] for row in high_rows}) == len(high_rows), "duplicate high frame", high_rows[0])
        previous = None
        for row in high_rows:
            frame, task = row["frame_index"], row["task_index"]
            check(row["memory_target_frame"] == frame, "future memory target frame", row)
            check(row["memory_input_frame"] < frame, "input memory not from previous observation", row)
            check(row["memory_update"].startswith(f"Task={task}; Completed="), "memory schema/task identity", row)
            expected_memory = make_memory(str(task), [event for event in source_events if event["_end"] <= frame],
                                          [event for event in source_events if event["_start"] < frame < event["_end"]])
            check(row["memory_update"] == expected_memory, "memory contents disagree with CURRENT source-observed events", row)
            if row["memory_input_corruption"] == "synthetic_text_premature_known_intent_completion":
                prior_frame = row["memory_input_frame"]
                known = [event for event in source_events if event["_end"] <= prior_frame]
                known += [event for event in source_events if event["_start"] <= prior_frame and event["_end"] > frame]
                check(row["memory"] == make_memory(str(task), known), "corrupted input exposes unknown future primitives", row)
            elif row["memory_input_corruption"] == "synthetic_text_missing_history":
                check(row["memory"] == f"Task={task}; Completed=none.", "missing-history corruption changed task information", row)
            elif row["memory_input_corruption"] != "none":
                check(False, "unreviewed memory corruption type", row)
            if row["intent_status"] == "CONTINUE":
                check(frame in lows and lows[frame]["intent"] == row["intent"], "HL/LL intent mismatch", row)
            if previous:
                check(row["previous_intent"] == previous["intent"], "previous intent chain mismatch", row)
                check(row["memory_input_frame"] == previous["frame_index"], "input observation chain mismatch", row)
                if row["memory_input_corruption"] == "none":
                    check(row["memory"] == previous["memory_update"], "clean memory chain mismatch", row)
            else:
                check(row["memory"] == f"Task={task}; Completed=none.", "noncanonical initial memory", row)
                check(row["previous_intent"] == "None", "nonempty initial intent", row)
            target = f'Intent: {row["intent"]}|Updated Memory: {row["memory_update"]}|Status: {row["intent_status"]}|<HL_END>'
            try:
                parse_memlite_output(target)
            except MemLiteProtocolError as exc:
                check(False, "target rejected by runtime parser: " + str(exc), row)
            output_tokens = len(tokenizer.encode(target, add_special_tokens=False))
            # Conservative budget estimate: text plus 192 image tokens,
            # proprio/format tokens, and 200 task tokens. Real encoder smoke
            # additionally verifies the full train sequence before launch.
            context = row["memory"] + " Previous intent: " + row["previous_intent"] + target
            estimated_tokens = len(tokenizer.encode(context, add_special_tokens=False)) + 600
            for key, value in (("output_tokens", output_tokens), ("estimated_total_tokens", estimated_tokens),
                               ("intent_chars", len(row["intent"])), ("memory_chars", len(row["memory_update"]))):
                if value > maxima[key][0]:
                    maxima[key] = (value, {"episode": current_ep, "frame": frame, "task": task})
            check(output_tokens <= 1024, "HL output exceeds generation budget", row)
            check(estimated_tokens <= 4096, "estimated train sequence exceeds budget", row)
            check(len(row["intent"]) <= 2048 and len(row["memory_update"]) <= 4096, "training target exceeds strict runtime schema limits", row)
            tags = []
            if previous is None:
                tags.append("start")
            elif row["intent_status"] == "DONE":
                tags.append("terminal")
            elif row["intent"] != previous["intent"]:
                tags.append("transition")
            else:
                tags.append("ongoing")
            if row["memory_input_corruption"] != "none":
                tags.append("corrupt_memory")
                counts["corrupted_high"] += 1
            if len(row.get("skill_idxes", [])) > 1:
                tags.append("parallel")
            for tag in tags:
                key = (task, tag)
                categories_seen[key] += 1
                candidate = {**row, "review_category": tag, "output_tokens": output_tokens,
                             "estimated_total_tokens": estimated_tokens,
                             "paired_low_intent": lows.get(frame, {}).get("intent"),
                             "previous_low_intent": lows.get(frame - 1, {}).get("intent")}
                bucket = reservoir[key]
                if len(bucket) < 6:
                    bucket.append(candidate)
                else:
                    index = rng.randrange(categories_seen[key])
                    if index < 6:
                        bucket[index] = candidate
            previous = row
        counts["checked_episodes"] += 1

    file = pq.ParquetFile(root / "meta/memlite_annotations.parquet")
    for batch in file.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            episode = row["episode_index"]
            if episode != current_ep:
                finish_episode()
                current_ep, lows, highs = episode, {}, []
            counts["rows"] += 1
            branch = row["memlite_branch"]
            counts[branch] += 1
            by_task[row["task_index"]][branch] += 1
            check(0 <= row["frame_index"] < episodes[episode]["length"], "frame outside actual episode", row)
            if branch == "low":
                check(row["frame_index"] not in lows, "duplicate low frame", row)
                check(row["intent_status"] == "CONTINUE" and row["intent"] != "Task complete", "terminal low action target", row)
                check(row["frame_index"] < row["action_horizon_end"] <= row["segment_end"], "invalid action supervision horizon", row)
                lows[row["frame_index"]] = {"intent": row["intent"]}
            else:
                highs.append(row)
                if row["intent_status"] == "DONE":
                    counts["done_high"] += 1
        if counts["rows"] % 524288 == 0:
            print(json.dumps({"rows_checked": counts["rows"], "episodes_checked": counts["checked_episodes"]}), flush=True)
    finish_episode()
    selected = [row for key in sorted(reservoir) for row in reservoir[key]]
    with (output / "manual_review_samples.jsonl").open("w") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Shorter, readable rows retain all actual text. Do not replace text with
    # a script's subjective "correct" classification.
    fields = ("task_index", "episode_index", "frame_index", "review_category", "previous_intent", "memory_input_frame",
              "memory", "intent", "memory_update", "intent_status", "memory_input_corruption", "paired_low_intent")
    with (output / "manual_review_readable.jsonl").open("w") as fh:
        for row in selected:
            fh.write(json.dumps({key: row[key] for key in fields}, ensure_ascii=False) + "\n")
    report = {"counts": dict(counts), "by_task": {str(k): dict(v) for k, v in by_task.items()},
              "maxima": dict(maxima), "errors": errors, "selected_samples": len(selected),
              "categories": {f"{k[0]}:{k[1]}": v for k, v in categories_seen.items()},
              "excluded_episodes": [row for row in episodes.values() if row["status"] != "ok"],
              "human_review_status": "PENDING: selection is not manual approval"}
    (output / "structural_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
