#!/usr/bin/env python3
"""Read-only evidence extraction from MEM-Lite policy and official sim traces.

This is NOT a success scorer or an automatic root-cause diagnosis. It does not
recompute simulator predicates, query hidden state, or change policy behavior.
Counters distinguish generated/hold chunks, accepted/rejected proposals, and
model DONE declarations from official success returned by env.step.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


TASKS = ("turning_on_radio", "picking_up_trash", "putting_away_Halloween_decorations",
         "cleaning_up_plates_and_food", "can_meat")
PARTS = {"base": (0, 3), "trunk": (3, 7), "left_arm": (7, 14),
         "left_gripper": (14, 15), "right_arm": (15, 22), "right_gripper": (22, 23)}


def records(path: Path, issues: list):
    if not path.is_file():
        issues.append({"path": str(path), "issue": "missing_trace"})
        return
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not an object")
            except (ValueError, TypeError):
                issues.append({"path": str(path), "line": number, "issue": "invalid_json_record"})
                continue
            yield number, row


def pointer(line, row):
    return {"line": line, "request": row.get("request_count"),
            "chunk": row.get("chunk_index"), "wall_time_unix": row.get("wall_time_unix")}


def merge_norm(summary, values):
    for part, fields in values.items():
        if not isinstance(fields, dict):
            continue
        output = summary.setdefault(part, {"proposals": 0, "count": 0, "clipped_count": 0})
        output["proposals"] += 1
        for key in ("count", "clipped_count"):
            output[key] += int(fields.get(key, 0))
        for key in ("max_normalized_excess", "decoded_abs_max", "delta_abs_max", "raw_abs_max"):
            value = fields.get(key)
            if isinstance(value, (float, int)) and math.isfinite(value):
                output[key] = max(output.get(key, 0), float(value))


def analyze_policy(path: Path) -> dict:
    issues, events, statuses, hold_reasons, rejected = [], Counter(), Counter(), Counter(), Counter()
    chunk_modes, served_modes, modes_horizons = {}, Counter(), defaultdict(Counter)
    done, transitions, images, rejection_examples = [], [], [], []
    proposal_norm, accepted_norm = {}, {}
    part_max, part_max_locations, fallback_keys, grip_changes = defaultdict(float), {}, Counter(), Counter()
    previous_grips = {}
    previous_intent_status = None
    task_id, task_text = None, None
    invalid_vectors = zero_base_commands = generated_terminal_actions = 0
    max_request = 0
    first_bridge_exception = None
    for line, row in records(path, issues):
        event = row.get("event", "<missing>")
        events[event] += 1
        request = row.get("request_count")
        if type(request) is int:
            max_request = max(max_request, request)
        if event == "episode_start":
            task_id, task_text = row.get("task_id"), row.get("task")
        elif event in ("reset", "bridge_reset"):
            chunk_modes.clear()
            previous_grips.clear()
            previous_intent_status = None
        elif event == "high_level":
            status, intent = row.get("intent_status"), row.get("output_intent")
            statuses[str(status)] += 1
            point = {**pointer(line, row), "status": status, "intent": intent,
                     "trigger": row.get("trigger")}
            if (intent, status) != previous_intent_status:
                transitions.append(point)
                previous_intent_status = (intent, status)
            if status == "DONE":
                done.append(point)
        elif event == "action_hold":
            hold_reasons[str(row.get("reason"))] += 1
        elif event in ("hold_chunk", "low_level_chunk"):
            mode = "hold" if event == "hold_chunk" else "generated"
            chunk_modes[row.get("chunk_index")] = {"mode": mode, "intent": row.get("intent")}
            horizon = row.get("decoded_action_horizon")
            modes_horizons[mode][str(horizon)] += 1
        elif event == "proposal_rejected":
            key = str(row.get("exception_type")) + ": " + str(row.get("message"))
            rejected[key] += 1
            if rejected[key] == 1:
                rejection_examples.append({**pointer(line, row), "exception": key})
        elif event == "action_proposal":
            merge_norm(proposal_norm, row.get("normalization", {}))
        elif event == "action_normalization":
            merge_norm(accepted_norm, row.get("parts", {}))
        elif event == "rgb_snapshot":
            images.append({**pointer(line, row), "label": row.get("label"), "images": row.get("images", [])})
        elif event == "bridge_exception" and first_bridge_exception is None:
            first_bridge_exception = {**pointer(line, row), "type": row.get("exception_type"), "message": row.get("message")}
        elif event == "official_action":
            chunk = chunk_modes.get(row.get("chunk_index"), {"mode": "unknown", "intent": None})
            served_modes[chunk["mode"]] += 1
            # A hold may carry mem_state.intent='Task complete' in telemetry;
            # that must NOT be misreported as a low-level generated action.
            if chunk["mode"] == "generated" and str(chunk["intent"]).strip().lower().rstrip(".") == "task complete":
                generated_terminal_actions += 1
            fallback_keys.update(row.get("hold_or_zero_fallback_keys", []))
            vector = row.get("actual_action_23d")
            if not isinstance(vector, list) or len(vector) != 23 or not all(
                type(value) in (int, float) and math.isfinite(value) for value in vector
            ):
                invalid_vectors += 1
                continue
            if all(abs(value) <= 1e-8 for value in vector[:3]):
                zero_base_commands += 1
            for part, (start, end) in PARTS.items():
                magnitude = max(abs(value) for value in vector[start:end])
                if part not in part_max_locations or magnitude > part_max[part]:
                    part_max[part] = magnitude
                    part_max_locations[part] = {**pointer(line, row), "command": vector[start:end]}
            for side, index in (("left", 14), ("right", 22)):
                if side in previous_grips and abs(vector[index] - previous_grips[side]) > 1e-5:
                    grip_changes[side] += 1
                previous_grips[side] = vector[index]
    return {
        "path": str(path), "task_id": task_id, "task_text": task_text, "max_request": max_request,
        "events": dict(events), "accepted_high_statuses": dict(statuses),
        "intent_status_transitions": transitions, "done_declarations": done,
        "hold_chunk_reasons": dict(hold_reasons), "served_action_modes": dict(served_modes),
        "chunk_horizons_by_mode": {key: dict(value) for key, value in modes_horizons.items()},
        "generated_actions_conditioned_on_task_complete": generated_terminal_actions,
        "rejection_counts": dict(rejected), "first_rejection_of_each_kind": rejection_examples,
        "proposal_normalization": proposal_norm, "accepted_normalization": accepted_norm,
        "bridge_fallback_key_counts": dict(fallback_keys), "invalid_action_vectors": invalid_vectors,
        "command_abs_max_by_part": dict(part_max), "zero_base_command_count": zero_base_commands,
        "command_abs_max_locations": part_max_locations,
        "gripper_target_change_counts": dict(grip_changes), "first_bridge_exception": first_bridge_exception,
        "image_index": images, "trace_issues": issues,
    }


def analyze_sim(path: Path, declarations: list) -> dict:
    issues, changes, matches = [], [], []
    waiting = sorted((item for item in declarations if type(item.get("wall_time_unix")) in (float, int)),
                     key=lambda item: item["wall_time_unix"])
    next_done = steps = success_steps = terminal_steps = 0
    first, last, previous_goal = None, None, object()
    for line, row in records(path, issues):
        if row.get("event") != "env_step":
            continue
        steps += 1
        point = {"trace_step": steps, "line": line, "wall_time_unix": row.get("wall_time_unix"),
                 "success": row.get("success"), "goal_status": row.get("goal_status"),
                 "terminated": row.get("terminated"), "truncated": row.get("truncated"),
                 "timeout": row.get("timeout")}
        first = point if first is None else first
        last = point
        success_steps += row.get("success") is True
        terminal_steps += row.get("terminated") is True or row.get("truncated") is True
        if row.get("goal_status") != previous_goal:
            changes.append(point)
            previous_goal = row.get("goal_status")
        timestamp = row.get("wall_time_unix")
        if type(timestamp) in (float, int):
            while next_done < len(waiting) and waiting[next_done]["wall_time_unix"] <= timestamp:
                declaration = waiting[next_done]
                matches.append({"declaration": declaration, "first_following_env_step": point,
                                "done_confirmed_at_next_step": row.get("success") is True})
                next_done += 1
    return {"path": str(path), "env_step_records": steps, "success_step_records": success_steps,
            "terminal_step_records": terminal_steps, "first": first, "last": last,
            "goal_status_changes": changes, "done_next_step_checks": matches,
            "declarations_without_following_sim_record": waiting[next_done:], "trace_issues": issues}


def analyze_root(root: Path) -> dict:
    policies = [analyze_policy(path) for path in sorted((root / "traces").rglob("policy_trace.jsonl"))]
    output = []
    for task_id, task in enumerate(TASKS):
        candidates = [policy for policy in policies if type(policy["task_id"]) is int and policy["task_id"] == task_id]
        declarations = [item for policy in candidates for item in policy["done_declarations"]]
        output.append({"task": task, "task_id": task_id, "policy_trace_count": len(candidates),
                       "policy_traces": candidates,
                       "sim_trace": analyze_sim(root / "tasks" / task / "sim_trace.jsonl", declarations)})
    return {"eval_root": str(root), "per_task": output,
            "unassigned_policy_traces": [policy["path"] for policy in policies if type(policy["task_id"]) is not int or not 0 <= policy["task_id"] < 5],
            "limitations": ["This report is evidence extraction, not official success scoring or causal diagnosis.",
                            "Commands are targets, not measured joint motion, grasp success or object displacement.",
                            "DONE not confirmed at the next step is a candidate for manual review, not a hidden-state oracle.",
                            "Goal predicate IDs need their task definition; satisfied-count is not official q-score.",
                            "Trace step is an ordinal of recorded env_step events; missing traces can break alignment."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_root(args.eval_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output_json), "tasks": [
        {"task": row["task"], "policy_traces": row["policy_trace_count"],
         "sim_records": row["sim_trace"]["env_step_records"]} for row in result["per_task"]]}, indent=2))


if __name__ == "__main__":
    main()
