#!/usr/bin/env python3
"""Strict summary of the fixed five-task official MEM-Lite diagnostic run.

Do not infer completion from a result file alone, reduce the denominator when
jobs are missing, or confuse model-declared DONE with official simulator success.
This module is CPU-only and does not change raw evaluation artifacts or policy state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TASKS = (
    "turning_on_radio",
    "picking_up_trash",
    "putting_away_Halloween_decorations",
    "cleaning_up_plates_and_food",
    "can_meat",
)


def load_task_result(root: Path, task: str) -> dict:
    task_root = root / "tasks" / task
    result = {
        "task": task, "state": "pending", "exit_code": None, "path": None,
        "success": None, "steps": None, "q_score_final": None, "issues": [],
    }
    code_path = task_root / "exit_code.txt"
    if code_path.exists():
        try:
            result["exit_code"] = int(code_path.read_text().strip())
        except ValueError:
            result["issues"].append("invalid_exit_code")
    candidates = sorted((task_root / "output" / "json").glob("*.json"))
    if len(candidates) != 1:
        result["issues"].append(f"expected_one_official_json:got_{len(candidates)}")
    else:
        path = candidates[0]
        result["path"] = str(path)
        try:
            official = json.loads(path.read_text())
            if not isinstance(official, dict):
                raise ValueError("official JSON is not an object")
            for key, expected in (("task", task), ("instance_id", 301), ("rollout_id", 0)):
                if type(official.get(key)) is not type(expected) or official[key] != expected:
                    result["issues"].append(f"wrong_{key}")
            if type(official.get("success")) is not bool:
                result["issues"].append("success_must_be_json_boolean")
            else:
                result["success"] = official["success"]
            steps = official.get("steps")
            if type(steps) is not int or steps < 0:
                result["issues"].append("steps_must_be_nonnegative_integer")
            else:
                result["steps"] = steps
            q_score = official.get("q_score")
            q_final = q_score.get("final") if isinstance(q_score, dict) else None
            if type(q_final) not in (int, float) or not math.isfinite(q_final):
                result["issues"].append("q_score_final_must_be_finite_number")
            else:
                result["q_score_final"] = float(q_final)
        except (OSError, ValueError) as error:
            result["issues"].append(f"unreadable_official_json:{type(error).__name__}")
    if result["exit_code"] is None:
        result["issues"].append("normal_exit_not_confirmed")
        result["state"] = "invalid_result" if code_path.exists() else "pending"
    elif result["exit_code"] != 0:
        result["state"] = "runtime_crash"
    elif result["issues"]:
        result["state"] = "invalid_result"
    else:
        result["state"] = "success" if result["success"] else "task_failed"
    return result


def summarize(root: Path) -> dict:
    rows = [load_task_result(root, task) for task in TASKS]
    completed = [row for row in rows if row["state"] in ("success", "task_failed")]
    successes = sum(row["success"] is True for row in completed)
    valid = len(completed) == len(TASKS)
    return {
        "memlite_root": str(root),
        "protocol": {"split": "public_test", "instance_id": 301, "rollout_id": 0,
                     "rollouts_per_task": 1, "expected_tasks": len(TASKS)},
        "overall": {"valid": valid, "successes": successes,
                    "completed_tasks": len(completed), "expected_tasks": len(TASKS),
                    "runtime_crashes": sum(row["state"] == "runtime_crash" for row in rows),
                    "success_rate": successes / len(TASKS) if valid else None},
        "per_task": [{"task": row["task"], "result": row} for row in rows],
        "limitations": ["One rollout per task is diagnostic, not a leaderboard estimate.",
                        "No non-MEM baseline or ablation: changes cannot isolate MEM's benefit.",
                        "Only the official JSON success flag determines task success."],
    }


def markdown(report: dict) -> str:
    lines = ["# MEM-Lite: official public-test instance 301, rollout 0", "",
             "| Task | State | Success | Steps | Final q-score | Exit |",
             "|---|---|---:|---:|---:|---:|"]
    for item in report["per_task"]:
        row = item["result"]
        fields = [row[key] for key in ("task", "state", "success", "steps", "q_score_final", "exit_code")]
        lines.append("| " + " | ".join("missing" if value is None else str(value) for value in fields) + " |")
    score = report["overall"]
    lines += ["", (f"Success: {score['successes']}/5 ({score['success_rate']:.1%})." if score["valid"]
                   else f"INCOMPLETE / INVALID: only {score['completed_tasks']}/5 verified normal exits with valid results; success rate is not reported."), ""]
    for item in report["per_task"]:
        if item["result"]["issues"]:
            lines.append(f"- {item['task']}: " + ", ".join(item["result"]["issues"]))
    lines += ["", *report["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.mem_root)
    for path in (args.output_json, args.output_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    args.output_markdown.write_text(markdown(report))
    if not report["overall"]["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
