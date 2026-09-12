import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("summary_v9", Path(__file__).parents[1] / "scripts/summarize_memlite_v9.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def make_result(root, task_name, code="0", **overrides):
    task_root = root / "tasks" / task_name
    output = task_root / "output/json"
    output.mkdir(parents=True)
    values = {"task": task_name, "instance_id": 301, "rollout_id": 0,
              "steps": 100, "success": False, "q_score": {"final": 0.0}}
    values.update(overrides)
    (output / "result.json").write_text(json.dumps(values))
    if code is not None:
        (task_root / "exit_code.txt").write_text(code)
    return output


def test_all_five_confirmed_results_have_fixed_denominator(tmp_path):
    for index, task in enumerate(summary.TASKS):
        make_result(tmp_path, task, success=index < 2)
    report = summary.summarize(tmp_path)
    assert report["overall"]["valid"] is True
    assert report["overall"]["success_rate"] == 0.4
    assert "2/5 (40.0%)" in summary.markdown(report)


@pytest.mark.parametrize("code,state", [(None, "pending"), ("bad", "invalid_result"), ("1", "runtime_crash")])
def test_json_success_alone_never_proves_normal_completion(tmp_path, code, state):
    for index, task in enumerate(summary.TASKS):
        make_result(tmp_path, task, code=code if index == 0 else "0", success=True)
    report = summary.summarize(tmp_path)
    assert report["per_task"][0]["result"]["state"] == state
    assert report["overall"]["success_rate"] is None
    assert report["overall"]["completed_tasks"] == 4
    assert report["overall"]["successes"] == 4


@pytest.mark.parametrize("overrides", [
    {"task": "wrong_task"}, {"instance_id": 302}, {"rollout_id": 1},
    {"rollout_id": False}, {"success": 1}, {"success": "false"},
    {"steps": -1}, {"steps": True}, {"q_score": {}},
    {"q_score": {"final": float("nan")}}, {"q_score": {"final": False}},
])
def test_official_result_identity_and_types_are_checked(tmp_path, overrides):
    task = summary.TASKS[0]
    make_result(tmp_path, task, **overrides)
    result = summary.load_task_result(tmp_path, task)
    assert result["state"] == "invalid_result"


def test_duplicate_or_truncated_output_cannot_count_as_completed(tmp_path):
    task = summary.TASKS[0]
    output = make_result(tmp_path, task)
    (output / "duplicate.json").write_text("{}")
    assert summary.load_task_result(tmp_path, task)["state"] == "invalid_result"
    (output / "duplicate.json").unlink()
    (output / "result.json").write_text("{")
    assert summary.load_task_result(tmp_path, task)["state"] == "invalid_result"


def test_empty_run_never_reports_zero_percent_success(tmp_path):
    report = summary.summarize(tmp_path)
    assert report["overall"]["success_rate"] is None
    assert report["overall"]["runtime_crashes"] == 0
    assert report["overall"]["completed_tasks"] == 0
    assert "success rate is not reported" in summary.markdown(report)
