import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("trace_analysis", Path(__file__).parents[1] / "scripts/analyze_memlite_rollout_traces.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def action(chunk, request, vector=None):
    return {"event": "official_action", "chunk_index": chunk, "request_count": request,
            "low_level_intent": "Task complete", "actual_action_23d": vector or [0.0] * 23,
            "hold_or_zero_fallback_keys": []}


def test_hold_intent_is_not_mistaken_for_generated_terminal_actions(tmp_path):
    result = analysis.analyze_policy(trace(tmp_path / "policy.jsonl", [
        {"event": "episode_start", "task_id": 0, "task": "radio"},
        {"event": "high_level", "intent_status": "DONE", "output_intent": "Task complete", "request_count": 1},
        {"event": "action_hold", "reason": "non_executable_status:DONE"},
        {"event": "hold_chunk", "chunk_index": 1, "intent": "Task complete", "decoded_action_horizon": 16},
        action(1, 1), action(1, 2),
        {"event": "low_level_chunk", "chunk_index": 2, "intent": "Task complete", "decoded_action_horizon": 32},
        action(2, 3),
    ]))
    assert result["served_action_modes"] == {"hold": 2, "generated": 1}
    assert result["generated_actions_conditioned_on_task_complete"] == 1
    assert result["hold_chunk_reasons"] == {"non_executable_status:DONE": 1}
    assert result["chunk_horizons_by_mode"] == {"hold": {"16": 1}, "generated": {"32": 1}}


def test_proposed_and_accepted_normalization_are_not_double_counted(tmp_path):
    norm = {"arm": {"count": 100, "clipped_count": 4, "max_normalized_excess": 0.4}}
    rejected_norm = {"arm": {"count": 100, "clipped_count": 10, "max_normalized_excess": 1.5}}
    result = analysis.analyze_policy(trace(tmp_path / "policy.jsonl", [
        {"event": "action_proposal", "normalization": norm},
        {"event": "action_normalization", "parts": norm},
        {"event": "action_proposal", "normalization": rejected_norm},
        {"event": "proposal_rejected", "exception_type": "ActionAdmissionError", "message": "too large", "request_count": 17},
    ]))
    assert result["proposal_normalization"]["arm"]["count"] == 200
    assert result["accepted_normalization"]["arm"]["count"] == 100
    assert result["proposal_normalization"]["arm"]["clipped_count"] == 14
    assert result["accepted_normalization"]["arm"]["clipped_count"] == 4
    assert result["first_rejection_of_each_kind"][0]["request"] == 17


def test_reset_and_missing_chunk_are_unknown_not_fabricated_hold(tmp_path):
    path = trace(tmp_path / "policy.jsonl", [
        {"event": "low_level_chunk", "chunk_index": 1, "intent": "pick up radio"},
        action(1, 1), {"event": "reset"}, action(1, 2),
    ])
    with path.open("a") as stream:
        stream.write("{truncated\n")
    result = analysis.analyze_policy(path)
    assert result["served_action_modes"] == {"generated": 1, "unknown": 1}
    assert len(result["trace_issues"]) == 1
    assert result["trace_issues"][0]["line"] == 5


def test_command_extrema_are_targets_with_original_record_location(tmp_path):
    vector = [0.0] * 23
    vector[7] = 9.0
    result = analysis.analyze_policy(trace(tmp_path / "policy.jsonl", [action(1, 1), action(1, 2, vector)]))
    assert result["command_abs_max_by_part"]["left_arm"] == 9.0
    assert result["command_abs_max_locations"]["left_arm"]["request"] == 2
    assert "grasp_success" not in result
    assert "joint_displacement" not in result


def test_done_is_matched_to_next_official_step_not_future_success(tmp_path):
    rows = [{"event": "env_step", "wall_time_unix": stamp, "success": success,
             "terminated": success, "truncated": False,
             "goal_status": {"satisfied": [0] if success else [], "unsatisfied": [] if success else [0]}}
            for stamp, success in ((1.0, False), (2.0, False), (3.0, True))]
    result = analysis.analyze_sim(trace(tmp_path / "sim.jsonl", rows),
                                  [{"wall_time_unix": 1.5, "request": 1}, {"wall_time_unix": 3.5, "request": 4}])
    assert result["done_next_step_checks"][0]["first_following_env_step"]["trace_step"] == 2
    assert result["done_next_step_checks"][0]["done_confirmed_at_next_step"] is False
    assert len(result["declarations_without_following_sim_record"]) == 1
    assert result["success_step_records"] == 1
    assert len(result["goal_status_changes"]) == 2


def test_task_identity_comes_from_payload_not_sanitized_folder_name(tmp_path):
    trace(tmp_path / "traces/policy_gpu0/not_a_task_name/episode_0001/policy_trace.jsonl", [
        {"event": "episode_start", "task_id": 2, "task": "Halloween prompt"}])
    result = analysis.analyze_root(tmp_path)
    assert result["per_task"][2]["task"] == "putting_away_Halloween_decorations"
    assert result["per_task"][2]["policy_trace_count"] == 1
    assert result["per_task"][0]["policy_trace_count"] == 0
    assert result["per_task"][2]["sim_trace"]["trace_issues"][0]["issue"] == "missing_trace"


def test_missing_trace_is_not_reported_as_zero_failures_with_complete_evidence(tmp_path):
    result = analysis.analyze_policy(tmp_path / "absent.jsonl")
    assert result["events"] == {}
    assert result["trace_issues"] == [{"path": str(tmp_path / "absent.jsonl"), "issue": "missing_trace"}]
