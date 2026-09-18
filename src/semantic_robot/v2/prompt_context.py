"""Bounded actor summaries; full-precision control and audit receipts stay intact."""
from collections import Counter
from dataclasses import asdict
import json
import math


def selected(value, keys):
    return {key: value[key] for key in keys if key in value}


def readable(value):
    """Round only language-model display numbers, NEVER control inputs."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite actor context")
        return round(value, 4)
    if isinstance(value, dict):
        return {key: readable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [readable(item) for item in value]
    return value


def contact_summary(value):
    result = selected(value, ("valid", "reason", "point_base_m", "surface_point_not_object_pose",
        "distance_to_active_closing_center_m", "mean_contact_distance_m", "per_hand_distance_m",
        "target_minus_center_base_m", "point_is_display_midpoint_not_grasp_target", "navigation_workspace_check"))
    result["views"] = [selected(row, ("view", "target_uv", "valid", "reason", "depth_m", "spread_m"))
                       for row in value.get("views", [])]
    result["projected_checks_not_identity_proof"] = [selected(row, ("view", "status"))
                       for row in value.get("projected_same_point_checks", [])]
    if "hand_contacts" in value:
        result["hand_contacts"] = {hand: contact_summary(row) for hand, row in value["hand_contacts"].items()}
    return result


def actor_context(harness, state, bundle, allowed=()):
    """Keep the supplied recent outcomes and all commands, not debug dumps.

    No result is fabricated or silently made positive. The executor continues
    to use original geometry, full receipts and its own strict authorization.
    """
    context = harness.context()
    scope = selected(context, ("goal_index", "goal", "stage", "held_target_claims",
        "holding_verified_by_observation_and_proprio", "carry_constraints", "unverified_close_latches",
        "recoveries", "strategy_replans", "recent_replans", "stop_reason", "events", "search", "active_grasp_probe"))
    scope["target_surface_estimate"] = contact_summary(context.get("target_surface_estimate", {}))
    scope["egocentric_motion"] = selected(context.get("egocentric_motion", {}),
        ("valid", "reason", "body_delta", "body_translation_z_m"))
    scope["recent_executed"] = [{"action": row.get("action"), "stage": row.get("stage"),
        "feedback": selected(row.get("feedback", {}), ("status", "control_ticks", "joint_limit_ticks",
        "target_error_m", "orientation_error_deg", "eef_delta_m", "finger_mean_m", "empty_grasp_suspected",
        "base_integral", "base_motion_source", "visual_base_residual", "carry", "holding"))}
        for row in context.get("recent_executed", [])[-8:]]
    receipt = getattr(harness, "candidate_receipt", {})
    tested = receipt.get("tested", [])
    scores = []
    # Explicit indices bind scores to the unchanged, separately transmitted
    # canonical commands; rejected or merely tested actions cannot be offered.
    for index, action in enumerate(allowed):
        row = next((row for row in tested if row.get("action") == asdict(action)), None)
        scores.append({"command_index": index, **(selected(row, ("accepted", "reason", "planned_ticks",
            "predicted_distance_gain_m", "predicted_per_hand_distance_m", "navigation_after")) if row else {})})
    preflight = {"scores_for_allowed_commands": scores,
        "rejected_reason_counts": dict(Counter(row.get("reason", "UNKNOWN") for row in tested if not row.get("accepted"))),
        "navigation": receipt.get("navigation"), "command_grip_latch": receipt.get("command_grip_latch"),
        "fresh_execution_recheck_required": True,
        "depth_guard": selected(receipt.get("depth_guard", {}),
            ("visible_depth_points", "nonrobot_obstacle_points", "unseen_space_not_certified")),
        "geometric_gain_is_not_grasp_success": True}
    guides = {view: {hand: selected(row, ("eef_uv", "grasp_center_uv", "grasp_center_in_frame",
                    "base_axis_pixel_deltas_for_1cm")) for hand, row in hands.items()}
              for view, hands in bundle.geometry.items()}
    value = {"harness": scope, "current_visual_evidence": None if harness.observation is None else asdict(harness.observation),
        "robot": {"finger_mean_mm_not_total_gap": (state.gripper * 1000).tolist(),
                  "base_axis_projection_guides": guides, "width_alone_does_not_verify_holding": True},
        "CURRENT preflight receipt": preflight,
        "display_precision_m": .0001, "full_precision_geometry_and_receipts_remain_in_executor": True}
    return json.dumps(readable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
