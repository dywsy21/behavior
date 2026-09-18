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
        "target_minus_center_base_m", "target_minus_center_tool_m", "point_is_display_midpoint_not_grasp_target", "navigation_workspace_check"))
    result["views"] = [selected(row, ("view", "target_uv", "valid", "reason", "depth_m", "spread_m"))
                       for row in value.get("views", [])]
    result["projected_checks_not_identity_proof"] = [selected(row, ("view", "status"))
                       for row in value.get("projected_same_point_checks", [])]
    if "hand_contacts" in value:
        result["hand_contacts"] = {hand: contact_summary(row) for hand, row in value["hand_contacts"].items()}
    return result


def columnar_scores(scores):
    """Lossless display table: every offered command/field remains represented.

    Dotted columns name fields of inspection_after. Missing cells are null,
    never implicit positive evidence. This changes no executor receipt/ranking.
    """
    flattened=[];columns=[]
    for score in scores:
        row={}
        for key,value in score.items():
            if key=="inspection_after":
                row.update({"inspection_after."+k:v for k,v in value.items()})
            else:row[key]=value
        for key in row:
            if key not in columns:columns.append(key)
        flattened.append(row)
    return {"encoding":"Each row follows columns; null means missing/unknown, not true",
            "columns":columns,"rows":[[row.get(key) for key in columns] for row in flattened]}


def actor_context(harness, state, bundle, allowed=()):
    """Keep the supplied recent outcomes and all commands, not debug dumps.

    No result is fabricated or silently made positive. The executor continues
    to use original geometry, full receipts and its own strict authorization.
    """
    context = harness.context()
    scope = selected(context, ("goal_index", "goal", "stage", "held_target_claims",
        "holding_verified_by_observation_and_proprio", "carry_constraints", "unverified_close_latches",
        "recoveries", "strategy_replans", "recent_replans", "stop_reason", "events", "search", "active_grasp_probe", "approach_progress", "target_reference", "held_inspection", "possible_contact_after_any_close"))
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
        score={"command_index": index, **(selected(row, ("accepted", "reason", "planned_ticks",
            "predicted_distance_gain_m", "predicted_per_hand_distance_m", "navigation_after", "inspection_after", "reorientation_after")) if row else {})}
        if getattr(harness,"multicamera_inspection",False) and "inspection_after" in score:
            # Hand identity is already bound by the canonical command and the
            # single shared reference. Do not repeat it for all 42 candidates.
            score["inspection_after"]=selected(score["inspection_after"],("observer_camera", "bearing_after_deg",
                "pointing_gain_deg", "in_image_bounds", "range_m", "prior_anchor_uv", "side_separation_from_head_deg",
                "relative_pose_novelty_margin"))
        scores.append(score)
    compact=bool(getattr(harness,"multicamera_inspection",False) and any("inspection_after" in s for s in scores))
    preflight = {"scores_for_allowed_commands": columnar_scores(scores) if compact else scores,
        "rejected_reason_counts": dict(Counter(row.get("reason", "UNKNOWN") for row in tested if not row.get("accepted"))),
        "navigation": receipt.get("navigation"), "command_grip_latch": receipt.get("command_grip_latch"),
        "fresh_execution_recheck_required": True,
        "depth_guard": selected(receipt.get("depth_guard", {}),
            ("visible_depth_points", "nonrobot_obstacle_points", "unseen_space_not_certified")),
        "geometric_gain_is_not_grasp_success": True}
    if getattr(harness,"multicamera_inspection",False):
        preflight["inspection_anchor_is_not_affordance_or_visibility_evidence"]=True
    if getattr(harness,"approach_reorientation",False):
        posture=receipt.get("approach_reorientation",{})
        lookahead=posture.get("preview") or {}
        preflight["approach_reorientation"]={"eligible":posture.get("eligible",False),
            **selected(lookahead,("trigger","source","scene_truth","future_state_is_prediction","future_actions_not_authorized"))}
    guides = {view: {hand: selected(row, ("eef_uv", "grasp_center_uv", "grasp_center_in_frame",
                    "base_axis_pixel_deltas_for_1cm", "finger_contact_region")) for hand, row in hands.items()}
              for view, hands in bundle.geometry.items()}
    value = {"harness": scope, "current_visual_evidence": None if harness.observation is None else asdict(harness.observation),
        "robot": {"finger_mean_mm_not_total_gap": (state.gripper * 1000).tolist(),
                  "base_axis_projection_guides": guides, "width_alone_does_not_verify_holding": True},
        "CURRENT preflight receipt": preflight,
        "display_precision_m": .0001, "full_precision_geometry_and_receipts_remain_in_executor": True}
    return json.dumps(readable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
