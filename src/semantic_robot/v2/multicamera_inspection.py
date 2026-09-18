"""Opt-in held-reference inspection with an independently movable free camera.

Shared attempts stay bounded. Load presentation and free-arm observation have
separate measured motion accounts; neither camera framing nor pose novelty is
treated as an affordance observation or a completed task.
"""
from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from .arm_observation_guard import ObservingArmGuard
from .grounding import observed_cloud
from .held_inspection import HeldInspection, inspection_palette
from .observer_geometry import anchor_camera_views, angle_degrees
from .protocol import Action, HOLD, ROTATIONS, TRANSLATIONS
from .servo import SafeServo


def free_observing_hand(harness):
    reference = harness.search_reference.removeprefix("held_")
    if reference not in ("left", "right") or not harness.hold_verified.get(reference):
        return None
    other = "left" if reference == "right" else "right"
    if harness.held.get(other) is not None or harness.hold_verified.get(other) or harness.pending_grasp.get(other):
        return None
    return other


def multicamera_palette(harness):
    reference = harness.search_reference.removeprefix("held_")
    if reference not in ("left", "right") or not harness.hold_verified.get(reference):
        return (HOLD,)
    actions = list(inspection_palette(reference, harness.carry, True))
    free = free_observing_hand(harness)
    if free:
        # Only this confirmed-free arm moves; load-carrying joints stay fixed.
        actions += [Action(free, move, scale, "tool" if move in ROTATIONS else "base")
                    for scale in ("coarse", "fine") for move in (*TRANSLATIONS, *ROTATIONS)]
    return tuple(actions)


def inspection_carry(harness, action):
    """A free camera may rotate while the OTHER hand remains level and fixed."""
    free = free_observing_hand(harness)
    if (getattr(harness, "multicamera_inspection", False) and free == action.part and
            harness.stage in ("SEARCH", "RECOVER") and harness.observation and not harness.observation.visible and
            action.move in (*TRANSLATIONS, *ROTATIONS)):
        return False
    return harness.carry


class MultiCameraInspection(HeldInspection):
    max_free_path_m = .20
    max_free_rotation_rad = np.deg2rad(120.)

    def __init__(self):
        super().__init__(budget_aware=True)
        self.observers = {}
        self.free_guard = None

    def observe(self, model, state, harness, depths=None, self_geometry=None):
        reference = harness.search_reference.removeprefix("held_")
        free = free_observing_hand(harness)
        fresh, base = super().observe(model, state, harness)
        if not base.get("valid"):
            return fresh, base
        row = self.observers.setdefault(harness.index, {"free_hand": free, "last_free": None,
            "free_path_m": 0., "free_rotation_rad": 0., "visited": {}, "best_bearing": {}, "recorded_attempts": -1})
        if row["free_hand"] != free:
            harness.stop_reason = "OBSERVER_HAND_OWNERSHIP_CHANGED"
            return False, {"valid": False, "reason": harness.stop_reason}
        head = model.forward(state.q, "camera_head")
        if free:
            relative = np.linalg.inv(head) @ model.forward(state.q, free)
            if row["last_free"] is not None:
                row["free_path_m"] += float(np.linalg.norm(relative[:3, 3] - row["last_free"][:3, 3]))
                row["free_rotation_rad"] += float(np.linalg.norm(Rotation.from_matrix(relative[:3, :3] @ row["last_free"][:3, :3].T).as_rotvec()))
            row["last_free"] = relative.copy()
        views = anchor_camera_views(model, state.q, reference, self.anchors[reference]["point_hand_m"], depths)
        attempts = self.states[harness.index]["attempts"]
        for view, item in views.items():
            relative = np.linalg.inv(model.forward(state.q, "camera_" + view)) @ model.forward(state.q, reference)
            history = row["visited"].setdefault(view, [])
            bearing = item["bearing_error_deg"]
            best = row["best_bearing"].get(view, bearing)
            # Measured camera pointing progress can sustain an observation
            # attempt; it never changes evidence.visible or any success flag.
            if best - bearing >= 2.:
                fresh = True
                row["best_bearing"][view] = bearing
            else:
                row["best_bearing"].setdefault(view, bearing)
            if row["recorded_attempts"] != attempts:
                if len(history) >= self.max_attempts + 1:
                    raise ValueError("Observer history exceeded shared attempt budget")
                history.append(relative.copy())
        row["recorded_attempts"] = attempts
        row["views"] = views
        self.free_guard = (ObservingArmGuard(model, state.q, observed_cloud(depths, model, state.q, stride=6), self_geometry, free)
                           if free and depths is not None else None)
        return bool(fresh), self.context(model, state, harness)

    def context(self, model, state, harness):
        value = super().context(model, state, harness)
        row = self.observers.get(harness.index)
        if not value.get("valid") or row is None:
            return value
        value.update(observer_selection_enabled=True, free_observing_hand=row["free_hand"],
                     camera_views=row.get("views", {}), free_arm_path_m=row["free_path_m"],
                     free_arm_rotation_deg=float(np.rad2deg(row["free_rotation_rad"])),
                     max_free_arm_path_m=self.max_free_path_m, max_free_arm_rotation_deg=120.,
                     attempts_shared_by_all_observers=True, free_motion_holds_reference_joints_and_grip=True,
                     environment_collision_check="FREE_ARM_VISIBLE_DEPTH_VETO_ONLY_NOT_COMPLETE_CLEARANCE")
        return value

    def candidates(self, model, state, harness, servo, depth_guard):
        held = self.states.get(harness.index)
        row = self.observers.get(harness.index)
        reference = harness.search_reference.removeprefix("held_")
        free = free_observing_hand(harness)
        receipt = {"source": "independent_camera_held_reference_inspection", "tested": [],
                   "command_grip_latch": servo.grips.tolist(), "depth_guard": depth_guard.receipt(),
                   "scene_truth": False, "environment_collision_check": "FREE_ARM_VISIBLE_DEPTH_VETO_ONLY_NOT_COMPLETE_CLEARANCE"}
        if held is None or row is None or reference not in self.anchors or not harness.hold_verified.get(reference):
            harness.stop_reason = "NO_VERIFIED_HELD_ANCHOR"
            return (HOLD,), receipt
        if held["attempts"] >= self.max_attempts:
            harness.stop_reason = "HELD_INSPECTION_BUDGET_REACHED"
            return (HOLD,), receipt
        allowed = [HOLD]
        for action in harness.palette():
            if action == HOLD:
                continue
            if action.part not in (reference, free) or action.move not in (*TRANSLATIONS, *ROTATIONS):
                raise ValueError("Unscoped camera inspection command")
            observing_free = action.part == free
            camera = free + "_wrist" if observing_free else "head"
            carry = inspection_carry(harness, action)
            path = row["free_path_m"] if observing_free else held["path_m"]
            rotation = row["free_rotation_rad"] if observing_free else held["rotation_rad"]
            path_limit = self.max_free_path_m if observing_free else self.max_path_m
            angle_limit = self.max_free_rotation_rad if observing_free else self.max_rotation_rad
            ok = (path < path_limit and rotation < angle_limit and
                  (path + action.amount(carry) <= path_limit if action.move in TRANSLATIONS else
                   rotation + action.amount(carry) <= angle_limit))
            reason = "INSPECTION_MOTION_BOUND"
            trial = SafeServo(model, state, servo.grips.copy(), servo.limits)
            if ok:
                ok = trial.begin(action, state, carry)
                reason = trial.status
            entry = {"action": asdict(action), "accepted": bool(ok), "reason": reason}
            if ok:
                predicted = trial.joint_plan[-1]
                views = anchor_camera_views(model, predicted, reference, self.anchors[reference]["point_hand_m"])
                current, after = row["views"][camera], views[camera]
                relative = np.linalg.inv(model.forward(predicted, "camera_" + camera)) @ model.forward(predicted, reference)
                novelty = self.novelty(relative, row["visited"][camera])
                entry.update(planned_ticks=trial.total_ticks, predicted_cameras=views,
                    inspection_after={"observer_camera": camera, "motion_hand": action.part, "reference_hand": reference,
                        "bearing_after_deg": after["bearing_error_deg"],
                        "pointing_gain_deg": current["bearing_error_deg"] - after["bearing_error_deg"],
                        "in_image_bounds": after["in_image_bounds"], "range_m": after["range_m"],
                        "prior_anchor_uv": after["prior_anchor_uv"],
                        "side_separation_from_head_deg": angle_degrees(views["head"]["camera_direction_in_reference_hand"], after["camera_direction_in_reference_hand"]),
                        "relative_pose_novelty_margin": novelty, "not_affordance_or_visibility_evidence": True})
                if novelty is not None and novelty < 1.:
                    entry.update(accepted=False, reason="PREVIOUSLY_OBSERVED_RELATIVE_VIEW")
                elif observing_free:
                    if self.free_guard is None or self.free_guard.arm != free:
                        entry.update(accepted=False, reason="CURRENT_FREE_ARM_DEPTH_GUARD_REQUIRED")
                    else:
                        safe, check = self.free_guard.check(trial.joint_plan)
                        entry["visible_arm_sweep"] = check
                        if not safe:
                            entry.update(accepted=False, reason=check["reason"])
                if entry["accepted"]:
                    allowed.append(action)
            receipt["tested"].append(entry)
        if len(allowed) == 1:
            harness.stop_reason = "NO_SAFE_HELD_INSPECTION_ACTION"
        return tuple(allowed), receipt
