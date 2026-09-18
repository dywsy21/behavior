"""Negative-only visible-depth veto for a known-free observing arm.

Robot visual boxes and sampled joint paths are conservative approximations.
They do not certify unobserved space, continuous collisions, or held-object
geometry. Do not reuse this module as a general collision-free motion planner.
"""
import numpy as np

from .grasp_motion import robot_point_mask


def box_distance(points, transform, lower, upper):
    local = (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]
    displacement = np.abs(local - (lower + upper) / 2) - (upper - lower) / 2
    return np.linalg.norm(np.maximum(displacement, 0), axis=1) + np.minimum(np.max(displacement, axis=1), 0)


class ObservingArmGuard:
    margin_m = .008

    def __init__(self, model, q, points, self_geometry, arm):
        if arm not in ("left", "right"):
            raise ValueError("Single free observing arm required")
        self.model, self.q, self.arm = model, np.asarray(q).copy(), arm
        self.valid = False
        self.reason = "ACTUAL_ROBOT_SELF_GEOMETRY_REQUIRED"
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not np.isfinite(self.points).all():
            raise ValueError("Finite measured depth points required")
        self.boxes = []
        if not self_geometry:
            return
        boxes = [box for box in self_geometry.get("boxes", []) if box["link"].startswith(arm + "_")]
        own = robot_point_mask(self.points, {**self_geometry, "boxes": boxes})
        if own is None:
            return
        # Only the known-free arm's current volume is masked. In particular,
        # depth on the OTHER hand/load is retained, not deleted as "self".
        self.points = self.points[~own]
        if len(self.points) < 40:
            self.reason = "INSUFFICIENT_CURRENT_DEPTH"
            return
        for box in boxes:
            name = "link:" + box["link"]
            if name not in model.links:
                self.reason = "MISSING_MOVING_LINK_KINEMATICS"
                return
            actual = np.asarray(box["T_base_link"])
            lower, upper = np.asarray(box["lower"]), np.asarray(box["upper"])
            # Actual finger FK may differ from calibration opening. Aperture
            # stays latched; propagate the link's relative rigid arm motion.
            offset = np.linalg.inv(model.forward(q, name)) @ actual
            self.boxes.append((name, offset, lower, upper, box_distance(self.points, actual, lower, upper)))
        self.valid, self.reason = True, "CURRENT_DEPTH_AND_FREE_ARM_BOXES_AVAILABLE"

    def check(self, joint_plan):
        receipt = {"source": "onboard_depth_free_arm_visual_box_sweep_veto", "scene_truth": False,
                   "unobserved_space_not_certified": True, "continuous_collision_clearance_not_certified": True,
                   "free_arm_current_box_volume_masked": True, "other_hand_and_load_points_retained": True,
                   "margin_m": self.margin_m, "points": len(self.points)}
        if not self.valid:
            return False, {**receipt, "reason": self.reason}
        plan = np.asarray(joint_plan, dtype=float)
        if plan.ndim != 2 or plan.shape[1] != 18 or not 1 <= len(plan) <= 64 or not np.isfinite(plan).all():
            raise ValueError("Finite bounded servo plan required")
        moving = np.arange(4, 11) if self.arm == "left" else np.arange(11, 18)
        fixed = np.setdiff1d(np.arange(18), moving)
        if np.max(np.abs(plan[:, fixed] - self.q[fixed])) > 1e-8:
            return False, {**receipt, "reason": "OTHER_ARM_OR_TORSO_WOULD_MOVE"}
        previous = np.vstack([self.q, plan[:-1]])
        samples = np.stack([(previous + plan) / 2, plan], axis=1).reshape(-1, 18)
        minimum = float("inf")
        for name, offset, lo, hi, start_distance in self.boxes:
            transforms = [self.model.forward(q, name) @ offset for q in samples]
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            swept = np.concatenate([corners @ t[:3, :3].T + t[:3, 3] for t in transforms])
            selected = np.all((self.points >= swept.min(axis=0) - self.margin_m) &
                              (self.points <= swept.max(axis=0) + self.margin_m), axis=1)
            if not selected.any():
                continue
            for index, transform in enumerate(transforms):
                distance = box_distance(self.points[selected], transform, lo, hi)
                minimum = min(minimum, float(distance.min()))
                # No new incursion into the margin, and no material worsening
                # of an already near-surface state. Existing proximity is not
                # silently upgraded to a clearance certificate.
                if np.any((distance < self.margin_m) & (distance < start_distance[selected] - .002)):
                    return False, {**receipt, "reason": "OBSERVED_FREE_ARM_SWEEP_OBSTACLE", "link": name,
                                   "sample": index, "min_observed_distance_m": minimum}
        return True, {**receipt, "reason": "NO_NEW_SAMPLED_VISIBLE_DEPTH_INCURSION",
                      "samples": len(samples), "min_observed_distance_m": None if not np.isfinite(minimum) else minimum}
