"""Robot-only hand envelopes against chassis/torso oriented boxes.

The R1Pro parallel jaws close inside their fully-open hand envelope. Using that
envelope also covers aperture changes without querying scene contacts. Visual
bounds overapproximate meshes; this is a conservative veto, not full collision
planning or a certificate about the environment or an object held in a hand.
"""
import numpy as np


def corners(lower, upper):
    return np.array([[x, y, z] for x in (lower[0], upper[0])
                     for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])


def obb_gap(ta, lo_a, hi_a, tb, lo_b, hi_b):
    """Maximum separating-axis gap; negative iff the closed boxes overlap."""
    ra, rb = ta[:3, :3], tb[:3, :3]
    center_a = ta[:3, 3] + ra @ ((lo_a + hi_a) / 2)
    center_b = tb[:3, 3] + rb @ ((lo_b + hi_b) / 2)
    axes = np.concatenate((ra.T, rb.T, np.cross(ra.T[:, None, :], rb.T[None, :, :]).reshape(-1, 3)))
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-9] / lengths[lengths > 1e-9, None]
    radius_a = np.abs(axes @ ra) @ ((hi_a - lo_a) / 2)
    radius_b = np.abs(axes @ rb) @ ((hi_b - lo_b) / 2)
    return float(np.max(np.abs(axes @ (center_b - center_a)) - radius_a - radius_b))


class HandBodyGuard:
    margin_m = .004

    def __init__(self, model):
        self.model = model
        metadata = model.spec["metadata"]
        geometry = metadata.get("robot_visual_boxes_reference")
        if (not geometry or geometry.get("scene_truth") is not False or
                geometry.get("source") != "robot_visual_link_boxes_actual_joint_fk" or
                metadata.get("parallel_gripper_open_envelope") != "R1Pro_parallel_prismatic_jaws" or
                not all(metadata.get("grasp_region_reference_fully_open", {}).get(a) is True for a in ("left", "right"))):
            raise ValueError("Robot-only fully-open R1Pro hand calibration required")
        boxes = {}
        for entry in geometry["boxes"]:
            t, lo, hi = (np.asarray(entry[k], dtype=float) for k in ("T_base_link", "lower", "upper"))
            if (t.shape != (4, 4) or lo.shape != (3,) or hi.shape != (3,) or
                    not all(np.isfinite(v).all() for v in (t, lo, hi)) or np.any(lo > hi) or
                    not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8) or
                    not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6) or
                    np.linalg.det(t[:3, :3]) < .999999):
                raise ValueError("Finite rigid robot visual bounds required")
            boxes[entry["link"]] = (t, lo, hi)
        self.hands, self.bodies = {}, {}
        for arm in ("left", "right"):
            parent = model.forward(model.reference, arm)
            points = []
            for name in (arm + "_gripper_link", arm + "_gripper_finger_link1", arm + "_gripper_finger_link2"):
                t, lo, hi = boxes[name]
                local = np.linalg.inv(parent) @ t
                points.extend(corners(lo, hi) @ local[:3, :3].T + local[:3, 3])
            points = np.asarray(points)
            self.hands[arm] = (points.min(axis=0), points.max(axis=0))
        for name in ("base_link", "torso_link1", "torso_link2", "torso_link3", "torso_link4"):
            t, lo, hi = boxes[name]
            key = "link:" + name
            if name != "base_link" and key not in model.links:
                raise ValueError("Body link kinematics missing")
            reference = np.eye(4) if name == "base_link" else model.forward(model.reference, key)
            self.bodies[name] = (np.linalg.inv(reference) @ t, lo, hi)

    def clearance(self, q):
        minimum, pair = float("inf"), None
        bodies = {name: ((np.eye(4) if name == "base_link" else self.model.forward(q, "link:" + name)) @ offset, lo, hi)
                  for name, (offset, lo, hi) in self.bodies.items()}
        for arm, (lo, hi) in self.hands.items():
            hand = self.model.forward(q, arm)
            for name, (body, blo, bhi) in bodies.items():
                gap = obb_gap(hand, lo, hi, body, blo, bhi)
                if gap < minimum:
                    minimum, pair = gap, (arm, name)
        return minimum - self.margin_m, pair
