"""Robot-only two-prismatic-finger FK; no guessed aperture or target geometry.

Asset grasp strips are retained for calibration checks. They are NOT certified
collision surfaces, a pressing tool, contact evidence, or a successful grasp.
"""
import copy

import numpy as np

from semantic_robot.control import finite, named_finger_positions
from .kinematics import adjoint, se3_exp


SOURCE = "robot_only_named_finger_joint_reference_and_jacobians"


def rigid(value):
    T = finite(value, (4, 4))
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6, rtol=0) or
            not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError("Rigid robot-frame transform required")
    return T


def link_reference(T_base_eef, T_base_link, com_jacobian, local_com):
    """Convert native COM geometric J to an EEF-frame link-origin screw model."""
    eef, link = rigid(T_base_eef), rigid(T_base_link)
    J = finite(com_jacobian, (6, 2))
    J[:3] -= np.cross(J[3:].T, link[:3, :3] @ finite(local_com, (3,))).T
    J[:3] -= np.cross(J[3:].T, link[:3, 3]).T
    inverse = np.linalg.inv(eef)
    screws = np.column_stack([adjoint(inverse, J[:, i]) for i in range(2)])
    return {"T_reference": (inverse @ link).tolist(), "screws": screws.tolist()}


class FingerKinematics:
    """A separate named-joint model; q18 arm/trunk FK stays unchanged."""

    def __init__(self, spec):
        if (not isinstance(spec, dict) or type(spec.get("version")) is not int or spec["version"] != 1 or
                spec.get("frame") != "eef" or spec.get("source") != SOURCE or
                spec.get("scene_truth") is not False or
                spec.get("embodiment") != "R1Pro_parallel_prismatic_jaws" or
                set(spec.get("arms", {})) != {"left", "right"}):
            raise ValueError("Supported robot-only finger calibration required")
        self.spec = copy.deepcopy(spec)
        self.arms, all_names = {}, set()
        for arm, row in spec["arms"].items():
            names = row.get("joint_names")
            if (not isinstance(names, list) or len(names) != 2 or
                    any(not isinstance(n, str) or not n for n in names) or
                    len(set(names)) != 2 or all_names.intersection(names)):
                raise ValueError("Two distinct, globally named finger joints per arm required")
            all_names.update(names)
            reference = finite(row["q_reference"], (2,))
            lower, upper = finite(row["lower"], (2,)), finite(row["upper"], (2,))
            if (np.any(lower >= upper) or np.any(reference < lower - 1e-5) or
                    np.any(reference > upper + 1e-5)):
                raise ValueError("Invalid finger joint reference/bounds")
            if not isinstance(row.get("links"), dict) or len(row["links"]) != 2:
                raise ValueError("Two independently articulated finger links required")
            links, active_joints = {}, set()
            for name, link in row["links"].items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Named finger link required")
                home, screws = rigid(link["T_reference"]), finite(link["screws"], (6, 2))
                norms = np.linalg.norm(screws[:3], axis=0)
                active = np.flatnonzero(norms > 1e-7)
                if (np.max(np.abs(screws[3:])) > 1e-7 or len(active) != 1 or
                        not np.isclose(norms[active[0]], 1, atol=1e-5, rtol=0)):
                    raise ValueError("Only one independent unit prismatic joint per finger is supported")
                active_joints.add(int(active[0]))
                points = np.asarray(link["grasp_strip_points_local_m"], dtype=float)
                if (points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 64 or
                        not np.isfinite(points).all()):
                    raise ValueError("Finite robot asset grasp-strip points required")
                links[name] = (home, screws, points.copy())
            if active_joints != {0, 1}:
                raise ValueError("Both independently measured finger joints must be represented")
            self.arms[arm] = (tuple(names), reference, lower, upper, links)
        self.joint_names = frozenset(all_names)

    def geometry(self, eef_poses, positions):
        positions = named_finger_positions(positions)
        if positions is None or set(positions) != self.joint_names:
            raise ValueError("Exact current named finger positions required; no mean-opening fallback")
        result = {"valid": True, "source": SOURCE, "frame": "robot_base",
                  "scene_truth": False, "finger_joint_positions_m": positions,
                  "not_contact_surface_or_holding_evidence": True, "arms": {}}
        for arm, (names, reference, lower, upper, links) in self.arms.items():
            current = np.array([positions[name] for name in names])
            if np.any(current < lower - 1e-5) or np.any(current > upper + 1e-5):
                raise ValueError("Measured finger position outside calibrated joint limits")
            eef = rigid(eef_poses[arm])
            result["arms"][arm] = {}
            for name, (home, screws, points) in links.items():
                T = np.eye(4)
                for i, delta in enumerate(current - reference):
                    T = T @ se3_exp(screws[:, i], delta)
                T = eef @ T @ home
                result["arms"][arm][name] = {
                    "T_base_link": T.tolist(),
                    "asset_grasp_strip_base_m": (points @ T[:3, :3].T + T[:3, 3]).tolist(),
                }
        return result
