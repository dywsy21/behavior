"""Portable product-of-exponentials FK/Jacobians from a robot-only calibration.

No simulator imports, scene objects, world localization, mesh truth or reward.
The calibration stores local link poses and joint screws at one reference pose.
It is accepted only after prediction is tested at other poses against the robot.
"""
import hashlib
import json
import math
from collections import OrderedDict

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.control import RobotState, finite


def skew(x):
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def transform(p, q):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q).as_matrix()
    result[:3, 3] = p
    return result


def link_origin_jacobian(com_jacobian, link_quaternion, local_com):
    """PhysX geometric J is at link COM, but link poses are at prim origins.

    Translate the *point*, not the axes: v_origin = v_com - omega x R*c.
    Both velocity blocks are already expressed in the robot-base frame.
    """
    J = finite(com_jacobian, (6, 18)).copy()
    offset = Rotation.from_quat(link_quaternion).apply(finite(local_com, (3,)))
    J[:3] -= np.cross(J[3:].T, offset).T
    return J


def se3_exp(screw, value):
    v, w = screw[:3], screw[3:]
    result = np.eye(4)
    norm = np.linalg.norm(w)
    if norm < 1e-10:
        result[:3, 3] = v * value
    else:
        axis, theta = w / norm, norm * value
        wx = skew(axis)
        result[:3, :3] = Rotation.from_rotvec(axis * theta).as_matrix()
        result[:3, 3] = (np.eye(3) * theta + (1-np.cos(theta))*wx + (theta-np.sin(theta))*(wx@wx)) @ (v/norm)
    return result


def adjoint(T, screw):
    R, p = T[:3, :3], T[:3, 3]
    w = R @ screw[3:]
    return np.r_[R @ screw[:3] + np.cross(p, w), w]


def _compile_twist(screw):
    """Constant terms of the same SE(3) exponential as se3_exp.

    A robot calibration's screws do not change with q. Precompute their
    normalization and products instead of rebuilding them for every IK
    collision probe. Keep se3_exp/evaluate as the independent Jacobian path.
    """
    v, w = screw[:3], screw[3:]
    norm = float(np.linalg.norm(w))
    if norm < 1e-10:
        return (norm, v.copy(), None, None, None, None)
    wx = skew(w / norm)
    square = wx @ wx
    vn = v / norm
    return (norm, vn, wx, square, wx @ vn, square @ vn)


def _compiled_exp(terms, value):
    norm, v, wx, square, wv, wwv = terms
    result = np.eye(4)
    if norm < 1e-10:
        result[:3, 3] = v * value
    else:
        theta = norm * value
        sine, one_minus_cosine = math.sin(theta), 1. - math.cos(theta)
        result[:3, :3] += sine * wx + one_minus_cosine * square
        result[:3, 3] = theta * v + one_minus_cosine * wv + (theta - sine) * wwv
    return result


class RobotModel:
    def __init__(self, calibration):
        self.spec = calibration
        if calibration.get("frame") != "robot_base" or calibration.get("version") != 2:
            raise ValueError("Unsupported local kinematic calibration")
        self.reference = finite(calibration["q_reference"], (18,))
        self.lower = finite(calibration["lower"], (18,))
        self.upper = finite(calibration["upper"], (18,))
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid joint bounds")
        self.links = {}
        for name, link in calibration["links"].items():
            self.links[name] = (finite(link["T_reference"], (4, 4)), finite(link["screws"], (6, 18)))
        for name in ("left", "right", "torso"):
            if name not in self.links:
                raise ValueError("Missing controlled link")
        self.sha = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
        # Preflight evaluates many candidate poses; collision FK does not need
        # Jacobians. A bounded, instance-local cache avoids recomputing them.
        self._fk_cache = OrderedDict()
        self._compiled_links = {}

    @staticmethod
    def from_reference(q, lower, upper, poses, jacobians, metadata=None):
        links = {}
        for name, (p, quat) in poses.items():
            J = np.asarray(jacobians[name])
            screws = J.copy()
            screws[:3] -= np.cross(J[3:].T, np.asarray(p)).T
            links[name] = {"T_reference": transform(p, quat).tolist(), "screws": screws.tolist()}
        return RobotModel({"version": 2, "frame": "robot_base", "q_reference": np.asarray(q).tolist(),
                           "lower": np.asarray(lower).tolist(), "upper": np.asarray(upper).tolist(),
                           "links": links, "metadata": metadata or {}})

    def evaluate(self, q, name):
        q = finite(q, (18,))
        home, screws = self.links[name]
        T, transformed = np.eye(4), np.zeros((6, 18))
        for i, value in enumerate(q - self.reference):
            if not np.any(screws[:, i]):
                continue
            transformed[:, i] = adjoint(T, screws[:, i])
            T = T @ se3_exp(screws[:, i], value)
        T = T @ home
        jac = transformed.copy()
        jac[:3] += np.cross(transformed[3:].T, T[:3, 3]).T
        return T, jac

    def poses(self, q, names=("left", "right", "torso")):
        return {name: (T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())
                for name in names for T in [self.forward(q, name)]}

    def forward(self, q, name):
        q = finite(q, (18,))
        home, screws = self.links[name]
        # Links are normally immutable calibration. Detect explicit replacement
        # AND in-place edits, so neither compiled constants nor pose-cache hits
        # can silently keep stale geometry in tests or calibration tooling.
        fingerprint = (home.tobytes(), screws.tobytes(), self.reference.tobytes())
        cached = self._compiled_links.get(name)
        if cached is None or cached[0] != fingerprint:
            if cached is not None:
                self._fk_cache.clear()
            terms = tuple((i, _compile_twist(screws[:, i])) for i in range(18)
                          if np.any(screws[:, i]))
            self._compiled_links[name] = (fingerprint, terms)
        else:
            terms = cached[1]
        key = (q.astype(np.float64).tobytes(), name)
        if key not in self._fk_cache:
            T = np.eye(4)
            delta = q - self.reference
            for i, constant in terms:
                if delta[i]:
                    T = T @ _compiled_exp(constant, delta[i])
            self._fk_cache[key] = T @ home
            if len(self._fk_cache) > 512:
                self._fk_cache.popitem(last=False)
        else:
            self._fk_cache.move_to_end(key)
        return self._fk_cache[key].copy()  # callers must not mutate cached geometry

    def grasp_centers(self, q):
        """Robot-defined closing-region centres, never estimated object poses."""
        points = self.spec.get("metadata", {}).get("grasp_centers_eef", {})
        return {arm: (self.forward(q, arm) @ np.r_[points.get(arm, [0., 0., 0.]), 1.])[:3]
                for arm in ("left", "right")}

    def state(self, q, gripper, base_velocity):
        poses, jac = {}, {}
        for name in ("left", "right", "torso"):
            T, jac[name] = self.evaluate(q, name)
            poses[name] = (T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())
        return RobotState(q, self.lower, self.upper, poses, jac, gripper, base_velocity)

    def grasp_regions(self, q, gripper):
        """Reference-opening finger regions only; changed aperture ABSTAINS.

        The portable model excludes individual finger DOFs. Never draw a stale
        open-jaw region over a closed or asymmetrically loaded hand.
        """
        meta=self.spec.get("metadata",{})
        regions=meta.get("grasp_regions_eef",{})
        reference=meta.get("grasp_region_reference_gripper_m")
        result={}
        for i,arm in enumerate(("left","right")):
            row={"valid":False,"reason":"REGION_GEOMETRY_OR_CURRENT_APERTURE_UNAVAILABLE"}
            result[arm]=row
            if (arm not in regions or reference is None or gripper is None or
                    not meta.get("grasp_region_reference_fully_open",{}).get(arm,False)):continue
            values=finite(gripper,(2,));refs=finite(reference,(2,))
            if abs(values[i]-refs[i])>.0005:
                row["reason"]="APERTURE_DIFFERS_FROM_CALIBRATED_REGION";continue
            sides=[np.asarray(x,dtype=float) for x in regions[arm]]
            if len(sides)!=2 or any(x.ndim!=2 or x.shape[1]!=3 or len(x)<2 or not np.isfinite(x).all() for x in sides):
                raise ValueError("Two finite robot contact strips required")
            T=self.forward(q,arm)
            row.update(valid=True,reason="ROBOT_REFERENCE_OPENING_GEOMETRY",source=meta.get("grasp_region_source"),
                sides_base_m=[(s@T[:3,:3].T+T[:3,3]).tolist() for s in sides],
                not_enclosure_or_holding_evidence=True)
        return result

    def finite_difference_error(self, q, eps=1e-6):
        errors = {}
        for name in ("left", "right", "torso"):
            T, J = self.evaluate(q, name)
            empirical = np.zeros((6, 18))
            for i in range(18):
                d = np.zeros(18); d[i] = eps
                plus, _ = self.evaluate(q + d, name)
                minus, _ = self.evaluate(q - d, name)
                empirical[:3, i] = (plus[:3, 3] - minus[:3, 3]) / (2 * eps)
                empirical[3:, i] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * eps)
            errors[name] = float(np.max(np.abs(empirical - J)))
        return errors
