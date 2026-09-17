"""Portable product-of-exponentials FK/Jacobians from a robot-only calibration.

No simulator imports, scene objects, world localization, mesh truth or reward.
The calibration stores local link poses and joint screws at one reference pose.
It is accepted only after prediction is tested at other poses against the robot.
"""
import hashlib
import json

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
                for name in names for T, _ in [self.evaluate(q, name)]}

    def state(self, q, gripper, base_velocity):
        poses, jac = {}, {}
        for name in ("left", "right", "torso"):
            T, jac[name] = self.evaluate(q, name)
            poses[name] = (T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())
        return RobotState(q, self.lower, self.upper, poses, jac, gripper, base_velocity)

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
