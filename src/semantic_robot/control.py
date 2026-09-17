"""Robot-relative Cartesian servo, producing the unchanged native 23D action.

All inputs are robot proprioception / robot kinematics; no scene object access.
Targets are latched per semantic unit, not repeatedly added each physics tick.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .actions import DIRECTIONS, ROTATIONS, displacement, rotation


def finite(value, shape):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite {shape}, got {value.shape}")
    return value.copy()


@dataclass
class RobotState:
    q: np.ndarray  # trunk[4], left[7], right[7]
    lower: np.ndarray
    upper: np.ndarray
    poses: dict  # left/right/torso: (position[3], quaternion_xyzw[4]), robot base frame
    jacobians: dict  # each [6,18], in the same frame, columns ordered as q
    gripper: np.ndarray  # measured finger openings, [left_mean,right_mean] meters
    base_velocity: np.ndarray  # body-local vx,vy,wz, no global position

    def __post_init__(self):
        for key in ("q", "lower", "upper"):
            setattr(self, key, finite(getattr(self, key), (18,)))
        if np.any(self.lower >= self.upper):
            raise ValueError("Bad joint limits")
        self.gripper = finite(self.gripper, (2,))
        self.base_velocity = finite(self.base_velocity, (3,))
        for name in ("left", "right", "torso"):
            pos, quat = self.poses[name]
            pos, quat = finite(pos, (3,)), finite(quat, (4,))
            if not .99 <= np.linalg.norm(quat) <= 1.01:
                raise ValueError("Expected normalized XYZW quaternion")
            self.poses[name] = (pos, quat)
            self.jacobians[name] = finite(self.jacobians[name], (6, 18))


def dls(jacobian, error, damping=.025, max_joint_step=.025):
    jacobian = np.asarray(jacobian)
    error = finite(error, (6,))
    if jacobian.shape[0] != 6 or not np.isfinite(jacobian).all():
        raise ValueError("Bad Jacobian")
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), error)
    # Uniform scaling preserves the joint-space direction, unlike per-axis clipping.
    delta *= min(1., max_joint_step / max(np.max(np.abs(delta)), 1e-9))
    return delta


def pose_error(target, actual):
    p, q = actual
    tp, tq = target
    return np.r_[tp - p, (Rotation.from_quat(tq) * Rotation.from_quat(q).inv()).as_rotvec()]


class R1ProServo:
    hz = 30

    def __init__(self, state: RobotState, gripper_command=None):
        self.carry = False
        self.grips = (np.clip(state.gripper / .05 * 2 - 1, -1, 1) if gripper_command is None
                      else finite(gripper_command, (2,)))
        if np.any(np.abs(self.grips) > 1):
            raise ValueError("Gripper command must be in [-1,1]")
        self.units = ()
        self.start = state
        self.targets = {}
        self.joint_hold = state.q.copy()
        self.base_delta = np.zeros(3)
        self.base_integral = np.zeros(3)
        self.ticks = 0
        self.total_ticks = 24
        self.clipped_ticks = 0

    def begin(self, units, state):
        # Construct/validate locally first. Rejected commands must not mutate targets/grippers.
        carry = self.carry
        targets = {k: (p.copy(), q.copy()) for k, (p, q) in state.poses.items()}
        grips = self.grips.copy()
        base_delta = np.zeros(3)
        if units[0].part == "MODE":
            carry = units[0].move == "CARRY"
        for unit in units:
            if unit.part in {"L", "R", "BOTH"}:
                arms = ["left", "right"] if unit.part == "BOTH" else [{"L": "left", "R": "right"}[unit.part]]
                for name in arms:
                    p, q = targets[name]
                    if unit.move in DIRECTIONS:
                        p = p + displacement(unit, carry)
                    elif unit.move in ROTATIONS:
                        q = (Rotation.from_rotvec(rotation(unit, carry)) * Rotation.from_quat(q)).as_quat()
                    elif unit.move in {"OPEN", "CLOSE"}:
                        grips[{"left": 0, "right": 1}[name]] = 1 if unit.move == "OPEN" else -1
                    targets[name] = (p, q)
            elif unit.part == "TORSO":
                p, q = targets["torso"]
                targets["torso"] = (p + displacement(unit, carry), q)
            elif unit.part == "BASE":
                if unit.move in ROTATIONS:
                    # CARRY permits a slow base turn without rotating the wrist relative to base.
                    base_delta[2] = np.deg2rad(3 if carry else (5 if unit.grain == "FINE" else 15)) * (1 if unit.move == "YAW_POS" else -1)
                else:
                    base_delta[:2] = displacement(unit, carry)[:2]
        if np.linalg.norm(targets["left"][0] - targets["right"][0]) < .08:
            raise ValueError("Requested wrist endpoints too close (<8cm); not a full collision check")
        self.carry, self.targets, self.grips = carry, targets, grips
        self.start, self.units = state, units
        self.joint_hold = state.q.copy()
        self.base_delta, self.base_integral = base_delta, np.zeros(3)
        self.ticks, self.clipped_ticks = 0, 0
        self.total_ticks = 30 if carry else 24

    def next_action(self, state: RobotState):
        if not self.targets or self.ticks >= self.total_ticks:
            raise RuntimeError("begin() required; semantic step already exhausted")
        # Quintic easing, then hold final target for last 1/4 of the micro-step.
        u = min(1., (self.ticks + 1) / (self.total_ticks * .75))
        s = 10*u**3 - 15*u**4 + 6*u**5
        q = self.joint_hold.copy()
        torso_active = any(unit.part == "TORSO" for unit in self.units)
        names = ["torso", "left", "right"] if torso_active else ["left", "right"]
        for name in names:
            cols = {"torso": slice(0, 4), "left": slice(4, 11), "right": slice(11, 18)}[name]
            active = torso_active or any(unit.part in ({"L", "BOTH"} if name == "left" else {"R", "BOTH"}) and unit.move in {*DIRECTIONS, *ROTATIONS} for unit in self.units)
            if not active:
                continue
            start_p, start_q = self.start.poses[name]
            end_p, end_q = self.targets[name]
            dr = (Rotation.from_quat(end_q) * Rotation.from_quat(start_q).inv()).as_rotvec()
            target = (start_p + s * (end_p - start_p), (Rotation.from_rotvec(s*dr) * Rotation.from_quat(start_q)).as_quat())
            err = pose_error(target, state.poses[name])
            # Bounded differential IK with orientation retained, including torso compensation.
            q[cols] = state.q[cols] + dls(state.jacobians[name][:, cols], .7*err,
                                        max_joint_step=.012 if self.carry else .025)
        clipped = np.clip(q, state.lower + .001, state.upper - .001)
        self.clipped_ticks += int(not np.array_equal(q, clipped))
        # One finite base pulse with smooth acceleration; the last tick is always zero.
        t = self.ticks / (self.total_ticks - 1)
        envelope = np.sin(np.pi*t)**2
        normalizer = (self.total_ticks - 1) / (2 * self.hz)
        velocity = self.base_delta / normalizer * envelope
        action = np.r_[velocity / [.75, .75, 1.], clipped[:4], clipped[4:11], self.grips[0], clipped[11:18], self.grips[1]]
        action[:3] = np.clip(action[:3], -1, 1)
        self.base_integral += state.base_velocity / self.hz
        self.ticks += 1
        return finite(action, (23,)).astype(np.float32)

    def feedback(self, state):
        return {"status": "EXECUTED_NOT_TASK_SUCCESS", "control_ticks": self.ticks,
                "joint_limit_ticks": self.clipped_ticks, "carry": self.carry,
                "eef_delta_m": {n: (state.poses[n][0]-self.start.poses[n][0]).round(4).tolist() for n in ("left", "right", "torso")},
                "target_error_m": {n: round(float(np.linalg.norm(self.targets[n][0]-state.poses[n][0])), 4) for n in self.targets},
                "orientation_error_deg": {n: round(float(np.rad2deg(np.linalg.norm(pose_error(self.targets[n], state.poses[n])[3:]))), 2) for n in self.targets},
                "base_local_velocity_integral": self.base_integral.round(4).tolist(),
                "measured_finger_opening_m": state.gripper.round(4).tolist(),
                "gripper_command": self.grips.tolist(), "object_holding": "UNKNOWN"}
