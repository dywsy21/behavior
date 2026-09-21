"""Bound-constrained IK, FK line search and 30 Hz fail-stop execution.

All collision geometry here is robot-only capsule approximation, not a claim of
complete environment collision avoidance. Scene RGB/depth belongs to perception.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from semantic_robot.control import finite, pose_error
from .protocol import Action, ROTATIONS, TRANSLATIONS, VIEWS


GRIPPER_COMPLETION_VERSION = "gripper-command-completion-v1"
GRIPPER_COMPLETED = "GRIPPER_COMMAND_COMPLETED"


def execution_completed(action, feedback):
    """Execution receipt, never a grasp/release/task-success predicate.

    Legacy positional receipts retain their meaning. A new gripper receipt is
    accepted only for its exact OPEN/CLOSE action, with the original pose errors
    still visible. Bare status strings and partial/unsafe receipts fail closed.
    """
    if not isinstance(feedback, dict) or "visual_gate_failure" in feedback:
        return False
    if feedback.get("status") == "TARGET_REACHED":
        return True
    if (feedback.get("status") != GRIPPER_COMPLETED or not isinstance(action, Action)
            or action.move not in ("open", "close") or action.part not in ("left", "right", "both")):
        return False
    try:
        r = feedback["gripper_execution"]
        expected = {"micro": 12, "fine": 18, "coarse": 24}[action.scale]
        if (r["version"] != GRIPPER_COMPLETION_VERSION or r["part"] != action.part or r["move"] != action.move
                or r["command_complete"] is not True or r["success_claim"] is not False
                or feedback["holding"] != "UNKNOWN"
                or feedback["official_success"] != "NOT_AVAILABLE_TO_ACTOR"
                or any(type(v) is not int or v != expected for v in
                       (r["planned_control_ticks"], r["executed_control_ticks"], feedback["control_ticks"]))
                or set(r["final_state_checks"]) != {"finite_proprio", "joint_bounds", "robot_collision_free",
                      "active_hand_within_divergence_envelope", "inactive_hand_within_pose_tolerance"}
                or any(v is not True for v in r["final_state_checks"].values())):
            return False
        limits = r["limits"]
        p, a, ip, ia = (float(limits[k]) for k in
                       ("active_position_m", "active_angle_deg", "inactive_position_m", "inactive_angle_deg"))
        if not (0 < p <= .018 and 0 < a <= 9.000000001 and ip == .0025 and ia == 1.5):
            return False
        active = ("left", "right") if action.part == "both" else (action.part,)
        nominal = True
        for arm in ("left", "right"):
            pe, ae = float(feedback["target_error_m"][arm]), float(feedback["orientation_error_deg"][arm])
            if not np.isfinite([pe, ae]).all() or min(pe, ae) < 0:
                return False
            nominal = nominal and pe <= ip and ae <= ia
            if pe > (p if arm in active else ip) or ae > (a if arm in active else ia):
                return False
        return feedback["pose_tracking_status"] == ("TARGET_REACHED" if nominal else "TRACKING_FAILED")
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def native_action(q, grips, velocity=(0., 0., 0.)):
    q, grips = finite(q, (18,)), finite(grips, (2,))
    vel = finite(velocity, (3,)) / [.75, .75, 1.]
    if np.any(np.abs(vel) > 1 + 1e-8) or np.any(np.abs(grips) > 1):
        raise ValueError("Controller range exceeded")
    return np.r_[vel, q[:4], q[4:11], grips[0], q[11:18], grips[1]].astype(np.float32)


def bounded_ik(J, error, q, lower, upper, step=.025):
    """Limits are INSIDE least squares. No post-hoc coordinate-wise clipping."""
    q = np.asarray(q)
    lo, hi = np.maximum(lower-q, -step), np.minimum(upper-q, step)
    if np.any(lo > hi) or not np.isfinite(J).all() or not np.isfinite(error).all():
        raise ValueError("Invalid IK state/bounds")
    fixed = (hi-lo) < 1e-10
    result = np.zeros(len(q)); result[fixed] = (lo[fixed]+hi[fixed])/2
    free = ~fixed
    if free.any():
        A = np.vstack([J[:, free], .025*np.eye(free.sum())])
        b = np.r_[error-J[:, fixed]@result[fixed], np.zeros(free.sum())]
        fit = lsq_linear(A, b, bounds=(lo[free], hi[free]), method="bvls", tol=1e-8, max_iter=100)
        if not fit.success or not np.isfinite(fit.x).all():
            raise ValueError("Bounded IK did not converge")
        result[free] = fit.x
    return result


def segment_distance(a, b, c, d):
    """Distance between closed line segments, including degenerate capsules."""
    u, v, w = b-a, d-c, a-c
    aa, bb, cc, dd, ee = u@u, u@v, v@v, u@w, v@w
    if aa < 1e-12:
        t = np.clip(ee/max(cc, 1e-12), 0, 1)
        return float(np.linalg.norm(a-(c+t*v)))
    if cc < 1e-12:
        s = np.clip(-dd/aa, 0, 1)
        return float(np.linalg.norm(a+s*u-c))
    denom = aa*cc-bb*bb
    s = np.clip((bb*ee-cc*dd)/denom, 0, 1) if denom > 1e-12 else 0.
    t = (bb*s+ee)/cc
    if t < 0:
        t, s = 0., np.clip(-dd/aa, 0, 1)
    elif t > 1:
        t, s = 1., np.clip((bb-dd)/aa, 0, 1)
    return float(np.linalg.norm(a+s*u-c-t*v))


class RobotCollisionGuard:
    def __init__(self, model, hand_body=False):
        self.model = model
        self.chains = model.spec.get("metadata", {}).get("arm_chains", {})
        from .hand_body_collision import HandBodyGuard
        self.hand_body = HandBodyGuard(model) if hand_body else None

    def clearance(self, q):
        names = {"left", "right"} | {name for chain in self.chains.values() for name in chain}
        poses = self.model.poses(q, names=names)
        # Always check both wrists. Additional arm link chains cover non-neighbor links.
        result = float(np.linalg.norm(poses["left"][0]-poses["right"][0])-.08)
        arms = {}
        for arm, names in self.chains.items():
            arms[arm] = [(poses[a][0], poses[b][0]) for a, b in zip(names[:-1], names[1:])]
        for arm, segments in arms.items():
            for i, (a, b) in enumerate(segments):
                for c, d in segments[i+3:]:
                    result = min(result, segment_distance(a, b, c, d)-.06)
        for a, b in arms.get("left", []):
            for c, d in arms.get("right", []):
                result = min(result, segment_distance(a, b, c, d)-.06)
        if self.hand_body is not None:
            result = min(result, self.hand_body.clearance(q)[0])
        return result


@dataclass(frozen=True)
class ServoLimits:
    robot_geometry_guards: bool = False
    gripper_completion_v1: bool = False
    joint_margin: float = .001
    joint_tick: float = .025
    orientation_weight: float = .15  # metres per radian, explicit unlike v1
    divergent_position: float = .018
    divergent_angle: float = np.deg2rad(9)
    stall_ticks: int = 5
    emergency_ticks: int = 3
    carry_duration_v1: bool = False

    def __post_init__(self):
        if type(self.carry_duration_v1) is not bool:
            raise ValueError("Carry duration v1 must be an explicit boolean")
        if self.gripper_completion_v1 and not (0 < self.divergent_position <= .018
                                               and 0 < self.divergent_angle <= np.deg2rad(9)):
            raise ValueError("Gripper completion v1 cannot enlarge the original divergence envelope")

    def action_tick_limit(self, action, carry=False):
        limit = {"micro": 24, "fine": 40, "coarse": 64}[action.scale]
        if (self.carry_duration_v1 and carry and action.part in ("left", "right", "both")
                and action.move in TRANSLATIONS):
            # Carry halves joint speed. Preserve the original finite joint-path
            # budget by doubling only moving time, not the five settling ticks.
            return 2 * (limit - 5) + 5
        return limit


class SafeServo:
    hz = 30

    def __init__(self, model, state, gripper_command=None, limits=None):
        self.model, self.limits = model, limits or ServoLimits()
        self.collision = RobotCollisionGuard(model, self.limits.robot_geometry_guards)
        self.grips = finite(gripper_command, (2,)) if gripper_command is not None else np.clip(state.gripper/.05*2-1, -1, 1)
        self.done, self.status, self.ticks = True, "IDLE", 0
        self.action = None

    def _errors(self, poses, targets):
        return {name: pose_error(targets[name], poses[name]) for name in targets}

    def _cost(self, poses, targets):
        return sum(float((error*np.r_[np.ones(3), np.full(3, self.limits.orientation_weight)]) @
                         (error*np.r_[np.ones(3), np.full(3, self.limits.orientation_weight)]))
                   for error in self._errors(poses, targets).values())

    def _solve(self, q, targets, step):
        poses, jac = {}, {}
        for name in targets:
            T, jac[name] = self.model.evaluate(q, name)
            poses[name] = (T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())
        weight = np.r_[np.ones(3), np.full(3, self.limits.orientation_weight)]
        J = np.vstack([jac[name][:, self.columns]*weight[:, None] for name in targets])
        error = np.concatenate([pose_error(targets[name], poses[name])*weight for name in targets])
        delta = bounded_ik(J, .8*error, q[self.columns], self.model.lower[self.columns]+self.limits.joint_margin,
                           self.model.upper[self.columns]-self.limits.joint_margin, step)
        cost = self._cost(poses, targets)
        for scale in (1., .5, .25, .125):
            candidate = q.copy(); candidate[self.columns] += scale*delta
            if self.collision.clearance(candidate) < 0:
                continue
            if self._cost(self.model.poses(candidate, targets), targets) <= cost+1e-12:
                return candidate
        return q.copy()

    def _within(self, poses):
        return all(np.linalg.norm(e[:3]) <= self.pos_tolerance and np.linalg.norm(e[3:]) <= self.rot_tolerance
                   for e in self._errors(poses, self.targets).values())

    def begin(self, action: Action, state, carry=False):
        if action.move in (*TRANSLATIONS, *ROTATIONS):
            action.amount(carry)  # reject before mutating any runtime state
        self.action, self.start, self.carry = action, state, bool(carry)
        self.joint_plan = None
        self.required_joint_trajectory_ticks = None
        self.ticks, self.stalled, self.divergent, self.limit_ticks = 0, 0, 0, 0
        self.total_ticks = {"micro": 12, "fine": 18, "coarse": 24}[action.scale]
        self.status, self.done, self.base_integral = "RUNNING", False, np.zeros(3)
        self.targets = {name: (p.copy(), q.copy()) for name, (p, q) in state.poses.items() if name in ("left", "right")}
        self.columns, self.velocity_delta = np.array([], dtype=int), np.zeros(3)
        self.previous_cost, self.last_expected = None, self.targets
        self.pos_tolerance, self.rot_tolerance = .0025, np.deg2rad(1.5)
        if np.any(state.q < self.model.lower+self.limits.joint_margin-1e-5) or np.any(state.q > self.model.upper-self.limits.joint_margin+1e-5):
            return self.abort("JOINT_STATE_OUT_OF_BOUNDS")
        if self.collision.clearance(state.q) < 0:
            return self.abort("ROBOT_COLLISION_RISK")
        if action.move == "hold":
            return True
        names = ["left", "right"] if action.part == "both" else [action.part]
        if action.move in ("open", "close"):
            for name in names:
                self.grips[("left", "right").index(name)] = 1 if action.move == "open" else -1
            return True
        amount = action.amount(carry)
        if action.part == "base":
            if action.move in TRANSLATIONS:
                self.velocity_delta[:2] = np.asarray(TRANSLATIONS[action.move])[:2]*amount
            else:
                self.velocity_delta[2] = amount*(1 if action.move == "yaw_plus" else -1)
            return True
        if action.part == "torso":
            self.targets["torso"] = tuple(v.copy() for v in state.poses["torso"])
            self.columns = np.arange(18)  # coupled trunk + arm compensation in ONE solve
        else:
            self.columns = np.concatenate([np.arange(4, 11) if n == "left" else np.arange(11, 18) for n in names])
        for name in names:
            p, quat = self.targets[name]
            frame = np.eye(3)
            if action.frame == "tool":
                frame = Rotation.from_quat(quat).as_matrix()
            elif action.frame in VIEWS:
                T, _ = self.model.evaluate(state.q, "camera_"+action.frame)
                # USD camera: right +X, up +Y, looks -Z. Semantic axes: forward,left,up.
                frame = T[:3, :3] @ np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
            if action.move in TRANSLATIONS:
                self.targets[name] = (p+frame@np.asarray(TRANSLATIONS[action.move])*amount, quat)
                self.pos_tolerance = max(.0007, min(.0025, amount*.25))
            else:
                self.targets[name] = (p, (Rotation.from_rotvec(frame@ROTATIONS[action.move]*amount)*Rotation.from_quat(quat)).as_quat())
                self.rot_tolerance = max(.003, min(np.deg2rad(1.5), amount*.25))
        # Non-mutating reachability gate before actuating anything.
        predicted = state.q.copy()
        for _ in range(64):
            errors = self._errors(self.model.poses(predicted, self.targets), self.targets)
            if all(np.linalg.norm(e[:3]) <= min(.0005,self.pos_tolerance*.25) and
                   np.linalg.norm(e[3:]) <= self.rot_tolerance*.25 for e in errors.values()):
                break
            predicted = self._solve(predicted, self.targets, .05)
        if not self._within(self.model.poses(predicted, self.targets)):
            return self.abort("UNREACHABLE_OR_COLLISION_BLOCKED")
        # Plan AND execute the same finite trajectory. Re-solving a lightly
        # damped Cartesian ramp once per tick does not reproduce the iterative
        # reachability solve, particularly at the folded-arm singularity.
        joint_tick = self.limits.joint_tick*(.5 if self.carry else 1)
        delta = predicted-state.q
        moving_ticks = max(int(np.ceil(self.total_ticks*.65)),
                           int(np.ceil(1.875*np.max(np.abs(delta))/joint_tick)))
        ticks = max(self.total_ticks,moving_ticks+5)
        self.required_joint_trajectory_ticks = ticks
        if ticks > self.limits.action_tick_limit(action, self.carry):
            return self.abort("DURATION_LIMIT_EXCEEDED")
        plan, previous = [], state.q.copy()
        for i in range(ticks):
            t = min(1.,(i+1)/moving_ticks)
            u = 10*t**3-15*t**4+6*t**5
            q = state.q+u*delta
            if (np.max(np.abs(q-previous)) > joint_tick+1e-9 or
                    self.collision.clearance(q) < 0 or self.collision.clearance((q+previous)/2) < 0):
                return self.abort("TRAJECTORY_LIMIT_OR_COLLISION")
            # Joint interpolation may curve in Cartesian space. Explicitly
            # reject paths leaving the bounded intended position/orientation
            # corridor, rather than silently changing the meaning of UP etc.
            for name,(p,quat) in self.model.poses(q,self.targets).items():
                p0,q0 = state.poses[name]; goal_p,goal_q = self.targets[name]
                rot = (Rotation.from_quat(goal_q)*Rotation.from_quat(q0).inv()).as_rotvec()
                expected = (p0+u*(goal_p-p0),(Rotation.from_rotvec(u*rot)*Rotation.from_quat(q0)).as_quat())
                error = pose_error(expected,(p,quat))
                if np.linalg.norm(error[:3]) > 2*self.pos_tolerance or np.linalg.norm(error[3:]) > 2*self.rot_tolerance:
                    return self.abort("CARTESIAN_PATH_DEVIATION")
            plan.append(q); previous = q
        self.joint_plan, self.total_ticks = np.asarray(plan), ticks
        return True

    def abort(self, reason):
        self.status, self.done = reason, True
        return False

    def safe_hold(self, state):
        return native_action(state.q, self.grips)

    def _gripper_v1(self):
        return bool(self.limits.gripper_completion_v1 and self.action is not None
                    and self.action.move in ("open", "close"))

    def _gripper_checks(self, state):
        """Public proprioception only; not contact or environment certification."""
        finite_state = (np.isfinite(state.q).all() and np.isfinite(state.gripper).all()
                        and np.isfinite(state.base_velocity).all()
                        and all(np.isfinite(np.r_[p, q]).all() and np.linalg.norm(q) > 1e-8
                                for p, q in state.poses.values()))
        bounds = bool(finite_state and np.all(state.q >= self.model.lower+self.limits.joint_margin-1e-5)
                      and np.all(state.q <= self.model.upper-self.limits.joint_margin+1e-5))
        collision_free = bool(finite_state and self.collision.clearance(state.q) >= 0)
        active = ("left", "right") if self.action.part == "both" else (self.action.part,)
        errors = self._errors(state.poses, self.targets) if finite_state else {}
        def within(arms, position, angle):
            return bool(finite_state and all(np.linalg.norm(errors[n][:3]) <= position
                                            and np.linalg.norm(errors[n][3:]) <= angle for n in arms))
        return {"finite_proprio": bool(finite_state), "joint_bounds": bounds,
                "robot_collision_free": collision_free,
                "active_hand_within_divergence_envelope": within(active, self.limits.divergent_position, self.limits.divergent_angle),
                "inactive_hand_within_pose_tolerance": within([n for n in ("left", "right") if n not in active],
                                                               self.pos_tolerance, self.rot_tolerance)}

    def next_action(self, state):
        if self.done:
            raise RuntimeError("No running micro-action")
        if not np.isfinite(state.q).all():
            raise ValueError("Non-finite proprioception; simulator must stop")
        if self._gripper_v1():
            checks = self._gripper_checks(state)
            if not checks["finite_proprio"]:
                raise ValueError("Non-finite gripper proprioception; simulator must stop")
            if not checks["joint_bounds"]:
                self.abort("JOINT_STATE_OUT_OF_BOUNDS")
                return self.safe_hold(state)
        errors = self._errors(state.poses, self.last_expected)
        bad = any(np.linalg.norm(e[:3]) > self.limits.divergent_position or np.linalg.norm(e[3:]) > self.limits.divergent_angle for e in errors.values())
        self.divergent = self.divergent+1 if bad else 0
        if self.divergent >= self.limits.emergency_ticks:
            self.abort("TRACKING_DIVERGED")
            return self.safe_hold(state)
        if self.collision.clearance(state.q) < 0:
            self.abort("ROBOT_COLLISION_RISK")
            return self.safe_hold(state)
        self.base_integral += state.base_velocity / self.hz
        u = min(1., (self.ticks+1)/(self.total_ticks*.65))
        s = 10*u**3-15*u**4+6*u**5
        current = {}
        for name, (p, q) in self.targets.items():
            p0, q0 = self.start.poses[name]
            rot = (Rotation.from_quat(q)*Rotation.from_quat(q0).inv()).as_rotvec()
            current[name] = (p0+s*(p-p0), (Rotation.from_rotvec(rot*s)*Rotation.from_quat(q0)).as_quat())
        q = self.start.q.copy()
        if self.columns.size:
            planned = self.joint_plan[self.ticks]
            current = self.model.poses(planned,self.targets)
            # Uniform rate limiting of the joint-space segment under physical
            # lag, NOT clipping independently solved Cartesian IK coordinates.
            delta = planned-state.q
            tick = self.limits.joint_tick*(.5 if self.carry else 1)
            q = state.q+delta*min(1.,tick/max(float(np.max(np.abs(delta))),1e-12))
            if self.collision.clearance(q) < 0:
                self.abort("ROBOT_COLLISION_RISK")
                return self.safe_hold(state)
            cost = self._cost(state.poses, self.targets)
            near_limit = np.any(q[self.columns] < self.model.lower[self.columns]+.003) or np.any(q[self.columns] > self.model.upper[self.columns]-.003)
            self.limit_ticks += int(near_limit)
            stalled = self.ticks >= self.total_ticks-5 and self.previous_cost is not None and cost >= self.previous_cost*.995 and not self._within(state.poses)
            self.stalled = self.stalled+1 if stalled else 0
            self.previous_cost = cost
            if self.stalled >= self.limits.stall_ticks:
                self.abort("JOINT_LIMIT_STALL" if near_limit else "NO_MOTION_PROGRESS")
                return self.safe_hold(state)
        self.last_expected = current
        t = self.ticks/(self.total_ticks-1)
        velocity = self.velocity_delta*np.sin(np.pi*t)**2 / ((self.total_ticks-1)/(2*self.hz))
        self.ticks += 1
        if self.ticks == self.total_ticks:
            velocity[:] = 0
        return native_action(q, self.grips, velocity)

    def finish(self, state):
        gripper_checks = self._gripper_checks(state) if self._gripper_v1() else None
        if self.status == "RUNNING":
            if self.ticks < self.total_ticks:
                self.abort("INTERRUPTED")
            elif gripper_checks is not None:
                # This is a completed command, not TARGET_REACHED and not a
                # grasp. Contact may deflect the held arm beyond IK precision.
                # Check the *post-step* state, including the final control tick.
                reasons = {"finite_proprio": "NONFINITE_PROPRIOCEPTION",
                           "joint_bounds": "JOINT_STATE_OUT_OF_BOUNDS",
                           "robot_collision_free": "ROBOT_COLLISION_RISK",
                           "active_hand_within_divergence_envelope": "TRACKING_DIVERGED",
                           "inactive_hand_within_pose_tolerance": "TRACKING_FAILED"}
                self.status = next((reasons[k] for k, ok in gripper_checks.items() if not ok), GRIPPER_COMPLETED)
            elif self.action.part == "base":
                residual = self.base_integral-self.velocity_delta
                self.status = "TARGET_REACHED" if np.linalg.norm(residual[:2]) < .012 and abs(residual[2]) < np.deg2rad(2) else "BASE_TRACKING_FAILED"
            else:
                self.status = "TARGET_REACHED" if self._within(state.poses) else "TRACKING_FAILED"
        self.done = True
        errors = self._errors(state.poses, self.targets)
        empty = [name for i, name in enumerate(("left", "right")) if self.grips[i] < 0 and state.gripper[i] < .0015]
        metadata=self.model.spec.get("metadata",{})
        opening=metadata.get("grasp_region_reference_gripper_m")
        calibrated_open=metadata.get("grasp_region_reference_fully_open",{})
        observed_open=[name for i,name in enumerate(("left","right")) if self.grips[i]>0
                       and opening is not None and calibrated_open.get(name,False)
                       and np.isfinite(state.gripper[i]) and abs(state.gripper[i]-opening[i])<=.0005]
        result = {"status": self.status, "control_ticks": self.ticks, "joint_limit_ticks": self.limit_ticks,
                "target_error_m": {n: float(np.linalg.norm(e[:3])) for n, e in errors.items()},
                "orientation_error_deg": {n: float(np.rad2deg(np.linalg.norm(e[3:]))) for n, e in errors.items()},
                "eef_delta_m": {n: (state.poses[n][0]-self.start.poses[n][0]).tolist() for n in ("left", "right")},
                "finger_mean_m": state.gripper.tolist(), "empty_grasp_suspected": empty,
                "gripper_open_at_calibrated_aperture": observed_open,
                "base_integral": self.base_integral.tolist(), "carry": self.carry,
                "holding": "UNKNOWN", "official_success": "NOT_AVAILABLE_TO_ACTOR"}
        if self.limits.carry_duration_v1:
            result["motion_timing"] = {
                "version": "carry_duration_v1",
                "maximum_control_ticks": self.limits.action_tick_limit(self.action, self.carry),
                "required_joint_trajectory_ticks": self.required_joint_trajectory_ticks,
                "joint_trajectory_planned": self.joint_plan is not None,
                "joint_step_limit": self.limits.joint_tick * (.5 if self.carry else 1),
                "joint_trajectory_settling_ticks": 5,
                "success_claim": False}
        if gripper_checks is not None:
            result["pose_tracking_status"] = "TARGET_REACHED" if self._within(state.poses) else "TRACKING_FAILED"
            result["gripper_execution"] = {"version": GRIPPER_COMPLETION_VERSION,
                "part": self.action.part, "move": self.action.move,
                "planned_control_ticks": self.total_ticks, "executed_control_ticks": self.ticks,
                "command_complete": self.ticks == self.total_ticks and self.status == GRIPPER_COMPLETED,
                "final_state_checks": gripper_checks, "success_claim": False,
                "limits": {"active_position_m": self.limits.divergent_position,
                           "active_angle_deg": float(np.rad2deg(self.limits.divergent_angle)),
                           "inactive_position_m": self.pos_tolerance, "inactive_angle_deg": 1.5}}
        return result
