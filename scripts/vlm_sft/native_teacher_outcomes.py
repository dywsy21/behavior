"""Strict TRAIN-only local outcomes. Never imported by the deployed actor.

Original implementation; no legacy oracle code is copied. One frame per real
physics control is mandatory. Missing evidence is UNKNOWN, never a label.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation


def rigid(value):
    a = np.asarray(value, float)
    if (a.shape != (4, 4) or not np.isfinite(a).all() or
            not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
            not np.allclose(a[:3, :3].T@a[:3, :3], np.eye(3), atol=1e-6, rtol=0) or
            abs(np.linalg.det(a[:3, :3])-1) > 1e-6):
        raise ValueError("Finite SE(3) teacher geometry required")
    return a


def stable(poses, translation=.004, angle_deg=3.):
    first = rigid(poses[0])
    for raw in poses[1:]:
        now = rigid(raw)
        if (np.linalg.norm(now[:3, 3]-first[:3, 3]) > translation or
                np.linalg.norm(Rotation.from_matrix(first[:3, :3].T@now[:3, :3]).as_rotvec()) > np.deg2rad(angle_deg)):
            return False
    return True


class LocalOutcome:
    def __init__(self, spec, stable_ticks=12):
        if spec["verb"] not in ("GRASP", "PRESS", "PLACE_IN", "PLACE_ON") or spec["hand"] not in ("left", "right"):
            raise ValueError("Bound supported local skill and hand required")
        if type(stable_ticks) is not int or stable_ticks < 12:
            raise ValueError("At least twelve distinct physics ticks of stability")
        self.spec, self.required = spec, stable_ticks
        self.frames = deque(maxlen=stable_ticks)
        self.first = self.last = None
        self.closed = self.opened = self.edge_seen = False
        self.close_pose = None
        self.failure = None
        self.result = {"outcome": "UNKNOWN", "reason": "NO_PHYSICS_EVIDENCE"}

    def update(self, frame, token=None):
        tick = frame.get("tick")
        if type(tick) is not int or tick < 0 or frame.get("target_uid") != self.spec["target"]:
            raise ValueError("Wrong target identity or physics clock")
        if self.last is not None:
            if tick == self.last["tick"]:
                if frame != self.last:
                    raise ValueError("Same physics clock has conflicting teacher evidence")
                return dict(self.result)  # Rendering/polling cannot earn stability.
            if tick != self.last["tick"]+1:
                raise ValueError("Missing/reordered physical teacher samples")
        rigid(frame["target_pose"])
        for arm in ("left", "right"):
            rigid(frame["hand_poses"][arm])
        previous = self.last
        self.last = frame
        if self.first is None:
            self.first = frame
        if frame.get("forbidden_contacts"):
            self.failure = "NEW_FORBIDDEN_PHYSICAL_CONTACT"
        if frame.get("payload_ok") is False:
            self.failure = "PROTECTED_PAYLOAD_LOST"
        if self.failure:
            self.result = {"outcome": "FAILED", "reason": self.failure}
            return dict(self.result)
        if frame.get("contacts_known") is not True or frame.get("payload_ok") is not True:
            self.frames.clear()
            self.result = {"outcome": "UNKNOWN", "reason": "MISSING_CONTACT_OR_PAYLOAD_EVIDENCE"}
            return dict(self.result)
        arm, verb = self.spec["hand"], self.spec["verb"]
        if token == arm.upper()+"_CLOSE" and not self.closed:
            self.closed = True
            self.close_pose = rigid(frame["hand_poses"][arm]).copy()
        if token == arm.upper()+"_OPEN": self.opened = True
        held = frame["held"]
        if any(type(held.get(a)) is not bool for a in ("left", "right")):
            self.frames.clear()
            self.result = {"outcome": "UNKNOWN", "reason": "UNKNOWN_TARGET_HOLD"}
            return dict(self.result)
        if self.spec.get("support_hand"):
            support = self.spec["support_hand"]
            support_poses = [np.linalg.inv(rigid(f["hand_poses"][support]))@rigid(f["target_pose"])
                             for f in (self.first, frame)]
            if held.get(support) is not True or not stable(support_poses, .008, 5.):
                self.failure = "SUPPORT_HAND_LOST_TARGET"
                self.result = {"outcome": "FAILED", "reason": self.failure}
                return dict(self.result)
        positive, reason = False, "LOCAL_SKILL_IN_PROGRESS"
        if verb == "GRASP":
            if any(self.first["held"].get(a) is not False for a in ("left", "right")):
                reason = "INITIAL_HOLD_NOT_KNOWN_FALSE_NO_NEW_CREDIT"
            else:
                rise = rigid(frame["target_pose"])[2, 3]-rigid(self.first["target_pose"])[2, 3]
                hand_rise = 0. if self.close_pose is None else rigid(frame["hand_poses"][arm])[2, 3]-self.close_pose[2, 3]
                positive = (self.closed and held[arm] is True and frame["finger_contact"].get(arm) is True
                            and rise >= .03 and hand_rise >= .025)
        elif verb == "PRESS":
            value = frame.get("toggled")
            if type(value) is not bool or type(self.first.get("toggled")) is not bool:
                self.frames.clear()
                self.result = {"outcome": "UNKNOWN", "reason": "TOGGLE_UNAVAILABLE"}
                return dict(self.result)
            if (previous is not None and previous.get("toggled") is False and
                    self.first["toggled"] is False and value is True and frame["finger_contact"].get(arm) is True):
                self.edge_seen = True
            positive = self.edge_seen and value is True
            if self.first["toggled"] is True: reason = "INITIAL_TOGGLE_TRUE_NO_NEW_CREDIT"
        else:
            lin, ang = np.asarray(frame.get("linear_velocity", []), float), np.asarray(frame.get("angular_velocity", []), float)
            if (lin.shape != (3,) or ang.shape != (3,) or not np.isfinite(lin).all() or not np.isfinite(ang).all() or
                    type(frame.get("relation")) is not bool or type(frame.get("supported")) is not bool or
                    (verb == "PLACE_IN" and type(frame.get("corners_inside")) is not bool)):
                self.frames.clear()
                self.result = {"outcome": "UNKNOWN", "reason": "PLACEMENT_MEASUREMENT_UNAVAILABLE"}
                return dict(self.result)
            positive = (self.first["held"].get(arm) is True and self.opened and
                        all(held[a] is False for a in ("left", "right")) and
                        frame.get("finger_opening", {}).get(arm, -1) >= .045 and
                        frame["relation"] is True and frame["supported"] is True and
                        (verb != "PLACE_IN" or frame["corners_inside"] is True) and
                        np.linalg.norm(lin) <= .01 and np.linalg.norm(ang) <= .05)
        if positive:
            self.frames.append(frame)
        else:
            self.frames.clear()
        good = len(self.frames) >= self.required
        if good and verb == "GRASP":
            poses = [np.linalg.inv(rigid(f["hand_poses"][arm]))@rigid(f["target_pose"]) for f in self.frames]
            good = stable(poses)
        if good and verb.startswith("PLACE"):
            good = stable([f["target_pose"] for f in self.frames], .003, 2.)
        self.result = {"outcome": "SUCCEEDED" if good else "IN_PROGRESS", "reason": "CAUSAL_LOCAL_SKILL" if good else reason,
                       "stable_ticks": len(self.frames), "required_ticks": self.required,
                       "target_uid": self.spec["target"], "tick": tick, "official_task_success": None}
        return dict(self.result)
