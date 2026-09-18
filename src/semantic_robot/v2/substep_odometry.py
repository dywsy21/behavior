"""Measure within an action, then deliver ONE complete action displacement.

No solver fallback, velocity integration, scene state or partial-motion credit.
The caller owns the control clock and must persist every sampled RGB-D frame.
"""
import copy
import hashlib

import numpy as np

from .grounding import validate_depth


def signature(images, depths, model, q):
    rgb = np.asarray(images["head_rgb"])
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    camera = model.spec["metadata"]["cameras"]["head"]
    if rgb.shape != (camera["height"], camera["width"], 3) or rgb.dtype != np.uint8:
        raise ValueError("Calibrated raw head RGB required")
    depth = validate_depth(depths["head"], camera)
    T = np.asarray(model.forward(q, "camera_head"))
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Finite current camera FK required")
    return {"rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
            "camera_fk": T.tolist()}


def checked_transform(receipt):
    T = np.asarray(receipt.get("body_transform_current_in_previous"), dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-6)):
        raise ValueError("Finite rigid measured transform required")
    delta = np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])
    claimed = np.asarray(receipt.get("body_delta"), dtype=float)
    if (claimed.shape != (3,) or not np.isfinite(claimed).all()
            or not np.allclose(claimed, delta, atol=1e-8)
            or np.linalg.norm(delta[:2]) > .18 or abs(delta[2]) > .30 or abs(T[2, 3]) > .035):
        raise ValueError("Measured transform/delta or original motion bound mismatch")
    return T


class SubstepMotion:
    def __init__(self, estimator, interval=6):
        if interval != 6:
            raise ValueError("Only the registered fixed six-control interval is supported")
        self.estimator = estimator
        self.interval = interval
        self.reference = None
        self.active = False
        self.pending = None
        self.failed = False

    def observe(self, images, depths, model, q):
        if self.active:
            raise ValueError("An action must finish before delivery")
        current = signature(images, depths, model, q)
        if self.pending is not None:
            # Runner reuses the exact end-of-action snapshot at the SAME control
            # count. Do not recompute only the final subinterval as the full move.
            if current != self.reference:
                raise ValueError("End snapshot/FK changed before action-motion delivery")
            receipt, self.pending = self.pending, None
            return copy.deepcopy(receipt)
        if self.failed:
            raise ValueError("A failed motion chain cannot be restarted")
        receipt = self.estimator.observe(images, depths, model, q)
        self.reference = current
        if not receipt.get("valid"):
            self.failed = True
        return receipt

    def begin(self, control):
        if (self.reference is None or self.active or self.pending is not None
                or self.failed or type(control) is not int or control < 0):
            raise ValueError("Valid consumed reference and integer action start required")
        if hasattr(self, "last") and control != self.last:
            raise ValueError("Unmeasured controls between actions are forbidden")
        self.active = True
        self.start = self.last = control
        self.before = copy.deepcopy(self.reference)
        self.total = np.eye(4)
        self.segments = []

    def sample(self, images, depths, model, q, control):
        if (not self.active or self.failed or type(control) is not int
                or not 0 < control-self.last <= self.interval):
            raise ValueError("Positive bounded control gap required; no skipped interval")
        current = signature(images, depths, model, q)
        receipt = copy.deepcopy(self.estimator.observe(images, depths, model, q))
        valid = receipt.get("valid") is True and not receipt.get("initial", False)
        if valid:
            self.total = self.total @ checked_transform(receipt)
        else:
            self.failed = True
        self.segments.append({"control_start": self.last, "control_end": control,
                              "before": self.reference, "after": current, "measurement": receipt})
        self.last = control
        self.reference = current
        return {"valid": valid, "control": control, "measurement": receipt}

    def finish(self, control, *, interrupted=False):
        if not self.active or control != self.last or not self.segments:
            raise ValueError("Every executed tick including the last tail must be sampled")
        if type(interrupted) is not bool:
            raise ValueError("Explicit interruption status required")
        self.active = False
        result = {"valid": False, "source": "fixed_six_control_RGBD_SE3_chain",
                  "no_scene_truth": True, "velocity_integral_not_used": True,
                  "control_start": self.start, "control_end": control,
                  "substep_interval_controls": self.interval, "segments": copy.deepcopy(self.segments),
                  "previous_rgb_sha256": self.before["rgb_sha256"],
                  "current_rgb_sha256": self.reference["rgb_sha256"],
                  "current_depth_sha256": self.reference["depth_sha256"]}
        if self.failed:
            result["reason"] = "SUBSTEP_RGBD_QUALITY_FAILED_NO_PARTIAL_CREDIT"
        elif interrupted:
            self.failed = True
            result["reason"] = "ACTION_INTERRUPTED_NO_FULL_MOTION_CERTIFICATE"
        else:
            T = self.total
            delta = [float(T[0, 3]), float(T[1, 3]), float(np.arctan2(T[1, 0], T[0, 0]))]
            if np.linalg.norm(delta[:2]) > .18 or abs(delta[2]) > .30 or abs(T[2, 3]) > .035:
                self.failed = True
                result["reason"] = "OUTSIDE_SINGLE_MICRO_ACTION_MOTION_BOUND"
            else:
                result.update(valid=True, reason="COMPLETE_ACTION_MEASURED_RGBD_CHAIN",
                              body_delta=delta, body_translation_z_m=float(T[2, 3]),
                              body_transform_current_in_previous=T.tolist())
        # Detailed chain goes to action_motion.json, not the model prompt.
        self.pending = copy.deepcopy({key: value for key, value in result.items() if key != "segments"})
        self.pending["segment_count"] = len(self.segments)
        if not self.failed:
            self.pending["minimum_segment_inliers"] = min(s["measurement"].get("inliers", 0) for s in self.segments)
            self.pending["maximum_segment_median_reprojection_px"] = max(s["measurement"].get("median_reprojection_px", 0.) for s in self.segments)
            self.pending["maximum_segment_median_depth_correspondence_m"] = max(s["measurement"].get("median_depth_correspondence_m", 0.) for s in self.segments)
        return result
