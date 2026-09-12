"""Causal RGB/velocity no-progress feedback. No simulator privileged state."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


REPEATED_ROTATION_FEEDBACK = "Repeated rotation without approach progress; pause and recover toward the current target."


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = False
    fps: float = 30.
    min_rotation_rad: float = 6.2
    max_translation_path_m: float = .30
    image_repeat_mae: float = .10
    min_seconds: float = 15.
    window_seconds: float = 45.
    cooldown_seconds: float = 20.
    recovery_settle_steps: int = 16


class MotionProgressMonitor:
    def __init__(self, config=None):
        self.cfg = ProgressConfig(**dict(config or {}))
        vals = [v for k, v in vars(self.cfg).items() if k != "enabled"]
        if not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError("Progress monitor thresholds must be finite and positive")
        if self.cfg.window_seconds <= self.cfg.min_seconds:
            raise ValueError("Progress window must exceed minimum observation time")
        self.reset()

    def reset(self):
        self.history = deque()
        self.last_step = None
        self.yaw_integral = self.xy_path = 0.
        self.cooldown_until = -1
        self.feedback = "none"
        self.trigger_count = 0
        self.recovery_translation_start = 0.
        self.current_intent = None
        self.stationary_since = None

    @staticmethod
    def image_signature(image):
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[0] != 3:
            raise ValueError("Progress input must use serving's RGB CHW contract")
        # Fixed coarse photometric signature; never used as a target detector.
        y = np.linspace(0, value.shape[1]-1, 24).astype(int)
        x = np.linspace(0, value.shape[2]-1, 24).astype(int)
        return value[:, y[:, None], x[None, :]].astype(np.float32) / 255.

    def observe(self, *, executed_steps, velocity, head_rgb, intent):
        if not self.cfg.enabled:
            return None
        step = int(executed_steps)
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("Real base velocity is required for progress monitoring")
        if self.last_step is not None and step < self.last_step:
            raise ValueError("Progress time went backwards without reset")
        if self.last_step == step:
            return None  # repeated/cached requests cannot inflate elapsed motion
        intent = str(intent)
        if self.current_intent != intent:
            # Matching only the endpoints would join two different intent
            # intervals after a planner changes away and then back again.
            self.history.clear()
            self.current_intent = intent
        dt = 0. if self.last_step is None else (step-self.last_step)/self.cfg.fps
        self.last_step = step
        if np.max(np.abs(velocity)) <= .04:
            if self.stationary_since is None:
                self.stationary_since = step
        else:
            self.stationary_since = None
        self.yaw_integral += float(velocity[2]) * dt
        self.xy_path += float(np.linalg.norm(velocity[:2])) * dt
        signature = self.image_signature(head_rgb)
        while self.history and step - self.history[0][0] > self.cfg.window_seconds*self.cfg.fps:
            self.history.popleft()
        # Keep feedback through the pause/reorientation. Clear only once actual
        # translation is observed; the text never asserts that task succeeded.
        if self.feedback != "none" and self.xy_path - self.recovery_translation_start > .20:
            self.feedback = "none"
        event = None
        if step >= self.cooldown_until:
            for old_step, old_yaw, old_xy, old_image, old_intent in self.history:
                if step-old_step < self.cfg.min_seconds*self.cfg.fps or old_intent != intent:
                    continue
                rotation = abs(self.yaw_integral-old_yaw)
                translation = self.xy_path-old_xy
                repeat_mae = float(np.abs(signature-old_image).mean())
                if (rotation >= self.cfg.min_rotation_rad and translation <= self.cfg.max_translation_path_m
                        and repeat_mae <= self.cfg.image_repeat_mae):
                    self.feedback = REPEATED_ROTATION_FEEDBACK
                    self.cooldown_until = step + int(self.cfg.cooldown_seconds*self.cfg.fps)
                    self.recovery_translation_start = self.xy_path
                    self.trigger_count += 1
                    event = {"reason": "repeated_rotation", "rotation_rad": rotation,
                             "translation_path_m": translation, "image_repeat_mae": repeat_mae,
                             "elapsed_steps": step-old_step, "feedback": self.feedback}
                    break
        self.history.append((step, self.yaw_integral, self.xy_path, signature, str(intent)))
        return event

    def acknowledge_recovery_exit(self, *, previous_intent, next_intent, status):
        """Retire stale loop feedback after an explicit planner exit AND a real stop.

        A near-target reset may need orientation but no translation. This does
        not certify target facing or task success; it only acknowledges that
        the observed spin has stopped and the planner exited local recovery.
        A later new loop is still independently monitored.
        """
        prefix = "stop rotating, face ["
        if (not self.cfg.enabled or self.feedback == "none" or status != "CONTINUE"
                or not str(previous_intent).startswith(prefix) or str(next_intent).startswith(prefix)
                or not str(next_intent).strip() or self.stationary_since is None
                or self.last_step-self.stationary_since < self.cfg.recovery_settle_steps):
            return False
        self.feedback = "none"
        self.history.clear()
        return True
