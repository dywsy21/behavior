"""Strict, factorized micro-actions and observable evidence, never executable code."""
from dataclasses import asdict, dataclass
import json
import math

import numpy as np

PARTS = ("left", "right", "both", "base", "torso", "all")
VIEWS = ("head", "left_wrist", "right_wrist")
TRANSLATIONS = {"forward": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0),
                "right": (0, -1, 0), "up": (0, 0, 1), "down": (0, 0, -1)}
ROTATIONS = {f"{axis}_{sign}": np.eye(3)[i] * value
             for i, axis in enumerate(("roll", "pitch", "yaw"))
             for sign, value in (("plus", 1), ("minus", -1))}
SCALES = ("micro", "fine", "coarse")


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError("Expected bounded JSON text")
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


@dataclass(frozen=True)
class Action:
    part: str
    move: str
    scale: str = "fine"
    frame: str = "base"

    def __post_init__(self):
        if self.part not in PARTS or self.scale not in SCALES or self.frame not in ("base", "tool", *VIEWS):
            raise ValueError("Unknown action field")
        if self.move not in (*TRANSLATIONS, *ROTATIONS, "open", "close", "hold"):
            raise ValueError("Unknown motion")
        if self.part == "all" and self.move != "hold":
            raise ValueError("Only HOLD may address all joints")
        if self.part in ("base", "torso"):
            allowed = ("forward", "back", "left", "right", "yaw_plus", "yaw_minus") if self.part == "base" else ("forward", "back", "up", "down")
            if self.move not in allowed or self.frame != "base":
                raise ValueError("Body controls require their explicit base frame")
        if self.part == "both" and self.frame != "base":
            raise ValueError("Synchronized arms use a shared base frame")
        if self.move in ("open", "close", "hold") and (self.scale != "fine" or self.frame != "base"):
            raise ValueError("Gripper/HOLD has canonical fine/base fields, no hidden displacement")

    @classmethod
    def parse(cls, text):
        value = strict_json(text)
        if not isinstance(value, dict) or set(value) != {"part", "move", "scale", "frame"}:
            raise ValueError("Exactly part, move, scale, frame required")
        if not all(isinstance(v, str) for v in value.values()):
            raise ValueError("Action fields must be strings")
        return cls(**value)

    def text(self):
        return json.dumps(asdict(self), separators=(",", ":"))

    def amount(self, carry=False):
        i = SCALES.index(self.scale)
        if self.move in ROTATIONS:
            if carry and self.part in ("left", "right", "both"):
                raise ValueError("Carry orientation is locked")
            return math.radians((1., 3., 8.)[i]) * (0.5 if carry else 1.)
        steps = {"base": (.02, .06, .15), "torso": (.003, .01, .025)}.get(self.part, (.002, .01, .03))
        return min(steps[i], .01 if self.part != "base" else .04) if carry else steps[i]


HOLD = Action("all", "hold")


@dataclass(frozen=True)
class Evidence:
    """VLM observations are claims, not physical or official ground truth."""
    visible: bool
    view: str
    target_uv: tuple | None
    enclosed: bool | None
    co_moving: bool | None
    supported: bool | None
    effect: bool | None
    hazard: str
    note: str

    @classmethod
    def parse(cls, text):
        value = strict_json(text)
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Evidence fields missing or unknown")
        if type(value["visible"]) is not bool:
            raise ValueError("visible must be boolean")
        for key in ("enclosed", "co_moving", "supported", "effect"):
            if value[key] is not None and type(value[key]) is not bool:
                raise ValueError(f"{key}: boolean or null required")
        if value["view"] not in (*VIEWS, "none") or value["hazard"] not in ("none", "occluded", "slip", "collision"):
            raise ValueError("Bad view or hazard")
        if not isinstance(value["note"], str) or len(value["note"]) > 400:
            raise ValueError("Bounded visible evidence required")
        uv = value["target_uv"]
        if uv is not None:
            if not isinstance(uv, list) or len(uv) != 2 or not all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in uv):
                raise ValueError("target_uv must be normalized image coordinates")
            value["target_uv"] = tuple(uv)
        if value["visible"] and (value["view"] == "none" or uv is None):
            raise ValueError("Visible target needs view and location")
        if not value["visible"] and (value["view"] != "none" or uv is not None):
            raise ValueError("Invisible target must not invent its location")
        return cls(**value)

    def as_dict(self):
        return asdict(self)
