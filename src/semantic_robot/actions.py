"""Small, auditable language surface. No object-level grasp/place macros."""
from dataclasses import dataclass
import re

import numpy as np

DIRECTIONS = {
    "FWD": (1, 0, 0), "BACK": (-1, 0, 0), "LEFT": (0, 1, 0),
    "RIGHT": (0, -1, 0), "UP": (0, 0, 1), "DOWN": (0, 0, -1),
}
ROTATIONS = {f"{axis}_{sign}": tuple(value if i == k else 0 for i in range(3))
             for k, axis in enumerate(("ROLL", "PITCH", "YAW"))
             for sign, value in (("POS", 1), ("NEG", -1))}


@dataclass(frozen=True)
class Unit:
    part: str
    move: str
    grain: str = "FINE"


def parse_action(text: str) -> tuple[Unit, ...]:
    """Reject ambiguity, arbitrary code/numbers, compound base motion and duplicates."""
    if not isinstance(text, str) or len(text) > 160:
        raise ValueError("Expected one short action line")
    text = text.strip()
    if text in {"HOLD", "DONE"}:
        return (Unit("ALL", text),)
    if text in {"MODE CARRY", "MODE NORMAL"}:
        return (Unit("MODE", text.split()[1]),)
    units = []
    for clause in text.split(";"):
        words = clause.strip().split()
        if len(words) not in (2, 3) or not all(re.fullmatch(r"[A-Z_]+", w) for w in words):
            raise ValueError("Use PART MOVE [FINE|COARSE], at most two independent arms")
        part, move = words[:2]
        grain = words[2] if len(words) == 3 else "FINE"
        if grain not in {"FINE", "COARSE"}:
            raise ValueError("Unknown grain")
        if part in {"L", "R", "BOTH"}:
            if move not in {*DIRECTIONS, *ROTATIONS, "OPEN", "CLOSE", "HOLD"}:
                raise ValueError("Unknown arm move")
            if move in {"OPEN", "CLOSE", "HOLD"} and len(words) != 2:
                raise ValueError("Gripper/HOLD has no grain")
            if move in {*DIRECTIONS, *ROTATIONS} and len(words) != 3:
                raise ValueError("Motion requires explicit grain")
        elif part == "BASE":
            if move not in {"FWD", "BACK", "LEFT", "RIGHT", "YAW_POS", "YAW_NEG"} or len(words) != 3:
                raise ValueError("Base allows planar translation or yaw only")
        elif part == "TORSO":
            if move not in {"UP", "DOWN", "FWD", "BACK"} or len(words) != 3:
                raise ValueError("Torso allows bounded sagittal translation only")
        else:
            raise ValueError("Unknown part")
        units.append(Unit(part, move, grain))
    if not 1 <= len(units) <= 2:
        raise ValueError("One unit or two synchronized independent arm units only")
    if len(units) == 2 and {u.part for u in units} != {"L", "R"}:
        raise ValueError("Only separate L and R units can be paired")
    return tuple(units)


def displacement(unit: Unit, carry: bool = False) -> np.ndarray:
    magnitude = .01 if carry else {"FINE": .02, "COARSE": .05}[unit.grain]
    if unit.part == "BASE":
        magnitude = .04 if carry else {"FINE": .06, "COARSE": .15}[unit.grain]
    if unit.part == "TORSO":
        magnitude = .01 if carry else {"FINE": .015, "COARSE": .03}[unit.grain]
    return np.asarray(DIRECTIONS[unit.move], dtype=float) * magnitude


def rotation(unit: Unit, carry: bool = False) -> np.ndarray:
    if carry:
        raise ValueError("CARRY locks wrist orientation; exit explicitly before rotating")
    degrees = {"FINE": 5., "COARSE": 15.}[unit.grain]
    return np.asarray(ROTATIONS[unit.move], dtype=float) * np.deg2rad(degrees)


def action_language(include_pairs=True):
    """Finite grammar for optional constrained decoding; no task/object heuristic."""
    arms = {part: [f"{part} {move} {grain}" for move in (*DIRECTIONS, *ROTATIONS)
                   for grain in ("FINE", "COARSE")] + [f"{part} {move}" for move in ("OPEN", "CLOSE", "HOLD")]
            for part in ("L", "R", "BOTH")}
    lines = [line for group in arms.values() for line in group]
    lines += [f"BASE {move} {grain}" for move in ("FWD", "BACK", "LEFT", "RIGHT", "YAW_POS", "YAW_NEG") for grain in ("FINE", "COARSE")]
    lines += [f"TORSO {move} {grain}" for move in ("UP", "DOWN", "FWD", "BACK") for grain in ("FINE", "COARSE")]
    lines += ["HOLD", "DONE", "MODE CARRY", "MODE NORMAL"]
    if include_pairs:
        lines += [left + " ; " + right for left in arms["L"] for right in arms["R"]]
    return tuple(sorted(lines))
