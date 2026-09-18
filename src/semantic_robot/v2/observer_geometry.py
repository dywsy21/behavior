"""Camera pointing and relative viewpoints from robot FK and an old visual anchor.

No affordance, current attachment, object orientation or visibility is inferred.
Changing camera orientation can improve framing without revealing another side.
"""
import numpy as np

from .grounding import validate_depth
from .vision import project


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or length < 1e-6:
        raise ValueError("A finite nonzero three-vector is required")
    return vector / length


def angle_degrees(a, b):
    return float(np.rad2deg(np.arccos(np.clip(unit(a) @ unit(b), -1., 1.))))


def camera_anchor_geometry(camera_transform, hand_transform, anchor_hand, camera):
    anchor_hand = np.asarray(anchor_hand, dtype=float)
    if anchor_hand.shape != (3,) or not np.isfinite(anchor_hand).all():
        raise ValueError("Finite previously observed hand-local anchor required")
    for transform in (camera_transform, hand_transform):
        if np.shape(transform) != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Finite robot FK transforms required")
    point = hand_transform[:3, :3] @ anchor_hand + hand_transform[:3, 3]
    delta = point - camera_transform[:3, 3]
    direction = unit(hand_transform[:3, :3].T @ -delta)
    uv = project(point, camera_transform, camera["K"])
    normalized = None if uv is None else uv / [camera["width"] - 1, camera["height"] - 1]
    in_frame = normalized is not None and bool(np.all(normalized >= 0) and np.all(normalized <= 1))
    return {
        "range_m": float(np.linalg.norm(delta)),
        "bearing_error_deg": angle_degrees(delta, -camera_transform[:3, 2]),
        "prior_anchor_uv": None if normalized is None else normalized.tolist(),
        "in_image_bounds": in_frame,
        "optical_depth_m": float(delta @ -camera_transform[:3, 2]),
        "camera_direction_in_reference_hand": direction.tolist(),
        "framing_cost": None if normalized is None else float(np.linalg.norm(normalized - [.5, .5])),
        "not_current_affordance_or_visibility_evidence": True,
    }


def projected_anchor_depth(geometry, depth, camera):
    """Fresh measured ray compatibility only, not identity or free-space safety."""
    if not geometry["in_image_bounds"]:
        return {"status": "OUTSIDE_IMAGE"}
    raw = validate_depth(depth, camera)
    x, y = np.rint(np.asarray(geometry["prior_anchor_uv"]) * [camera["width"] - 1, camera["height"] - 1]).astype(int)
    if not (2 <= x < camera["width"] - 2 and 2 <= y < camera["height"] - 2):
        return {"status": "IMAGE_EDGE_UNKNOWN"}
    patch = raw[y - 2:y + 3, x - 2:x + 3]
    values = patch[np.isfinite(patch) & (patch > .025) & (patch < 4.)]
    if len(values) < 15:
        return {"status": "DEPTH_UNKNOWN"}
    observed = float(np.median(values))
    spread = float(np.quantile(values, .9) - np.quantile(values, .1))
    expected = geometry["optical_depth_m"]
    tolerance = max(.015, .04 * expected)
    status = ("DEPTH_EDGE_UNKNOWN" if spread > max(.015, .04 * observed) else
              "OCCLUDED_BY_NEARER_SURFACE" if observed < expected - tolerance else
              "OBSERVED_FREE_SPACE_CONTRADICTION" if observed > expected + tolerance else
              "SURFACE_DEPTH_COMPATIBLE_NOT_IDENTITY_PROOF")
    return {"status": status, "observed_depth_m": observed, "expected_depth_m": expected,
            "spread_m": spread, "tolerance_m": tolerance}


def anchor_camera_views(model, q, reference_hand, anchor_hand, depths=None):
    if reference_hand not in ("left", "right"):
        raise ValueError("A verified reference hand is required")
    hand = model.forward(q, reference_hand)
    result = {}
    for view, camera in model.spec["metadata"]["cameras"].items():
        row = camera_anchor_geometry(model.forward(q, "camera_" + view), hand, anchor_hand, camera)
        if depths is not None and view in depths:
            row["current_depth_check"] = projected_anchor_depth(row, depths[view], camera)
        result[view] = row
    return result
