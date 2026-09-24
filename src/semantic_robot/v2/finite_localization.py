"""Current-frame target localization through two bounded visual choices.

A 3x3 region choice is followed by a choice among at most 12 RGB-D surface
samples spread across that region. Neither valid depth nor a region choice
establishes object identity. Both questions permit abstention; there is no
nearest-surface fallback or free-coordinate answer. Results are static VLM
claims, never holding, bilateral contact, motion, task effect, or success.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .affordance import SurfaceChoice
from .grounding import GroundedEvidence, localize_target, validate_depth
from .harness import Goal
from .protocol import VIEWS


REGION_SYSTEM = """Locate the requested target in this CURRENT camera image. The first image is unmarked RGB; the second is the SAME image with a numbered 3-by-3 grid. Grid cells are NOT object detections. Select the cell containing a clearly visible part of the requested object. For a button or handle, locate that specific part, not merely its parent object. If the requested target is not identifiable, select null. Output exactly one listed JSON choice. Do not infer presence from the request or from the grid. This is only a static region selection, not an action or a holding/success judgement."""

SURFACE_SYSTEM = """Select a visible surface ON the requested target. You receive the CURRENT unmarked full image, a separate overview marking where the crop came from, the unmarked crop, and the same crop with numbered dots. The crop rectangle and dots are NOT object detections. Dots are evenly distributed depth-valid sensor samples and may be on background or another object. Use the full image and unmarked crop to identify what each dot belongs to. For a button or handle, the dot must be on that specific part. Choose null if no dot is clearly on the requested target. Output exactly one listed JSON choice; never invent coordinates or pick the nearest point just because depth exists. Selecting a surface does not establish a grasp pose, enclosure, holding, motion, support, task effect or success."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _pixels(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": _hash(np.ascontiguousarray(array).tobytes())}


def bind_frame(frame_id, goal, raw, depths, model, state):
    """Bind a caller's public capture clock/ID and all used sensor geometry.

    This checks data consistency, not capture freshness at the simulator. The
    caller must supply the reviewed capture barrier and use the same frame ID
    when consuming the result. A changed ID invalidates even identical pixels.
    """
    if not isinstance(frame_id, str) or not 1 <= len(frame_id) <= 256:
        raise ValueError("Explicit bounded current capture identity required")
    if not isinstance(goal, Goal):
        raise ValueError("Validated public goal required")
    cameras = model.spec["metadata"]["cameras"]
    if (not isinstance(raw, dict) or not isinstance(depths, dict) or
            not raw or set(raw) != set(depths) or not set(raw) <= set(VIEWS)):
        raise ValueError("Matching named onboard RGB-D views required")
    # RobotModel caches compiled FK: changing its spec in place is not a new
    # calibration. Reject it rather than combining new metadata with old FK.
    calibration_sha = _hash(json.dumps(model.spec, sort_keys=True).encode())
    if calibration_sha != model.sha:
        raise ValueError("Robot calibration changed in place")
    # FK reads live arrays, not spec: also reject edits to those independent
    # arrays even when the serialized calibration and its hash stayed intact.
    for name, key in (("reference", "q_reference"), ("lower", "lower"), ("upper", "upper")):
        if not np.array_equal(getattr(model, name), np.asarray(model.spec[key])):
            raise ValueError("Live robot reference/limits differ from calibration")
    if set(model.links) != set(model.spec["links"]):
        raise ValueError("Live robot link set differs from calibration")
    for name, (transform, screws) in model.links.items():
        expected = model.spec["links"][name]
        if (not np.array_equal(transform, np.asarray(expected["T_reference"])) or
                not np.array_equal(screws, np.asarray(expected["screws"]))):
            raise ValueError("Live robot FK arrays differ from calibration")
    q = np.asarray(state.q)
    grip = np.asarray(state.gripper)
    if (q.shape != model.reference.shape or grip.shape != (2,) or
            q.dtype.kind not in "fi" or grip.dtype.kind not in "fi" or
            not np.isfinite(q).all() or not np.isfinite(grip).all()):
        raise ValueError("Finite current robot state required")
    sensors = {}
    for view in raw:
        camera = cameras[view]
        width, height = camera["width"], camera["height"]
        rgb = np.asarray(raw[view])
        if (type(width) is not int or type(height) is not int or min(width, height) < 24 or
                max(width, height) > 1024 or rgb.dtype != np.uint8 or rgb.shape != (height, width, 3)):
            raise ValueError("Native calibrated uint8 RGB resolution required")
        depth = validate_depth(depths[view], camera)
        sensors[view] = {"rgb": _pixels(rgb), "depth": _pixels(depth)}
    payload = {"version": "h50-finite-localization-v1", "capture_id": frame_id,
               "goal": asdict(goal), "model_sha256": model.sha,
               "q": _pixels(q), "gripper": _pixels(grip), "sensors": sensors}
    return _hash(_canonical(payload).encode())


def region_boxes(width, height):
    """Nine half-open cells cover the image exactly, also for odd dimensions."""
    if any(type(v) is not int or v < 24 or v > 1024 for v in (width, height)):
        raise ValueError("Bounded native image dimensions required")
    return tuple((col * width // 3, row * height // 3,
                  (col + 1) * width // 3, (row + 1) * height // 3)
                 for row in range(3) for col in range(3))


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


@dataclass(frozen=True)
class SelectionRequest:
    phase: str
    binding: str
    system: str
    text: str
    images: tuple
    labels: tuple
    allowed: tuple

    def receipt(self):
        return {"phase": self.phase, "binding": self.binding, "system": self.system,
                "text": self.text, "images": [{"label": label, **_pixels(image)}
                                               for label, image in zip(self.labels, self.images)],
                "allowed": [choice.text() for choice in self.allowed]}

    @property
    def sha256(self):
        return _hash(_canonical(self.receipt()).encode())


@dataclass(frozen=True)
class SelectionReply:
    request_sha256: str
    text: str


@dataclass(frozen=True)
class LocatedTarget:
    binding: str
    evidence: GroundedEvidence
    receipt: dict

    def for_frame(self, frame_id, goal, raw, depths, model, state):
        if bind_frame(frame_id, goal, raw, depths, model, state) != self.binding:
            raise ValueError("Cannot use a target from a different capture/goal/state")
        return self.evidence


def region_request(goal, view, raw, binding):
    original = Image.fromarray(raw).copy()
    guide = original.copy()
    draw = ImageDraw.Draw(guide)
    font = _font(max(14, min(original.size) // 24))
    cells = region_boxes(*original.size)
    for index, (x0, y0, x1, y1) in enumerate(cells):
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="cyan", width=2)
        draw.text((x0 + 5, y0 + 3), str(index), font=font, fill="white",
                  stroke_width=2, stroke_fill="black")
    text = _canonical({"target": goal.target, "goal_kind": goal.kind, "view": view,
                       "grid": "3 rows x 3 columns, row-major IDs 0..8",
                       "region_is_not_object_detection": True})
    allowed = (SurfaceChoice(), *(SurfaceChoice(i) for i in range(9)))
    return SelectionRequest("region", binding, REGION_SYSTEM, text, (original, guide),
                            ("CURRENT_" + view.upper() + "_RAW",
                             "CURRENT_" + view.upper() + "_REGION_GRID_NOT_OBJECT_LABELS"), allowed)


def region_surfaces(view, box, depths, model, state):
    """Uniform 4x3 samples, not nearest-first samples around a guessed UV."""
    camera = model.spec["metadata"]["cameras"][view]
    width, height = camera["width"], camera["height"]
    if tuple(box) not in region_boxes(width, height):
        raise ValueError("Surface ROI must be one of this frame's exact grid cells")
    x0, y0, x1, y1 = box
    rows = []
    for row in range(3):
        for col in range(4):
            x = x0 + int((col + .5) * (x1 - x0) / 4)
            y = y0 + int((row + .5) * (y1 - y0) / 3)
            evidence = _evidence(view, (x / (width - 1), y / (height - 1)))
            target = localize_target(evidence, depths, model, state.q)
            if target["valid"]:
                rows.append({"id": row * 4 + col, "pixel_xy": [x, y],
                             "target_uv": list(evidence.target_uv),
                             "point_base_m": target["point_base_m"],
                             "identity_not_established_by_depth": True})
    return rows


def surface_request(goal, view, raw, binding, box, candidates):
    original = Image.fromarray(raw).copy()
    overview = original.copy()
    draw = ImageDraw.Draw(overview)
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="cyan", width=3)
    crop = original.crop(box)
    scale = 512 / max(crop.size)
    crop = crop.resize(tuple(max(1, round(s * scale)) for s in crop.size), Image.Resampling.NEAREST)
    guide = crop.copy()
    draw = ImageDraw.Draw(guide)
    for candidate in candidates:
        # Pixel-centre mapping; projection coordinates themselves stay native.
        x = (candidate["pixel_xy"][0] - x0 + .5) * crop.width / (x1 - x0) - .5
        y = (candidate["pixel_xy"][1] - y0 + .5) * crop.height / (y1 - y0) - .5
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="cyan", outline="black")
        draw.text((min(crop.width - 30, x + 6), max(0, y - 23)), str(candidate["id"]),
                  font=_font(22), fill="white", stroke_width=2, stroke_fill="black")
    text = _canonical({"target": goal.target, "goal_kind": goal.kind, "view": view,
                       "crop_box_pixels": list(box), "image_size_pixels": list(original.size),
                       "candidates": [{"id": c["id"], "pixel_xy": c["pixel_xy"]} for c in candidates],
                       "crop_and_depth_candidates_are_not_object_detections": True})
    labels = tuple("CURRENT_" + view.upper() + suffix for suffix in
                   ("_RAW", "_CROP_LOCATION_NOT_OBJECT_BOX", "_CROP_RAW", "_SURFACE_CANDIDATES_NOT_OBJECT_LABELS"))
    allowed = (SurfaceChoice(), *(SurfaceChoice(c["id"]) for c in candidates))
    return SelectionRequest("surface", binding, SURFACE_SYSTEM, text,
                            (original, overview, crop, guide), labels, allowed)


def _evidence(view="none", uv=None):
    return GroundedEvidence(visible=uv is not None, view=view, target_uv=uv,
                            enclosed=None, co_moving=None, supported=None, effect=None,
                            hazard="none", note="Static surface-selection claim only; temporal feedback unknown.",
                            other_views=(), target_reference="unknown")


def locate_target(*, frame_id, goal, raw, depths, model, state, views, choose, deadline):
    """At most two choices per supplied view, at most three views, no retry.

    ``choose`` must count every issued call in its global budget and return a
    SelectionReply tied to this exact request. Its transport/worker must enforce
    a timeout; this function checks the caller's deadline before/after each call.
    No production control path is changed by importing or calling this locator.
    """
    if (not isinstance(views, tuple) or not 1 <= len(views) <= 3 or len(set(views)) != len(views) or
            not set(views) <= set(raw) or any(view not in VIEWS for view in views)):
        raise ValueError("Explicit unique current camera order required")
    if type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise ValueError("Finite monotonic deadline required")
    binding = bind_frame(frame_id, goal, raw, depths, model, state)
    receipt = {"version": "h50-finite-localization-v1", "binding": binding, "views": list(views),
               "calls": [], "attempts": [], "scene_truth_used": False,
               "temporal_claims_established": False, "robot_controls": 0}

    def check_current():
        if time.monotonic() >= deadline:
            raise TimeoutError("Finite localization deadline reached")
        if bind_frame(frame_id, goal, raw, depths, model, state) != binding:
            raise ValueError("Capture changed during localization")

    def select(request):
        check_current()
        identity = request.sha256
        row = {"request_sha256": identity, "request": request.receipt(), "status": "issued"}
        receipt["calls"].append(row)
        reply = choose(request)
        check_current()
        if not isinstance(reply, SelectionReply) or reply.request_sha256 != identity or request.sha256 != identity:
            raise ValueError("Selection reply/request identity mismatch")
        choice = SurfaceChoice.parse(reply.text)
        if choice not in request.allowed or reply.text != choice.text():
            raise ValueError("Noncanonical or unoffered finite choice")
        row.update(status="completed", text=reply.text)
        return choice

    check_current()
    for view in views:
        attempt = {"view": view}
        receipt["attempts"].append(attempt)
        region = select(region_request(goal, view, raw[view], binding))
        attempt["region_choice"] = asdict(region)
        if region.candidate_id is None:
            attempt["reason"] = "REGION_ABSTAINED"
            continue
        box = region_boxes(raw[view].shape[1], raw[view].shape[0])[region.candidate_id]
        candidates = region_surfaces(view, box, depths, model, state)
        attempt.update(crop_box_pixels=list(box), candidates=candidates)
        if not candidates:
            attempt["reason"] = "NO_VALID_SURFACE_SAMPLES"
            continue
        choice = select(surface_request(goal, view, raw[view], binding, box, candidates))
        attempt["surface_choice"] = asdict(choice)
        if choice.candidate_id is None:
            attempt["reason"] = "SURFACE_ABSTAINED"
            continue
        selected = next(c for c in candidates if c["id"] == choice.candidate_id)
        evidence = _evidence(view, tuple(selected["target_uv"]))
        check_current()
        attempt["reason"] = "MODEL_SELECTED_SURFACE_NOT_IDENTITY_PROOF"
        receipt.update(reason=attempt["reason"], selected_view=view, evidence=asdict(evidence))
        return LocatedTarget(binding, evidence, receipt)
    check_current()
    evidence = _evidence()
    receipt.update(reason="NO_CONFIRMED_TARGET_SURFACE", evidence=asdict(evidence))
    return LocatedTarget(binding, evidence, receipt)
