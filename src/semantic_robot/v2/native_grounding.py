"""Current RAW grounding using Qwen's native 0..1000 point/box protocol.

A point is only a model-selected visible surface hypothesis. A box is only an
image region and has no implicit centre/grasp point. Neither proves identity,
holding or task completion. No scene objects or private evaluator state enter
this module. Coordinates use the official image-width/height convention, then
convert explicitly to this repository's (width-1)/(height-1) sensor UVs.
"""
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
from PIL import Image

from .finite_localization import SelectionReply, SelectionRequest, bind_frame
from .grounding import GroundedEvidence, localize_target
from .harness import Goal
from .protocol import VIEWS, strict_json


SYSTEM = "You are a helpful assistant."
FIELDS = {"point": ("point_2d", 2), "box": ("bbox_2d", 4)}


@dataclass(frozen=True)
class NativeDetection:
    label: str
    coordinates: tuple


def parse_native(text, mode):
    """Allow one complete JSON fence, never repair fields/coordinates/prose."""
    if mode not in FIELDS or not isinstance(text, str) or len(text) > 8192:
        raise ValueError("Bounded native point/box JSON required")
    normalized = text.strip()
    lines = normalized.splitlines()
    transport = None
    if len(lines) >= 3 and lines[0] in ("```json", "```") and lines[-1] == "```":
        normalized = "\n".join(lines[1:-1]).strip()
        transport = "single_markdown_json_fence"
    value = strict_json(normalized)
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Native output must be a bounded JSON array, empty for abstention")
    key, size = FIELDS[mode]
    detections = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {key, "label"}:
            raise ValueError("Exact native coordinates and label fields required")
        coordinates, label = row[key], row["label"]
        if (not isinstance(label, str) or not label.strip() or len(label) > 160 or
                not isinstance(coordinates, list) or len(coordinates) != size or
                any(type(v) is not int or not 0 <= v <= 1000 for v in coordinates)):
            raise ValueError("Explicit integer 0..1000 coordinates and bounded label required")
        if mode == "box" and (coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]):
            raise ValueError("Ordered nonzero bbox required; no coordinate swapping")
        detections.append(NativeDetection(label, tuple(coordinates)))
    return tuple(detections), transport


def native_request(goal, view, raw, binding, mode):
    if (not isinstance(goal, Goal) or view not in VIEWS or mode not in FIELDS or
            not isinstance(binding, str) or len(binding) != 64):
        raise ValueError("Explicit bound goal/view/native mode required")
    rgb = np.asarray(raw)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or
            min(rgb.shape[:2]) < 24 or max(rgb.shape[:2]) > 1024):
        raise ValueError("Unmarked native RGB image required")
    key, _ = FIELDS[mode]
    form = '[{"point_2d":[x,y],"label":"target name"}]' if mode == "point" else \
           '[{"bbox_2d":[x1,y1,x2,y2],"label":"target name"}]'
    text = ("Locate this target in the CURRENT image: " + json.dumps(goal.target) + ". "
            "Report " + key + " in a JSON array with this format: " + form + ". "
            "Use integer coordinates normalized from 0 to 1000 relative to the full image width and height. "
            "For a button or handle, locate that specific part, not its parent object. "
            "Return [] if the requested target is absent or cannot be identified unambiguously. "
            "Return at most one target and only JSON.")
    if mode == "point":
        text += " The point must be on a visible surface of the requested target, not on background."
    return SelectionRequest("native_" + mode, binding, SYSTEM, text,
                            (Image.fromarray(rgb).copy(),), ("CURRENT_" + view.upper() + "_RAW",), ())


def native_pixels(detection, mode, width, height):
    """Pixel indices for points; half-open pixel rectangle for boxes.

    Point coordinates at 1000 map outside the last pixel and are rejected,
    not silently clipped. Box right/bottom edges at 1000 legitimately equal
    width/height. Non-square images use separate scale factors.
    """
    if (mode not in FIELDS or not isinstance(detection, NativeDetection) or
            any(type(v) is not int or not 24 <= v <= 1024 for v in (width, height))):
        raise ValueError("Validated detection and bounded sensor dimensions required")
    # Direct construction must not bypass parser validation.
    key, size = FIELDS[mode]
    parse_native(json.dumps([{key: list(detection.coordinates), "label": detection.label}]), mode)
    c = detection.coordinates
    if mode == "point":
        pixel = (c[0] * width // 1000, c[1] * height // 1000)
        if not 0 <= pixel[0] < width or not 0 <= pixel[1] < height:
            raise ValueError("Point lies outside native sensor pixels")
        return pixel
    return (c[0] * width // 1000, c[1] * height // 1000,
            (c[2] * width + 999) // 1000, (c[3] * height + 999) // 1000)


def _evidence(view="none", uv=None):
    return GroundedEvidence(visible=uv is not None, view=view, target_uv=uv,
        enclosed=None, co_moving=None, supported=None, effect=None, hazard="none",
        note="Static native-model point hypothesis only; no identity, grasp or outcome proof.",
        other_views=(), target_reference="unknown")


@dataclass(frozen=True)
class NativeLocalization:
    binding: str
    mode: str
    evidence: GroundedEvidence | None
    bbox_pixels: tuple | None
    receipt: dict

    def for_frame(self, frame_id, goal, raw, depths, model, state):
        if bind_frame(frame_id, goal, raw, depths, model, state) != self.binding:
            raise ValueError("Native localization belongs to another capture/goal/geometry")
        return self

    def point_evidence(self):
        if self.mode != "point" or self.evidence is None:
            raise ValueError("A bounding box cannot be converted implicitly to a contact point")
        return self.evidence


def locate_native(*, frame_id, goal, raw, depths, model, state, view, mode, choose, deadline):
    """One current-image request; callback/worker owns its hard timeout/budget."""
    if (view not in raw or view not in VIEWS or mode not in FIELDS or
            type(deadline) not in (float, int) or not math.isfinite(deadline)):
        raise ValueError("Explicit current view, native mode and finite deadline required")
    binding = bind_frame(frame_id, goal, raw, depths, model, state)
    def current():
        if time.monotonic() >= deadline:
            raise TimeoutError("Native grounding deadline reached")
        if bind_frame(frame_id, goal, raw, depths, model, state) != binding:
            raise ValueError("Public capture changed during native grounding")
    current()
    request = native_request(goal, view, raw[view], binding, mode)
    identity = request.sha256
    reply = choose(request)
    current()
    if not isinstance(reply, SelectionReply) or reply.request_sha256 != identity or request.sha256 != identity:
        raise ValueError("Native reply/request identity mismatch")
    detections, transport = parse_native(reply.text, mode)
    receipt = {"version": "h51-native-grounding-v1", "binding": binding, "mode": mode,
        "view": view, "request": request.receipt(), "request_sha256": identity,
        "raw_text": reply.text, "transport_normalization": transport,
        "detections": [asdict(d) for d in detections], "scene_truth_used": False,
        "temporal_claims_established": False, "robot_controls": 0, "model_calls": 1}
    evidence, box = (_evidence() if mode == "point" else None), None
    if len(detections) != 1:
        receipt["reason"] = "MODEL_ABSTAINED" if not detections else "AMBIGUOUS_MULTIPLE_TARGETS"
    else:
        camera = model.spec["metadata"]["cameras"][view]
        width, height = camera["width"], camera["height"]
        try:
            pixels = native_pixels(detections[0], mode, width, height)
        except ValueError:
            receipt["reason"] = "POINT_OUTSIDE_SENSOR_PIXELS"
        else:
            receipt["native_pixels"] = list(pixels)
            if mode == "box":
                box = pixels
                receipt["reason"] = "MODEL_REGION_ONLY_NOT_A_CONTACT"
            else:
                selected = _evidence(view, (pixels[0] / (width - 1), pixels[1] / (height - 1)))
                geometry = localize_target(selected, depths, model, state.q)
                receipt["geometry"] = geometry
                if geometry["valid"]:
                    evidence = selected
                    receipt["reason"] = "MODEL_POINT_PLUS_VALID_DEPTH_NOT_IDENTITY_PROOF"
                else:
                    receipt["reason"] = "POINT_DEPTH_REJECTED"
    current()
    return NativeLocalization(binding, mode, evidence, box, receipt)
