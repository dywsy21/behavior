"""An appearance hint is not a current detection, coordinate or grasp state."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
from PIL import Image

from .contact_review import apply_review
from .vision import VisualBundle

LABEL = "REFERENCE_APPEARANCE_NOT_CURRENT"
VIEWS = ("head", "left_wrist", "right_wrist")
SYSTEM = """Locate the CURRENT subgoal's identifiable object or affordance in the supplied CURRENT RAW camera images. This is only a visual localization question, not a robot action or a completion judgment.
An additional REFERENCE_APPEARANCE_NOT_CURRENT image is an enlarged raw crop around a previously confirmed target surface. It is only an appearance cue, not an object bounding box or proof of current visibility. It may include background and robot parts. Identify the target anew in the CURRENT images. Do not report coordinates in the reference image and do not copy its position. A partial view is usable when its visible appearance identifies the same target; otherwise report unknown.
Choose the CURRENT view that most clearly identifies a visible physical contact surface on that target. Do not prefer a view merely because it is on the working hand. For pick, use a visible graspable part such as a rim or handle when identifiable; a hole, background or gripper is not an object surface. Use only current visible evidence for the returned point. If the target cannot be identified in any CURRENT image, report visible=false, view=none and target_uv=null.
Return one JSON object with exactly these keys:
visible: boolean; view: head, left_wrist, right_wrist or none; target_uv: [x,y] normalized to [0,1] in that CURRENT RAW image, x from left and y from top, or null if not visible; enclosed: null; co_moving: null; supported: null; effect: null; hazard: none or occluded; note: one short sentence describing the current visible identity cue or absence; other_views: []; target_reference: unknown.
No enclosure, motion, support, holding or task success is established by this localization question."""


def eligible(harness):
    return (harness.goal.kind == "pick" and harness.goal.hand in ("left", "right")
            and harness.stage in ("SEARCH", "RECOVER", "APPROACH", "ALIGN")
            and not harness.stop_reason and not harness.carry
            and not any(harness.pending_grasp.values())
            and not any(harness.hold_verified.values())
            and not any(harness.possible_contact_after_close.values()))


def goal_key(harness):
    # Recovery can replace a goal without changing its list index.
    return json.dumps([harness.index, asdict(harness.goal)], sort_keys=True, allow_nan=False)


def snapshot(harness, state, raw):
    if not isinstance(raw, dict) or set(raw) != set(VIEWS):
        raise ValueError("Appearance memory requires three exact CURRENT RAW images")
    pixels = {}
    for name in VIEWS:
        image = np.asarray(raw[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Exact uint8 RGB required for appearance provenance")
        pixels[name] = [list(image.shape), hashlib.sha256(image.tobytes()).hexdigest()]
    value = [goal_key(harness), harness.stage, np.asarray(state.q).tolist(),
             np.asarray(state.gripper).tolist(), pixels]
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class AppearanceMemory:
    max_age = 8

    def __init__(self):
        self.entry = None
        self.serial = 0
        self.pending = None
        self.used = False

    def begin(self, harness, state, bundle):
        """One invocation per ordinary observation; never adds a model call."""
        self.serial += 1
        self.used = False
        self.pending = snapshot(harness, state, bundle.current_raw)
        if (not eligible(harness) or self.entry is not None and
                (self.entry["goal"] != goal_key(harness) or
                 self.serial-self.entry["serial"] > self.max_age)):
            self.entry = None
        if self.entry is None:
            return None
        self.used = True
        images = [Image.fromarray(bundle.current_raw[v]) for v in VIEWS]
        images.append(self.entry["image"].copy())
        labels = ["CURRENT_"+v.upper()+"_RAW" for v in VIEWS] + [LABEL]
        views = VisualBundle(images, labels, {}, bundle.current_raw)
        context = {"current_goal": asdict(harness.goal),
                   "images": "Three CURRENT RAW views from this same instant, then one NON-CURRENT appearance reference."}
        provenance = {"mode": "appearance_reference_v1", "age_observations": self.serial-self.entry["serial"],
                      "current_snapshot_binding": self.pending,
                      "reference_snapshot_binding": self.entry["binding"],
                      "reference_pixels_sha256": self.entry["pixels_sha256"],
                      "current_detection_or_grasp_claim": False}
        return views, json.dumps(context), provenance

    def used_for(self, harness, state, raw):
        if self.pending != snapshot(harness, state, raw):
            raise ValueError("Appearance observation/refinement snapshot mismatch")
        return self.used

    def update(self, harness, state, evidence, target, bundle, model, receipt):
        """Only an actual selected current surface may refresh the raw crop."""
        self.used_for(harness, state, bundle.current_raw)
        if not eligible(harness) or not evidence.visible:
            self.entry = None
            return {"updated": False, "reason": "INELIGIBLE_OR_NOT_VISIBLE"}
        checked = apply_review(target, receipt.get("near_contact_review"), harness,
                               state, evidence, bundle.current_raw)
        selected = receipt.get("attempted") is True and receipt.get("choice", {}).get("candidate_id") is not None
        if not checked["valid"] or receipt.get("attempted") and not selected:
            self.entry = None
            return {"updated": False, "reason": "CURRENT_SURFACE_UNCONFIRMED"}
        if np.linalg.norm(np.asarray(target["point_base_m"])-model.grasp_centers(state.q)[harness.goal.hand]) > .25:
            self.entry = None
            return {"updated": False, "reason": "OUTSIDE_NEAR_FIELD"}
        if not selected:
            return {"updated": False, "reason": "NO_NEW_SEMANTIC_SELECTION"}
        proposals = receipt["proposals"]
        choice = receipt["choice"]["candidate_id"]
        candidates = [row for row in proposals["candidates"] if row["id"] == choice]
        if (len(candidates) != 1 or proposals["view"] != evidence.view or
                tuple(candidates[0]["target_uv"]) != tuple(evidence.target_uv)):
            raise ValueError("Appearance crop requires the actual selected current candidate")
        original = Image.fromarray(bundle.current_raw[evidence.view])
        box = proposals["crop_box_pixels"]
        if (list(original.size) != proposals["image_size_pixels"] or len(box) != 4 or
                any(type(x) is not int for x in box) or not
                (0 <= box[0] < box[2] <= original.width and 0 <= box[1] < box[3] <= original.height)):
            raise ValueError("Exact in-frame current crop required")
        crop = original.crop(tuple(box))
        scale = 512/max(crop.size)
        crop = crop.resize(tuple(max(1, round(s*scale)) for s in crop.size), Image.Resampling.NEAREST)
        pixels_sha = hashlib.sha256(crop.tobytes()).hexdigest()
        self.entry = {"image": crop, "goal": goal_key(harness), "serial": self.serial,
                      "binding": self.pending, "pixels_sha256": pixels_sha}
        return {"updated": True, "reason": "CURRENT_SEMANTIC_SURFACE_RAW_CROP",
                "snapshot_binding": self.pending, "pixels_sha256": pixels_sha,
                "size": list(crop.size), "max_age_observations": self.max_age,
                "current_detection_or_grasp_claim": False}
