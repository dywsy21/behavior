"""Finite, visible RGB-D surface proposals; VLM must identify the target.

Never snap a rejected pixel to the nearest surface automatically: it may be
background. Numbered candidates are sensor hypotheses, NOT object detections.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .grounding import localize_target, validate_depth
from .protocol import strict_json
from .vision import VisualBundle


@dataclass(frozen=True)
class SurfaceChoice:
    candidate_id: object = None

    def __post_init__(self):
        if self.candidate_id is not None and (type(self.candidate_id) is not int or not 0 <= self.candidate_id < 12):
            raise ValueError("Surface choice must abstain or name a bounded candidate")

    def text(self):
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def parse(cls, text):
        value=strict_json(text)
        if not isinstance(value,dict) or set(value)!={"candidate_id"}:
            raise ValueError("Exactly candidate_id required; no inferred pixels")
        return cls(**value)


def surface_candidates(evidence, depths, model, q, max_candidates=12):
    if type(max_candidates) is not int or not 1 <= max_candidates <= 12:
        raise ValueError("Bounded surface proposal count required")
    receipt={"view":evidence.view,"requested_uv":evidence.target_uv,"candidates":[],
             "scene_truth_used":False,"automatic_target_assignment":False}
    if not evidence.visible or evidence.view not in depths:
        receipt["reason"]="NO_VISIBLE_TARGET_OR_DEPTH";return receipt
    camera=model.spec["metadata"]["cameras"][evidence.view]
    depth=validate_depth(depths[evidence.view],camera)
    width,height=camera["width"],camera["height"]
    receipt["image_size_pixels"]=[width,height]
    cx,cy=map(int,np.rint(np.asarray(evidence.target_uv)*[width-1,height-1]).astype(int))
    radius=max(20,round(.09*min(width,height)))
    x0,y0,x1,y1=max(0,cx-radius),max(0,cy-radius),min(width,cx+radius+1),min(height,cy+radius+1)
    receipt["crop_box_pixels"]=[x0,y0,x1,y1]
    receipt["depth_sha256"]=hashlib.sha256(depth.tobytes()).hexdigest()
    step=max(3,round(radius/12))
    locations=[(x,y) for y in range(y0+2,y1-2,step) for x in range(x0+2,x1-2,step)]
    locations.sort(key=lambda p:((p[0]-cx)**2+(p[1]-cy)**2,p))
    spacing=max(7,round(.018*min(width,height)))
    for x,y in locations:
        if any(np.linalg.norm(np.asarray([x,y])-c["pixel_xy"])<spacing for c in receipt["candidates"]):
            continue
        proposal=replace(evidence,target_uv=(x/(width-1),y/(height-1)),other_views=())
        point=localize_target(proposal,depths,model,q)
        if not point["valid"]:continue
        view=point["views"][0]
        receipt["candidates"].append({"id":len(receipt["candidates"]),"pixel_xy":[x,y],
            "target_uv":list(proposal.target_uv),"point_base_m":point["point_base_m"],
            "depth_m":view["depth_m"],"spread_m":view["spread_m"],
            "candidate_is_not_object_detection":True})
        if len(receipt["candidates"])==max_candidates:break
    receipt["reason"]="VLM_SELECTION_REQUIRED" if receipt["candidates"] else "NO_STABLE_OBSERVED_SURFACE"
    return receipt


def refinement_bundle(bundle, receipt):
    view=receipt["view"]
    original=Image.fromarray(bundle.current_raw[view])
    if list(original.size)!=receipt["image_size_pixels"]:
        raise ValueError("RGB/depth crop resolutions must match exactly")
    raw=original.crop(tuple(receipt["crop_box_pixels"]))
    scale=512/max(raw.size)
    crop=raw.resize(tuple(max(1,round(s*scale)) for s in raw.size),Image.Resampling.NEAREST)
    guide=crop.copy(); draw=ImageDraw.Draw(guide)
    try:font=ImageFont.truetype("DejaVuSans.ttf",18)
    except OSError:font=ImageFont.load_default()
    x0,y0,_,_=receipt["crop_box_pixels"]
    sx,sy=guide.width/raw.width,guide.height/raw.height
    for c in receipt["candidates"]:
        x,y=(c["pixel_xy"][0]-x0)*sx,(c["pixel_xy"][1]-y0)*sy
        draw.ellipse((x-3,y-3,x+3,y+3),fill="cyan",outline="black",width=1)
        text=str(c["id"])
        tx,ty=min(guide.width-30,x+5),max(0,y-20)
        draw.text((tx,ty),text,fill="white",stroke_width=2,stroke_fill="black",font=font)
    images=[Image.fromarray(v) for v in bundle.current_raw.values()]+[crop,guide]
    labels=["CURRENT_"+v.upper()+"_RAW" for v in bundle.current_raw]+[
        "CURRENT_"+view.upper()+"_CROP_RAW",
        "CURRENT_"+view.upper()+"_SURFACE_CANDIDATES_NOT_OBJECT_DETECTIONS"]
    return VisualBundle(images,labels,bundle.geometry,bundle.current_raw)


def select_surface(evidence, choice, receipt):
    if choice.candidate_id is None:return evidence
    selected=next((c for c in receipt["candidates"] if c["id"]==choice.candidate_id),None)
    if selected is None:raise ValueError("Choice is not in this snapshot's candidate list")
    if not evidence.visible or receipt["view"]!=evidence.view:
        raise ValueError("Cannot refine an invisible or different-camera target")
    # Correspondence / motion is not established by selecting a new surface.
    # Never retain potentially mismatched cross-view points or certify a grasp.
    return replace(evidence,target_uv=tuple(selected["target_uv"]),other_views=(),co_moving=None)
