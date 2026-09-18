"""Robot-geometry overlays and bounded before/after views. No object GT."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


def project(point, T_camera, K):
    local = T_camera[:3, :3].T @ (np.asarray(point)-T_camera[:3, 3])
    optical = np.array([local[0], -local[1], -local[2]])
    if optical[2] <= .005:
        return None
    uvw = np.asarray(K) @ optical
    return uvw[:2]/uvw[2]


def rgb_image(value):
    x = np.asarray(value)
    if x.ndim != 3:
        raise ValueError("Expected RGB image")
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.transpose(1, 2, 0)
    if x.dtype != np.uint8 or x.shape[-1] != 3:
        raise ValueError("RGB must be uint8 HWC or CHW, never silently rescaled")
    return Image.fromarray(x)


@dataclass
class VisualBundle:
    images: list
    labels: list
    geometry: dict
    current_raw: dict


def prepare_views(images, model, q, previous=None, grounded=False):
    metadata = model.spec["metadata"]["cameras"]
    result, labels, geometry, raw = [], [], {}, {}
    poses = model.poses(q)
    centers = model.grasp_centers(q) if grounded else {}
    for view, camera in metadata.items():
        original = rgb_image(images[view+"_rgb"])
        if original.size != (camera["width"], camera["height"]):
            raise ValueError("Camera resolution drift; projection refused")
        raw[view] = np.asarray(original).copy()
        annotated = original.copy(); draw = ImageDraw.Draw(annotated)
        T, _ = model.evaluate(q, "camera_"+view)
        K = camera["K"]; entries = {}
        for arm, (p, _) in poses.items():
            if arm not in ("left", "right"):
                continue
            uv = project(p, T, K)
            record = {"eef_uv": None, "base_axis_pixel_deltas_for_1cm": {}}
            if uv is not None:
                record["eef_uv"] = (uv/np.array(original.size)).tolist()
                if 0 <= uv[0] < original.width and 0 <= uv[1] < original.height:
                    xy = tuple(uv); color = "cyan" if arm == "left" else "magenta"
                    draw.ellipse((xy[0]-4, xy[1]-4, xy[0]+4, xy[1]+4), outline=color, width=2)
                    draw.text((xy[0]+5, xy[1]), arm+" EEF", fill=color)
                for axis, vec, color in (("FWD", (1,0,0), "red"), ("LEFT", (0,1,0), "lime"), ("UP", (0,0,1), "deepskyblue")):
                    end = project(p+.01*np.asarray(vec), T, K)
                    if end is not None:
                        record["base_axis_pixel_deltas_for_1cm"][axis] = (end-uv).round(2).tolist()
                        # Mark scale-expanded arrows for legibility; never use them as target pixels.
                        tip = uv+np.clip((end-uv)*3, -60, 60)
                        if np.all(uv >= 0) and np.all(uv < original.size):
                            draw.line((*uv, *tip), fill=color, width=2)
                            draw.text(tuple(tip), axis, fill=color)
            entries[arm] = record
            if grounded:
                center_uv = project(centers[arm],T,K)
                record["grasp_center_uv"] = None if center_uv is None else (center_uv/np.array(original.size)).tolist()
                record["grasp_center_source"] = model.spec["metadata"].get("grasp_center_source","EEF_FALLBACK_NOT_FINGERTIP")
                if center_uv is not None:
                    in_frame=bool(np.all(center_uv>=0) and np.all(center_uv<original.size))
                    record["grasp_center_in_frame"]=in_frame
                    color="cyan" if arm=="left" else "magenta"
                    if in_frame:
                        x,y=center_uv
                        draw.line((x-7,y,x+7,y),fill=color,width=2)
                        draw.line((x,y-7,x,y+7),fill=color,width=2)
                        draw.text((x+8,y-14),arm+" CLOSING CENTER",fill=color)
                    else:
                        # A border label is NOT a clamped target/hand marker.
                        draw.text((8,28 if arm=="left" else 44),arm+" closing center OFFSCREEN",fill=color)
        geometry[view] = entries
        if previous is not None:
            result.append(Image.fromarray(previous[view])); labels.append("PREVIOUS_"+view.upper()+"_RAW")
        result.append(original); labels.append("CURRENT_"+view.upper()+"_RAW")
        # One geometric guide per camera; raw evidence is never covered by markings.
        result.append(annotated); labels.append("CURRENT_"+view.upper()+"_ROBOT_GUIDE_NOT_OBJECT_LABELS")
    return VisualBundle(result, labels, geometry, raw)
