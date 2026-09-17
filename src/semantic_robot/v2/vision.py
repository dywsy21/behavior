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


def prepare_views(images, model, q, previous=None):
    metadata = model.spec["metadata"]["cameras"]
    result, labels, geometry, raw = [], [], {}, {}
    poses = model.poses(q)
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
        geometry[view] = entries
        if previous is not None:
            result.append(Image.fromarray(previous[view])); labels.append("PREVIOUS_"+view.upper()+"_RAW")
        result.append(original); labels.append("CURRENT_"+view.upper()+"_RAW")
        # One geometric guide per camera; raw evidence is never covered by markings.
        result.append(annotated); labels.append("CURRENT_"+view.upper()+"_ROBOT_GUIDE_NOT_OBJECT_LABELS")
    return VisualBundle(result, labels, geometry, raw)
