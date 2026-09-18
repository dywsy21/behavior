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


def clip_segment(a, b, size):
    """Intersect a projected segment with the image, never clamp its vertices.

    Offscreen endpoints are ordinary for close wrist cameras. Clipping preserves
    the true visible line; two offscreen points on the same side draw nothing.
    Points behind the camera remain unknown, not invented projections.
    """
    if a is None or b is None:
        return None
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if not np.isfinite([a,b]).all():return None
    d=b-a;lo,hi=0.,1.
    for axis,maximum in enumerate(np.asarray(size)-1):
        if abs(d[axis])<1e-12:
            if not 0<=a[axis]<=maximum:return None
            continue
        enter,leave=sorted((-a[axis]/d[axis],(maximum-a[axis])/d[axis]))
        lo,hi=max(lo,enter),min(hi,leave)
        if lo>hi:return None
    return tuple(a+lo*d),tuple(a+hi*d)


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


def prepare_views(images, model, q, previous=None, grounded=False, gripper=None,show_finger_regions=True):
    metadata = model.spec["metadata"]["cameras"]
    result, labels, geometry, raw = [], [], {}, {}
    poses = model.poses(q)
    centers = model.grasp_centers(q) if grounded else {}
    regions = model.grasp_regions(q,gripper) if grounded else {}
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
                region=regions[arm] if show_finger_regions else {"valid":False,"reason":"FINGER_GUIDE_DISABLED_FOR_ABLATION"}
                record["finger_contact_region"]={"valid":region["valid"],"reason":region["reason"],
                    "robot_geometry_not_target_or_enclosure":True}
                if region["valid"]:
                    strips=[]
                    for side in region["sides_base_m"]:
                        strip=[project(point,T,K) for point in side]
                        strips.append(strip)
                    record["finger_contact_region"]["sides_uv"]=[[
                        None if point is None else (point/np.asarray(original.size)).tolist() for point in side] for side in strips]
                    points=[point for side in strips for point in side]
                    # Draw only actual visible portions, not border-clamped
                    # fake endpoints or a fabricated closed border polygon.
                    drawn=[]
                    if all(point is not None for point in points):
                        color="cyan" if arm=="left" else "magenta"
                        pts=np.asarray(points);middle=pts.mean(axis=0)
                        order=np.argsort(np.arctan2(pts[:,1]-middle[1],pts[:,0]-middle[0]))
                        outline=[tuple(pts[j]) for j in order]
                        for a,b in zip(outline,outline[1:]+outline[:1]):
                            segment=clip_segment(a,b,original.size)
                            if segment is not None:draw.line(segment,fill=color,width=2)
                        for side in strips:
                            for a,b in zip(side,side[1:]):
                                segment=clip_segment(a,b,original.size)
                                if segment is not None:
                                    draw.line(segment,fill="yellow",width=3)
                                    drawn.append([list(p) for p in segment])
                        if drawn and np.all(middle>=0) and np.all(middle<original.size):
                            draw.text((middle[0]+8,middle[1]+18),arm+" FINGER REGION (robot only)",fill=color)
                    record["finger_contact_region"]["visible_strip_segments_px"]=drawn
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
