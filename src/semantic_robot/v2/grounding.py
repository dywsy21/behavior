"""Ground VLM-selected pixels using *onboard* depth, never scene truth.

The output is an uncertain visible surface point, not an object pose or grasp
certificate. Occluded/depth-discontinuous/mutually inconsistent points abstain.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .protocol import Evidence, VIEWS, strict_json, TRANSLATIONS
from .vision import project


@dataclass(frozen=True)
class GroundedEvidence(Evidence):
    other_views: tuple = ()

    @classmethod
    def parse(cls, text):
        value = strict_json(text)
        # No corroborating view supplied means NO extra evidence. This is the
        # dataclass's abstaining default, not a guessed pixel / repaired fact.
        # All core evidence fields, unknown keys and malformed supplied views
        # remain strict. The policy records this default without changing raw text.
        if isinstance(value, dict) and set(value) == set(Evidence.__dataclass_fields__):
            value["other_views"] = []
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Exact grounded observation fields required")
        extra = value.pop("other_views")
        import json
        base = Evidence.parse(json.dumps(value, allow_nan=False))
        if not isinstance(extra, list) or len(extra) > 2 or (extra and not base.visible):
            raise ValueError("At most two CURRENT corroborating views of a visible target")
        seen, checked = {base.view}, []
        for entry in extra:
            if not isinstance(entry, dict) or set(entry) != {"view", "target_uv"}:
                raise ValueError("Corroborating views require only view and target_uv")
            if entry["view"] not in VIEWS or entry["view"] in seen:
                raise ValueError("Unique camera views required")
            fields = asdict(base); fields.update(entry)
            validated = Evidence.parse(json.dumps(fields, allow_nan=False))
            checked.append({"view":validated.view, "target_uv":validated.target_uv})
            seen.add(validated.view)
        return cls(**asdict(base), other_views=tuple(checked))


def validate_depth(value, camera):
    depth = np.asarray(value)
    if depth.shape == (camera["height"], camera["width"], 1):
        depth = depth[..., 0]
    if depth.shape != (camera["height"], camera["width"]) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("Depth must be floating metres with the calibrated RGB resolution")
    # Invalid pixels remain invalid. Never convert holes/infinity into obstacles
    # or invent a far surface by clipping.
    return depth.astype(np.float32, copy=False)


def unproject(uv_pixels, z, K, T_camera):
    rays = np.linalg.solve(np.asarray(K), np.c_[np.atleast_2d(uv_pixels), np.ones(len(np.atleast_2d(uv_pixels)))].T).T
    optical = rays * np.asarray(z).reshape(-1, 1)
    usd = optical * [1., -1., -1.]
    return usd @ T_camera[:3, :3].T + T_camera[:3, 3]


def check_projected_contact(point, primary_view, depths, model, q):
    """Check the SAME 3D surface ray in other calibrated cameras.

    Independently named object pixels are not point correspondences. Occlusion,
    FOV and depth discontinuities abstain; clear observed free space at the
    projected point contradicts it. Agreement is geometry, not object identity.
    """
    checks=[]
    for view,camera in model.spec["metadata"]["cameras"].items():
        if view==primary_view or view not in depths:continue
        T=model.forward(q,"camera_"+view)
        uv=project(point,T,camera["K"])
        expected=-float((np.asarray(point)-T[:3,3])@T[:3,2])
        row={"view":view,"expected_depth_m":expected,"status":"OUTSIDE_VIEW",
             "pixel_is_geometry_projection_not_model_correspondence":True}
        checks.append(row)
        if uv is None or np.any(uv<2) or np.any(uv>=[camera["width"]-2,camera["height"]-2]):continue
        row["projected_uv"]=(uv/[camera["width"]-1,camera["height"]-1]).tolist()
        x,y=np.rint(uv).astype(int);depth=validate_depth(depths[view],camera)
        patch=depth[y-2:y+3,x-2:x+3]
        values=patch[np.isfinite(patch)&(patch>.025)&(patch<4.)]
        if len(values)<15:
            row["status"]="DEPTH_UNKNOWN";continue
        measured=float(np.median(values));spread=float(np.quantile(values,.9)-np.quantile(values,.1))
        tolerance=max(.015,expected*.04)
        row.update(observed_depth_m=measured,spread_m=spread,tolerance_m=tolerance)
        if spread>max(.015,measured*.04):row["status"]="DEPTH_EDGE_UNKNOWN"
        elif measured<expected-tolerance:row["status"]="OCCLUDED_BY_NEARER_SURFACE"
        elif measured>expected+tolerance:row["status"]="OBSERVED_FREE_SPACE_CONTRADICTION"
        else:row["status"]="SURFACE_DEPTH_AGREES_NOT_IDENTITY_PROOF"
    return checks


def localize_target(evidence, depth_images, model, q):
    result = {"valid":False, "source":"VLM_pixel_plus_onboard_depth",
              "surface_point_not_object_pose":True, "views":[], "reason":"TARGET_NOT_VISIBLE"}
    if not evidence.visible:
        return result
    entries = [{"view":evidence.view, "target_uv":evidence.target_uv}, *getattr(evidence,"other_views",())]
    points, rows = [], []
    cameras = model.spec["metadata"]["cameras"]
    for entry in entries:
        view, uv = entry["view"], np.asarray(entry["target_uv"])
        row = {"view":view, "target_uv":uv.tolist(), "valid":False}
        rows.append(row)
        if view not in depth_images:
            row["reason"] = "DEPTH_MISSING"; continue
        camera = cameras[view]; depth = validate_depth(depth_images[view], camera)
        x,y = np.rint(uv*(np.array([camera["width"],camera["height"]])-1)).astype(int)
        patch = depth[max(0,y-2):y+3,max(0,x-2):x+3]
        valid = patch[np.isfinite(patch) & (patch > .025) & (patch < 4.)]
        if valid.size < max(5, .6*patch.size):
            row["reason"] = "DEPTH_HOLE_OR_RANGE"; continue
        z = float(np.median(valid)); spread = float(np.quantile(valid,.9)-np.quantile(valid,.1))
        if spread > max(.015, z*.04):
            row["reason"] = "DEPTH_EDGE_AMBIGUOUS"; row["spread_m"]=spread; continue
        T = model.forward(q,"camera_"+view)
        point = unproject([[x,y]],[z],camera["K"],T)[0]
        row.update(valid=True, depth_m=z, spread_m=spread, point_base_m=point.tolist())
        points.append(point)
    result["views"] = rows
    if not points:
        result["reason"]="NO_VALID_TARGET_DEPTH"; return result
    if len(points)>1 and max(np.linalg.norm(a-b) for a in points for b in points) > .07:
        result["reason"]="MULTIVIEW_TARGET_DISAGREEMENT"; return result
    result.update(valid=True, reason="OBSERVED_SURFACE_ESTIMATE",
                  point_base_m=np.mean(points,axis=0).tolist(), valid_views=len(points))
    if not getattr(evidence,"other_views",()):
        checks=check_projected_contact(result["point_base_m"],evidence.view,depth_images,model,q)
        result["projected_same_point_checks"]=checks
        result["model_cameras_not_independently_averaged"]=True
        if any(row["status"]=="OBSERVED_FREE_SPACE_CONTRADICTION" for row in checks):
            result.update(valid=False,reason="PROJECTED_CONTACT_DEPTH_CONTRADICTION")
    return result


def observed_cloud(depth_images, model, q, stride=12):
    """Sparse CURRENT visible-depth points in robot coordinates, not full scene."""
    points = []
    for view, raw in depth_images.items():
        camera=model.spec["metadata"]["cameras"][view]; depth=validate_depth(raw,camera)
        yy,xx=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
        z=depth[::stride,::stride]; good=np.isfinite(z)&(z>.04)&(z<4.)
        if good.any():
            points.append(unproject(np.c_[xx[good],yy[good]],z[good],camera["K"],model.forward(q,"camera_"+view)))
    return np.concatenate(points) if points else np.empty((0,3))


class LocalDepthGuard:
    """Conservative local base-motion veto; not full mesh collision planning.

    Only visible depth is used. Blind translation is rejected. Rotation checks
    observed arm obstacles as well as a circular body footprint; unseen geometry
    remains a limitation, explicitly present in every receipt.
    """
    def __init__(self, points, model, q, depth_images, body_radius=.34):
        self.model, self.q = model, q
        self.radius = body_radius
        self.depth_images = depth_images
        self.points = np.asarray(points).reshape(-1,3)
        self.segments=[]
        for chain in model.spec.get("metadata",{}).get("arm_chains",{}).values():
            xyz=[model.forward(q,n)[:3,3] for n in chain]
            self.segments.extend(zip(xyz[:-1],xyz[1:]))
        # Remove points falling on known robot arms/grippers, not target masks.
        from .self_filter import chassis_depth_mask
        own_chassis,self.self_filter_receipt=chassis_depth_mask(model,q,self.points)
        keep=~own_chassis
        for a,b in self.segments:
            keep &= self.segment_distances(self.points,a,b) > .07
        for p in model.grasp_centers(q).values():
            keep &= np.linalg.norm(self.points-p,axis=1) > .09
        self.environment_points=self.points[keep]
        self.obstacles=self.points[keep & (self.points[:,2]>.10) & (self.points[:,2]<1.8)]

    @staticmethod
    def segment_distances(points,a,b):
        v=b-a
        u=np.clip((points-a)@v/max(float(v@v),1e-12),0,1)
        return np.linalg.norm(points-(a+u[:,None]*v),axis=1)

    def check(self, action, carry=False):
        if action.part != "base" or action.move == "hold":
            return True,"NOT_A_BASE_MOVE"
        if len(self.environment_points)<40:
            return False,"BASE_DEPTH_UNKNOWN"
        amount=action.amount(carry)
        translation=np.zeros(3); yaw=0.
        if action.move in TRANSLATIONS:
            translation=np.asarray(TRANSLATIONS[action.move])*amount
            # A valid depth ray must actually observe the intended heading;
            # absence of obstacle points is not a free-space certificate.
            heading=math.atan2(translation[1],translation[0])
            ray_heading=np.arctan2(self.environment_points[:,1],self.environment_points[:,0])
            diff=np.arctan2(np.sin(ray_heading-heading),np.cos(ray_heading-heading))
            seen=(np.abs(diff)<math.radians(25))&(np.linalg.norm(self.environment_points[:,:2],axis=1)>.45)
            if np.count_nonzero(seen)<12:
                return False,"UNOBSERVED_BASE_TRANSLATION"
        else:
            yaw=amount*(1 if action.move=="yaw_plus" else -1)
        if not len(self.obstacles):
            return True,"VISIBLE_DEPTH_ONLY_NO_OBSTACLE"
        start_body=np.linalg.norm(self.obstacles[:,:2],axis=1)-self.radius
        start_arms=[self.segment_distances(self.obstacles,a,b)-.065 for a,b in self.segments]
        for u in np.linspace(0.,1.,6)[1:]:
            angle=-yaw*u; c,s=math.cos(angle),math.sin(angle)
            R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            points=(self.obstacles-u*translation)@R.T
            body=np.linalg.norm(points[:,:2],axis=1)-self.radius
            body_band=points[:,2]<.75
            if np.any(body_band & (body<.015) & (body<start_body-.002)):
                return False,"OBSERVED_BASE_OBSTACLE"
            for (a,b),start in zip(self.segments,start_arms):
                distance=self.segment_distances(points,a,b)-.065
                if np.any((distance<0)&(distance<start-.002)):
                    return False,"OBSERVED_ARM_SWEEP_OBSTACLE"
        return True,"VISIBLE_DEPTH_SWEEP_ONLY"

    def receipt(self):
        return {"visible_depth_points":len(self.points),"nonrobot_obstacle_points":len(self.obstacles),
                "chassis_self_depth":self.self_filter_receipt,"environment_depth_points":len(self.environment_points),
                "body_radius_m":self.radius,"unseen_space_not_certified":True,
                "scene_truth_used":False}
