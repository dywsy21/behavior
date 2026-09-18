"""Observable RGB-D point tracks relative to the hand; no simulator object API.

Single-pair scores are diagnostic. Only the bounded multi-lift verifier can
authorize an observational grasp, with current robot self-geometry mandatory.
"""
import hashlib
import numpy as np
from .grounding import unproject,validate_depth,MIN_CONTACT_DEPTH_M


def robot_point_mask(points,geometry):
    if (not geometry or geometry.get("source")!="robot_visual_link_boxes_actual_joint_fk"
            or geometry.get("scene_truth") is not False or not geometry.get("includes_actual_finger_positions")
            or not geometry.get("boxes")):
        return None
    points=np.asarray(points,dtype=float).reshape(-1,3);mask=np.zeros(len(points),dtype=bool)
    margin=float(geometry["margin_m"])
    if not .002<=margin<=.01:raise ValueError("Robot exclusion margin outside conservative bound")
    for box in geometry["boxes"]:
        t=np.asarray(box["T_base_link"],dtype=float);lo=np.asarray(box["lower"]);hi=np.asarray(box["upper"])
        if t.shape!=(4,4) or lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite(t).all() or not np.isfinite([lo,hi]).all() or np.any(lo>hi):raise ValueError("Invalid robot-only box")
        if (not np.allclose(t[3], [0,0,0,1], atol=1e-8, rtol=0) or
                not np.allclose(t[:3,:3].T@t[:3,:3], np.eye(3), atol=1e-6, rtol=0) or
                not np.isclose(np.linalg.det(t[:3,:3]), 1., atol=1e-6, rtol=0)):
            raise ValueError("Rigid SE(3) robot-only box transform required")
        local=(points-t[:3,3])@t[:3,:3]
        mask|=np.all((local>=lo-margin)&(local<=hi+margin),axis=1)
    return mask


def continued_features(previous,detected,mask,limit=160,min_distance=7):
    """Keep measured feature identity; recheck the current target/self mask.

    Neither stale coordinates nor prior consistency bypass the new optical,
    depth, robot-exclusion, spread or rigid-following checks below.
    """
    old=np.asarray(previous,dtype=np.float32)
    if old.ndim!=2 or old.shape[1]!=2 or len(old)>limit or not np.isfinite(old).all():
        raise ValueError("Bounded finite Nx2 previous feature coordinates required")
    new=np.empty((0,2),np.float32) if detected is None else np.asarray(detected,dtype=np.float32).reshape(-1,2)
    kept=[];inherited=0;h,w=mask.shape
    for origin,rows in (("continued",old),("detected",new)):
        for p in rows:
            x,y=np.rint(p).astype(int)
            if not 0<=x<w or not 0<=y<h or not mask[y,x]:continue
            if kept and np.min(np.linalg.norm(np.asarray(kept)-p,axis=1))<min_distance:continue
            kept.append(p)
            if origin=="continued":inherited+=1
            if len(kept)==limit:break
        if len(kept)==limit:break
    return (np.asarray(kept,dtype=np.float32).reshape(-1,1,2) if kept else None),inherited


def spatial_features(gray,mask):
    """Fixed 4x4 spatial quota; local contrast cannot suppress distant corners.

    Detection is not evidence: every candidate still passes the original optical
    flow, depth, self-exclusion and rigid-motion checks. The 160 point/7px budget
    is unchanged. Whole-image derivatives avoid artificial crop-edge corners.
    """
    import cv2
    h,w=mask.shape;groups=[]
    for iy in range(4):
        for ix in range(4):
            tile=np.zeros_like(mask)
            ys=slice(iy*h//4,(iy+1)*h//4);xs=slice(ix*w//4,(ix+1)*w//4)
            tile[ys,xs]=mask[ys,xs]
            found=cv2.goodFeaturesToTrack(gray,maxCorners=10,qualityLevel=.015,minDistance=7,mask=tile,blockSize=5)
            groups.append([] if found is None else found.reshape(-1,2).tolist())
    # Round-robin preserves coverage when adjacent tiles share nearby corners.
    candidates=[g[i] for i in range(10) for g in groups if i<len(g)]
    return continued_features(np.empty((0,2),np.float32),candidates or None,mask)[0]


def point_tracks(before_rgb,after_rgb,before_depth,after_depth,camera,before_camera,after_camera,seed,before_self=None,after_self=None,previous_features=None,stable_seed_depth=False,spatial_seed_features=False):
    import cv2
    old,new=np.asarray(before_rgb),np.asarray(after_rgb)
    shape=(camera["height"],camera["width"],3)
    if old.shape!=shape or new.shape!=shape or old.dtype!=np.uint8 or new.dtype!=np.uint8:
        raise ValueError("Calibrated raw RGB required")
    result={"source":"onboard_RGBD_forward_backward_feature_tracks","scene_truth":False,
            "valid":False,"reason":"INSUFFICIENT_TARGET_FEATURES",
            "previous_rgb_sha256":hashlib.sha256(old.tobytes()).hexdigest(),
            "current_rgb_sha256":hashlib.sha256(new.tobytes()).hexdigest()}
    if np.array_equal(old,new):
        return {**result,"reason":"IDENTICAL_FULL_IMAGE_NO_FRESH_MOTION_EVIDENCE"},None,None
    d0,d1=validate_depth(before_depth,camera),validate_depth(after_depth,camera)
    h,w=d0.shape;K=np.asarray(camera["K"]);yy,xx=np.mgrid[:h,:w]
    valid=np.isfinite(d0)&(d0>MIN_CONTACT_DEPTH_M)&(d0<3.)
    uv=np.c_[xx[valid],yy[valid]];xyz=unproject(uv,d0[valid],K,before_camera)
    # A seed is a VLM-localized target surface, not an object segmentation mask.
    # Keep a small 3D neighborhood; later measurements remain corroboration.
    near=np.linalg.norm(xyz-np.asarray(seed),axis=1)<.035
    own=robot_point_mask(xyz,before_self)
    result.update(depth_range_m=[MIN_CONTACT_DEPTH_M,3.],depth_valid_pixels=int(valid.sum()),
                  local_seed_pixels_before_self_exclusion=int(near.sum()),
                  robot_seed_pixels_excluded=int((near&own).sum()) if own is not None else None)
    if own is not None:near&=~own
    mask=np.zeros((h,w),np.uint8);mask[yy[valid][near],xx[valid][near]]=255
    if stable_seed_depth:
        # The very same 3x3 finite/stable-depth requirement is checked again
        # after optical flow. Apply it before corner ranking too, so rejected
        # depth-edge corners cannot set the detector's relative-quality scale.
        finite=np.isfinite(d0);z=np.where(finite,d0,0).astype(np.float32);kernel=np.ones((3,3),np.uint8)
        stable=(cv2.erode(finite.astype(np.uint8),kernel)>0)&(cv2.dilate(z,kernel)-cv2.erode(z,kernel)<=.01)
        stable[:2]=False;stable[-2:]=False;stable[:,:2]=False;stable[:,-2:]=False
        mask[~stable]=0
        result["eligible_stable_depth_seed_pixels"]=int(np.count_nonzero(mask))
    gray0=cv2.cvtColor(old,cv2.COLOR_RGB2GRAY);gray1=cv2.cvtColor(new,cv2.COLOR_RGB2GRAY)
    points=(spatial_features(gray0,mask) if spatial_seed_features else
            cv2.goodFeaturesToTrack(gray0,maxCorners=160,qualityLevel=.015,minDistance=7,mask=mask,blockSize=5))
    if spatial_seed_features:result["feature_selection"]="fixed_4x4_local_contrast_10_per_tile_max160_min_distance7"
    if previous_features is not None:
        detected=0 if points is None else len(points)
        points,inherited=continued_features(previous_features,points,mask)
        result.update(previous_features_supplied=len(previous_features),continued_features_in_current_mask=inherited,
                      newly_detected_features=detected,feature_budget=160)
    result.update(eligible_feature_pixels=int(near.sum()),seed_feature_count=0 if points is None else len(points))
    if points is None or len(points)<8:return result,None,None
    params=dict(winSize=(17,17),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    after,ok,err=cv2.calcOpticalFlowPyrLK(gray0,gray1,points,None,**params)
    if after is None or ok is None or err is None:return result,None,None
    back,ok_back,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,after,None,**params)
    if back is None or ok_back is None:return result,None,None
    good=ok.ravel().astype(bool)&ok_back.ravel().astype(bool)&(np.linalg.norm(back-points,axis=2).ravel()<.5)&(err.ravel()<12.)
    a,b=points.reshape(-1,2)[good],after.reshape(-1,2)[good]
    kept=[];xyz0=[];xyz1=[]
    for u,v in zip(a,b):
        zs=[]
        for pixel,depth in ((u,d0),(v,d1)):
            x,y=np.rint(pixel).astype(int)
            if not 2<=x<w-2 or not 2<=y<h-2:break
            patch=depth[y-1:y+2,x-1:x+2]
            if not np.isfinite(patch).all() or np.ptp(patch)>.01:break
            z=float(np.median(patch))
            if not MIN_CONTACT_DEPTH_M<z<3.:break
            zs.append(z)
        if len(zs)!=2:continue
        xyz0.append(unproject([u],[zs[0]],K,before_camera)[0]);xyz1.append(unproject([v],[zs[1]],K,after_camera)[0]);kept.append((u,v))
    self0,self1=robot_point_mask(xyz0,before_self),robot_point_mask(xyz1,after_self)
    if self0 is not None and self1 is not None:
        external=~(self0|self1)
        result["robot_tracks_removed"]=int((~external).sum())
        kept=[x for x,k in zip(kept,external) if k]
        xyz0=np.asarray(xyz0)[external];xyz1=np.asarray(xyz1)[external]
    result.update(seed_feature_count=len(points),rgbd_tracks=len(kept),robot_self_exclusion_checked=self0 is not None and self1 is not None,
                  tracked_pixels=[{"before":u.tolist(),"after":v.tolist()} for u,v in kept])
    if len(kept)<8:return result,None,None
    spread=np.ptp(np.asarray([v for u,v in kept]),axis=0)
    if min(spread)<12.:return {**result,"reason":"TRACKS_TOO_SPATIALLY_CONCENTRATED"},None,None
    result.update(valid=True,reason="MATCHED_LOCAL_SURFACE_FEATURES",pixel_span=spread.tolist())
    return result,np.asarray(xyz0),np.asarray(xyz1)


def measure_grasp_motion(before,after,model,arm,seed,body_transform,previous_features=None,stable_seed_depth=False,spatial_seed_features=False):
    """Compare matched points in robot and hand coordinates; no VLM co-motion.

    The transformation maps the current body to the previous body, obtained
    independently from head RGB-D and robot camera FK, not command integration.
    """
    view=arm+"_wrist";camera=model.spec["metadata"]["cameras"][view]
    t0,t1=(model.forward(row["q"],"camera_"+view) for row in (before,after))
    receipt,p0,p1=point_tracks(before["rgb"][view],after["rgb"][view],before["depth"][view],after["depth"][view],camera,t0,t1,seed,before.get("self_geometry"),after.get("self_geometry"),previous_features,stable_seed_depth,spatial_seed_features)
    if not receipt["valid"]:return receipt
    e0,e1=(model.forward(row["q"],arm) for row in (before,after))
    body=np.asarray(body_transform,dtype=float)
    if body.shape!=(4,4) or not np.isfinite(body).all():raise ValueError("Finite measured rigid body transform required")
    actual=body@e1;hand_delta=actual[:3,3]-e0[:3,3];travel=float(np.linalg.norm(hand_delta))
    local0=(p0-e0[:3,3])@e0[:3,:3];local1=(p1-e1[:3,3])@e1[:3,:3]
    residual=np.linalg.norm(local1-local0,axis=1)
    world1=p1@body[:3,:3].T+body[:3,3];target_delta=world1-p0
    stationary=np.linalg.norm(target_delta,axis=1)
    coherence=np.linalg.norm(target_delta-hand_delta,axis=1)
    passing=(residual<.003)&(coherence<.004)&(stationary>.5*travel)&((target_delta@hand_delta)>.5*travel*travel)
    fraction=float(passing.mean())
    self_checked=receipt.get("robot_self_exclusion_checked",False)
    # Individual small lifts are corroboration only. Authorization below needs
    # repeated registered lifts and >=15mm cumulative measured displacement.
    consistent=travel>=.003 and hand_delta[2]>=.002 and fraction>=.8
    valid=travel>=.006 and hand_delta[2]>=.005 and fraction>=.8
    receipt.update(valid=bool(valid),reason="LOCAL_TARGET_FOLLOWS_MEASURED_HAND_LIFT" if valid else "INSUFFICIENT_OR_NONRIGID_HAND_FOLLOWING",
        hand_delta_m=hand_delta.tolist(),hand_travel_m=travel,
        median_hand_relative_residual_m=float(np.median(residual)),p90_hand_relative_residual_m=float(np.quantile(residual,.9)),
        median_target_displacement_m=float(np.median(stationary)),passing_fraction=fraction,
        target_seed_is_not_segmentation=True,robot_self_exclusion_checked=self_checked,
        registered_pair_consistent=bool(consistent and self_checked),
        diagnostic_only_not_grasp_authorization=True)
    return receipt


class GraspMotionVerifier:
    """Bounded rigid following, not a simulator holding predicate or task truth."""
    def __init__(self,persistent_tracks=False,spatial_seed_features=False):
        self.previous=None;self.lifts={};self.displacement={};self.last_execution=None
        self.persistent_tracks=bool(persistent_tracks);self.tracks={}
        self.spatial_seed_features=bool(spatial_seed_features)
        if self.spatial_seed_features and not self.persistent_tracks:
            raise ValueError("Spatial features require the identity-bound persistent tracking mode")

    def observe(self,model,state,images,depths,self_geometry,harness,targets,motion,evidence):
        frame={"q":state.q.copy(),"gripper":state.gripper.copy(),"rgb":images,"depth":depths,
               "self_geometry":self_geometry,"goal_index":harness.index,"targets":targets,
               "execution":harness.executions}
        old,self.previous=self.previous,frame
        result={"source":"registered_onboard_RGBD_grasp_motion","verified":False,"per_hand":{},
                "scene_truth":False,"not_official_task_success":True}
        action=harness.last_action
        eligible=(old is not None and old["goal_index"]==harness.index and
            harness.goal.kind=="pick" and harness.stage=="VERIFY_GRASP" and
            action is not None and action.part==harness.goal.hand and action.move=="up" and
            evidence.visible and evidence.enclosed is not False and evidence.co_moving is not False and
            evidence.hazard not in ("slip","collision") and
            harness.feedback and harness.feedback["status"]=="TARGET_REACHED" and
            motion.get("valid") and "body_transform_current_in_previous" in motion)
        if not eligible:
            self.lifts={};self.displacement={};self.tracks={};result["reason"]="NO_ELIGIBLE_VERIFICATION_LIFT";return result
        if harness.executions==self.last_execution:
            result["reason"]="NO_NEW_EXECUTED_LIFT";return result
        self.last_execution=harness.executions
        for arm in harness.arms:
            index=("left","right").index(arm);seed=old["targets"].get(arm,{})
            good=(seed.get("valid") and targets.get(arm,{}).get("valid") and
                  harness.pending_grasp[arm] and state.gripper[index]>=.0015 and old["gripper"][index]>=.0015)
            row={"registered_pair_consistent":False,"reason":"MISSING_TARGET_OR_NONEMPTY_PENDING_HAND"}
            if good:
                arguments=(old,frame,model,arm,seed["point_base_m"],motion["body_transform_current_in_previous"])
                if self.persistent_tracks:
                    history=self.tracks.get(arm);features=None;reason="NO_PREVIOUS_QUALIFIED_TRACKS"
                    if history:
                        digest=hashlib.sha256(np.asarray(old["rgb"][arm+"_wrist"]).tobytes()).hexdigest()
                        linked=(history["execution"]==old["execution"] and history["goal_index"]==old["goal_index"]
                                and history["image_sha"]==digest and old["execution"]+1==harness.executions)
                        if linked:features=history["points"];reason="PREVIOUS_QUALIFIED_PAIR_SAME_IMAGE_GOAL_EXECUTION"
                        else:
                            self.lifts[arm]=0;self.displacement[arm]=np.zeros(3)
                            reason="HISTORY_IDENTITY_MISMATCH_RESET"
                    row=measure_grasp_motion(*arguments,features,True,spatial_seed_features=self.spatial_seed_features)
                    row["feature_history_binding"]={"continued":features is not None,"reason":reason,"stable_depth_before_detection":True}
                else:row=measure_grasp_motion(*arguments)
            if row.get("registered_pair_consistent"):
                self.lifts[arm]=self.lifts.get(arm,0)+1
                self.displacement[arm]=self.displacement.get(arm,np.zeros(3))+np.asarray(row["hand_delta_m"])
            else:
                self.lifts[arm]=0;self.displacement[arm]=np.zeros(3)
            self.tracks.pop(arm,None)
            if self.persistent_tracks and row.get("registered_pair_consistent") and row.get("tracked_pixels"):
                self.tracks[arm]={"execution":harness.executions,"goal_index":harness.index,
                    "image_sha":row["current_rgb_sha256"],"points":[p["after"] for p in row["tracked_pixels"]]}
            row["consecutive_registered_lifts"]=self.lifts[arm]
            row["cumulative_hand_delta_m"]=self.displacement[arm].tolist()
            row["verified"]=bool(self.lifts[arm]>=2 and np.linalg.norm(self.displacement[arm])>=.015 and self.displacement[arm][2]>=.012)
            result["per_hand"][arm]=row
        result["verified"]=all(row["verified"] for row in result["per_hand"].values())
        result["reason"]="REPEATED_REGISTERED_TARGET_FOLLOWING" if result["verified"] else "REGISTRATION_INSUFFICIENT"
        return result
