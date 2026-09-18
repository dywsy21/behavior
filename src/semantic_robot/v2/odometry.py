"""Quality-gated egocentric RGB-D odometry; no simulator/world-pose API.

An instantaneous base joint velocity was not a reliable displacement sensor in
H-08. Do not integrate it into fictitious search coverage or fit a magic scale.
"""
import hashlib

import numpy as np

from .grounding import validate_depth


def body_motion(camera_before, camera_after, optical_old_to_new):
    S=np.diag([1.,-1.,-1.,1.])
    transform=np.linalg.inv(camera_after@S@optical_old_to_new@S@np.linalg.inv(camera_before))
    return transform,np.array([transform[0,3],transform[1,3],np.arctan2(transform[1,0],transform[0,0])])


def solve_correspondences(points_before, pixels_after, points_after, K, camera_before, camera_after):
    """Robust PnP with independently observed depth agreement and finite bounds."""
    import cv2
    xyz,uv,newxyz=map(lambda x:np.asarray(x,dtype=np.float64),(points_before,pixels_after,points_after))
    result={"valid":False,"reason":"INSUFFICIENT_STABLE_CORRESPONDENCES","matches":len(xyz)}
    if len(xyz)<25:return result
    if (xyz.shape!=(len(xyz),3) or newxyz.shape!=xyz.shape or uv.shape!=(len(xyz),2)
            or not all(np.isfinite(a).all() for a in (xyz,uv,newxyz))):
        raise ValueError("Finite aligned RGB-D correspondences required")
    cv2.setRNGSeed(31)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,np.asarray(K),None,iterationsCount=200,
        reprojectionError=1.8,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<25:
        result["reason"]="PNP_NOT_SUPPORTED";return result
    sel=inliers.ravel();result.update(inliers=len(sel),inlier_fraction=len(sel)/len(xyz))
    if len(sel)/len(xyz)<.45:
        result["reason"]="MATCHES_NOT_GEOMETRICALLY_COHERENT";return result
    rvec,tvec=cv2.solvePnPRefineLM(xyz[sel],uv[sel],np.asarray(K),None,rvec,tvec)
    rel=np.eye(4);rel[:3,:3]=cv2.Rodrigues(rvec)[0];rel[:3,3]=tvec.ravel()
    predicted,_=cv2.projectPoints(xyz[sel],rvec,tvec,np.asarray(K),None)
    pixel_error=float(np.median(np.linalg.norm(predicted.reshape(-1,2)-uv[sel],axis=1)))
    point_error=float(np.median(np.linalg.norm(xyz[sel]@rel[:3,:3].T+rel[:3,3]-newxyz[sel],axis=1)))
    motion,delta=body_motion(camera_before,camera_after,rel)
    result.update(median_reprojection_px=pixel_error,median_depth_correspondence_m=point_error,
                  body_delta=delta.tolist(),body_translation_z_m=float(motion[2,3]),
                  body_transform_current_in_previous=motion.tolist())
    if pixel_error>1. or point_error>.015:
        result["reason"]="RGB_DEPTH_MOTION_DISAGREEMENT";return result
    if np.linalg.norm(delta[:2])>.18 or abs(delta[2])>.30 or abs(motion[2,3])>.035:
        result["reason"]="OUTSIDE_SINGLE_MICRO_ACTION_MOTION_BOUND";return result
    result.update(valid=True,reason="CURRENT_RGBD_GEOMETRIC_MOTION")
    return result


def rigid_fit(before,after):
    """Fixed-scale least-squares rigid registration; never a learned pose prior."""
    a,b=np.asarray(before),np.asarray(after)
    if len(a)<3 or a.shape!=b.shape or a.shape[1:]!=(3,):return None
    ca,cb=a.mean(axis=0),b.mean(axis=0)
    u,s,vt=np.linalg.svd((a-ca).T@(b-cb))
    if s[1]<1e-8:return None  # Collinear points do not constrain full rotation.
    correction=np.eye(3);correction[2,2]=np.linalg.det(vt.T@u.T)
    r=vt.T@correction@u.T
    t=cb-r@ca
    return r,t


def solve_rgbd_correspondences(points_before,pixels_after,points_after,K,camera_before,camera_after):
    """Experimental 3D--3D RANSAC with independent RGB reprojection checks.

    Same motion/reprojection/depth limits as PnP. This is not a fallback that
    keeps trying solvers until one says a stopped action was safe.
    """
    xyz,uv,newxyz=map(lambda x:np.asarray(x,dtype=np.float64),(points_before,pixels_after,points_after))
    result={"valid":False,"reason":"INSUFFICIENT_STABLE_CORRESPONDENCES","matches":len(xyz),
            "estimator":"RGBD_RIGID_RANSAC_EXPERIMENTAL"}
    if len(xyz)<25:return result
    if (xyz.shape!=(len(xyz),3) or newxyz.shape!=xyz.shape or uv.shape!=(len(xyz),2)
            or not all(np.isfinite(a).all() for a in (xyz,uv,newxyz))):
        raise ValueError("Finite aligned RGB-D correspondences required")
    rng=np.random.default_rng(31);best=None;best_score=(-1,-float("inf"))
    for _ in range(200):
        choice=rng.choice(len(xyz),3,replace=False);fit=rigid_fit(xyz[choice],newxyz[choice])
        if fit is None:continue
        r,t=fit;error=np.linalg.norm(xyz@r.T+t-newxyz,axis=1);sel=error<.01
        score=(int(sel.sum()),-float(np.median(error[sel])) if sel.any() else -float("inf"))
        if score>best_score:best,best_score=sel,score
    if best is None or best.sum()<25:
        result["reason"]="RGBD_RIGID_NOT_SUPPORTED";return result
    for _ in range(2):
        fit=rigid_fit(xyz[best],newxyz[best])
        if fit is None:return {**result,"reason":"DEGENERATE_3D_SUPPORT"}
        r,t=fit;error=np.linalg.norm(xyz@r.T+t-newxyz,axis=1);best=error<.01
        if best.sum()<25:return {**result,"reason":"RGBD_RIGID_NOT_SUPPORTED"}
    sel=np.flatnonzero(best);result.update(inliers=len(sel),inlier_fraction=len(sel)/len(xyz))
    if len(sel)/len(xyz)<.45:return {**result,"reason":"MATCHES_NOT_GEOMETRICALLY_COHERENT"}
    fitted=xyz[sel]@r.T+t
    if np.any(fitted[:,2]<=.01):return {**result,"reason":"INVALID_PROJECTED_DEPTH"}
    projected=fitted@np.asarray(K).T;projected=projected[:,:2]/projected[:,2,None]
    pixel_error=float(np.median(np.linalg.norm(projected-uv[sel],axis=1)))
    point_error=float(np.median(np.linalg.norm(fitted-newxyz[sel],axis=1)))
    rel=np.eye(4);rel[:3,:3]=r;rel[:3,3]=t
    motion,delta=body_motion(camera_before,camera_after,rel)
    result.update(median_reprojection_px=pixel_error,median_depth_correspondence_m=point_error,
        body_delta=delta.tolist(),body_translation_z_m=float(motion[2,3]),
        body_transform_current_in_previous=motion.tolist(),
        support_3d_singular_values=np.linalg.svd(xyz[sel]-xyz[sel].mean(axis=0),compute_uv=False).tolist(),
        inlier_pixel_span=np.ptp(uv[sel],axis=0).tolist())
    if pixel_error>1. or point_error>.015:return {**result,"reason":"RGB_DEPTH_MOTION_DISAGREEMENT"}
    if np.linalg.norm(delta[:2])>.18 or abs(delta[2])>.30 or abs(motion[2,3])>.035:
        return {**result,"reason":"OUTSIDE_SINGLE_MICRO_ACTION_MOTION_BOUND"}
    return {**result,"valid":True,"reason":"CURRENT_RGBD_GEOMETRIC_MOTION"}


class RGBDMotion:
    def __init__(self,estimator="pnp"):
        import cv2
        if estimator not in ("pnp","rgbd_rigid","rgbd_joint"):raise ValueError("Explicit RGB-D estimator required")
        if estimator=="rgbd_joint":
            from .joint_odometry import solve_joint_correspondences
            self.solve=solve_joint_correspondences
        else:
            self.solve=solve_correspondences if estimator=="pnp" else solve_rgbd_correspondences
        self.cv2=cv2
        cv2.setNumThreads(2)
        self.sift=cv2.SIFT_create(nfeatures=2000);self.matcher=cv2.BFMatcher()
        self.previous=None

    def observe(self,images,depths,model,q):
        cv2=self.cv2
        rgb=np.asarray(images["head_rgb"])
        if rgb.shape[0]==3:rgb=rgb.transpose(1,2,0)
        camera=model.spec["metadata"]["cameras"]["head"]
        if rgb.shape!=(camera["height"],camera["width"],3) or rgb.dtype!=np.uint8:
            raise ValueError("Current calibrated raw head RGB required for motion")
        depth=validate_depth(depths["head"],camera)
        keypoints,descriptors=self.sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
        T=model.forward(q,"camera_head");K=np.asarray(camera["K"])
        current={"keypoints":keypoints,"descriptors":descriptors,"depth":depth.copy(),"camera":T,
                 "rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),"depth_sha256":hashlib.sha256(depth.tobytes()).hexdigest()}
        previous,self.previous=self.previous,current
        receipt={"source":"onboard_head_RGBD_with_robot_camera_FK","no_scene_truth":True,
                 "velocity_integral_not_used":True,"current_rgb_sha256":current["rgb_sha256"],
                 "current_depth_sha256":current["depth_sha256"]}
        if previous is None:return {**receipt,"valid":True,"initial":True,"body_delta":[0.,0.,0.],"reason":"REFERENCE_ONLY_NO_COVERAGE_CLAIM"}
        receipt["previous_rgb_sha256"]=previous["rgb_sha256"]
        pairs=(self.matcher.knnMatch(previous["descriptors"],descriptors,k=2)
               if previous["descriptors"] is not None and descriptors is not None else [])
        oldpoints=[];newpoints=[];pixels=[]
        for pair in pairs:
            if len(pair)!=2 or pair[0].distance>=.7*pair[1].distance:continue
            match=pair[0]
            uv0=np.asarray(previous["keypoints"][match.queryIdx].pt)
            uv1=np.asarray(keypoints[match.trainIdx].pt)
            good=[]
            for uv,d in ((uv0,previous["depth"]),(uv1,depth)):
                x,y=np.rint(uv).astype(int);patch=d[max(0,y-1):y+2,max(0,x-1):x+2]
                z=float(np.median(patch))
                if not .05<z<5. or not np.isfinite(patch).all() or np.ptp(patch)>.03:break
                good.append(np.linalg.solve(K,np.r_[uv,1.])*z)
            if len(good)==2:oldpoints.append(good[0]);newpoints.append(good[1]);pixels.append(uv1)
        return {**receipt,**self.solve(oldpoints,pixels,newpoints,K,previous["camera"],T)}
