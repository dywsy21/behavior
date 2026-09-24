"""Diagnostic-only RGB-D samples inside an unchanged detector box.

Robot chassis overlap is neither target identity nor complete robot occlusion.
This module never accepts/rejects a target, chooses a grasp point, or acts.
"""
import numpy as np

from .finite_localization import bind_frame
from .grounding import MIN_CONTACT_DEPTH_M, unproject, validate_depth
from .self_filter import chassis_depth_mask


GRID = 12


def box_samples(box, width, height):
    if (not isinstance(box,(list,tuple)) or len(box)!=4 or
            any(type(x) not in (int,float) or not np.isfinite(x) for x in box) or
            box[2]<=box[0] or box[3]<=box[1]):
        raise ValueError('Finite nonempty detector xyxy box required')
    if any(type(n) is not int or not 24<=n<=1024 for n in (width,height)):
        raise ValueError('Bounded native image shape required')
    x0,y0=max(0.,box[0]),max(0.,box[1])
    x1,y1=min(float(width),box[2]),min(float(height),box[3])
    if x0>=x1 or y0>=y1:return np.empty((0,2),dtype=int)
    fractions=(np.arange(GRID)+.5)/GRID
    xx,yy=np.meshgrid(x0+fractions*(x1-x0),y0+fractions*(y1-y0))
    # Clip only the sampling domain to available pixels, never the stored box.
    pixels=np.floor(np.c_[xx.ravel(),yy.ravel()]).astype(int)
    return np.unique(np.clip(pixels,[0,0],[width-1,height-1]),axis=0)


def diagnose_box(*, frame_id, goal, raw, depths, model, state, view, box):
    binding=bind_frame(frame_id,goal,raw,depths,model,state)
    if view not in raw:raise ValueError('Calibrated current view required')
    camera=model.spec['metadata']['cameras'][view]
    pixels=box_samples(box,camera['width'],camera['height'])
    depth=validate_depth(depths[view],camera)
    z=depth[pixels[:,1],pixels[:,0]]
    valid=np.isfinite(z)&(z>MIN_CONTACT_DEPTH_M)&(z<3.)
    selected=pixels[valid]
    K=np.asarray(camera['K'],dtype=float)
    T=np.asarray(model.forward(state.q,'camera_'+view))
    if (K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or
            not np.allclose(K[2],[0,0,1],rtol=0,atol=1e-8) or
            T.shape!=(4,4) or not np.isfinite(T).all() or
            not np.allclose(T[3],[0,0,0,1],rtol=0,atol=1e-8) or
            not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),rtol=0,atol=1e-6) or
            not np.isclose(np.linalg.det(T[:3,:3]),1.,rtol=0,atol=1e-6)):
        raise ValueError('Finite calibrated pinhole and rigid camera FK required')
    points=unproject(selected,z[valid],K,T) if len(selected) else np.empty((0,3))
    own,receipt=chassis_depth_mask(model,state.q,points)
    known=receipt.get('available') is True
    return {'frame_binding':binding,'view':view,'box_xyxy_px':list(box),
        'diagnostic_only':True,'target_identity':None,'target_accepted':None,
        'not_complete_robot_mask':True,'grid':[GRID,GRID],
        'sampled_pixels':len(pixels),'depth_valid_samples':int(valid.sum()),
        'invalid_depth_samples':int((~valid).sum()),'depth_domain_m':[MIN_CONTACT_DEPTH_M,3.],
        'chassis_geometry':receipt,
        'chassis_samples':int(own.sum()) if known else None,
        'chassis_fraction_of_depth_samples':float(own.mean()) if known and len(own) else None,
        'samples':[{'pixel_xy':pixel.tolist(),'point_base_m':point.tolist(),
                    'on_chassis_surface':bool(is_own) if known else None}
                   for pixel,point,is_own in zip(selected,points,own)]}
