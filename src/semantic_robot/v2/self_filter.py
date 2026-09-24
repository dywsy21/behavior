"""Remove depth returns on the robot's own exported chassis visual surface.

Only robot asset vertices + robot FK. Not a scene mask, object query, bounding
box deletion, or permission to treat unobserved space as free.
"""
import numpy as np
from scipy.spatial import cKDTree


def paired_surface_distance(points, triangles):
    """Exact point-to-triangle distance for equally sized arrays, degeneracy safe."""
    p=np.asarray(points,dtype=float);t=np.asarray(triangles,dtype=float)
    a,b,c=t[:,0],t[:,1],t[:,2]
    ab,ac,ap=b-a,c-a,p-a
    dot=lambda x,y:np.einsum("ij,ij->i",x,y)
    aa,bb,cc=dot(ab,ab),dot(ab,ac),dot(ac,ac)
    d,e=dot(ap,ab),dot(ap,ac)
    denom=aa*cc-bb*bb
    safe=np.where(denom>1e-20,denom,1.)
    u=(cc*d-bb*e)/safe;v=(aa*e-bb*d)/safe
    inside=(denom>1e-20)&(u>=0)&(v>=0)&(u+v<=1)
    normal=np.cross(ab,ac)
    plane=np.abs(dot(ap,normal))/np.sqrt(np.maximum(dot(normal,normal),1e-20))
    result=np.where(inside,plane,np.inf)
    for x,y in ((a,b),(b,c),(c,a)):
        edge=y-x
        weight=np.clip(dot(p-x,edge)/np.maximum(dot(edge,edge),1e-20),0,1)
        result=np.minimum(result,np.linalg.norm(p-x-weight[:,None]*edge,axis=1))
    return result


class ChassisSurface:
    tolerance_m=.006

    def __init__(self, spec):
        vertices=np.asarray(spec["vertices"],dtype=float)
        raw_faces=np.asarray(spec["faces"])
        if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
                or raw_faces.ndim!=2 or raw_faces.shape[1]!=3 or not np.issubdtype(raw_faces.dtype,np.integer)
                or not len(raw_faces) or raw_faces.min()<0 or raw_faces.max()>=len(vertices)
                or len(raw_faces)>250000):
            raise ValueError("Bounded robot-only triangle mesh required")
        if spec.get("source")!="robot_base_visual_mesh_only" or spec.get("scene_truth") is not False:
            raise ValueError("Self geometry must be robot asset metadata")
        self.link=spec["link"]
        self.triangles=vertices[raw_faces]
        self.centers=self.triangles.mean(axis=1)
        self.radii=np.linalg.norm(self.triangles-self.centers[:,None],axis=2).max(axis=1)
        # A few large triangles must not give every tiny skin triangle a huge
        # query radius. Radius bins are an exact broad phase, not decimation:
        # every true near-surface triangle remains inside its bin's bound.
        exponents=np.ceil(np.log2(np.maximum(self.radii,.001))).astype(int)
        self.bins=[]
        for exponent in np.unique(exponents):
            ids=np.flatnonzero(exponents==exponent)
            self.bins.append((ids,cKDTree(self.centers[ids]),float(self.radii[ids].max())))
        self.lower,self.upper=vertices.min(axis=0),vertices.max(axis=0)

    def mask(self, points, T_link):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        T_link=np.asarray(T_link,dtype=float)
        if (not np.isfinite(points).all() or T_link.shape!=(4,4) or not np.isfinite(T_link).all()
                or not np.allclose(T_link[3],[0,0,0,1],rtol=0,atol=1e-8)
                or not np.allclose(T_link[:3,:3].T@T_link[:3,:3],np.eye(3),rtol=0,atol=1e-6)
                or not np.isclose(np.linalg.det(T_link[:3,:3]),1.,rtol=0,atol=1e-6)):
            raise ValueError("Finite points and current rigid robot FK required")
        local=(points-T_link[:3,3])@T_link[:3,:3]
        eligible=np.flatnonzero(np.all((local>=self.lower-self.tolerance_m)&(local<=self.upper+self.tolerance_m),axis=1))
        mask=np.zeros(len(points),dtype=bool)
        # Bound temporary memory even for a detailed chassis mesh.
        for start in range(0,len(eligible),64):
            ids=eligible[start:start+64]
            for triangle_ids,tree,radius in self.bins:
                remaining=ids[~mask[ids]]
                if not len(remaining):break
                nearby=tree.query_ball_point(local[remaining],radius+self.tolerance_m)
                point_rows=[];face_rows=[]
                for point_id,candidates in zip(remaining,nearby):
                    if not candidates:continue
                    candidates=triangle_ids[np.asarray(candidates,dtype=int)]
                    distance=np.linalg.norm(self.centers[candidates]-local[point_id],axis=1)
                    candidates=candidates[distance<=self.radii[candidates]+self.tolerance_m]
                    point_rows.extend([point_id]*len(candidates));face_rows.extend(candidates)
                if not point_rows:continue
                point_rows=np.asarray(point_rows,dtype=int);face_rows=np.asarray(face_rows,dtype=int)
                # Large exact broad-phase matches are chunked, never dropped.
                for block in range(0,len(point_rows),100000):
                    pids=point_rows[block:block+100000];fids=face_rows[block:block+100000]
                    close=paired_surface_distance(local[pids],self.triangles[fids])<=self.tolerance_m
                    mask[pids[close]]=True
        return mask


def chassis_depth_mask(model, q, points):
    spec=model.spec.get("metadata",{}).get("base_visual_surface")
    if spec is None:
        return np.zeros(len(points),dtype=bool),{"available":False,"masked_points":0}
    if not hasattr(model,"_chassis_surface"):
        model._chassis_surface=ChassisSurface(spec)
    surface=model._chassis_surface
    mask=surface.mask(points,model.forward(q,surface.link))
    return mask,{"available":True,"source":"robot_base_visual_mesh_only","link":surface.link,
                 "triangles":len(surface.triangles),"tolerance_m":surface.tolerance_m,
                 "masked_points":int(mask.sum()),"scene_truth":False}
