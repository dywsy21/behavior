import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.v2.self_filter import ChassisSurface, paired_surface_distance


class SelfSurfaceTests(unittest.TestCase):
    def spec(self):
        return {"source":"robot_base_visual_mesh_only","scene_truth":False,"link":"link:base_link",
                "vertices":[[-.2,-.3,.3],[.2,-.3,.3],[.2,.3,.3],[-.2,.3,.3]],"faces":[[0,1,2],[0,2,3]]}

    def test_exact_triangle_interior_edges_and_degenerate(self):
        tri=np.array([[[0,0,0],[1,0,0],[0,1,0]]]*4,dtype=float)
        points=np.array([[.2,.2,.1],[2,0,0],[-1,-1,0],[.7,.7,0]])
        np.testing.assert_allclose(paired_surface_distance(points,tri),[.1,1,np.sqrt(2),np.sqrt(.08)])
        tri[:]=0
        np.testing.assert_allclose(paired_surface_distance(points,tri),np.linalg.norm(points,axis=1))

    def test_only_actual_chassis_surface_not_its_box_or_surrounding_obstacles(self):
        surface=ChassisSurface(self.spec())
        points=np.array([[0,0,.3],[.1,.1,.304],[0,0,.312],[0,0,.1],[.25,0,.3]])
        np.testing.assert_array_equal(surface.mask(points,np.eye(4)),[True,True,False,False,False])

    def test_mesh_uses_robot_fk_and_frame_not_scene_pose(self):
        surface=ChassisSurface(self.spec())
        T=np.eye(4);T[:3,:3]=Rotation.from_euler("xyz",[.1,.2,.4]).as_matrix();T[:3,3]=[.1,.2,.5]
        local=np.array([[0,0,.3],[0,0,.32]])
        points=local@T[:3,:3].T+T[:3,3]
        np.testing.assert_array_equal(surface.mask(points,T),[True,False])

    def test_large_triangle_centroid_search_does_not_miss_close_corner(self):
        surface=ChassisSurface(self.spec())
        np.testing.assert_array_equal(surface.mask(np.array([[.199,.299,.3],[-.199,-.299,.3]]),np.eye(4)),[True,True])

    def test_invalid_or_scene_geometry_is_rejected(self):
        for changed in ({"scene_truth":True},{"source":"scene_mesh"},{"faces":[[0,1,99]]},
                        {"faces":[[0.,1.,2.]]},{"vertices":[[np.nan,0,0]]}):
            with self.assertRaises(ValueError):ChassisSurface({**self.spec(),**changed})

    def test_radius_bins_equal_brute_force_across_tiny_and_large_triangles(self):
        rng=np.random.default_rng(471)
        centers=rng.uniform(-.1,.1,(60,3))
        sizes=np.geomspace(.001,.6,60)
        triangles=centers[:,None]+rng.uniform(-1,1,(60,3,3))*sizes[:,None,None]
        spec=self.spec();spec["vertices"]=triangles.reshape(-1,3).tolist()
        spec["faces"]=np.arange(180).reshape(-1,3).tolist()
        surface=ChassisSurface(spec)
        points=np.r_[triangles.mean(axis=1)+rng.normal(0,.003,(60,3)),rng.uniform(-.3,.3,(50,3))]
        exact=np.array([np.min(paired_surface_distance(np.broadcast_to(p,(60,3)),triangles))<=surface.tolerance_m for p in points])
        np.testing.assert_array_equal(surface.mask(points,np.eye(4)),exact)


if __name__=="__main__":unittest.main()
