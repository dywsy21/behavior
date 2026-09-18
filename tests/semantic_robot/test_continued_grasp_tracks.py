import unittest
from unittest.mock import patch
import hashlib
import numpy as np
from semantic_robot.v2.grasp_motion import continued_features,GraspMotionVerifier,point_tracks,spatial_features
import test_grasp_motion as original
from test_v2 import evidence


class ContinuedFeatureTests(unittest.TestCase):
    def test_local_contrast_keeps_independent_weak_corners(self):
        import cv2
        gray=np.zeros((128,128),np.uint8);gray[8:20,8:20]=255;gray[100:112,100:112]=15
        mask=np.full_like(gray,255)
        global_points=cv2.goodFeaturesToTrack(gray,maxCorners=160,qualityLevel=.015,minDistance=7,mask=mask,blockSize=5)
        self.assertFalse(np.any(global_points.reshape(-1,2)[:,0]>90))
        points=spatial_features(gray,mask).reshape(-1,2)
        self.assertTrue(np.any(points[:,0]>90));self.assertTrue(np.any(points[:,0]<30))
        mask[96:,96:]=0
        self.assertFalse(np.any(spatial_features(gray,mask).reshape(-1,2)[:,0]>90))

    def test_spatial_candidate_cap_spacing_and_empty_mask(self):
        gray=np.random.default_rng(44).integers(0,256,(480,480),dtype=np.uint8)
        mask=np.full_like(gray,255);points=spatial_features(gray,mask).reshape(-1,2)
        self.assertLessEqual(len(points),160)
        distances=np.linalg.norm(points[:,None]-points[None,:],axis=2);np.fill_diagonal(distances,np.inf)
        self.assertGreaterEqual(distances.min(),7.)
        self.assertIsNone(spatial_features(gray,np.zeros_like(mask)))

    def test_ongoing_identity_has_priority_but_rechecks_current_mask(self):
        mask=np.ones((60,60),np.uint8);mask[10,10]=0
        old=[[10,10],[20,20],[30,30]];detected=np.asarray([[[21,21]],[[45,45]]],np.float32)
        points,n=continued_features(old,detected,mask)
        self.assertEqual(n,2)
        np.testing.assert_array_equal(points.reshape(-1,2),[[20,20],[30,30],[45,45]])

    def test_bounds_spacing_and_budget_do_not_inflate_feature_count(self):
        mask=np.ones((100,100),np.uint8)
        points,n=continued_features([[-1,0],[100,0],[10,10],[10.1,10.1]],None,mask)
        self.assertEqual(n,1);self.assertEqual(len(points),1)
        old=[[x,y] for x in range(0,100,8) for y in range(0,100,8)][:160]
        points,n=continued_features(old,None,mask)
        self.assertEqual(len(points),160)
        for bad in ([],[[0,0,0]],[[np.nan,0]],old+[[0,0]]):
            with self.assertRaises(ValueError):continued_features(bad,None,mask)

    def test_stable_depth_prefilter_preserves_valid_near_surface(self):
        camera={"width":100,"height":100,"K":[[100,0,50],[0,100,50],[0,0,1]]}
        old=np.random.default_rng(81).integers(0,256,(100,100,3),dtype=np.uint8);new=np.roll(old,1,axis=1)
        depth=np.full((100,100),.032,np.float32)
        r,_,_=point_tracks(old,new,depth,depth,camera,np.eye(4),np.eye(4),[0,0,-.032],stable_seed_depth=True)
        self.assertTrue(r["valid"]);self.assertGreaterEqual(r["rgbd_tracks"],8)

    def case(self):
        m,s,h,_,targets,motion=original.GraspMotionTests().make()
        return m,s,h,GraspMotionVerifier(persistent_tracks=True),targets,motion

    def observe(self,x,value=1,obs=None):
        m,s,h,v,t,motion=x;h.executions+=1
        rgb={a+"_wrist":np.full((100,100,3),value,np.uint8) for a in h.arms}
        return v.observe(m,s,rgb,{},original.boxes(),h,t,motion,obs or evidence(enclosed=True,co_moving=None))

    @staticmethod
    def measured(old,current,*args,**kwargs):
        return {"registered_pair_consistent":True,"hand_delta_m":[0,0,.008],
                "current_rgb_sha256":hashlib.sha256(current["rgb"]["right_wrist"].tobytes()).hexdigest(),
                "tracked_pixels":[{"before":[8*i,30],"after":[8*i,31]} for i in range(8)]}

    def test_cache_is_bound_to_actual_previous_image_goal_and_execution(self):
        x=self.case()
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",side_effect=self.measured) as measured:
            self.observe(x,0);self.assertFalse(self.observe(x,1)["verified"])
            r=self.observe(x,2);self.assertTrue(r["verified"])
            self.assertEqual(len(measured.call_args.args[-2]),8)
            self.assertIs(measured.call_args.args[-1],True)
            self.assertIs(measured.call_args.kwargs["spatial_seed_features"],False)
            self.assertTrue(r["per_hand"]["right"]["feature_history_binding"]["continued"])

    def test_spatial_registration_mode_is_explicit_and_passed_to_measurement(self):
        with self.assertRaises(ValueError):GraspMotionVerifier(spatial_seed_features=True)
        x=self.case();x[3].spatial_seed_features=True
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",side_effect=self.measured) as measured:
            self.observe(x,0);self.observe(x,1)
            self.assertIs(measured.call_args.kwargs["spatial_seed_features"],True)

    def test_hash_or_execution_gap_resets_chain_and_uses_cold_detection(self):
        for kind in ("hash","execution","goal"):
            x=self.case();h,v=x[2:4]
            with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",side_effect=self.measured):
                self.observe(x,0);self.observe(x,1)
                if kind=="hash":v.tracks["right"]["image_sha"]="0"*64
                elif kind=="execution":h.executions+=1
                else:v.tracks["right"]["goal_index"]+=1
                r=self.observe(x,2)
                self.assertFalse(r["verified"])
                self.assertEqual(r["per_hand"]["right"]["consecutive_registered_lifts"],1)
                self.assertEqual(r["per_hand"]["right"]["feature_history_binding"]["reason"],"HISTORY_IDENTITY_MISMATCH_RESET")

    def test_failed_measurement_or_explicit_denial_discards_history(self):
        x=self.case();v=x[3]
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",side_effect=self.measured):
            self.observe(x,0);self.observe(x,1)
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",return_value={"registered_pair_consistent":False}):
            self.assertFalse(self.observe(x,2)["verified"])
        self.assertEqual(v.tracks,{})
        with patch("semantic_robot.v2.grasp_motion.measure_grasp_motion",side_effect=self.measured):
            self.observe(x,3)
            self.assertFalse(self.observe(x,4,evidence(enclosed=False))["verified"])
        self.assertEqual(v.tracks,{})
