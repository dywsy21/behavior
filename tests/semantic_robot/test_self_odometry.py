import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from semantic_robot.v2.odometry import RGBDMotion
from semantic_robot.v2.self_odometry import make_frame, validate_frame, world_correspondences
from semantic_robot.v2.substep_odometry import SubstepMotion
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.servo import SafeServo
from test_substep_odometry import measured
from test_v2 import fixture


class SelfOdometryTests(unittest.TestCase):
    def setUp(self):
        self.model, self.state = fixture()
        T = np.eye(4); T[2,3] = 1.
        self.geometry = {"source":"robot_visual_link_boxes_actual_joint_fk", "scene_truth":False,
                         "includes_actual_finger_positions":True,"margin_m":.003,
                         "boxes":[{"link":"test_finger","lower":[-.08,-.08,-.04],
                                   "upper":[.08,.08,.04],"T_base_link":T.tolist()}]}
        self.model.spec["metadata"]["robot_visual_boxes_reference"] = copy.deepcopy(self.geometry)
        self.images = {"head_rgb":np.zeros((3,100,100),np.uint8)}
        self.depths = {"head":np.ones((100,100),np.float32)}
        self.args = self.images,self.depths,self.model,self.state.q

    def frame(self,control=0,geometry=None):
        return make_frame(self.images,self.depths,self.model,self.state,control,
                          self.geometry if geometry is None else geometry)

    def bound(self,control=0):
        return dict(robot_frame=self.frame(control),gripper=self.state.gripper,control=control)

    def test_same_frame_and_actual_finger_pose_not_static_open_transform(self):
        frame = self.frame()
        self.assertEqual(len(validate_frame(frame,*self.args,self.state.gripper,0)),64)
        moved = copy.deepcopy(self.geometry); moved["boxes"][0]["T_base_link"][0][3]=.03
        self.state.gripper[0]=.015
        new = self.frame(6,moved)
        self.assertEqual(new["geometry"]["boxes"][0]["T_base_link"][0][3],.03)
        with self.assertRaises(ValueError):validate_frame(frame,*self.args,self.state.gripper,0)
        self.assertEqual(frame["geometry"],self.geometry)

    def test_stale_missing_unknown_and_changed_sensor_frames_fail(self):
        for field,value in (("version","old"),("control",True),("control",6),("q",[0]*17),
                            ("gripper",[.01,.05]),("sensors",{}),("geometry",None)):
            frame=self.frame();frame[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_frame(frame,*self.args,self.state.gripper,0)
        for control in (None,True,0.,-1):
            with self.assertRaises(ValueError):validate_frame(self.frame(),*self.args,self.state.gripper,control)
        frame=self.frame();images=copy.deepcopy(self.images);images["head_rgb"][0,4,5]=1
        with self.assertRaises(ValueError):validate_frame(frame,images,self.depths,self.model,self.state.q,self.state.gripper,0)
        depths=copy.deepcopy(self.depths);depths["head"][4,5]=.9
        with self.assertRaises(ValueError):validate_frame(frame,self.images,depths,self.model,self.state.q,self.state.gripper,0)
        q=self.state.q.copy();q[5]=.01
        with self.assertRaises(ValueError):validate_frame(frame,self.images,self.depths,self.model,q,self.state.gripper,0)

    def test_asset_coverage_bounds_scene_truth_and_rigid_geometry_are_fixed(self):
        mutations=[lambda g:g.update(scene_truth=True),lambda g:g.update(includes_actual_finger_positions=1),
            lambda g:g.update(boxes=[]),lambda g:g["boxes"].append(copy.deepcopy(g["boxes"][0])),
            lambda g:g["boxes"][0].update(link="scene_table"),lambda g:g.update(margin_m=.009),
            lambda g:g["boxes"][0].update(upper=[9,9,9]),
            lambda g:g["boxes"][0]["T_base_link"][0].__setitem__(0,2.)]
        for change in mutations:
            g=copy.deepcopy(self.geometry);change(g)
            with self.subTest(change=change),self.assertRaises(ValueError):self.frame(0,g)

    def test_exclusion_uses_either_endpoint_actual_optical_to_base_transform(self):
        T=self.model.forward(self.state.q,"camera_head")
        old=np.array([[0,0,1],[.3,0,1],[.4,0,1]])
        new=np.array([[.3,0,1],[0,0,1],[.4,0,1]])
        a,u,b,receipt=world_correspondences(old,[[1,2],[3,4],[5,6]],new,T,T,self.geometry,self.geometry)
        np.testing.assert_array_equal(a,old[2:]);np.testing.assert_array_equal(b,new[2:])
        np.testing.assert_array_equal(u,[[5,6]])
        self.assertEqual(receipt["self_excluded_before"],1);self.assertEqual(receipt["self_excluded_after"],1)
        self.assertEqual(receipt["self_excluded_either"],2)
        with self.assertRaises(ValueError):world_correspondences(old,[[1,2]]*3,new,T,T,None,self.geometry)

    def test_actual_rgbd_observer_filters_before_unchanged_single_solver(self):
        odom=RGBDMotion("rgbd_joint",exclude_robot=True)
        points=[SimpleNamespace(pt=(float(x),float(y))) for x in range(10,91,10) for y in range(10,91,10)]
        odom.sift=Mock();odom.sift.detectAndCompute.return_value=(points,np.ones((len(points),128),np.float32))
        odom.matcher=Mock();odom.matcher.knnMatch.return_value=[
            (SimpleNamespace(distance=1.,queryIdx=i,trainIdx=i),SimpleNamespace(distance=10.)) for i in range(len(points))]
        odom.solve=Mock(return_value=measured())
        self.assertTrue(odom.observe(*self.args,**self.bound())["initial"])
        result=odom.observe(*self.args,**self.bound(6))
        self.assertEqual(odom.solve.call_count,1)
        self.assertTrue(result["robot_self_exclusion"])
        self.assertEqual(result["robot_self_filter"]["raw_depth_matches"],81)
        self.assertEqual(result["robot_self_filter"]["self_excluded_either"],1)
        self.assertEqual(len(odom.solve.call_args.args[0]),80)
        self.assertNotEqual(result["previous_robot_frame_sha256"],result["current_robot_frame_sha256"])

    def test_no_geometry_is_not_silent_legacy_and_default_stays_off(self):
        for kind in ("pnp","rgbd_rigid"):
            with self.assertRaises(ValueError):RGBDMotion(kind,exclude_robot=True)
        new=RGBDMotion("rgbd_joint",exclude_robot=True)
        with self.assertRaises(ValueError):new.observe(*self.args)
        self.assertIsNone(new.previous)
        old=RGBDMotion("rgbd_joint")
        with self.assertRaises(ValueError):old.observe(*self.args,**self.bound())
        self.assertTrue(old.observe(*self.args)["initial"])

    def test_bound_substeps_deliver_only_same_geometry_gripper_and_clock(self):
        base=Mock();base.exclude_robot=True
        base.observe.side_effect=[{"valid":True,"initial":True},measured()]
        motion=SubstepMotion(base)
        motion.observe(*self.args,**self.bound())
        with self.assertRaises(ValueError):motion.begin(1)
        motion.begin(0)
        end=self.frame(6)
        motion.sample(*self.args,6,robot_frame=end,gripper=self.state.gripper)
        result=motion.finish(6)
        self.assertTrue(result["valid"]);self.assertTrue(result["robot_self_exclusion"])
        for key in ("clock","gripper","geometry"):
            altered=copy.deepcopy(end);gripper=self.state.gripper.copy();control=6
            if key=="clock":control=7;altered["control"]=7
            if key=="gripper":gripper[0]=.03;altered["gripper"]=gripper.tolist()
            if key=="geometry":altered["geometry"]["boxes"][0]["T_base_link"][0][3]=.02
            with self.subTest(key=key),self.assertRaises(ValueError):
                motion.observe(*self.args,robot_frame=altered,gripper=gripper,control=control)
            self.assertIsNotNone(motion.pending)
        delivered=motion.observe(*self.args,robot_frame=end,gripper=self.state.gripper,control=6)
        self.assertEqual(delivered["segment_count"],1);self.assertEqual(base.observe.call_count,2)

    def test_all_self_matches_cannot_create_false_world_support(self):
        T=self.model.forward(self.state.q,"camera_head")
        a,u,b,receipt=world_correspondences([[0,0,1]]*40,[[50,50]]*40,[[0,0,1]]*40,T,T,self.geometry,self.geometry)
        from semantic_robot.v2.joint_odometry import solve_joint_correspondences
        result=solve_joint_correspondences(a,u,b,np.array([[70,0,50],[0,70,50],[0,0,1]]),T,T)
        self.assertEqual(receipt["remaining_world_matches"],0)
        self.assertFalse(result["valid"]);self.assertNotIn("body_delta",result)

    def test_reanchor_preserves_mask_and_binds_all_three_new_frames(self):
        from unittest.mock import patch
        from test_search_reanchor import SearchReanchorTests
        old=SearchReanchorTests();old.setup()
        old.model.spec["metadata"]["robot_visual_boxes_reference"]=copy.deepcopy(self.geometry)
        old.c.odometry_self_exclusion=True
        frames=[];factories=[]
        def save(control,images,depths,sensor,state):
            frame=make_frame(images,depths,old.model,state,control,self.geometry)
            frames.append(frame);return frame
        def factory(kind,**kwargs):
            self.assertEqual(kind,"rgbd_joint");self.assertEqual(kwargs,{"exclude_robot":True})
            base=Mock();base.exclude_robot=True
            base.observe.side_effect=[{"valid":True,"initial":True},measured(),measured()]
            factories.append(base);return base
        with patch("semantic_robot.v2.search_reanchor.RGBDMotion",side_effect=factory):
            result,snapshot=old.c.search_recovery.attempt(old.c,old.failed,controls=36,action_limit=60,
                terminal=False,deadline=None,observe=lambda label:(old.images,old.depths,{}),
                state_now=lambda:old.state,issue_hold=lambda:False,save_snapshot=save)
        self.assertTrue(result["valid"]);self.assertEqual([f["control"] for f in frames],[36,42,48])
        self.assertEqual(len(factories),2);self.assertTrue(old.c.motion.exclude_robot)
        self.assertEqual(snapshot[6],frames[-1]);self.assertTrue(result["chain"]["robot_self_exclusion"])
        self.assertEqual([c.kwargs["control"] for c in factories[0].observe.call_args_list],[36,42,48])

    def test_actual_runner_capture_refuses_joint_or_gripper_changes(self):
        tree=ast.parse((Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="capture_self_frame")
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),"actual_self_capture","exec")
        kin=Mock();kin.native_self_boxes.return_value=self.geometry
        for key in ("q","gripper",None):
            after=copy.deepcopy(self.state)
            if key is not None:getattr(after,key)[0]+=.01
            state_now=Mock(side_effect=[self.state,after])
            ns={"np":np,"kin":kin,"model":self.model,"state_now":state_now};exec(code,ns)
            if key is None:
                self.assertEqual(ns["capture_self_frame"](0,self.images,self.depths,self.state)["geometry"],self.geometry)
            else:
                with self.assertRaises(RuntimeError):ns["capture_self_frame"](0,self.images,self.depths,self.state)

    def test_real_runner_flag_requires_six_control_joint_and_fresh_geometry(self):
        path=Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py"
        tree=ast.parse(path.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test).startswith("args.odometry_self_exclusion and (not"))
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec")
        good=dict(odometry_self_exclusion=True,grasp_motion=True,visual_odometry=True,
                  odometry_estimator="rgbd_joint",odometry_substep_controls=6)
        exec(code,{"args":SimpleNamespace(**good),"grounded":True})
        for key,value in (("grasp_motion",False),("visual_odometry",False),("odometry_estimator","pnp"),("odometry_substep_controls",0)):
            with self.assertRaises(ValueError):exec(code,{"args":SimpleNamespace(**{**good,key:value}),"grounded":True})
        h=GroundedHarness([Goal("pick","x","right","x")])
        with self.assertRaises(ValueError):GroundedController(self.model,SafeServo(self.model,self.state),h,odometry_self_exclusion=True)


if __name__=="__main__":unittest.main()
