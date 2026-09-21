import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

from semantic_robot.v2.feature_tracking import refine_matches, VERSION
from semantic_robot.v2.odometry import RGBDMotion
from semantic_robot.v2.grounded_harness import GroundedController,GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.self_odometry import make_frame
from test_v2 import fixture
from test_substep_odometry import measured


class FeatureTrackingTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(31)
        image=rng.integers(0,256,(120,160),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),.6)
        points=np.array([[x,y] for x in (35.,60.,85.,110.,135.) for y in (35.,60.,85.)])
        return image,points

    def test_known_subpixel_translation_corrects_initial_location(self):
        before,points=self.data();delta=np.array([1.5,-.75])
        after=cv2.warpAffine(before,np.array([[1.,0.,delta[0]],[0.,1.,delta[1]]]),(160,120))
        guess=points+delta+[.55,-.35];original=guess.copy()
        result,keep,receipt=refine_matches(before,after,points,guess)
        self.assertTrue(keep.all())
        np.testing.assert_allclose(result-points,np.tile(delta,(len(points),1)),atol=.16)
        np.testing.assert_array_equal(guess,original)
        self.assertEqual(receipt['version'],VERSION)
        self.assertFalse(receipt['unrefined_fallback'])

    def test_empty_has_no_opencv_call_and_no_fictitious_matches(self):
        image,_=self.data()
        with patch('cv2.calcOpticalFlowPyrLK') as tracking:
            result,keep,receipt=refine_matches(image,image,np.empty((0,2)),np.empty((0,2)))
        tracking.assert_not_called();self.assertEqual(result.shape,(0,2));self.assertEqual(keep.size,0)
        self.assertEqual(receipt['bidirectional_accepted'],0)

    def test_invalid_shape_dtype_nonfinite_are_rejected(self):
        image,points=self.data()
        variants=[(image.astype(float),image,points,points),(image,image[:100],points,points),
                  (image[:,:,None],image[:,:,None],points,points),(image,image,points[:,0],points[:,0]),
                  (image,image,points,points[:-1]),(image,image,points+np.nan,points)]
        for args in variants:
            with self.subTest(shape=[x.shape for x in args]),self.assertRaises(ValueError):refine_matches(*args)

    def test_border_and_off_image_coordinates_are_not_tracked(self):
        image,_=self.data();points=np.array([[0.,10.],[159.,10.],[12.,120.],[-1.,10.]])
        with patch('cv2.calcOpticalFlowPyrLK') as tracking:
            _,keep,_=refine_matches(image,image,points,points)
        tracking.assert_not_called();self.assertFalse(keep.any())

    def test_subpixel_rounding_cannot_shrink_the_three_by_three_depth_patch(self):
        image,_=self.data();old=np.array([[157.,60.]],np.float32);good=np.ones((1,1),np.uint8)
        for x,want in [(158.49,True),(158.5,True),(158.51,False),(158.75,False)]:
            forward=np.array([[[x,60.]]],np.float32)
            with patch('cv2.calcOpticalFlowPyrLK',side_effect=[(forward,good,None),(old[:,None],good,None)]) as tracker:
                _,keep,_=refine_matches(image,image,old,old)
            self.assertIs(bool(keep[0]),want)
            self.assertEqual(tracker.call_count,2 if want else 1)
        # The same rule applies to initial proposals, the old endpoint and y.
        for points in (np.array([[158.75,60.]]),np.array([[60.,118.75]])):
            with patch('cv2.calcOpticalFlowPyrLK') as tracker:
                _,keep,_=refine_matches(image,image,points,points)
            tracker.assert_not_called();self.assertFalse(keep.any())

    def test_forward_failure_nonfinite_and_outside_do_not_reach_backward(self):
        image,points=self.data();n=len(points)
        for out,status in [(points[:,None].astype(np.float32),np.zeros((n,1),np.uint8)),
                           (np.full((n,1,2),np.nan,np.float32),np.ones((n,1),np.uint8)),
                           (np.full((n,1,2),9999.,np.float32),np.ones((n,1),np.uint8))]:
            with patch('cv2.calcOpticalFlowPyrLK',return_value=(out,status,None)) as tracking:
                _,keep,_=refine_matches(image,image,points,points)
            self.assertEqual(tracking.call_count,1);self.assertFalse(keep.any())

    def test_reverse_inconsistency_and_failure_rejected_without_fallback(self):
        image,points=self.data();n=len(points);good=np.ones((n,1),np.uint8)
        for reverse,status in [(points[:,None]+[2.,0.],good),(points[:,None],good*0),
                               (points[:,None]*np.nan,good)]:
            with patch('cv2.calcOpticalFlowPyrLK',side_effect=[(points[:,None],good,None),(reverse,status,None)]):
                _,keep,receipt=refine_matches(image,image,points,points)
            self.assertFalse(keep.any());self.assertFalse(receipt['unrefined_fallback'])

    def test_error_or_none_returns_zero_valid_not_original_support(self):
        image,points=self.data()
        for side in [cv2.error('fixture'),[(None,None,None)]]:
            with patch('cv2.calcOpticalFlowPyrLK',side_effect=side):
                _,keep,receipt=refine_matches(image,image,points,points)
            self.assertFalse(keep.any());self.assertFalse(receipt['unrefined_fallback'])

    def test_malformed_tracker_shape_fails_closed(self):
        image,points=self.data()
        with patch('cv2.calcOpticalFlowPyrLK',return_value=(points[:1,None],np.ones((1,1)),None)):
            with self.assertRaises(ValueError):refine_matches(image,image,points,points)

    def test_explicit_joint_only_and_default_is_unchanged(self):
        self.assertFalse(RGBDMotion().refine_matches)
        self.assertTrue(RGBDMotion('rgbd_joint',refine_matches=True).refine_matches)
        for args in [dict(refine_matches=1),dict(refine_matches=None),dict(refine_matches=True),
                     dict(estimator='rgbd_rigid',refine_matches=True)]:
            with self.assertRaises(ValueError):RGBDMotion(**args)

    def test_controller_routes_same_option_and_requires_self_exclusion(self):
        model,state=fixture();h=GroundedHarness([Goal('pick','x','right','x')]);servo=SafeServo(model,state)
        good=dict(visual_odometry=True,odometry_estimator='rgbd_joint',odometry_self_exclusion=True,odometry_match_refinement=True)
        controller=GroundedController(model,servo,h,**good)
        self.assertTrue(controller.motion.refine_matches);self.assertTrue(controller.odometry_match_refinement)
        for key,value in [('visual_odometry',False),('odometry_estimator','pnp'),('odometry_self_exclusion',False),('odometry_match_refinement',1)]:
            with self.assertRaises(ValueError):GroundedController(model,servo,h,**{**good,key:value})

    def runner_tree(self):
        return ast.parse((Path(__file__).resolve().parents[2]/'scripts/semantic_robot/run_v2.py').read_text())

    def test_actual_runner_mode_requires_current_geometry_fixed_six_joint(self):
        node=next(n for n in ast.walk(self.runner_tree()) if isinstance(n,ast.If)
                  and ast.unparse(n.test).startswith('args.odometry_match_refinement and (not'))
        code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_mode_guard','exec')
        good=dict(odometry_match_refinement=True,odometry_self_exclusion=True,visual_odometry=True,
                  odometry_estimator='rgbd_joint',odometry_substep_controls=6)
        exec(code,{'args':SimpleNamespace(**good),'grounded':True})
        for key,value in [('odometry_self_exclusion',False),('visual_odometry',False),('odometry_estimator','pnp'),('odometry_substep_controls',0)]:
            with self.assertRaises(ValueError):exec(code,{'args':SimpleNamespace(**{**good,key:value}),'grounded':True})
        with self.assertRaises(ValueError):exec(code,{'args':SimpleNamespace(**good),'grounded':False})

    def test_gate_manifest_result_and_both_runner_factories_bind_option(self):
        tree=self.runner_tree()
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.Compare)
                  and ast.unparse(n.left)=="g.get('odometry_match_refinement', False)")
        code=compile(ast.Expression(body=node),'actual_gate_identity','eval')
        for gate,requested,want in [({},False,True),({},True,False),({'odometry_match_refinement':True},True,True),({'odometry_match_refinement':True},False,False)]:
            self.assertIs(eval(code,{'g':gate,'args':SimpleNamespace(odometry_match_refinement=requested)}),want)
        values=[v for n in ast.walk(tree) if isinstance(n,ast.Dict) for k,v in zip(n.keys,n.values)
                if isinstance(k,ast.Constant) and k.value=='odometry_match_refinement']
        self.assertEqual(len(values),2)
        self.assertTrue(all(ast.unparse(v)=='args.odometry_match_refinement' for v in values))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id in ('GroundedController','RGBDMotion')]
        self.assertEqual(len(calls),2)
        for n in calls:
            key='odometry_match_refinement' if n.func.id=='GroundedController' else 'refine_matches'
            self.assertEqual(ast.unparse(next(k.value for k in n.keywords if k.arg==key)),'args.odometry_match_refinement')

    def test_reanchor_probe_and_new_reference_both_keep_frontend(self):
        from test_self_odometry import SelfOdometryTests
        from test_search_reanchor import SearchReanchorTests
        geometry=SelfOdometryTests();geometry.setUp();old=SearchReanchorTests();old.setup()
        old.model.spec['metadata']['robot_visual_boxes_reference']=copy.deepcopy(geometry.geometry)
        old.c.odometry_self_exclusion=True;old.c.odometry_match_refinement=True;factories=[]
        def make(kind,**options):
            self.assertEqual(kind,'rgbd_joint');self.assertEqual(options,{'exclude_robot':True,'refine_matches':True})
            base=Mock();base.exclude_robot=True
            base.observe.side_effect=[{'valid':True,'initial':True},measured(),measured()]
            factories.append(base);return base
        def save(control,images,depths,sensor,state):return make_frame(images,depths,old.model,state,control,geometry.geometry)
        with patch('semantic_robot.v2.search_reanchor.RGBDMotion',side_effect=make):
            result,_=old.c.search_recovery.attempt(old.c,old.failed,controls=36,action_limit=60,terminal=False,
                deadline=None,observe=lambda label:(old.images,old.depths,{}),state_now=lambda:old.state,
                issue_hold=lambda:False,save_snapshot=save)
        self.assertTrue(result['valid']);self.assertEqual(len(factories),2)
        self.assertEqual([call.kwargs['control'] for call in factories[0].observe.call_args_list],[36,42,48])

    def test_observer_uses_refined_pixels_then_original_depth_and_self_guards(self):
        from test_self_odometry import SelfOdometryTests
        f=SelfOdometryTests();f.setUp()
        points=[SimpleNamespace(pt=(float(x),float(y))) for x in range(10,91,10) for y in range(10,91,10)]
        for enabled in (False,True):
            odom=RGBDMotion('rgbd_joint',exclude_robot=True,refine_matches=enabled)
            odom.sift=Mock();odom.sift.detectAndCompute.return_value=(points,np.ones((len(points),128),np.float32))
            odom.matcher=Mock();odom.matcher.knnMatch.return_value=[
                (SimpleNamespace(distance=1.,queryIdx=i,trainIdx=i),SimpleNamespace(distance=10.)) for i in range(len(points))]
            odom.solve=Mock(return_value=measured());odom.observe(*f.args,**f.bound())
            def refined(before,after,old,proposed):
                updated=proposed.copy();updated[0]=[20,10];mask=np.zeros(len(old),bool);mask[0]=True
                return updated,mask,{'version':VERSION}
            with patch('semantic_robot.v2.feature_tracking.refine_matches',side_effect=refined) as tracking:
                receipt=odom.observe(*f.args,**f.bound(6))
            self.assertEqual(tracking.call_count,int(enabled))
            self.assertEqual(odom.solve.call_count,1)
            self.assertEqual(receipt['robot_self_filter']['remaining_world_matches'],1 if enabled else 80)
            if enabled:np.testing.assert_array_equal(odom.solve.call_args.args[1],[[20,10]])


if __name__=='__main__':unittest.main()
