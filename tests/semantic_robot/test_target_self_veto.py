"""Robot-asset-only contact veto; synthetic cases, not task success tests."""
import copy
from dataclasses import asdict
import time
import unittest

import numpy as np

from semantic_robot.v2.affordance import SurfaceChoice
from semantic_robot.v2.finite_localization import SelectionReply, locate_target, region_boxes, region_surfaces
from semantic_robot.v2.grounding import localize_target
from semantic_robot.v2.bimanual import BimanualEvidence, HandContact
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.native_grounding import locate_native
from semantic_robot.v2.self_filter import ChassisSurface
from semantic_robot.v2.protocol import Action, HOLD
from semantic_robot.v2.servo import SafeServo
from test_grounded import grounded_evidence
from test_v2 import fixture


def setup_surface():
    model,state=fixture();spec=copy.deepcopy(model.spec)
    spec['metadata']['base_visual_surface']['vertices']=[[-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]]
    model=RobotModel(spec)
    return model,state,{v:np.ones((100,100),np.float32) for v in ('head','left_wrist','right_wrist')}


class TargetSelfVetoTests(unittest.TestCase):
    def test_chassis_point_rejected_in_every_view_not_promoted_to_target(self):
        model,state,depths=setup_surface()
        for view in depths:
            out=localize_target(grounded_evidence(view=view),{view:depths[view]},model,state.q)
            self.assertFalse(out['valid']);self.assertNotIn('point_base_m',out)
            self.assertEqual(out['reason'],'TARGET_ON_ROBOT_CHASSIS')
            self.assertTrue(out['views'][0]['chassis_self_check']['selected_pixel_on_chassis'])

    def test_millimetre_tolerance_does_not_delete_entire_box_or_envelope(self):
        model,state,depths=setup_surface()
        for delta,rejected in ((.0059,True),(.0061,False),(.02,False),(-.02,False)):
            depth=np.full((100,100),1.+delta,np.float32)
            out=localize_target(grounded_evidence(),{'right_wrist':depth},model,state.q)
            self.assertEqual(out['valid'],not rejected)
            self.assertFalse(out['target_identity_established_by_geometry'])
            self.assertFalse(out['complete_robot_exclusion'])

    def test_missing_robot_calibration_is_unknown_and_never_a_valid_contact(self):
        model,state,depths=setup_surface();spec=copy.deepcopy(model.spec)
        del spec['metadata']['base_visual_surface'];model=RobotModel(spec)
        out=localize_target(grounded_evidence(),depths,model,state.q)
        self.assertFalse(out['valid']);self.assertEqual(out['reason'],'ROBOT_SELF_GEOMETRY_UNAVAILABLE')
        self.assertIsNone(out['views'][0]['chassis_self_check']['selected_pixel_on_chassis'])

    def test_raw_selected_pixel_cannot_be_hidden_by_neighbourhood_median(self):
        model,state,_=setup_surface();depth=np.full((100,100),1.01,np.float32);depth[50,50]=1.
        out=localize_target(grounded_evidence(),{'right_wrist':depth},model,state.q)
        self.assertEqual(out['reason'],'TARGET_ON_ROBOT_CHASSIS')
        receipt=out['views'][0]['chassis_self_check']
        self.assertTrue(receipt['selected_pixel_on_chassis']);self.assertIsNone(receipt['aggregate_on_chassis'])
        self.assertEqual(receipt['checked_points'],1)  # immediate veto, median not consulted
        depth[:]=1.;depth[50,50]=1.01
        out=localize_target(grounded_evidence(),{'right_wrist':depth},model,state.q)
        receipt=out['views'][0]['chassis_self_check']
        self.assertFalse(receipt['selected_pixel_on_chassis']);self.assertTrue(receipt['aggregate_on_chassis'])
        self.assertFalse(out['valid'])

    def test_hole_at_requested_pixel_is_not_filled_from_adjacent_depth(self):
        model,state,_=setup_surface()
        for bad in (np.nan,np.inf,0.,4.):
            depth=np.full((100,100),1.2,np.float32);depth[50,50]=bad
            out=localize_target(grounded_evidence(),{'right_wrist':depth},model,state.q)
            self.assertFalse(out['valid']);self.assertEqual(out['views'][0]['reason'],'TARGET_PIXEL_DEPTH_MISSING')

    def test_secondary_view_cannot_launder_primary_self_point_or_vice_versa(self):
        model,state,depths=setup_surface()
        obs=grounded_evidence(other_views=[{'view':'head','target_uv':[.5,.5]}])
        for body_view in ('head','right_wrist'):
            depths['head'][:]=1.02;depths['right_wrist'][:]=1.02;depths[body_view][:]=1.
            out=localize_target(obs,depths,model,state.q)
            self.assertFalse(out['valid']);self.assertEqual(out['reason'],'TARGET_ON_ROBOT_CHASSIS')

    def test_patch_edge_or_holes_cannot_hide_self_hit_from_another_view(self):
        model,state,depths=setup_surface()
        obs=grounded_evidence(other_views=[{'view':'head','target_uv':[.5,.5]}])
        for body_view in ('head','right_wrist'):
            for kind in ('edge','holes'):
                for depth in depths.values():depth[:]=1.02
                depths[body_view][:]=1.1 if kind=='edge' else np.nan
                depths[body_view][48:50,48:53]=1.
                depths[body_view][50,48:51]=1.
                out=localize_target(obs,depths,model,state.q)
                with self.subTest(body_view=body_view,kind=kind):
                    self.assertFalse(out['valid']);self.assertEqual(out['reason'],'TARGET_ON_ROBOT_CHASSIS')
                    hit=next(row for row in out['views'] if row['view']==body_view)
                    self.assertTrue(hit['chassis_self_check']['selected_pixel_on_chassis'])

    def test_explicit_other_view_may_supply_a_ray_when_original_pixel_is_a_hole(self):
        model,state,depths=setup_surface()
        for depth in depths.values():depth[:]=1.02
        depths['right_wrist'][50,50]=np.nan
        obs=grounded_evidence(other_views=[{'view':'head','target_uv':[.5,.5]}])
        out=localize_target(obs,depths,model,state.q)
        self.assertTrue(out['valid']);self.assertEqual(out['valid_views'],1)
        self.assertFalse(out['views'][0]['valid']);self.assertTrue(out['views'][1]['valid'])
        self.assertEqual(out['views'][0]['reason'],'TARGET_PIXEL_DEPTH_MISSING')
        self.assertNotIn('point_base_m',out['views'][0])  # no synthesized hole ray

    def test_average_of_two_valid_nonself_points_can_still_be_on_chassis(self):
        model,state,depths=setup_surface();depths['head'][:]=.98;depths['right_wrist'][:]=1.02
        obs=grounded_evidence(other_views=[{'view':'head','target_uv':[.5,.5]}])
        out=localize_target(obs,depths,model,state.q)
        self.assertTrue(all(r['valid'] for r in out['views']))
        self.assertFalse(out['valid']);self.assertTrue(out['chassis_self_check']['mean_point_on_chassis'])

    def test_current_fk_moves_robot_surface_not_scene_or_target_point(self):
        model,state,depths=setup_surface();spec=copy.deepcopy(model.spec)
        spec['links']['link:base_link']['screws'][2][0]=1.
        model=RobotModel(spec);q=state.q.copy();q[0]=.02
        self.assertFalse(localize_target(grounded_evidence(),depths,model,state.q)['valid'])
        self.assertTrue(localize_target(grounded_evidence(),depths,model,q)['valid'])

    def test_nonfinite_points_or_nonrigid_fk_fail_closed(self):
        model,_,_=setup_surface();surface=ChassisSurface(model.spec['metadata']['base_visual_surface'])
        with self.assertRaises(ValueError):surface.mask([[np.nan,0,1]],np.eye(4))
        for kind in ('scale','reflection','row','nan'):
            transform=np.eye(4)
            if kind=='scale':transform[0,0]=2.
            elif kind=='reflection':transform[0,0]=-1.
            elif kind=='row':transform[3,0]=1.
            else:transform[0,0]=np.nan
            with self.subTest(kind=kind),self.assertRaises(ValueError):surface.mask([[0,0,1]],transform)

    def test_region_locator_never_offers_self_surface_ids(self):
        model,state,depths=setup_surface();raw={v:np.zeros((100,100,3),np.uint8) for v in depths}
        self.assertEqual(region_surfaces('head',region_boxes(100,100)[4],depths,model,state),[])
        requests=[]
        def choose(request):
            requests.append(request);return SelectionReply(request.sha256,SurfaceChoice(4).text())
        result=locate_target(frame_id='self-frame',goal=Goal('pick','object','right','independent result'),
            raw=raw,depths=depths,model=model,state=state,views=('head',),choose=choose,deadline=time.monotonic()+10)
        self.assertEqual(len(requests),1);self.assertFalse(result.evidence.visible)
        self.assertEqual(result.receipt['attempts'][0]['reason'],'NO_VALID_SURFACE_SAMPLES')

    def test_native_locator_does_not_convert_self_point_to_actor_evidence(self):
        model,state,depths=setup_surface();raw={v:np.zeros((100,100,3),np.uint8) for v in depths}
        def choose(request):return SelectionReply(request.sha256,'[{"point_2d":[500,500],"label":"object"}]')
        result=locate_native(frame_id='self-frame',goal=Goal('pick','object','right','independent result'),
            raw=raw,depths=depths,model=model,state=state,view='head',mode='point',choose=choose,deadline=time.monotonic()+10)
        self.assertFalse(result.point_evidence().visible)
        self.assertEqual(result.receipt['geometry']['reason'],'TARGET_ON_ROBOT_CHASSIS')

    def test_existing_manipulation_stages_cannot_act_or_confirm_from_self_claims(self):
        for kind,stage in (('pick','GRASP'),('pick','VERIFY_GRASP'),('press','INTERACT'),
                           ('press','VERIFY_EFFECT'),('place','ALIGN'),('place','VERIFY_SUPPORT'),
                           ('place','RELEASE'),('place','VERIFY_PLACE')):
            model,state,depths=setup_surface()
            manager=GroundedHarness([Goal(kind,'object','right','independent result')])
            controller=GroundedController(model,SafeServo(model,state),manager)
            manager.stage=stage;manager.confirmations=1;manager.last_action=Action('right','up')
            if kind=='place':manager.hold_verified['right']=True;manager.held['right']='existing load'
            obs=grounded_evidence(enclosed=True,co_moving=True,supported=True,effect=True)
            for _ in range(2):controller.observe(obs,state,depths,{'head':{'valid_fraction':1.}})
            with self.subTest(kind=kind,stage=stage):
                self.assertEqual(manager.completed,[]);self.assertEqual(manager.index,0)
                self.assertEqual(manager.confirmations,0);self.assertIsNone(manager.observation.effect)
                self.assertEqual(manager.context()['target_geometry_veto']['raw_evidence'],asdict(obs))
                self.assertTrue(obs.effect)  # original claim was not mutated
                self.assertEqual(manager.palette(),(HOLD,));self.assertEqual(controller.candidates(state),(HOLD,))
                self.assertEqual(controller.search_action(state)[0],HOLD)
                self.assertEqual(controller.inspection_candidates(state),(HOLD,))
                self.assertEqual(controller.verification_action((HOLD,))[0],HOLD)
                if kind=='place':self.assertTrue(manager.hold_verified['right'])

    def test_missing_geometry_and_review_wrapper_cannot_restore_old_grasp_palette(self):
        from semantic_robot.v2.grounding import target_self_veto_reason
        model,state,depths=setup_surface();spec=copy.deepcopy(model.spec)
        del spec['metadata']['base_visual_surface'];model=RobotModel(spec)
        manager=GroundedHarness([Goal('pick','object','right','independent result')])
        controller=GroundedController(model,SafeServo(model,state),manager)
        controller.observe(grounded_evidence(enclosed=True),state,depths,{'head':{'valid_fraction':1.}})
        manager.stage='GRASP'
        self.assertEqual(controller.candidates(state),(HOLD,));self.assertEqual(manager.palette(),(HOLD,))
        self.assertEqual(target_self_veto_reason(controller.target),'ROBOT_SELF_GEOMETRY_UNAVAILABLE')
        wrapped={**controller.target,'reason':'VISIBLE_TARGET_CONTACT_UNCONFIRMED'}
        self.assertEqual(target_self_veto_reason(wrapped),'ROBOT_SELF_GEOMETRY_UNAVAILABLE')

    def test_bimanual_one_self_contact_blocks_both_hands_without_fabricating_load(self):
        model,state,depths=setup_surface();depths['right_wrist'][:]=1.2
        manager=GroundedHarness([Goal('pick','wide object','both','independent result')])
        controller=GroundedController(model,SafeServo(model,state),manager)
        obs=BimanualEvidence(**asdict(grounded_evidence()),hand_contacts=(
            HandContact('left','left_wrist',(.5,.5),True,True),
            HandContact('right','right_wrist',(.5,.5),True,True)))
        controller.observe(obs,state,depths,{'head':{'valid_fraction':1.}})
        self.assertEqual(controller.target['hand_contacts']['left']['reason'],'TARGET_ON_ROBOT_CHASSIS')
        self.assertEqual(controller.candidates(state),(HOLD,));self.assertFalse(any(manager.hold_verified.values()))
        self.assertEqual(manager.observation.hand_contacts,())
        self.assertEqual(len(manager.target_geometry_veto['raw_evidence']['hand_contacts']),2)

    def test_fresh_nonself_observation_clears_veto_without_claiming_completion(self):
        model,state,depths=setup_surface();manager=GroundedHarness([Goal('pick','object','right','independent result')])
        controller=GroundedController(model,SafeServo(model,state),manager)
        controller.observe(grounded_evidence(),state,depths,{'head':{'valid_fraction':1.}})
        self.assertIsNotNone(manager.target_geometry_veto)
        for depth in depths.values():depth[:]=1.2
        controller.observe(grounded_evidence(),state,depths,{'head':{'valid_fraction':1.}})
        self.assertTrue(controller.target['valid']);self.assertIsNone(manager.target_geometry_veto)
        self.assertEqual(manager.completed,[]);self.assertEqual(manager.stage,'APPROACH')


if __name__=='__main__':unittest.main()
