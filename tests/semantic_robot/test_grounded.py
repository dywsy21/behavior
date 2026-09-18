import copy
from dataclasses import asdict
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.v2.grounding import GroundedEvidence, LocalDepthGuard, localize_target, unproject, validate_depth
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController, metric_co_motion, parse_recovery
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.og_calibration import CalibratedRobot
from semantic_robot.v2.onboard import OnboardRGBD
from semantic_robot.v2.protocol import Action, HOLD
from semantic_robot.v2.policy import GROUNDED_OBSERVE_SYSTEM, perception_context
from semantic_robot.v2.search import CoverageSearch
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import project, prepare_views
from test_v2 import fixture, evidence, feedback


def grounded_evidence(**changes):
    extra=changes.pop("other_views",[])
    fields=asdict(evidence(**changes)); fields["other_views"]=extra
    return GroundedEvidence.parse(json.dumps(fields))


def setup_controller():
    model,state=fixture()
    manager=GroundedHarness([Goal("pick","radio","right","visible motion with hand")])
    servo=SafeServo(model,state,[-1,-1])
    controller=GroundedController(model,servo,manager)
    depths={v:np.full((100,100),1.2,dtype=np.float32) for v in ("head","left_wrist","right_wrist")}
    receipt={v:{"valid_fraction":1.} for v in depths}
    return model,state,manager,servo,controller,depths,receipt


class DepthTests(unittest.TestCase):
    def test_navigation_model_done_claim_cannot_override_wrong_measured_bearing(self):
        model,state=fixture();h=GroundedHarness([Goal("navigate","table","both","facing table")])
        h.stage="APPROACH";h.last_action=Action("base","left")
        obs=grounded_evidence(effect=True)
        h.observe(obs,state,measured_progress={"navigation_aligned":False})
        self.assertEqual(h.stage,"APPROACH");self.assertEqual(h.completed,[])
        h.observe(obs,state,measured_progress={"navigation_aligned":True})
        self.assertEqual(h.stage,"VERIFY_EFFECT")
        h.observe(obs,state,measured_progress={"navigation_aligned":False})
        self.assertEqual(h.stage,"APPROACH");self.assertEqual(h.completed,[])
    def test_search_reacquires_target_ray_not_previous_camera_center(self):
        search=CoverageSearch();search.heading=.5
        camera=np.eye(4);camera[:3,:3]=Rotation.from_euler("y",-np.pi/2).as_matrix()
        K=[[100,0,100],[0,100,100],[0,0,1]]
        search.observe(0,camera,K,200,1.,True,target_bearing_rad=.7)
        self.assertAlmostEqual(search.last_seen_heading,1.2)
        search.heading=.7
        search.observe(0,camera,K,200,1.,False)
        action,reason=search.propose()
        self.assertEqual(action,Action("base","yaw_plus","coarse"))
        self.assertEqual(reason,"REACQUIRE_LAST_OBSERVED_BEARING")
        with self.assertRaises(ValueError):search.observe(0,camera,K,200,1.,True,target_bearing_rad=float("nan"))
    def test_navigation_faces_observed_destination_before_sideways_chasing(self):
        model,state,manager,servo,controller,depths,receipt=setup_controller()
        manager.goals=[Goal("navigate","visible table","both","in front of table")]
        manager.stage="APPROACH";controller.centers=model.grasp_centers(state.q)
        controller.target={"valid":True,"point_base_m":[1.,1.,.8],"distance_to_active_closing_center_m":1.5}
        controller.depth_guard=SimpleNamespace(check=lambda *args:(True,"TEST_FREE_DEPTH"),receipt=lambda:{})
        allowed=controller.candidates(state)
        self.assertIn(Action("base","yaw_plus","coarse"),allowed)
        self.assertTrue(all(a==HOLD or (a.part=="base" and a.move=="yaw_plus") for a in allowed))
        self.assertEqual(manager.candidate_receipt["navigation"]["phase"],"FACE_TARGET_FIRST")
        self.assertAlmostEqual(manager.candidate_receipt["navigation"]["current"]["bearing_deg"],45.)
        self.assertEqual(manager.completed,[])
        controller.target["point_base_m"]=[1.,0.,.8]
        allowed=controller.candidates(state)
        self.assertIn(Action("base","forward","micro"),allowed)
        self.assertIn(Action("base","forward","fine"),allowed)

    def test_navigation_blocked_turn_retains_checked_reposition_not_blind_turn(self):
        model,state,manager,servo,controller,depths,receipt=setup_controller()
        manager.goals=[Goal("navigate","table","both","in front")];manager.stage="APPROACH"
        controller.centers=model.grasp_centers(state.q)
        controller.target={"valid":True,"point_base_m":[1.,1.,.8],"distance_to_active_closing_center_m":1.5}
        controller.depth_guard=SimpleNamespace(check=lambda a,*args:(a.move not in ("yaw_plus","yaw_minus"),"TEST_BLOCKED_TURN"),receipt=lambda:{})
        allowed=controller.candidates(state)
        self.assertFalse(any(a.move in ("yaw_plus","yaw_minus") for a in allowed))
        self.assertIn(Action("base","left","micro"),allowed)
    def test_primary_contact_reprojection_compares_same_point_not_model_clicks(self):
        model,state=fixture();depths={v:np.full((100,100),1.2,np.float32) for v in ("head","left_wrist","right_wrist")}
        result=localize_target(grounded_evidence(),depths,model,state.q)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["projected_same_point_checks"]),2)
        self.assertTrue(all(r["status"]=="SURFACE_DEPTH_AGREES_NOT_IDENTITY_PROOF" for r in result["projected_same_point_checks"]))
        self.assertEqual(result["valid_views"],1)  # not two independent identity confirmations

    def test_primary_contact_occluded_or_unknown_elsewhere_remains_single_view(self):
        model,state=fixture();depths={"right_wrist":np.full((100,100),1.2,np.float32),
                                  "head":np.full((100,100),.8,np.float32),
                                  "left_wrist":np.full((100,100),np.inf,np.float32)}
        result=localize_target(grounded_evidence(),depths,model,state.q)
        self.assertTrue(result["valid"])
        self.assertEqual({r["status"] for r in result["projected_same_point_checks"]},
                         {"OCCLUDED_BY_NEARER_SURFACE","DEPTH_UNKNOWN"})

    def test_primary_point_in_observed_free_space_of_other_camera_rejected(self):
        model,state=fixture();depths={"right_wrist":np.full((100,100),1.2,np.float32),
                                  "head":np.full((100,100),2.,np.float32)}
        result=localize_target(grounded_evidence(),depths,model,state.q)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"],"PROJECTED_CONTACT_DEPTH_CONTRADICTION")

    def test_projected_depth_edge_is_unknown_not_false_corroboration(self):
        model,state=fixture();depths={v:np.full((100,100),1.2,np.float32) for v in ("head","right_wrist")}
        depths["head"][49:52,:]=2.
        result=localize_target(grounded_evidence(),depths,model,state.q)
        self.assertTrue(result["valid"])
        self.assertEqual(result["projected_same_point_checks"][0]["status"],"DEPTH_EDGE_UNKNOWN")

    def test_grounded_prompt_example_is_complete_and_parser_valid(self):
        example=GROUNDED_OBSERVE_SYSTEM.split("Return only JSON with exactly these fields:\n",1)[1].split("\n",1)[0]
        fields=json.loads(example)
        self.assertEqual(set(fields),set(GroundedEvidence.__dataclass_fields__))
        self.assertEqual(fields["other_views"],[])
        self.assertFalse(GroundedEvidence.parse(example).visible)

    def test_missing_corroboration_is_abstention_not_invented_evidence(self):
        # Actual H-07 response shape: one pilot missed the field when invisible,
        # the other missed it despite a visible head-view target.
        for visible in (True,False):
            fields=asdict(evidence(visible=visible,view="head" if visible else "none",
                                  target_uv=(.67,.78) if visible else None))
            parsed=GroundedEvidence.parse(json.dumps(fields))
            self.assertEqual(parsed.other_views,())
            self.assertEqual(parsed.visible,visible)
            self.assertEqual(parsed.target_uv,fields["target_uv"])

    def test_missing_optional_view_does_not_relax_core_or_unknown_keys(self):
        valid=asdict(evidence(view="head"))
        for key in valid:
            fields=dict(valid); fields.pop(key)
            with self.subTest(missing=key),self.assertRaises(ValueError):
                GroundedEvidence.parse(json.dumps(fields))
        for fields in ({**valid,"object_pose":[0,0,0]}, {**valid,"other_views":None},
                       {**valid,"other_views":{}}, {**valid,"other_views":[{"view":"none","target_uv":None}]}):
            with self.assertRaises(ValueError):GroundedEvidence.parse(json.dumps(fields))

    def test_render_barrier_discards_pre_reset_sensor_buffers_without_controls(self):
        model,state=fixture()
        adapter=OnboardRGBD.__new__(OnboardRGBD)
        updates=[]
        class Sensor:
            def get_obs(self):
                # Emulate a renderer retaining an old view until its pipeline
                # has been drained; both RGB and depth must come from the new one.
                current=len(updates)>=4
                return {"rgb":np.full((100,100,3),77 if current else 12,dtype=np.uint8),
                        "depth_linear":np.full((100,100),1. if current else 3.,dtype=np.float32)},{}
        adapter.sensors={v:Sensor() for v in ("head","left_wrist","right_wrist")}
        adapter.snapshot_id=0
        images,depths,receipt=adapter.read(model,render=lambda:updates.append("render_only"))
        self.assertEqual(len(updates),4)
        for view in adapter.sensors:
            self.assertTrue(np.all(images[view+"_rgb"]==77))
            self.assertTrue(np.all(depths[view]==1.))
            self.assertEqual(receipt[view]["render_barrier_updates"],4)
            self.assertEqual(receipt[view]["control_steps_in_capture"],0)
        with self.assertRaises(ValueError):adapter.read(model,render=None)

    def test_onboard_adapter_never_reconfigures_live_camera_or_reloads_physics(self):
        class Sensor:
            def __init__(self,size):self.size=size
            modalities={"rgb","depth_linear"}
            intrinsic_matrix=np.eye(3)
            @property
            def image_height(self):return self.size
            @property
            def image_width(self):return self.size
            # No setters: redundant assignment itself must fail this regression.
        sensors={"zed_link:Camera":Sensor(720),"left_realsense_link:Camera":Sensor(480),"right_realsense_link:Camera":Sensor(480)}
        env=SimpleNamespace(robots=[SimpleNamespace(sensors=sensors)])
        adapter=OnboardRGBD(env)
        self.assertEqual(len(adapter.sensors),3)
        sensors["zed_link:Camera"].size=480
        with self.assertRaises(ValueError):OnboardRGBD(env)
        sensors["zed_link:Camera"].size=720
        sensors["zed_link:Camera"].modalities={"rgb"}
        with self.assertRaises(ValueError):OnboardRGBD(env)

    def test_unprojection_linear_depth_usd_sign_and_camera_rotation(self):
        model,state=fixture()
        T=np.eye(4);T[:3,:3]=Rotation.from_euler("xyz",[.2,.3,.4]).as_matrix();T[:3,3]=[.1,.2,.5]
        K=[[100,0,50],[0,100,50],[0,0,1]]
        points=unproject([[60,30],[0,0]],[1,2],K,T)
        for point,expected in zip(points,([60,30],[0,0])):
            np.testing.assert_allclose(project(point,T,K),expected,atol=1e-10)
        local=(points[0]-T[:3,3])@T[:3,:3]
        np.testing.assert_allclose(local,[.1,.2,-1])
        self.assertNotAlmostEqual(np.linalg.norm(local),1.)  # z is not radial range

    def test_depth_units_shape_dtype_holes(self):
        model,_=fixture();camera=model.spec["metadata"]["cameras"]["head"]
        for raw in (np.zeros((100,100),dtype=np.uint16),np.zeros((99,100),dtype=np.float32)):
            with self.assertRaises(ValueError):validate_depth(raw,camera)
        depth=np.full((100,100),np.inf,dtype=np.float32)
        self.assertTrue(np.isinf(validate_depth(depth,camera)).all())
        self.assertFalse(localize_target(grounded_evidence(),{"right_wrist":depth},model,model.reference)["valid"])

    def test_discontinuous_depth_rejected_not_background_average(self):
        model,state=fixture();depth=np.ones((100,100),dtype=np.float32)
        depth[49:52,:]=2.
        out=localize_target(grounded_evidence(),{"right_wrist":depth},model,state.q)
        self.assertFalse(out["valid"])
        self.assertEqual(out["views"][0]["reason"],"DEPTH_EDGE_AMBIGUOUS")

    def test_multiview_disagreement_abstains(self):
        model,state=fixture()
        obs=grounded_evidence(other_views=[{"view":"head","target_uv":[.5,.5]}])
        depths={"right_wrist":np.ones((100,100),dtype=np.float32),"head":np.full((100,100),1.2,dtype=np.float32)}
        out=localize_target(obs,depths,model,state.q)
        self.assertEqual(out["reason"],"MULTIVIEW_TARGET_DISAGREEMENT")
        depths["head"][:]=1.
        self.assertTrue(localize_target(obs,depths,model,state.q)["valid"])

    def test_multiview_contract_rejects_hidden_or_duplicate_views(self):
        for extra in ([{"view":"right_wrist","target_uv":[.5,.5]}],
                      [{"view":"head","target_uv":[.5,.5],"object_pose":[1,2,3]}],
                      [{"view":"head","target_uv":[2.,.5]}]):
            with self.assertRaises(ValueError):grounded_evidence(other_views=extra)
        with self.assertRaises(ValueError):grounded_evidence(visible=False,view="none",target_uv=None,
            other_views=[{"view":"head","target_uv":[.5,.5]}])

    def test_only_visible_surface_point_not_object_pose(self):
        model,state=fixture()
        out=localize_target(grounded_evidence(target_uv=[.5,.5]),{"right_wrist":np.ones((100,100),dtype=np.float32)},model,state.q)
        self.assertTrue(out["valid"]);self.assertTrue(out["surface_point_not_object_pose"])
        np.testing.assert_allclose(out["point_base_m"],[0,0,1])


class GeometryTests(unittest.TestCase):
    def test_cached_fk_matches_jacobian_path_and_no_mutable_alias(self):
        model,_=fixture()
        for i in range(5):
            q=np.random.default_rng(i).uniform(-.1,.1,18)
            for name in model.links:
                np.testing.assert_allclose(model.forward(q,name),model.evaluate(q,name)[0],atol=1e-12)
                t=model.forward(q,name);t[:]=0
                self.assertEqual(model.forward(q,name)[3,3],1.)

    def test_grasp_center_robot_asset_average_open_close_invariance(self):
        kin=CalibratedRobot.__new__(CalibratedRobot)
        point=lambda name:SimpleNamespace(link_name=name,position=np.array([0.,0.,.02]))
        kin.robot=SimpleNamespace(assisted_grasp_start_points={a:[point(a+"1")] for a in ("left","right")},
                                  assisted_grasp_end_points={a:[point(a+"2")] for a in ("left","right")})
        kin.path="robot"
        centers=[]
        for opening in (0.,.02,.05):
            kin.api=SimpleNamespace(get_link_relative_position_orientation=lambda path,name:
                (np.array([.4,opening if name.endswith("1") else -opening,.8]),np.array([0,0,0,1])))
            centers.append(kin.native_grasp_centers()["right"])
        np.testing.assert_allclose(centers,np.tile([.4,0,.82],(3,1)))

    def test_overlay_closing_center_is_distinct_and_raw_preserved(self):
        model,state=fixture()
        spec=copy.deepcopy(model.spec);spec["metadata"]["grasp_centers_eef"]={"right":[.04,0,0],"left":[0,0,0]}
        model=RobotModel(spec)
        images={v+"_rgb":np.zeros((100,100,3),dtype=np.uint8) for v in ("head","left_wrist","right_wrist")}
        bundle=prepare_views(images,model,state.q,grounded=True)
        entry=bundle.geometry["head"]["right"]
        self.assertNotEqual(entry["grasp_center_uv"],entry["eef_uv"])
        self.assertTrue(entry["grasp_center_in_frame"])
        self.assertEqual(np.count_nonzero(bundle.current_raw["head"]),0)


class SearchTests(unittest.TestCase):
    def make_camera(self):
        T=np.eye(4);T[:3,:3]=Rotation.from_euler("y",-math.pi/2).as_matrix()
        return T

    def observe(self,search,visible=False):
        return search.observe(0,self.make_camera(),[[70,0,50],[0,70,50],[0,0,1]],100,1.,visible)

    def test_commands_do_not_count_as_coverage_or_motion(self):
        search=CoverageSearch();self.assertTrue(self.observe(search))
        count=len(search.covered)
        for _ in range(20):
            action,_=search.propose()
            self.assertEqual(action.scale,"coarse")
            self.assertFalse(self.observe(search))
        self.assertEqual(len(search.covered),count);self.assertEqual(search.rotation_rad,0.)

    def test_actual_rotation_expands_coverage_and_stops_after_sweep(self):
        search=CoverageSearch()
        for i in range(48):
            self.observe(search)
            if search.complete:break
            search.executed({"base_integral":[0,0,math.radians(8)]})
        self.assertTrue(search.complete)
        action,reason=search.propose()
        self.assertIsNone(action);self.assertIn("SWEEP_COMPLETE",reason)

    def test_invalid_or_vertical_view_does_not_certify_coverage(self):
        search=CoverageSearch()
        search.observe(0,np.eye(4),[[70,0,50],[0,70,50],[0,0,1]],100,1.,False)
        self.assertEqual(search.covered,set())
        search.observe(0,self.make_camera(),[[70,0,50],[0,70,50],[0,0,1]],100,.1,False)
        self.assertEqual(search.covered,set())

    def test_base_depth_obstacle_and_blind_backward_refused(self):
        model,state=fixture()
        floor=np.c_[np.linspace(.5,2,100),np.zeros(100),np.zeros(100)]
        guard=LocalDepthGuard(floor,model,state.q,{})
        self.assertTrue(guard.check(Action("base","forward"))[0])
        self.assertFalse(guard.check(Action("base","back"))[0])
        obstacles=np.vstack([floor,[[.38,0,.3]]])
        guard=LocalDepthGuard(obstacles,model,state.q,{})
        self.assertEqual(guard.check(Action("base","forward")),(False,"OBSERVED_BASE_OBSTACLE"))

    def test_depth_unknown_is_not_empty_free_space(self):
        model,state=fixture();guard=LocalDepthGuard(np.empty((0,3)),model,state.q,{})
        self.assertFalse(guard.check(Action("base","forward"))[0])


class OrchestrationTests(unittest.TestCase):
    def test_failed_coarse_approach_checks_feasible_middle_scale_before_micro(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        c.observe(grounded_evidence(),state,depths,receipt);h.stage="APPROACH"
        center=c.centers["right"];point=center+np.array([0.,0.,.12])
        c.target.update(valid=True,point_base_m=point.tolist(),distance_to_active_closing_center_m=.12)
        c.target["mean_contact_distance_m"]=.12
        coarse=Action("right","up","coarse");middle=Action("right","up","fine");micro=Action("right","up","micro")
        real=SafeServo.begin
        def reject_coarse(instance,action,current,carry=False):
            if action.part=="right" and action.move=="up" and action.scale=="coarse":
                return instance.abort("UNREACHABLE_OR_COLLISION_BLOCKED")
            return real(instance,action,current,carry)
        with patch.object(SafeServo,"begin",reject_coarse), patch.object(h,"palette",return_value=(HOLD,coarse,middle,micro)):
            allowed=c.candidates(state)
        tested=[Action(**r["action"]) for r in h.candidate_receipt["tested"]]
        self.assertIn(middle,allowed)
        self.assertLess(tested.index(coarse),tested.index(middle));self.assertLess(tested.index(middle),tested.index(micro))
        self.assertLessEqual(len(tested),25)

    def test_perception_does_not_receive_previous_target_or_action_scores(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        h.grounding={"target_uv":[.12345,.98765],"point_base_m":[7,8,9]}
        h.candidate_receipt={"tested":["OLD_ACTION_SCORES"]}
        h.search_context={"covered_heading_bins":[3,5]}
        h.last_action=Action("right","up")
        h.feedback=feedback();h.feedback["eef_delta_m"]["right"]=[0,0,.01]
        text=perception_context(h,state,SimpleNamespace(geometry={"head":{"robot_only":True}}))
        value=json.loads(text)
        self.assertEqual(value["current_goal"]["target"],"radio")
        self.assertEqual(value["last_executed_action"]["move"],"up")
        self.assertEqual(value["measured_last_motion_for_before_after_comparison"]["eef_delta_m"]["right"],[0,0,.01])
        for forbidden in ("target_uv","point_base_m","OLD_ACTION_SCORES","covered_heading_bins"):
            self.assertNotIn(forbidden,text)

    def test_alignment_keeps_body_fallback_only_for_unheld_pick(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        h.stage="ALIGN"
        self.assertIn(Action("base","forward","micro"),h.palette())
        self.assertNotIn(Action("base","forward","fine"),h.palette())
        h.hold_verified["left"]=True
        self.assertNotIn(Action("base","forward","micro"),h.palette())

    def test_far_contact_leaves_alignment_without_certifying_grasp(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        h.stage="ALIGN"
        c.observe(grounded_evidence(enclosed=True),state,depths,receipt)
        self.assertEqual(h.stage,"APPROACH")
        self.assertEqual(h.index,0)
        self.assertNotIn(Action("right","close"),h.palette())

    def test_alignment_samples_alternate_directions_and_fine_rotations(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        c.observe(grounded_evidence(),state,depths,receipt)
        h.stage="ALIGN"
        center=c.centers["right"]
        point=center+np.array([.04,.015,.03])
        c.target.update(valid=True,point_base_m=point.tolist(),distance_to_active_closing_center_m=float(np.linalg.norm(point-center)))
        c.candidates(state)
        tested=[Action(**r["action"]) for r in h.candidate_receipt["tested"]]
        self.assertIn(Action("right","forward","micro"),tested)
        self.assertTrue(any(a.part=="right" and a.scale=="fine" and a.move in ("forward","back","left","right","up","down")
                            and c._translation_direction(a,state)[2]>.99 for a in tested))
        for move in ("roll_plus","roll_minus","pitch_plus","pitch_minus","yaw_plus","yaw_minus"):
            self.assertIn(Action("right",move,"fine","tool"),tested)
            self.assertIn(Action("right",move,"micro","tool"),tested)
        self.assertLessEqual(len(tested),25)

    def test_unknown_depth_cannot_advance_to_grasp(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        h.stage="ALIGN"
        for depth in depths.values():depth[:]=np.inf
        c.observe(grounded_evidence(enclosed=True),state,depths,receipt)
        self.assertNotEqual(h.stage,"GRASP")
        allowed=c.candidates(state)
        self.assertEqual(allowed,(HOLD,))

    def test_valid_depth_still_does_not_certify_grasp(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        c.observe(grounded_evidence(enclosed=False),state,depths,receipt)
        self.assertFalse(any(h.hold_verified.values()));self.assertEqual(h.index,0)

    def test_preflight_preserves_latch_and_only_releases_allowed_commands(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        c.observe(grounded_evidence(),state,depths,receipt)
        allowed=c.candidates(state)
        np.testing.assert_array_equal(servo.grips,[-1,-1])
        self.assertLessEqual(len(c.harness.candidate_receipt["tested"]),c.max_preflights+1)
        self.assertTrue(all(a in h.palette() for a in allowed))
        self.assertNotIn(Action("right","close"),allowed)
        self.assertEqual(servo.status,"IDLE")

    def test_near_limit_micro_survives_rejected_fine(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        spec=copy.deepcopy(model.spec);spec["upper"][11]=.004
        model=RobotModel(spec);state=model.state(state.q,state.gripper,state.base_velocity)
        servo=SafeServo(model,state,[-1,-1]);c=GroundedController(model,servo,h)
        c.observe(grounded_evidence(),state,depths,receipt)
        c.target.update(valid=True,point_base_m=[.6,-.3,.8],distance_to_active_closing_center_m=.2)
        allowed=c.candidates(state)
        self.assertIn(Action("right","forward","micro"),allowed)
        self.assertNotIn(Action("right","forward","fine"),allowed)

    def test_replans_cannot_remove_goals_or_reset_safety_stops(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        obs=grounded_evidence(visible=False,view="none",target_uv=None)
        c.observe(obs,state,depths,receipt)
        goals=list(h.goals)
        for i in range(2):
            h.stop_reason="RECOVERY_BUDGET_EXHAUSTED";c.replan_needed=h.stop_reason
            c.apply_recovery(parse_recovery('{"strategy":"scan_left","visible_reason":"new views are needed"}'))
            self.assertEqual(h.goals,goals);self.assertEqual(h.index,0)
            self.assertFalse(any(h.hold_verified.values()))
        with self.assertRaises(ValueError):c.apply_recovery({"strategy":"scan_left","visible_reason":"again"})
        with self.assertRaises(ValueError):parse_recovery('{"strategy":"done","visible_reason":"looks fine"}')
        with self.assertRaises(ValueError):parse_recovery('{"strategy":"hold","visible_reason":"safe","goals":[]}')

    def test_fresh_coverage_delays_no_progress_but_does_not_reset_global_budget(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        invisible=grounded_evidence(visible=False,view="none",target_uv=None)
        for _ in range(30):
            h.observe(invisible,state,measured_progress={"new_search_coverage":True,"target_distance_m":math.inf})
        self.assertEqual(h.recoveries,0);self.assertEqual(h.index,0)
        for _ in range(9):h.observe(invisible,state,measured_progress={"new_search_coverage":False})
        self.assertGreater(h.recoveries,0)

    def test_metric_motion_static_object_and_camera_motion_cannot_pass(self):
        model,state=fixture();centers=model.grasp_centers(state.q)
        target={"valid":True,"views":[{"view":"right_wrist","valid":True}],"point_base_m":[.4,-.3,.8]}
        previous={"target":copy.deepcopy(target),"centers":copy.deepcopy(centers)}
        centers["right"][2]+=.01
        fb={"base_integral":[0,0,0]}
        self.assertIs(metric_co_motion(previous,target,state,centers,fb,"right"),False)
        target["point_base_m"][2]+=.01
        self.assertIs(metric_co_motion(previous,target,state,centers,fb,"right"),True)
        self.assertIsNone(metric_co_motion(previous,target,state,centers,{"base_integral":[0,0,.1]},"right"))

    def test_visible_target_recovery_strategy_is_actually_queued(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        c.observe(grounded_evidence(),state,depths,receipt)
        c.replan_needed="RECOVERY_BUDGET_EXHAUSTED"
        c.apply_recovery(parse_recovery('{"strategy":"scan_right","visible_reason":"need a different visible angle"}'))
        self.assertEqual(c.reposition,Action("base","yaw_minus","coarse"))
        self.assertEqual(c.reposition_left,1)
        self.assertIn(c.reposition,h.palette())

    def test_navigation_keeps_rotation_candidates_and_not_arm_actions(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        h.goals=[Goal("navigate","table","right","table close enough")]
        c.observe(grounded_evidence(),state,depths,receipt)
        c.candidates(state)
        tested=[Action(**row["action"]) for row in h.candidate_receipt["tested"]]
        self.assertIn(Action("base","yaw_plus"),tested)
        self.assertIn(Action("base","yaw_minus"),tested)
        self.assertTrue(all(a.part in ("base","all") for a in tested))


if __name__=="__main__":unittest.main()
