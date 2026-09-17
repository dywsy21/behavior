import copy
from dataclasses import asdict
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_robot.control import RobotState
from semantic_robot.v2.harness import Goal, TaskHarness, parse_plan
from semantic_robot.v2.kinematics import RobotModel, link_origin_jacobian
from semantic_robot.v2.protocol import Action, Evidence, HOLD, strict_json
from semantic_robot.v2.servo import SafeServo, bounded_ik, native_action, segment_distance
from semantic_robot.v2.vision import prepare_views, project


def fixture():
    q = np.zeros(18)
    poses = {"left": (np.array([.4,.3,.8]), np.array([0.,0.,0.,1.])),
             "right": (np.array([.4,-.3,.8]), np.array([0.,0.,0.,1.])),
             "torso": (np.array([0.,0.,.8]), np.array([0.,0.,0.,1.]))}
    jac = {name: np.zeros((6, 18)) for name in poses}
    jac["left"][:,4:10] = np.eye(6)
    jac["right"][:,11:17] = np.eye(6)
    for name in poses:
        jac[name][:3,:3] = np.eye(3)
    for view in ("head", "left_wrist", "right_wrist"):
        poses["camera_"+view] = (np.array([0.,0.,2.]), np.array([0.,0.,0.,1.]))
        jac["camera_"+view] = np.zeros((6,18))
    meta = {"cameras": {view: {"K": [[70,0,50],[0,70,50],[0,0,1]], "width":100, "height":100}
                        for view in ("head", "left_wrist", "right_wrist")}}
    model = RobotModel.from_reference(q, np.full(18,-2.), np.full(18,2.), poses, jac, meta)
    return model, model.state(q, np.array([.05,.05]), np.zeros(3))


def evidence(**kwargs):
    fields = dict(visible=True, view="right_wrist", target_uv=[.5,.5], enclosed=False,
                  co_moving=None, supported=None, effect=None, hazard="none", note="Visible evidence.")
    fields.update(kwargs)
    return Evidence.parse(json.dumps(fields))


def feedback(status="TARGET_REACHED", delta=.01):
    return {"status": status, "eef_delta_m": {"left":[0,0,0], "right":[0,0,delta]}}


class ProtocolTests(unittest.TestCase):
    def test_roundtrip_complete_commands(self):
        for part in ("left", "right", "both"):
            for move in ("forward", "up", "open", "close", "hold"):
                action = Action(part,move)
                self.assertEqual(Action.parse(action.text()), action)

    def test_unknown_ambiguous_and_extra_fields_rejected(self):
        for text in ('L|BOTH OPEN FINE', '{"part":"right"}', '[]',
                     '{"part":"right","part":"left","move":"up","scale":"fine","frame":"base"}',
                     '{"part":"right","move":"up","scale":10,"frame":"base"}',
                     '{"part":"right","move":"up","scale":"fine","frame":"world"}'):
            with self.subTest(text=text), self.assertRaises(ValueError): Action.parse(text)

    def test_bounds_and_body_frame(self):
        for fields in (("base","up"), ("torso","left"), ("all","close"), ("right","open","micro"), ("both","up","fine","tool")):
            with self.assertRaises(ValueError): Action(*fields)
        self.assertEqual(Action("right","up","micro").amount(), .002)
        self.assertEqual(Action("right","up","coarse").amount(True), .01)
        with self.assertRaises(ValueError): Action("right","roll_plus").amount(True)

    def test_evidence_strict_nullable_and_localization(self):
        self.assertIsNone(evidence().co_moving)
        for changes in ({"visible":"yes"}, {"target_uv":[float("nan"),.5]}, {"view":"world"},
                        {"visible":False}, {"enclosed":1}, {"target_uv":[-1,.5]}):
            with self.assertRaises(ValueError): evidence(**changes)

    def test_no_privileged_fields(self):
        fields = asdict(evidence()); fields["object_pose_gt"] = [1,2,3]
        with self.assertRaises(ValueError): Evidence.parse(json.dumps(fields))

    def test_plan_schema(self):
        goal = Goal("pick", "radio", "right", "moves with hand")
        self.assertEqual(parse_plan(json.dumps([asdict(goal)])), [goal])
        for text in ('[]', '[{"kind":"teleport"}]', '{}'):
            with self.assertRaises(ValueError): parse_plan(text)

    def test_plan_resource_ownership(self):
        for goals in ([Goal("place","shelf","right","supported")],
                      [Goal("pick","radio","right","held"),Goal("pick","bowl","right","held")]):
            with self.assertRaises(ValueError): parse_plan(json.dumps([asdict(g) for g in goals]))


class KinematicsTests(unittest.TestCase):
    def test_com_jacobian_translates_point_in_rotated_link(self):
        model, state = fixture()
        rng = np.random.default_rng(52)
        local_com = np.array([.027,-.018,.12])
        for _ in range(8):
            quat = Rotation.random(random_state=rng).as_quat()
            offset = Rotation.from_quat(quat).apply(local_com)
            true = rng.normal(size=(6,18))
            measured = true.copy()
            measured[:3] += np.cross(true[3:].T, offset).T
            np.testing.assert_allclose(link_origin_jacobian(measured,quat,local_com),true,atol=1e-14)
            np.testing.assert_array_equal(measured[3:],true[3:])

    def test_com_correction_recovers_fk_not_just_self_consistent_jacobian(self):
        pose = (np.array([.4,.1,.8]), np.array([0.,0.,0.,1.]))
        true = np.zeros((6,18)); true[5,4] = 1
        true[:3,4] = [-.1,.4,0]
        com = np.array([.03,.02,0.])
        measured = true.copy(); measured[:3] += np.cross(true[3:].T,com).T
        poses = {name:pose for name in ("left","right","torso")}
        corrected = link_origin_jacobian(measured,pose[1],com)
        model = RobotModel.from_reference(np.zeros(18),np.full(18,-2.),np.full(18,2.),poses,
                                         {name:corrected for name in poses})
        q = np.zeros(18); q[4] = .6
        actual = Rotation.from_rotvec([0,0,.6]).apply(pose[0])
        np.testing.assert_allclose(model.poses(q)["left"][0],actual,atol=1e-12)

    def test_fk_reference_roundtrip(self):
        model, state = fixture()
        restored = RobotModel(json.loads(json.dumps(model.spec)))
        self.assertEqual(restored.sha, model.sha)
        for name in state.poses:
            np.testing.assert_allclose(restored.poses(state.q)[name][0],state.poses[name][0])

    def test_analytic_jacobian_finite_difference_multiple_poses(self):
        model, state = fixture()
        for seed in range(4):
            q = np.random.default_rng(seed).uniform(-.2,.2,18)
            self.assertLess(max(model.finite_difference_error(q).values()), 1e-7)

    def test_only_selected_arm_moves(self):
        model, state = fixture(); q = state.q.copy(); q[11] += .02
        changed = model.poses(q)
        np.testing.assert_allclose(changed["right"][0]-state.poses["right"][0], [.02,0,0], atol=1e-10)
        np.testing.assert_array_equal(changed["left"][0],state.poses["left"][0])

    def test_trunk_affects_both_arms(self):
        model, state = fixture(); q = state.q.copy(); q[2] += .02
        for name, (p, _) in model.poses(q).items():
            self.assertAlmostEqual(p[2]-state.poses[name][0][2], .02)

    def test_projection_usd_camera_not_mirrored(self):
        K = np.array([[100,0,50],[0,100,50],[0,0,1]])
        np.testing.assert_allclose(project([.1,.2,-1],np.eye(4),K), [60,30])
        self.assertIsNone(project([0,0,1],np.eye(4),K))

    def test_before_after_guide_contract(self):
        model, state = fixture()
        images = {view+"_rgb":np.zeros((100,100,3),dtype=np.uint8) for view in ("head","left_wrist","right_wrist")}
        current = prepare_views(images,model,state.q)
        self.assertEqual(len(current.images),6)
        next_views = prepare_views(images,model,state.q,current.current_raw)
        self.assertEqual(len(next_views.images),9)
        self.assertEqual(sum("PREVIOUS" in x for x in next_views.labels),3)
        np.testing.assert_array_equal(next_views.current_raw["head"],images["head_rgb"])
        images["head_rgb"] = images["head_rgb"][:90]
        with self.assertRaises(ValueError): prepare_views(images,model,state.q)


class ServoTests(unittest.TestCase):
    def test_real_folded_arm_executes_the_plan_that_passed_preflight(self):
        model = RobotModel(json.loads((Path(__file__).parent/"fixtures/r1pro_folded_fk.json").read_text()))
        state = model.state(model.reference,np.full(2,.05),np.zeros(3))
        servo = SafeServo(model,state)
        self.assertTrue(servo.begin(Action("right","up","fine"),state))
        self.assertGreater(servo.total_ticks,18)
        self.assertLessEqual(servo.total_ticks,40)
        path = [state.q.copy()]
        while not servo.done and servo.ticks<servo.total_ticks:
            a = servo.next_action(state)
            q = np.r_[a[3:14],a[15:22]]; path.append(q)
            state = model.state(q,state.gripper,np.zeros(3))
        result = servo.finish(state)
        self.assertEqual(result["status"],"TARGET_REACHED")
        self.assertLess(result["target_error_m"]["right"],servo.pos_tolerance)
        self.assertGreater(result["eef_delta_m"]["right"][2],.01-servo.pos_tolerance)
        self.assertLessEqual(np.abs(np.diff(path,axis=0)).max(),.025001)

    def test_slow_joint_limit_cannot_silently_exceed_action_horizon(self):
        from semantic_robot.v2.servo import ServoLimits
        model,state=fixture()
        servo=SafeServo(model,state,limits=ServoLimits(joint_tick=.00001))
        self.assertFalse(servo.begin(Action("right","up"),state))
        self.assertEqual(servo.status,"DURATION_LIMIT_EXCEEDED")
        self.assertEqual(servo.ticks,0)

    def simulate(self, action, carry=False):
        model, state = fixture(); servo = SafeServo(model,state)
        self.assertTrue(servo.begin(action,state,carry))
        actions = []
        while not servo.done and servo.ticks < servo.total_ticks:
            a = servo.next_action(state); actions.append(a)
            q = np.r_[a[3:14],a[15:22]]
            state = model.state(q, np.array([.05 if a[14]>0 else 0,.05 if a[22]>0 else 0]),a[:3]*[.75,.75,1.])
        return model, state, servo, np.asarray(actions), servo.finish(state)

    def test_bounded_solver_reallocates_instead_of_clipping(self):
        result = bounded_ik(np.array([[1.,1.]]),np.array([1.8]),np.zeros(2),np.zeros(2),np.array([.1,2.]),step=2.)
        self.assertLessEqual(result[0],.1+1e-10)
        self.assertGreater(result[1],1.6)

    def test_native23_exact_order(self):
        q = np.arange(18)/100
        a = native_action(q,[-1,1])
        np.testing.assert_allclose(np.r_[a[3:14],a[15:22]],q)
        self.assertEqual(a[14],-1); self.assertEqual(a[22],1)

    def test_all_directions_and_magnitudes(self):
        for move in ("forward","back","left","right","up","down"):
            for scale in ("micro","fine","coarse"):
                with self.subTest(move=move,scale=scale):
                    _, _, servo, actions, result = self.simulate(Action("right",move,scale))
                    self.assertEqual(result["status"],"TARGET_REACHED")
                    self.assertLessEqual(abs(np.diff(actions[:,15:22],axis=0)).max(),.026)

    def test_trunk_arms_coupled(self):
        _, _, _, _, result = self.simulate(Action("torso","up"))
        self.assertEqual(result["status"],"TARGET_REACHED")
        self.assertLess(np.linalg.norm(result["eef_delta_m"]["right"]),.002)

    def test_camera_relative_left_is_not_base_left(self):
        _, _, _, _, result = self.simulate(Action("right","left","fine","head"))
        self.assertLess(result["eef_delta_m"]["right"][0],-.009)
        self.assertAlmostEqual(result["eef_delta_m"]["right"][1],0.,places=4)

    def test_unreachable_rejected_before_motion(self):
        model,state=fixture()
        spec=copy.deepcopy(model.spec); spec["upper"][11]=.0011
        servo=SafeServo(RobotModel(spec),state)
        self.assertFalse(servo.begin(Action("right","forward"),state))
        self.assertEqual(servo.ticks,0)
        self.assertIn("UNREACHABLE",servo.status)

    def test_stalled_feedback_not_executed_success(self):
        model,state=fixture(); servo=SafeServo(model,state)
        servo.begin(Action("right","up"),state)
        while not servo.done and servo.ticks<servo.total_ticks:
            servo.next_action(state)
        result=servo.finish(state)
        self.assertNotEqual(result["status"],"TARGET_REACHED")
        np.testing.assert_array_equal(servo.safe_hold(state)[:3],0)

    def test_divergence_stops(self):
        model,state=fixture(); servo=SafeServo(model,state)
        servo.begin(Action("right","up"),state)
        moved=copy.deepcopy(state); moved.poses["right"][0][0]+=.1
        for _ in range(3): action=servo.next_action(moved)
        self.assertTrue(servo.done)
        self.assertEqual(servo.status,"TRACKING_DIVERGED")
        np.testing.assert_allclose(action[:3],0)

    def test_carry_and_gripper_latch(self):
        model,state=fixture(); servo=SafeServo(model,state,[-1,-1])
        servo.begin(HOLD,state,carry=True)
        a=servo.next_action(state)
        self.assertEqual(a[14],-1); self.assertEqual(a[22],-1)
        with self.assertRaises(ValueError): servo.begin(Action("right","roll_plus"),state,True)

    def test_base_pulse_stops_and_integrates(self):
        _,_,_,actions,result=self.simulate(Action("base","forward","fine"))
        self.assertAlmostEqual(actions[:,0].sum()*.75/30,.06,places=6)
        np.testing.assert_array_equal(actions[-1,:3],0)
        self.assertEqual(result["status"],"TARGET_REACHED")

    def test_capsule_degenerate_and_crossing(self):
        self.assertAlmostEqual(segment_distance(np.zeros(3),np.zeros(3),np.ones(3),np.ones(3)),np.sqrt(3))
        self.assertAlmostEqual(segment_distance(np.array([-1,0,0]),np.array([1,0,0]),np.array([0,-1,0]),np.array([0,1,0])),0.)


class HarnessTests(unittest.TestCase):
    def make(self):
        return TaskHarness([Goal("pick","radio","right","moves with right hand"),Goal("press","radio power button","left","power indicator changes")])

    def test_search_disallows_manipulation_and_never_done(self):
        h=self.make()
        self.assertTrue(all(a.part in ("all","base","torso") for a in h.palette()))
        with self.assertRaises(ValueError): h.authorize(Action("left","close"))

    def test_stage_and_active_hand(self):
        h=self.make(); _,s=fixture()
        h.observe(evidence(),s); self.assertEqual(h.stage,"APPROACH")
        h.observe(evidence(),s); self.assertEqual(h.stage,"ALIGN")
        self.assertFalse(any(a.part in ("left","both") for a in h.palette()))
        self.assertFalse(any(a.move=="close" for a in h.palette()))
        h.observe(evidence(enclosed=True),s); self.assertEqual(h.stage,"GRASP")
        self.assertIn(Action("right","close"),h.palette())

    def test_close_does_not_complete_pick(self):
        h=self.make(); h.stage="GRASP"
        h.executed(Action("right","close"),feedback(delta=0))
        self.assertEqual(h.stage,"VERIFY_GRASP"); self.assertEqual(h.index,0)

    def test_empty_width_vetoes_hallucinated_hold(self):
        h=self.make(); _,s=fixture(); s.gripper[1]=0
        h.stage="VERIFY_GRASP"; h.feedback=feedback()
        h.observe(evidence(enclosed=True,co_moving=True),s)
        self.assertEqual(h.stage,"RECOVER"); self.assertEqual(h.index,0)
        self.assertFalse(h.hold_verified["right"])

    def test_width_alone_and_stationary_claim_not_hold(self):
        h=self.make(); _,s=fixture(); s.gripper[1]=.02
        h.stage="VERIFY_GRASP"; h.feedback=feedback(delta=0)
        h.observe(evidence(enclosed=True,co_moving=True),s)
        self.assertEqual(h.index,0)

    def test_temporal_evidence_plus_motion_and_width(self):
        h=self.make(); _,s=fixture(); s.gripper[1]=.02
        h.stage="VERIFY_GRASP"; h.observation=evidence(enclosed=True)
        h.feedback=feedback(delta=.01)
        h.observe(evidence(enclosed=True,co_moving=True),s)
        self.assertEqual(h.index,1); self.assertTrue(h.hold_verified["right"])
        self.assertEqual(h.stage,"SEARCH"); self.assertIsNone(h.stop_reason)

    def test_failure_and_finite_recovery(self):
        h=self.make(); _,s=fixture()
        for _ in range(4):
            h.feedback=feedback("JOINT_LIMIT_STALL"); h.observe(evidence(),s)
        self.assertEqual(h.stop_reason,"RECOVERY_BUDGET_EXHAUSTED")
        self.assertEqual(h.palette(),(HOLD,))

    def test_noop_loop_does_not_run_48_times(self):
        h=self.make()
        for _ in range(4): h.executed(HOLD,feedback(delta=0))
        self.assertEqual(h.stage,"RECOVER")

    def test_valid_repeated_moves_are_not_blanket_banned(self):
        h=self.make(); h.stage="APPROACH"
        for _ in range(4): h.executed(Action("right","forward"),feedback())
        self.assertEqual(h.stage,"APPROACH")

    def test_recovery_requires_new_successful_recovery_action(self):
        h=self.make(); _,s=fixture()
        h.executed(Action("right","forward"),feedback("TRACKING_FAILED"))
        h.observe(evidence(),s); self.assertEqual(h.stage,"RECOVER")
        h.observe(evidence(),s); self.assertEqual(h.stage,"RECOVER")
        h.executed(Action("right","open"),feedback(delta=0))
        h.observe(evidence(),s); self.assertEqual(h.stage,"ALIGN")

    def test_approach_allows_smaller_step_after_reachability_failure(self):
        h=self.make(); h.stage="APPROACH"; h.observation=evidence()
        for scale in ("micro","fine","coarse"):
            h.authorize(Action("right","forward",scale,"base"))
        with self.assertRaises(ValueError):
            h.authorize(Action("left","forward","micro","base"))

    def test_each_recovery_attempt_has_a_fresh_bounded_window(self):
        h=TaskHarness([Goal("navigate","table","both","table visible")]); _,s=fixture()
        unseen=evidence(visible=False,view="none",target_uv=None)
        h.recover("SEARCH_NO_PROGRESS")
        for attempt in range(1,4):
            self.assertEqual(h.recoveries,attempt)
            self.assertEqual(h.stage_age,0)
            for _ in range(3):
                h.observe(unseen,s)
                self.assertEqual(h.recoveries,attempt)
                self.assertIsNone(h.stop_reason)
            h.observe(unseen,s)
        self.assertEqual(h.recoveries,4)
        self.assertEqual(h.stop_reason,"RECOVERY_BUDGET_EXHAUSTED")
        self.assertEqual(h.palette(),(HOLD,))

    def test_effect_confirmation_holds_instead_of_pushing_again(self):
        h=TaskHarness([Goal("press","button","left","indicator changes")]); _,s=fixture()
        h.stage="INTERACT"; h.executed(Action("left","forward","micro"),feedback())
        h.observe(evidence(effect=True),s)
        self.assertEqual(h.stage,"VERIFY_EFFECT"); self.assertEqual(h.palette(),(HOLD,))
        h.executed(HOLD,feedback(delta=0)); h.observe(evidence(effect=True),s)
        self.assertEqual(h.stop_reason,"PLAN_EXHAUSTED_NOT_OFFICIAL_SUCCESS")

    def test_support_requires_confirmation_before_open(self):
        h=TaskHarness([Goal("place","shelf","right","supported")]); _,s=fixture(); h.stage="ALIGN"
        h.observe(evidence(supported=True),s)
        self.assertEqual(h.stage,"VERIFY_SUPPORT"); self.assertEqual(h.palette(),(HOLD,))
        h.executed(HOLD,feedback(delta=0)); h.observe(evidence(supported=True),s)
        self.assertEqual(h.stage,"RELEASE")

    def test_invisible_target_no_fabricated_uv(self):
        h=self.make(); _,s=fixture(); h.stage="ALIGN"
        h.observe(evidence(visible=False,view="none",target_uv=None,hazard="occluded"),s)
        self.assertEqual(h.stage,"SEARCH")

    def test_place_requires_release_and_repeated_support(self):
        h=TaskHarness([Goal("place","shelf","right","rests on shelf")]); _,s=fixture()
        h.stage="VERIFY_PLACE"
        h.observe(evidence(enclosed=False,supported=True),s)
        self.assertIsNone(h.stop_reason)
        h.observe(evidence(enclosed=False,supported=True),s)
        self.assertEqual(h.stop_reason,"PLAN_EXHAUSTED_NOT_OFFICIAL_SUCCESS")


if __name__=="__main__":
    unittest.main()
