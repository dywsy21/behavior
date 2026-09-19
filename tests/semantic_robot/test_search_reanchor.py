import ast
import copy
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.search import CoverageSearch
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.substep_odometry import SubstepMotion
from semantic_robot.v2.wall_budget import expired
from test_substep_odometry import measured
from test_v2 import fixture, evidence, feedback


class SearchReanchorTests(unittest.TestCase):
    def setup(self):
        self.model, self.state = fixture()
        self.model.spec["metadata"].update(grasp_region_reference_gripper_m=[.05,.05],
            grasp_region_reference_fully_open={"left":True,"right":True})
        self.h = GroundedHarness([Goal("navigate","table","both","visible" )])
        self.h.bind_reference("world","test-language")
        self.h.observation = evidence(visible=False,view="none",target_uv=None)
        self.c = GroundedController(self.model, SafeServo(self.model,self.state,[1,1]), self.h,
            visual_odometry=True, odometry_estimator="rgbd_joint", search_motion_recovery=True,
            approach_progress=True, grasp_motion=True, persistent_grasp_tracks=True)
        self.images={"head_rgb":np.zeros((3,100,100),np.uint8)}
        self.depths={"head":np.ones((100,100),np.float32)}
        base=Mock();base.observe.side_effect=[{"valid":True,"initial":True},measured(yaw=.02),{"valid":False,"reason":"LOW_TEXTURE"}]
        self.old=SubstepMotion(base);self.c.motion=self.old
        self.old.observe(self.images,self.depths,self.model,self.state.q)
        self.old.begin(24)
        self.old.sample(self.images,self.depths,self.model,self.state.q,30)
        self.old.sample(self.images,self.depths,self.model,self.state.q,36)
        self.failed=self.old.finish(36,interrupted=True)
        self.c.executed(Action("base","yaw_plus","coarse"),feedback("INTERRUPTED"))
        self.h.stop_reason="VISUAL_ODOMETRY_UNCERTAIN"
        self.c.search.goal_index=0
        self.c.search.heading=2.;self.c.search.xy[:]=[.5,.1]
        self.c.search.covered={1,2,3};self.c.search.nodes=[{"xy":[0,0],"bins":[1]}]
        self.c.search.search_travel_m=.4;self.c.search.travel_m=.4
        self.c.search.search_rotation_rad=2.;self.c.search.rotation_rad=2.
        self.c.search.last_seen_heading=1.;self.c.search.direction=-1
        self.c.previous={"stale":True};self.c.target={"valid":True,"stale":True}
        self.c.grasp_verifier.previous={"stale":True};self.c.grasp_verifier.tracks={"right":{"stale":True}}
        self.issued=[];self.captures=[];self.factories=[]

    def attempt(self, receipts=None, **changes):
        def factory():
            base=Mock();base.observe.side_effect=[{"valid":True,"initial":True},*(receipts or [measured(),measured()])]
            result=SubstepMotion(base);self.factories.append(result);return result
        def issue():
            self.issued.append(1);return False
        kwargs=dict(controls=36,action_limit=60,terminal=False,deadline=None,
            observe=lambda label:(self.images,self.depths,{"raw":"fixture"}),
            state_now=lambda:self.state,issue_hold=issue,
            save_snapshot=lambda control,*args:self.captures.append(control),motion_factory=factory)
        kwargs.update(changes)
        return self.c.search_recovery.attempt(self.c,self.failed,**kwargs)

    def test_success_opens_only_new_epoch_without_old_motion_or_goal_credit(self):
        self.setup(); history=copy.deepcopy(list(self.h.history));pending=copy.deepcopy(self.c.pending_motion)
        result,snapshot=self.attempt()
        self.assertTrue(result["valid"]);self.assertEqual(len(self.issued),12)
        self.assertEqual(self.captures,[36,42,48]);self.assertEqual(snapshot[0],48)
        self.assertEqual(result["unknown_failed_span"],[24,36])
        self.assertNotIn("body_delta",result)
        self.assertTrue(self.old.failed);self.assertFalse(self.old.pending["valid"])
        self.assertIsNone(self.c.pending_motion);self.assertIsNot(self.c.motion,self.old)
        self.assertIsNone(self.c.previous);self.assertFalse(self.c.target["valid"])
        self.assertIsNone(self.c.grasp_verifier.previous);self.assertEqual(self.c.grasp_verifier.tracks,{})
        self.assertEqual(list(self.h.history),history);self.assertEqual(self.h.executions,1)
        self.assertEqual(self.h.index,0);self.assertEqual(self.h.completed,[])
        self.assertIsNone(self.h.stop_reason);self.assertIsNone(self.h.observation)
        self.assertEqual(self.c.search.heading,0);np.testing.assert_array_equal(self.c.search.xy,[0,0])
        self.assertEqual(self.c.search.covered,set());self.assertEqual(self.c.search.nodes,[])
        self.assertIsNone(self.c.search.last_seen_heading);self.assertEqual(self.c.search.direction,-1)
        self.assertEqual(self.c.search.travel_m,.4);self.assertEqual(self.c.search.rotation_rad,2.)
        self.assertEqual(self.c.search.search_travel_m,.4);self.assertEqual(self.c.search.search_rotation_rad,2.)
        self.assertEqual(self.c.search.unknown_budget_travel_m,.18)
        self.assertEqual(self.c.search.unknown_budget_rotation_rad,.30)
        self.assertFalse(self.c.search.observe(0,self.model.forward(self.state.q,"camera_head"),
            self.model.spec["metadata"]["cameras"]["head"]["K"],100,1.,False))
        receipt=self.c.update_motion(self.images,self.depths,self.state)
        self.assertTrue(receipt["initial"]);self.assertEqual(self.c.search.heading,0)
        self.assertEqual(pending["status"],"INTERRUPTED")

    def test_ineligible_has_no_sensor_call_no_control_and_no_state_reset(self):
        mutations=[lambda:setattr(self.h,"stage","APPROACH"),
            lambda:setattr(self.h,"stop_reason","SELF_COLLISION"),
            lambda:setattr(self.h,"observation",evidence()),
            lambda:setattr(self.h,"observation",evidence(visible=False,view="none",target_uv=None,hazard="collision")),
            lambda:self.h.target_references.update({0:"unknown"}),
            lambda:self.h.pending_grasp.update(right=True),
            lambda:self.h.pending_grasp.update(right=None),
            lambda:self.h.hold_verified.update(left=True),
            lambda:self.h.possible_contact_after_close.update(left=True),
            lambda:self.h.held.update(right="radio"),
            lambda:self.h.completed.append({"some":"goal"}),
            lambda:setattr(self.c.search_recovery,"ever_closed",True),
            lambda:setattr(self.c.search_recovery,"attempts",2),
            lambda:self.c.servo.grips.__setitem__(0,-1),
            lambda:self.state.gripper.__setitem__(0,.02),
            lambda:self.model.spec["metadata"].pop("grasp_region_reference_gripper_m"),
            lambda:self.model.spec["metadata"]["grasp_region_reference_fully_open"].update(right=1),
            lambda:setattr(self.h,"last_action",Action("right","up")),
            lambda:setattr(self.c,"reposition_left",1),
            lambda:self.c.approach_monitor.blocked.update(forward={}),
            lambda:self.failed.update(reason="ACTION_INTERRUPTED_NO_FULL_MOTION_CERTIFICATE"),
            lambda:setattr(self.c,"pending_motion",None),
            lambda:self.old.pending.update(control_end=35)]
        for i,mutation in enumerate(mutations):
            with self.subTest(i=i):
                self.setup();mutation();old_stop=self.h.stop_reason
                result,snapshot=self.attempt()
                self.assertFalse(result["valid"]);self.assertIsNone(snapshot)
                self.assertEqual(self.issued,[]);self.assertEqual(self.captures,[])
                self.assertIs(self.c.motion,self.old);self.assertEqual(self.h.stop_reason,old_stop)

    def test_full_probe_budget_clock_and_termination_are_strict(self):
        for change in ({"controls":35},{"controls":36.0},{"action_limit":47},{"terminal":True},{"terminal":None},{"deadline":0}):
            self.setup();result,_=self.attempt(**change)
            self.assertFalse(result["valid"]);self.assertEqual(self.issued,[])

    def test_probe_quality_failure_keeps_stop_unknown_chain_and_attempt_spend(self):
        self.setup();result,_=self.attempt([measured(),{"valid":False,"reason":"LOW_TEXTURE"}])
        self.assertFalse(result["valid"]);self.assertEqual(len(self.issued),12)
        self.assertFalse(result["chain"]["valid"]);self.assertEqual(self.c.search.local_epoch,0)
        self.assertIs(self.c.motion,self.old);self.assertEqual(self.c.search_recovery.attempts,1)
        self.assertEqual(self.h.stop_reason,"VISUAL_ODOMETRY_UNCERTAIN")

    def test_valid_but_moving_hold_never_reanchors(self):
        for delta in (measured(x=.009),measured(yaw=.015)):
            self.setup();result,_=self.attempt([delta,delta]);self.assertFalse(result["valid"])
            self.assertEqual(result["reason"],"HOLD_NOT_OBSERVED_AT_REST")
            self.assertEqual(self.c.search.local_epoch,0)

    def test_mid_probe_budget_grip_or_terminal_cannot_resume(self):
        for reason in ("terminal","grip","deadline"):
            self.setup();clock=[0.]
            def issue():
                self.issued.append(1)
                if reason=="grip":self.state.gripper[0]=.03
                if reason=="deadline":clock[0]=2.
                return reason=="terminal"
            with patch("semantic_robot.v2.wall_budget.time.perf_counter",side_effect=lambda:clock[0]):
                result,_=self.attempt(issue_hold=issue,deadline=1.)
            self.assertFalse(result["valid"]);self.assertEqual(len(self.issued),1)
            self.assertIs(self.c.motion,self.old);self.assertEqual(self.c.search.local_epoch,0)

    def test_unknown_budget_tax_not_forgotten_or_presented_as_measured_pose(self):
        search=CoverageSearch();search.search_travel_m=1.1;search.travel_m=1.1
        search.new_local_epoch(0,6)
        self.assertEqual(search.propose(),(None,"SEARCH_TRAVEL_BUDGET"))
        self.assertEqual(search.context()["travel_m"],1.1)
        self.assertEqual(search.context()["unknown_motion_control_spans"],[[0,6]])
        search.new_local_epoch(30,36)
        with self.assertRaises(ValueError):search.new_local_epoch(60,66)

    def test_close_latch_and_default_off(self):
        self.setup();self.c.pending_motion=None;self.c.executed(Action("right","close"),feedback())
        self.assertTrue(self.c.search_recovery.ever_closed)
        model,state=fixture();h=GroundedHarness([Goal("pick","x","right","x")])
        self.assertIsNone(GroundedController(model,SafeServo(model,state),h).search_recovery)
        with self.assertRaises(ValueError):GroundedController(model,SafeServo(model,state),h,search_motion_recovery=True)

    def test_runner_records_controls_and_all_cameras_and_rebinds_new_clock(self):
        path=Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py"
        tree=ast.parse(path.read_text())
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
            ast.unparse(n.test)=="controller and controller.search_recovery is not None and motion_fault and (not budget_interrupted)")
        text=ast.unparse(block)
        for required in ("nonlocal controls","controls += 1","action23","save_reanchor_snapshot",
                         "substep_motion = controller.motion","saved_end_snapshot = new_snapshot","previous = None"):
            self.assertIn(required,text)
        snapshot=next(n for n in ast.walk(block) if isinstance(n,ast.FunctionDef) and n.name=="save_reanchor_snapshot")
        self.assertIn("('head', 'left_wrist', 'right_wrist')",ast.unparse(snapshot))

    def test_actual_runner_recovery_block_issues_exact_hold_and_reuses_final_snapshot(self):
        self.setup()
        tree=ast.parse((Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text())
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
            ast.unparse(n.test)=="controller and controller.search_recovery is not None and motion_fault and (not budget_interrupted)")
        function=ast.parse("def actual():\n controls=36\n pass").body[0]
        function.body=[function.body[0],block,*ast.parse("return controls, previous, saved_end_snapshot, substep_motion").body]
        code=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),"actual_runner_recovery","exec")
        images={v+"_rgb":self.images["head_rgb"] for v in ("head","left_wrist","right_wrist")}
        depths={v:self.depths["head"] for v in ("head","left_wrist","right_wrist")}
        trace=StringIO();commands=[]
        def base_factory(kind):
            self.assertEqual(kind,"rgbd_joint")
            base=Mock();base.observe.side_effect=[{"valid":True,"initial":True},measured(),measured()]
            return base
        def step(command,render=True):commands.append(command.copy());return False
        with tempfile.TemporaryDirectory() as folder:
            ns={"controller":self.c,"manager":self.h,"servo":self.c.servo,"directory":Path(folder),
                "motion_fault":True,"budget_interrupted":False,"substep_result":self.failed,
                "action_control_limit":60,"deadline":None,"terminal":False,"decision":19,
                "state_now":lambda:self.state,"observation_now":lambda label:(images,depths,{}),
                "step":step,"trace":trace,"capture":lambda label:None,"expired":expired,
                "Image":Image,"np":np,"json":json,"row":{},
                "write":lambda p,value:p.write_text(json.dumps(value))}
            with patch("semantic_robot.v2.search_reanchor.RGBDMotion",side_effect=base_factory):
                exec(code,ns);controls,previous,snapshot,motion=ns["actual"]()
            self.assertEqual(controls,48);self.assertIsNone(previous);self.assertEqual(snapshot[0],48)
            self.assertIs(motion,self.c.motion);self.assertIsNot(motion,self.old)
            self.assertEqual(len(commands),12)
            for command in commands:
                np.testing.assert_array_equal(command[:3],[0,0,0])
                np.testing.assert_array_equal(command[[14,22]],[1,1])
            rows=[json.loads(line) for line in trace.getvalue().splitlines()]
            self.assertEqual([r["control"] for r in rows],list(range(37,49)))
            self.assertTrue(all(r["reanchor_after_decision"]==19 and "action23" in r for r in rows))
            self.assertEqual(len(list(Path(folder).rglob("*_RAW.png"))),9)
            self.assertTrue(json.loads((Path(folder)/"search_reanchor/receipt.json").read_text())["valid"])

    def test_actual_runner_option_guard_has_no_implicit_prefix_or_estimator_fallback(self):
        tree=ast.parse((Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text())
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
            ast.unparse(n.test).startswith("args.search_motion_recovery and"))
        code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),"runner_recovery_options","exec")
        good=dict(search_motion_recovery=True,robot_geometry_guards=True,held_object_inspection=True,
                  odometry_estimator="rgbd_joint",odometry_substep_controls=6,prefix=0,replay_prefix_spec=None)
        exec(code,{"args":SimpleNamespace(**good),"grounded":True})
        for key,value in (("robot_geometry_guards",False),("held_object_inspection",False),
                          ("odometry_estimator","pnp"),("odometry_substep_controls",0),
                          ("prefix",1),("replay_prefix_spec","saved")):
            with self.subTest(key=key),self.assertRaises(ValueError):
                exec(code,{"args":SimpleNamespace(**{**good,key:value}),"grounded":True})
        with self.assertRaises(ValueError):exec(code,{"args":SimpleNamespace(**good),"grounded":False})


if __name__=="__main__":unittest.main()
