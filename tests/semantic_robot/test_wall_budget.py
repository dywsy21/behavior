from io import BytesIO
import ast
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semantic_robot.v2.policy import VLMPolicy,call_service
from semantic_robot.v2.protocol import HOLD
from semantic_robot.v2.substep_odometry import SubstepMotion
from semantic_robot.v2.wall_budget import WallTimeBudgetReached,expired,require_time,stop_before_motion


class WallBudgetTests(unittest.TestCase):
    def test_no_deadline_or_finite_boundary(self):
        self.assertIsNone(require_time(None));self.assertFalse(expired(None))
        with patch("semantic_robot.v2.wall_budget.time.perf_counter",return_value=10):
            self.assertEqual(require_time(12),2)
            self.assertTrue(expired(10))
            with self.assertRaises(WallTimeBudgetReached):require_time(9)
        with self.assertRaises(ValueError):require_time(float("nan"))

    def test_expired_policy_does_not_spend_another_call(self):
        p=object.__new__(VLMPolicy);p.calls=0;p.max_calls=5;p.deadline=0
        with patch("semantic_robot.v2.policy.call_service") as call:
            with self.assertRaises(WallTimeBudgetReached):p._call("act")
            self.assertEqual(p.calls,0);call.assert_not_called()

    def test_timeout_uses_remaining_after_image_encoding(self):
        clock=SimpleNamespace(now=10.)
        class Image:
            def save(self,file,format):file.write(b"fixture");clock.now=14.
        bundle=SimpleNamespace(labels=["head"],images=[Image()])
        response=BytesIO(json.dumps({"text":"fixture"}).encode())
        with patch("semantic_robot.v2.wall_budget.time.perf_counter",side_effect=lambda:clock.now):
            with patch("semantic_robot.v2.policy.urlopen",return_value=response) as call:
                call_service("http://localhost/","act","","",bundle,deadline=16)
                self.assertEqual(call.call_args.kwargs["timeout"],2.)

    def test_encoding_expiration_does_not_issue_http(self):
        clock=SimpleNamespace(now=10.)
        class Image:
            def save(self,file,format):file.write(b"fixture");clock.now=16.
        with patch("semantic_robot.v2.wall_budget.time.perf_counter",side_effect=lambda:clock.now):
            with patch("semantic_robot.v2.policy.urlopen") as call:
                with self.assertRaises(WallTimeBudgetReached):
                    call_service("http://localhost/","act","","",SimpleNamespace(labels=["head"],images=[Image()]),deadline=15)
                call.assert_not_called()

    def test_late_http_result_is_saved_but_not_returned_for_actuation(self):
        clock=SimpleNamespace(now=10.)
        class Late(BytesIO):
            def read(self,*a):clock.now=12.;return super().read(*a)
        response=Late(json.dumps({"text":"late action"}).encode())
        p=object.__new__(VLMPolicy);p.calls=0;p.max_calls=1;p.deadline=11.;p.uri="http://localhost/"
        with patch("semantic_robot.v2.wall_budget.time.perf_counter",side_effect=lambda:clock.now):
            with patch("semantic_robot.v2.policy.urlopen",return_value=response):
                with self.assertRaises(WallTimeBudgetReached):
                    p._call("act","","",SimpleNamespace(labels=[],images=[]))
        self.assertEqual(p.calls,1)
        self.assertTrue(p.last_call["cancelled_after_deadline"])
        self.assertEqual(p.last_call["result"]["text"],"late action")

    def test_selected_action_is_logged_without_moving_after_deadline(self):
        manager=SimpleNamespace(stop_reason=None);rows=[];row={"action":{"move":"close"},"control_start":23}
        self.assertTrue(stop_before_motion(0,23,row,rows,manager))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]["control_end"],23)
        self.assertEqual(rows[0]["feedback"]["control_ticks"],0)
        self.assertFalse(rows[0]["accepted_before_motion"])
        self.assertEqual(manager.stop_reason,"WALL_TIME_BUDGET_REACHED_BEFORE_ACTION")

    def test_real_runner_checks_both_before_and_after_servo_preparation(self):
        # Execute the actual selected-action/preflight section, not a copy.
        tree=ast.parse((Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=="decision")
        first=next(i for i,n in enumerate(loop.body) if ast.unparse(n).startswith("row['action'] ="))
        last=next(i for i,n in enumerate(loop.body) if ast.unparse(n).startswith("previous = bundle.current_raw"))
        stub=ast.parse("for _ in range(1):\n    pass").body[0];stub.body=loop.body[first:last]
        code=compile(ast.fix_missing_locations(ast.Module(body=[stub],type_ignores=[])),"runner_deadline_section","exec")
        for before in (True,False):
            clock=SimpleNamespace(now=2. if before else 0.);calls=[]
            def begin(*a,**kw):calls.append(1);clock.now=2.;return True
            ns={"action":HOLD,"row":{"control_start":7},"controls":7,"decisions":[],"deadline":1.,
                "manager":SimpleNamespace(carry=False,stop_reason=None),"servo":SimpleNamespace(begin=begin),
                "controller":None,"args":SimpleNamespace(multicamera_inspection=False,workspace_posture=False,mode="agent"),
                "asdict":asdict,"time":SimpleNamespace(perf_counter=lambda:clock.now),"state_now":lambda:None,
                "stop_before_motion":stop_before_motion}
            with patch("semantic_robot.v2.wall_budget.time.perf_counter",side_effect=lambda:clock.now):exec(code,ns)
            self.assertEqual(len(calls),0 if before else 1)
            self.assertEqual(ns["decisions"][0]["feedback"]["control_ticks"],0)
            self.assertEqual(ns["decisions"][0]["control_end"],7)

    def test_real_runner_zero_tick_deadline_race_never_opens_or_finishes_chain(self):
        tree=ast.parse((Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text())
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="accepted"
                   and any(isinstance(child,ast.While) for child in n.body))
        loop=ast.parse("for _ in range(1):\n    pass").body[0];loop.body=[block]
        code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),"actual_zero_tick_race","exec")
        motion=SubstepMotion(None);motion.reference={"fixture":"unchanged"}
        ns={"accepted":True,"servo":SimpleNamespace(done=False,ticks=0,total_ticks=18),
            "controls":7,"action_control_limit":20,"row":{"control_start":7},"decisions":[],
            "deadline":0,"expired":expired,"budget_interrupted":False,"substep_motion":motion,
            "manager":SimpleNamespace(stop_reason=None),"stop_before_motion":stop_before_motion}
        with patch.object(motion,"begin",wraps=motion.begin) as begin,patch.object(motion,"finish",wraps=motion.finish) as finish:
            exec(code,ns);begin.assert_not_called();finish.assert_not_called()
        self.assertFalse(motion.active);self.assertIsNone(motion.pending)
        self.assertTrue(ns["budget_interrupted"]);self.assertEqual(ns["controls"],7)
        self.assertEqual(ns["decisions"][0]["feedback"]["status"],"NOT_STARTED_WALL_TIME_BUDGET")


if __name__=="__main__":unittest.main()
