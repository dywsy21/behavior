"""Real local HTTP + image serialization + policy/harness contracts, no GPU."""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from semantic_robot.v2.harness import Goal, TaskHarness
from semantic_robot.v2.policy import VLMPolicy
from semantic_robot.v2.policy import GroundedPolicy
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.vision import VisualBundle
from test_v2 import fixture, evidence


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # This workstation has a global HTTP proxy; test loopback stays local.
        self.proxy_environment=patch.dict(os.environ,{"no_proxy":"127.0.0.1,localhost",
                                                     "NO_PROXY":"127.0.0.1,localhost"})
        self.proxy_environment.start()
        owner = self
        self.requests, self.cap, self.forbidden = [], False, False
        self.grounded=False
        self.action_override=None
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass

            def send(self,value):
                raw=json.dumps(value).encode()
                self.send_response(200);self.send_header("Content-Length",str(len(raw)))
                self.end_headers();self.wfile.write(raw)

            def do_GET(self):
                self.send({"protocol":"semantic-v2","revision":"test-pinned-revision"})

            def do_POST(self):
                value=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(value)
                if value["kind"]=="plan":
                    text=(json.dumps({"strategy":"scan_left","visible_reason":"current table is not identified"})
                          if "Replan only" in value["system"] else
                          json.dumps([asdict(Goal("pick","radio","right","visible co-motion after micro lift"))]))
                elif value["kind"]=="observe":
                    observation=asdict(evidence(view="head"))
                    if owner.grounded:observation["other_views"]=[]
                    text=json.dumps(observation)
                else:
                    candidates=[Action.parse(x) for x in value["allowed"]]
                    text=(owner.action_override if owner.action_override else Action("right","close") if owner.forbidden else
                          next((a for a in candidates if a.part=="right" and a.move=="up"),candidates[0])).text()
                self.send({"text":text,"hit_token_cap":owner.cap})
        self.server=HTTPServer(("127.0.0.1",0),Handler)
        self.thread=threading.Thread(target=lambda:self.server.serve_forever(poll_interval=.01),daemon=True)
        self.thread.start()
        self.uri=f"http://127.0.0.1:{self.server.server_port}"
        self.bundle=VisualBundle([Image.new("RGB",(32,32),"red")],["CURRENT_HEAD_RAW"],{},
                                 {"head":np.zeros((32,32,3),dtype=np.uint8)})
        self.model,self.state=fixture()

    def tearDown(self):
        self.server.shutdown();self.thread.join(timeout=2);self.server.server_close()
        self.proxy_environment.stop()

    def test_plan_observe_stage_scope_and_real_image_payload(self):
        policy=VLMPolicy(self.uri,"test-pinned-revision",max_calls=3)
        goals,_=policy.plan(0,"Turn on a radio",self.bundle)
        manager=TaskHarness(goals)
        obs,_=policy.observe(manager,self.state,self.bundle)
        manager.observe(obs,self.state,self.bundle.geometry)
        action,_=policy.act(manager,self.state,self.bundle)
        self.assertEqual(manager.stage,"APPROACH")
        self.assertEqual(action.part,"right")
        self.assertEqual(policy.calls,3)
        self.assertEqual([x["kind"] for x in self.requests],["plan","observe","act"])
        self.assertTrue(self.requests[0]["images"][0]["png"].startswith("iVBOR"))
        self.assertEqual(set(self.requests[0]),{"kind","system","text","images","allowed"})
        self.assertFalse(any(x in json.dumps(self.requests) for x in ("object_pose_gt","goal_status")))
        with self.assertRaises(RuntimeError):policy.observe(manager,self.state,self.bundle)
        self.assertEqual(len(self.requests),3)

    def test_identity_rejected_before_neural_or_actuator_calls(self):
        with self.assertRaises(ValueError):VLMPolicy(self.uri,"wrong-revision")
        self.assertEqual(self.requests,[])

    def test_truncation_consumes_one_call_and_never_retries(self):
        self.cap=True
        policy=VLMPolicy(self.uri,"test-pinned-revision")
        with self.assertRaises(ValueError):policy.plan(0,"Turn on a radio",self.bundle)
        self.assertEqual(policy.calls,1);self.assertEqual(len(self.requests),1)

    def test_remote_legal_but_stage_forbidden_action_rejected(self):
        policy=VLMPolicy(self.uri,"test-pinned-revision")
        manager=TaskHarness([Goal("pick","radio","right","held")])
        manager.observe(evidence(view="head"),self.state)
        self.forbidden=True
        with self.assertRaises(ValueError):policy.act(manager,self.state,self.bundle)
        self.assertEqual(manager.executions,0)

    def test_grounded_schema_preflight_and_bounded_recovery_real_transport(self):
        self.grounded=True
        policy=GroundedPolicy(self.uri,"test-pinned-revision",max_calls=3)
        manager=GroundedHarness([Goal("pick","radio","right","held")])
        servo=SafeServo(self.model,self.state,[-1,-1])
        controller=GroundedController(self.model,servo,manager)
        obs,_=policy.observe(manager,self.state,self.bundle)
        depths={v:np.ones((100,100),dtype=np.float32) for v in ("head","left_wrist","right_wrist")}
        controller.observe(obs,self.state,depths,{"head":{"valid_fraction":1.}})
        allowed=controller.candidates(self.state)
        action,_=policy.act_feasible(manager,self.state,self.bundle,allowed)
        self.assertIn(action,allowed)
        recovery,_=policy.recover(manager,self.state,self.bundle,"schema test")
        self.assertEqual(recovery["strategy"],"scan_left")
        self.assertEqual(policy.calls,3)
        self.assertEqual(manager.executions,0)
        self.assertIn("CURRENT preflight receipt",self.requests[1]["text"])

    def test_stage_legal_but_not_preflighted_action_rejected(self):
        policy=GroundedPolicy(self.uri,"test-pinned-revision")
        manager=GroundedHarness([Goal("pick","radio","right","held")]);manager.stage="APPROACH"
        manager.observation=evidence()
        self.action_override=Action("right","down")
        allowed=(Action("right","up"),)
        self.assertIn(self.action_override,manager.palette())
        with self.assertRaises(ValueError):policy.act_feasible(manager,self.state,self.bundle,allowed)
        self.assertEqual(manager.executions,0)


if __name__=="__main__":
    unittest.main()
