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
from semantic_robot.v2.policy import GroundedPolicy, RefinedGroundedPolicy
from semantic_robot.v2.affordance import SurfaceChoice
from semantic_robot.v2.grounding import GroundedEvidence, localize_target
from semantic_robot.v2.grounded_harness import GroundedHarness, GroundedController
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.protocol import Action
from semantic_robot.v2.vision import VisualBundle
from semantic_robot.v2.bimanual import BimanualEvidence, HandContact
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
        self.observation_override=None
        self.surface_override=SurfaceChoice(0)
        self.finite_choice_kinds=["act","ground"]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass

            def send(self,value):
                raw=json.dumps(value).encode()
                self.send_response(200);self.send_header("Content-Length",str(len(raw)))
                self.end_headers();self.wfile.write(raw)

            def do_GET(self):
                self.send({"protocol":"semantic-v2","revision":"test-pinned-revision",
                           "finite_choice_kinds":owner.finite_choice_kinds})

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
                    text=owner.observation_override if owner.observation_override is not None else json.dumps(observation)
                elif value["kind"]=="ground":text=owner.surface_override.text()
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

    def test_actual_invisible_response_reaches_search_without_retry_or_extra_evidence(self):
        # Verbatim server_27b_v2 call 3, H-07 plates_27b_v1, 2026-09-18.
        self.observation_override='{"visible":false,"view":"none","target_uv":null,"enclosed":null,"co_moving":null,"supported":null,"effect":false,"hazard":"none","note":"Target not identified in current views."}'
        policy=GroundedPolicy(self.uri,"test-pinned-revision",max_calls=1)
        manager=GroundedHarness([Goal("navigate","breakfast table","both","visibly near table")])
        servo=SafeServo(self.model,self.state)
        controller=GroundedController(self.model,servo,manager)
        obs,call=policy.observe(manager,self.state,self.bundle)
        self.assertEqual(call["result"]["text"],self.observation_override)
        self.assertEqual(call["validation"]["defaulted_fields"],["other_views"])
        self.assertEqual(call["validation"]["retries"],0)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(policy.calls,1)
        depths={v:np.ones((100,100),dtype=np.float32) for v in ("head","left_wrist","right_wrist")}
        controller.observe(obs,self.state,depths,{"head":{"valid_fraction":1.}})
        action,selection=controller.search_action(self.state)
        self.assertEqual(action,Action("base","yaw_plus","coarse"))
        self.assertEqual(manager.stage,"SEARCH")
        self.assertFalse(controller.target["valid"])
        self.assertEqual(manager.executions,0)  # preflight only, not a new rollout

    def test_actual_visible_response_raw_preserved_and_no_defaults_when_supplied(self):
        self.observation_override='{"visible":true,"view":"head","target_uv":[0.67,0.78],"enclosed":false,"co_moving":null,"supported":true,"effect":false,"hazard":"none","note":"Red and white radio is visible on the table. Right gripper is open and not in contact with the radio."}'
        policy=GroundedPolicy(self.uri,"test-pinned-revision",max_calls=2)
        manager=GroundedHarness([Goal("pick","radio","right","held")])
        obs,call=policy.observe(manager,self.state,self.bundle)
        self.assertEqual(call["result"]["text"],self.observation_override)
        self.assertEqual(obs.other_views,())
        self.assertEqual(obs.target_uv,(.67,.78))
        fields=json.loads(self.observation_override);fields["other_views"]=[]
        self.observation_override=json.dumps(fields)
        _,call=policy.observe(manager,self.state,self.bundle)
        self.assertEqual(call["validation"]["defaulted_fields"],[])

    def test_bimanual_observer_has_two_explicit_contacts_and_no_centroid_fallback(self):
        fields=asdict(evidence(view="head",enclosed=True));fields["other_views"]=[]
        fields["hand_contacts"]=[asdict(HandContact("left","head",(.4,.5),True,None)),
                                 asdict(HandContact("right","none",None,None,None))]
        self.observation_override=json.dumps(fields)
        policy=GroundedPolicy(self.uri,"test-pinned-revision",max_calls=2)
        manager=GroundedHarness([Goal("pick","wide tray","both","both hands support it",True)])
        obs,call=policy.observe(manager,self.state,self.bundle)
        self.assertIsInstance(obs,BimanualEvidence)
        self.assertIn("two graspable regions",self.requests[0]["system"])
        self.assertEqual(obs.hand_contacts[1].view,"none")
        self.assertEqual(call["validation"]["defaulted_fields"],[])
        fields.pop("hand_contacts");self.observation_override=json.dumps(fields)
        obs,call=policy.observe(manager,self.state,self.bundle)
        self.assertEqual(obs.hand_contacts,())
        self.assertEqual(call["validation"]["defaulted_fields"],["hand_contacts"])
        self.assertEqual(policy.calls,2)

    def test_live_contact_contract_rejects_invented_cross_view_correspondence(self):
        fields=asdict(evidence(view="right_wrist"))
        fields["other_views"]=[{"view":"head","target_uv":[.7,.8]}]
        self.observation_override=json.dumps(fields)
        policy=GroundedPolicy(self.uri,"test-pinned-revision",max_calls=1)
        manager=GroundedHarness([Goal("pick","radio","right","held")])
        with self.assertRaisesRegex(ValueError,"guessed metric correspondences"):
            policy.observe(manager,self.state,self.bundle)
        self.assertEqual(policy.calls,1)
        self.assertIn("MUST be []",self.requests[0]["system"])

    def test_bimanual_refinement_is_single_hand_and_shared_bounded_budget(self):
        original,_,state,bundle,depths,model=self.refinement_inputs()
        contacts=(HandContact("left","head",(.5,.5),False,None),
                  HandContact("right","right_wrist",(.6,.6),False,None))
        obs=BimanualEvidence(**asdict(original),hand_contacts=contacts)
        manager=GroundedHarness([Goal("pick","wide tray","both","both hands support it",True)])
        manager.stage="APPROACH"
        policy=RefinedGroundedPolicy(self.uri,"test-pinned-revision",max_refinements=1)
        changed,call,receipt,_=policy.refine(obs,manager,state,bundle,depths,model)
        self.assertTrue(receipt["attempted"])
        self.assertEqual(receipt["hand"],"left")
        self.assertNotEqual(changed.hand_contacts[0].target_uv,contacts[0].target_uv)
        self.assertEqual(changed.hand_contacts[1],contacts[1])
        self.assertEqual(BimanualEvidence.parse(json.dumps(asdict(changed))),changed)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(json.loads(self.requests[0]["text"].split("\nChoose exactly one:")[0])["current_goal"]["hand"],"left")
        _,call,receipt,_=policy.refine(obs,manager,state,bundle,depths,model)
        self.assertIsNone(call);self.assertEqual(len(self.requests),1)
        self.assertEqual(receipt["reason"],"REFINEMENT_BUDGET_EXHAUSTED")
        unknown=BimanualEvidence(**asdict(original),hand_contacts=())
        same,call,_,_=policy.refine(unknown,manager,state,bundle,depths,model)
        self.assertIs(same,unknown);self.assertIsNone(call)

    def refinement_inputs(self):
        obs=GroundedEvidence(**asdict(evidence(view="head")))
        manager=GroundedHarness([Goal("pick","radio","right","held")]);manager.stage="APPROACH"
        depths={v:np.ones((100,100),np.float32) for v in ("head","left_wrist","right_wrist")}
        depths["head"][49:52,:]=2.
        bundle=VisualBundle([],[],{},{v:np.zeros((100,100,3),np.uint8) for v in depths})
        return obs,manager,self.state,bundle,depths,self.model

    def test_real_surface_choice_transport_abstention_and_budget(self):
        policy=RefinedGroundedPolicy(self.uri,"test-pinned-revision",max_refinements=1)
        inputs=self.refinement_inputs()
        selected,call,receipt,views=policy.refine(*inputs)
        self.assertEqual(call["result"]["text"],'{"candidate_id":0}')
        self.assertTrue(receipt["refined_target"]["valid"])
        self.assertEqual(receipt["robot_controls"],0)
        self.assertNotEqual(selected.target_uv,inputs[0].target_uv)
        self.assertEqual(self.requests[0]["kind"],"ground")
        self.assertIn('{"candidate_id":null}',self.requests[0]["allowed"])
        self.assertEqual(len(self.requests[0]["images"]),6)
        self.assertNotIn("rejected_depth",self.requests[0]["text"])
        self.assertNotIn("point_base_m",self.requests[0]["text"])
        same,call,receipt,_=policy.refine(*inputs)
        self.assertIs(same,inputs[0]);self.assertIsNone(call)
        self.assertEqual(receipt["reason"],"REFINEMENT_BUDGET_EXHAUSTED")
        self.assertEqual(len(self.requests),1)
        self.surface_override=SurfaceChoice()
        other=RefinedGroundedPolicy(self.uri,"test-pinned-revision")
        selected,_,receipt,_=other.refine(*inputs)
        self.assertIs(selected,inputs[0]);self.assertEqual(receipt["reason"],"MODEL_ABSTAINED")
        self.assertEqual(inputs[1].executions,0)

    def test_refinement_skips_verification_and_rejects_old_service(self):
        inputs=self.refinement_inputs()
        policy=RefinedGroundedPolicy(self.uri,"test-pinned-revision")
        for stage in ("GRASP","VERIFY_GRASP","RELEASE","VERIFY_PLACE","VERIFY_SUPPORT","VERIFY_EFFECT"):
            inputs[1].stage=stage
            self.assertIsNone(policy.refine(*inputs)[1])
        inputs[1].stage="APPROACH";inputs[4]["head"][:]=1.
        self.assertIsNone(policy.refine(*inputs)[1])
        self.finite_choice_kinds=["act"]
        with self.assertRaises(ValueError):RefinedGroundedPolicy(self.uri,"test-pinned-revision")
        self.assertEqual(self.requests,[])


if __name__=="__main__":
    unittest.main()
