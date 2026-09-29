import copy
from dataclasses import asdict
import io
from http.client import BadStatusLine, IncompleteRead
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from semantic_robot.v2.jev_client import (
    ENDPOINT, MODEL, JevClient, JevError, NoRedirect, choice, load_key, strict_response,
    validate_response,
)
from semantic_robot.v2.jev_policy import (
    JevDecisionPolicy, JevGroundedPolicy, command_rubric, decision_state, load_task_plan,
)
from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action, HOLD
from semantic_robot.v2.wall_budget import WallTimeBudgetReached
from test_grounded import setup_controller, grounded_evidence


KEY = "dummy-unit-test-only-secret"


def answer(questions, selections=None):
    selections = selections or {}
    answers = {}
    for name, question in questions.items():
        selected = selections.get(name, next(iter(question["criteria"])))
        answers[name] = {"type": "choice", "choice": selected, "confidence": 1.,
                         "probabilities": {k: float(k == selected) for k in question["criteria"]}}
    return {"model": MODEL, "answers": answers, "usage": {"input_tokens": 12, "output_tokens": 4}}


class Response(io.BytesIO):
    def geturl(self):
        return ENDPOINT


class Opener:
    def __init__(self, selections=None, mutate=None):
        self.selections = selections or []
        self.requests = []
        self.mutate = mutate

    def open(self, request, timeout):
        payload = json.loads(request.data)
        self.requests.append((request, timeout))
        selected = self.selections[len(self.requests)-1] if self.selections else {}
        value = answer(payload["questions"], selected)
        if self.mutate:
            self.mutate(value)
        return Response(json.dumps(value).encode())


def client(opener=None, **kw):
    return JevClient(api_key=KEY, opener=opener or Opener(), **kw)


def context():
    _, state, harness, _, _, _, _ = setup_controller()
    harness.stage = "APPROACH"
    harness.observation = grounded_evidence(enclosed=False, co_moving=None)
    allowed = (HOLD, Action("right", "up", "micro"), Action("right", "down", "micro"))
    harness.candidate_receipt = {"tested": [
        {"action": asdict(a), "accepted": True, "reason": "TEST_ONLY", "predicted_distance_gain_m": gain}
        for a, gain in zip(allowed, (0., .002, -.002))]}
    bundle = SimpleNamespace(geometry={})
    return harness, state, bundle, allowed


class ClientTests(unittest.TestCase):
    def test_durable_attempt_and_result_accounting_has_no_credentials(self):
        records=[]
        c=client(journal=records.append)
        c.evaluate({}, {"q":choice("Pick",{"a":"A","b":"B"})})
        self.assertEqual([r['event'] for r in records],['attempt','validated'])
        self.assertNotIn(KEY,json.dumps(records))
        records=[]
        opener=SimpleNamespace(open=lambda *a,**kw:(_ for _ in ()).throw(URLError(KEY)))
        with self.assertRaises(JevError): client(opener,journal=records.append).evaluate({}, {"q":choice("Pick",{"a":"A","b":"B"})})
        self.assertEqual([r['event'] for r in records],['attempt','network_error'])
        self.assertNotIn(KEY,json.dumps(records))

    def test_official_native_endpoint_no_credential_in_receipt(self):
        opener = Opener()
        c = client(opener)
        result, receipt = c.evaluate({"x": False}, {"q": choice("Pick", {"no": "No", "yes": "Yes"})})
        self.assertEqual(opener.requests[0][0].full_url, ENDPOINT)
        self.assertEqual(opener.requests[0][0].get_header("Authorization"), "Bearer " + KEY)
        self.assertNotIn(KEY, json.dumps(receipt))
        self.assertNotIn(KEY, repr(c))
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.input_tokens, 12)
        self.assertEqual(result["answers"]["q"]["choice"], "no")

    def test_failures_spend_budget_no_retry_no_secret_error(self):
        for error in (HTTPError(ENDPOINT, 401, KEY, {}, None), URLError(KEY), TimeoutError(KEY),
                      BadStatusLine(KEY), IncompleteRead(KEY.encode(), 999)):
            opener = SimpleNamespace(open=lambda *a, **kw: (_ for _ in ()).throw(error))
            c = client(opener, max_calls=1)
            for _ in range(2):
                with self.assertRaises(JevError) as caught:
                    c.evaluate({}, {"q": choice("Pick", {"a": "A", "b": "B"})})
                self.assertNotIn(KEY, str(caught.exception))
            self.assertEqual(c.calls, 1)
            self.assertIsNone(c.last_call)

    def test_redirect_refused(self):
        with self.assertRaises(JevError):
            NoRedirect().redirect_request(None, None, 307, "", {}, "https://evil.invalid")

    def test_bad_or_late_response_does_not_authorize(self):
        q = {"q": choice("Pick", {"a": "A", "b": "B"})}
        mutators = [
            lambda v: v.update(model="jev-latest"),
            lambda v: v["answers"]["q"].update(choice="injected"),
            lambda v: v["answers"]["q"].update(confidence=float("nan")),
            lambda v: v["answers"]["q"].update(probabilities={"a": .3, "b": .2}),
            lambda v: v["answers"]["q"].update(probabilities={"a": 0., "b": 1.}),
            lambda v: v["answers"]["q"].update(probabilities={"a": True, "b": 0.}),
            lambda v: v["usage"].update(input_tokens=-1),
            lambda v: v["answers"].update(extra={}),
        ]
        for mutate in mutators:
            with self.subTest(mutate=mutate), self.assertRaises(JevError):
                client(Opener(mutate=mutate)).evaluate({}, q)
        c = client()
        c.deadline = time.perf_counter() - 1
        with self.assertRaises(WallTimeBudgetReached):
            c.evaluate({}, q)
        self.assertEqual(c.calls, 0)
        def expire(_):
            c.deadline = time.perf_counter() - 1
        c = client(Opener(mutate=expire))
        with self.assertRaises(WallTimeBudgetReached):
            c.evaluate({}, q)
        self.assertEqual(c.calls, 1)
        self.assertIsNotNone(c.last_call)

    def test_bounded_json_and_sensitive_payload(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'not-json'):
            with self.assertRaises(JevError):
                strict_response(raw)
        c = client()
        q = {"q": choice("Pick", {"a": "A", "b": "B"})}
        with self.assertRaises(JevError):
            c.evaluate({"secret": KEY}, q)
        with self.assertRaises(ValueError):
            c.evaluate({"too_big": "x" * 100_001}, q)
        self.assertEqual(c.calls, 0)

    def test_native_two_decimal_probability_mass_kept_not_renormalized(self):
        q = {"q": choice("Pick", {str(i): "option" for i in range(8)})}
        value = answer(q)
        probs = {str(i): p for i, p in enumerate((.80, .10, .08, .01, 0., 0., 0., 0.))}
        value["answers"]["q"]["probabilities"] = probs
        result = validate_response(value, q, MODEL)
        self.assertEqual(result["answers"]["q"]["probabilities"], probs)
        self.assertFalse(result["probability_validation"]["q"]["locally_renormalized"])
        self.assertAlmostEqual(result["probability_validation"]["q"]["raw_probability_sum"], .99)
        for bad in (.75, .80333):
            value["answers"]["q"]["probabilities"]["0"] = bad
            with self.assertRaises(JevError):
                validate_response(value, q, MODEL)

    def test_private_file_and_env(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "key"
            path.write_text(KEY)
            path.chmod(0o600)
            self.assertEqual(load_key(path), KEY)
            path.chmod(0o644)
            with self.assertRaises(JevError):
                load_key(path)
            link = Path(root) / "alias"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                load_key(link)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": KEY}):
            self.assertEqual(load_key(), KEY)


class PolicyTests(unittest.TestCase):
    def test_context_has_no_sim_truth_and_computes_progress_in_code(self):
        h, state, bundle, allowed = context()
        h.grounding["debug_sim_object_position"] = [99, 99, 99]
        h.candidate_receipt["truth"] = "DO_NOT_SEND"
        raw = json.loads(actor_context(h, state, bundle, allowed))
        before = copy.deepcopy(raw)
        value = decision_state(raw, allowed)
        self.assertEqual(raw, before)
        self.assertNotIn("DO_NOT_SEND", json.dumps(value))
        self.assertNotIn("debug_sim_object_position", json.dumps(value))
        self.assertNotIn("base_axis_projection_guides", value["robot"])
        self.assertEqual(value["commands"]["command_001"]["computed_progress_not_grasp_quality"], "reduces_distance")
        self.assertEqual(value["commands"]["command_002"]["computed_progress_not_grasp_quality"], "increases_distance")
        self.assertIsNone(value["current_visual_evidence"]["co_moving"])

    def test_two_stage_choices_bind_whole_action_no_axis_recombination(self):
        h, state, bundle, allowed = context()
        opener = Opener([{"intent": "approach"}, {"command": "command_001"}])
        engine = JevDecisionPolicy(client(opener))
        chosen, receipt = engine.select(json.loads(actor_context(h, state, bundle, allowed)), allowed)
        self.assertEqual(chosen, allowed[1])
        self.assertEqual(len(opener.requests), 2)
        second = json.loads(opener.requests[1][0].data)
        self.assertEqual(second["state"]["chosen_tactic_not_new_evidence"], "approach")
        self.assertEqual(receipt["result"]["text"], allowed[1].text())

    def test_progress_facts_do_not_require_grasp_before_approach(self):
        h, state, bundle, allowed = context()
        h.grounding = {"valid": True}
        value = decision_state(json.loads(actor_context(h, state, bundle, allowed)), allowed)
        self.assertTrue(value["facts"]["approach_progress_available"])
        self.assertFalse(value["facts"]["verified_holding"])
        self.assertFalse(value["facts"]["immediate_hazard"])
        self.assertIn("gain 2.00 mm", command_rubric(value["commands"]["command_001"]))
        self.assertIn("do NOT select", command_rubric(value["commands"]["command_000"]))
        self.assertEqual(len(value["commands"]), len(allowed))

    def test_sole_hold_is_local_and_sole_move_has_abstention(self):
        h, state, bundle, allowed = context()
        c = client(Opener([{"intent": "wait"}, {"command": "abstain"}]))
        engine = JevDecisionPolicy(c)
        action, _ = engine.select(json.loads(actor_context(h, state, bundle, (HOLD,))), (HOLD,))
        self.assertEqual(action, HOLD)
        self.assertEqual(c.calls, 0)
        with self.assertRaisesRegex(JevError, "abstained"):
            engine.select(json.loads(actor_context(h, state, bundle, (allowed[1],))), (allowed[1],))

    def test_unpreflighted_and_privileged_context_rejected(self):
        h, state, bundle, allowed = context()
        value = json.loads(actor_context(h, state, bundle, allowed))
        value["CURRENT preflight receipt"]["scores_for_allowed_commands"][1]["accepted"] = False
        with self.assertRaises(ValueError):
            decision_state(value, allowed)
        with self.assertRaises(ValueError):
            decision_state({**value, "simulator_truth": {}}, allowed)

    def test_adapter_observer_never_plans_acts_or_recovers(self):
        h, state, bundle, allowed = context()
        observer = SimpleNamespace(identity={"revision": "test"}, calls=0, last_call=None, deadline=None)
        opener = Opener([{"intent": "approach"}, {"command": "command_001"}, {"recovery": "hold"}])
        plan = Path(__file__).resolve().parents[2] / "configs/semantic_robot/jev_task0_plan.json"
        policy = JevGroundedPolicy(observer, client(opener), task_id=0, task_plan=plan)
        policy.deadline = time.perf_counter() + 30
        goals, receipt = policy.plan(0, "turn on radio", bundle)
        self.assertEqual(goals[0].hand, "right")
        self.assertEqual(goals[1].hand, "left")
        self.assertEqual(policy.calls, 0)
        action, _ = policy.act_feasible(h, state, bundle, allowed)
        self.assertEqual(action, allowed[1])
        recovery, _ = policy.recover(h, state, bundle, "TEST")
        self.assertEqual(recovery["strategy"], "hold")
        self.assertEqual(observer.calls, 0)
        self.assertEqual(policy.calls, 3)
        self.assertEqual(h.completed, [])
        with self.assertRaises(JevError):
            policy.act(h, state, bundle)
        h.stop_reason = "STOPPED"
        with self.assertRaises(JevError):
            policy.act_feasible(h, state, bundle, allowed)
        self.assertEqual(policy.calls, 3)

    def test_task_plan_does_not_transfer_to_wrong_task(self):
        plan = Path(__file__).resolve().parents[2] / "configs/semantic_robot/jev_task0_plan.json"
        with self.assertRaises(ValueError):
            load_task_plan(plan, 3)

    def test_context_mutated_during_second_request_never_returns_action(self):
        h, state, bundle, allowed = context()
        plan = Path(__file__).resolve().parents[2] / "configs/semantic_robot/jev_task0_plan.json"
        def mutate(_):
            if len(opener.requests) == 2:
                h.candidate_receipt["tested"][1]["accepted"] = False
        opener = Opener([{"intent": "approach"}, {"command": "command_001"}], mutate)
        observer = SimpleNamespace(identity={}, calls=0)
        policy = JevGroundedPolicy(observer, client(opener), task_id=0, task_plan=plan)
        with self.assertRaisesRegex(JevError, "context changed"):
            policy.act_feasible(h, state, bundle, allowed)
        self.assertEqual(h.executions, 0)

    def test_failed_second_request_retains_separate_accounting(self):
        h, state, bundle, allowed = context()
        plan = Path(__file__).resolve().parents[2] / "configs/semantic_robot/jev_task0_plan.json"
        class SecondFails(Opener):
            def open(self, request, timeout):
                if len(self.requests) == 1:
                    raise URLError("unit-test-network-failure")
                return super().open(request, timeout)
        observer = SimpleNamespace(identity={}, calls=1)
        policy = JevGroundedPolicy(observer, client(SecondFails()), task_id=0, task_plan=plan)
        with self.assertRaises(JevError):
            policy.act_feasible(h, state, bundle, allowed)
        self.assertEqual(policy.calls, 3)
        self.assertEqual(policy.accounting()["observer_calls"], 1)
        self.assertEqual(policy.accounting()["jev_requests"], 2)
        self.assertEqual(policy.accounting()["jev_input_tokens"], 12)
        self.assertEqual(policy.accounting()["jev_requests_without_validated_response"], 1)

    def test_saved_state_requires_pinned_action_and_manifest(self):
        import importlib.util
        import hashlib
        script = Path(__file__).resolve().parents[2] / "scripts/semantic_robot/test_jev_saved.py"
        spec = importlib.util.spec_from_file_location("jev_saved_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        h, state, bundle, allowed = context()
        payload = {"request": {"kind": "act", "text": actor_context(h, state, bundle, allowed) + "\n" +
                   "\n".join(f"{i}: {a.text()}" for i, a in enumerate(allowed))}, "result": {"text": allowed[1].text()}}
        raw = json.dumps(payload).encode()
        manifest = {"actor_scene_truth": False, "code_commit": "test", "implementation_digest": "test",
                    "task": 0, "instance": 1, "window_sha": "test", "model_identity": {"model": "test"}}
        manifest_raw = json.dumps(manifest).encode()
        source = {"schema": "jev-saved-sources-v1", "sources": {"test": {
            **{k: v for k, v in manifest.items() if k != "actor_scene_truth"},
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest()}}, "states": [{
                "source": "test", "decision": 1, "action_sha256": hashlib.sha256(raw).hexdigest()}]}
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "decision_001/action.json"
            p.parent.mkdir()
            p.write_bytes(raw)
            m = Path(root) / "manifest.json"
            m.write_bytes(manifest_raw)
            self.assertEqual(len(module.saved_context(p, source)[1]), len(allowed))
            m.write_bytes(manifest_raw + b" ")
            with self.assertRaises(ValueError):
                module.saved_context(p, source)
            m.write_bytes(manifest_raw)
            p.write_bytes(raw + b" ")
            with self.assertRaises(ValueError):
                module.saved_context(p, source)

    def test_freshness_blocks_sensor_joint_clock_and_control_drift(self):
        import numpy as np
        from semantic_robot.v2.jev_freshness import snapshot, freshness_check, array_digest, VIEWS
        from semantic_robot.v2.render_batch import make_receipts
        from render_batch_fixture import batch_fixture, camera_receipts
        _, state, _, _ = context()
        images = {v + "_rgb": np.arange(48,dtype=np.uint8).reshape(3,4,4) for v in VIEWS}
        depths = {v: np.ones((4, 4), dtype=np.float32) for v in VIEWS}
        def receipts(capture):
            r = camera_receipts(make_receipts(batch_fixture(131+capture),
                {"simulation_time":1.,"physics_index":120},capture))
            for v in VIEWS:
                r[v]["rgb_sha256"] = array_digest(images[v+"_rgb"].transpose(1,2,0))["sha256"]
                r[v]["depth_sha256"] = array_digest(depths[v])["sha256"]
            return r
        before = snapshot(state, images, depths, receipts(1), 0)
        after = snapshot(state, images, depths, receipts(2), 0)
        self.assertTrue(freshness_check(before, after)["passed"])
        self.assertFalse(freshness_check(before, before)["passed"])
        for field in ("controls", "clock", "sensors", "proprio", "finger_qpos"):
            changed = copy.deepcopy(after)
            changed[field] = "changed"
            self.assertFalse(freshness_check(before, changed)["passed"])
        stale = receipts(2)
        depths["head"][0, 0] = 2.
        # Different renderer samples are valid only with verified new buffers.
        with self.assertRaises(ValueError): snapshot(state, images, depths, stale, 0)
        after = snapshot(state, images, depths, receipts(2), 0)
        self.assertTrue(freshness_check(before, after)["passed"])
        self.assertNotEqual(before["sensors"]["head"], after["sensors"]["head"])
        with self.assertRaises(ValueError):
            snapshot(state, images, depths, {v: {} for v in VIEWS}, 0)
        torn = receipts(2);torn["head"]["native_time"]["physics_index"] += 1
        with self.assertRaises(ValueError): snapshot(state, images, depths, torn, 0)
        invalid = receipts(2);invalid["head"]["native_time"]["render_batch_verified"] = False
        with self.assertRaises(ValueError): snapshot(state, images, depths, invalid, 0)
        changed = copy.deepcopy(after);changed["batch"]["cameras"]["head"]["render_product"] = "/OTHER"
        self.assertFalse(freshness_check(before, changed)["passed"])

    def test_reference_routes_holding_hand_not_working_hand(self):
        h, _, _, _ = context()
        h.held_inspection_enabled = h.reference_from_planner = True
        h.held["right"] = "radio"
        h.hold_verified["right"] = True
        plan = Path(__file__).resolve().parents[2] / "configs/semantic_robot/jev_task0_plan.json"
        opener = Opener([{"reference": "held_right"}])
        observer = SimpleNamespace(identity={}, calls=0)
        policy = JevGroundedPolicy(observer, client(opener), task_id=0, task_plan=plan)
        h.goals = policy.goals
        h.index = 1
        policy.resolve_target_reference(h)
        self.assertEqual(h.search_reference, "held_right")
        self.assertEqual(h.goal.hand, "left")
        questions = json.loads(opener.requests[0][0].data)["questions"]
        self.assertNotIn("held_left", questions["reference"]["criteria"])


if __name__ == "__main__":
    unittest.main()
