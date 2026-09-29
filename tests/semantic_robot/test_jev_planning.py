"""Exact-input, hypothetical-ordering and no-retry contracts; no network."""
import copy
import json
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from semantic_robot.v2.jev_client import JevAbstained, JevError
from semantic_robot.v2.jev_policy import JevGroundedPolicy, load_task_plan, planning_request
from test_jev import Opener, client
from semantic_robot.v2.jev_control import validate_actor_ownership

PLAN = Path(__file__).resolve().parents[2] / "configs/semantic_robot/jev_task0_plan.json"


class PlanningTests(unittest.TestCase):
    def policy(self, choices=None):
        opener = Opener(choices)
        policy = JevGroundedPolicy(SimpleNamespace(identity={}, calls=0), client(opener),
                                   task_id=0, task_plan=PLAN)
        return policy, opener

    def test_exact_official_instruction_enforced_before_api(self):
        policy, opener = self.policy()
        for wrong in ("turn on radio", policy.plan_identity["official_instruction"] + " ", None):
            with self.assertRaisesRegex(JevError, "Official task instruction"):
                policy.plan(0, wrong, None)
        self.assertFalse(opener.requests)
        self.assertFalse(policy.planning_started)

    def test_live_requests_equal_shared_builder_and_do_not_claim_execution(self):
        policy, opener = self.policy([{"next_goal": "goal_0"}, {"next_goal": "goal_1"}])
        instruction = policy.plan_identity["official_instruction"]
        goals, receipt = policy.plan(0, instruction, None)
        self.assertEqual(goals, policy.goals)
        for slot, prefix in enumerate(([], [0])):
            state, questions = planning_request(policy.goals, policy.plan_identity, instruction, prefix)
            actual = json.loads(opener.requests[slot][0].data)
            self.assertEqual(actual["state"], state)
            self.assertEqual(actual["questions"], questions)
            self.assertEqual(state["phase"], "ORDER_FUTURE_PLAN_NOT_EXECUTION")
            self.assertIn("No step has executed", state["current_execution_evidence"])
            self.assertEqual(state["official_task_instruction_verbatim"], instruction)
            self.assertIn("abstain", questions["next_goal"]["criteria"])
        self.assertEqual(receipt["result"]["planning_status"], "complete")
        self.assertEqual(receipt["result"]["selected_goal_ids"], ["goal_0", "goal_1"])
        self.assertEqual(policy.observer.calls, 0)

    def test_abstention_preserves_both_receipts_and_cannot_be_retried(self):
        policy, opener = self.policy([{"next_goal": "goal_0"}, {"next_goal": "abstain"}])
        with self.assertRaises(JevAbstained):
            policy.plan(0, policy.plan_identity["official_instruction"], None)
        result = policy.last_call["result"]
        self.assertEqual(result["planning_status"], "abstained_no_action")
        self.assertEqual(result["selected_goal_ids"], ["goal_0"])
        self.assertEqual(len(result["planning_calls"]), 2)
        self.assertIsNone(policy.planned_goals)
        with self.assertRaises(JevError):
            policy.plan(0, policy.plan_identity["official_instruction"], None)
        self.assertEqual(len(opener.requests), 2)

    def test_prefix_checks_types_dependencies_duplicates_and_completion(self):
        goals, identity = load_task_plan(PLAN, 0)
        for bad in ([1], [False], [0, 0], [0, 1], [-1], [2]):
            with self.assertRaises(ValueError):
                planning_request(goals, identity, identity["official_instruction"], bad)
        original = copy.deepcopy(identity)
        state, questions = planning_request(goals, identity, identity["official_instruction"], [0])
        self.assertEqual(identity, original)
        self.assertEqual(state["eligible_next_goal_ids"], ["goal_1"])
        self.assertEqual(set(questions["next_goal"]["criteria"]), {"goal_1", "abstain"})

    def test_runner_records_first_and_later_plan_abstention_as_zero_action_outcome(self):
        from run_v2 import select_initial_plan
        for choices in ([{"next_goal": "abstain"}], [{"next_goal": "goal_0"}, {"next_goal": "abstain"}]):
            ledger = []
            policy, _ = self.policy(choices)
            policy.client._journal = ledger.append
            goals, receipt, proof = select_initial_plan(policy, 0, policy.plan_identity["official_instruction"], None)
            self.assertIsNone(goals)
            self.assertEqual(receipt["result"]["planning_status"], "abstained_no_action")
            result = dict(stop_reason="JEV_PLAN_ABSTAINED", planning_abstention=proof,
                          decisions=[], controls=1, jev_requests=len(choices), observer_calls=0,
                          prefix_controls=0, diagnostic_replay_controls=0)
            ticks = [dict(control=1, safety_stop=True)]
            audit = validate_actor_ownership(result, ledger, ticks)
            self.assertFalse(audit["control_integration_validated"])
            self.assertEqual(audit["jev_owned_actions"], 0)
            self.assertTrue(audit["all_control_ticks_checked"])
            for bad in ({**result, "controls": 2}, {**result, "observer_calls": 1},
                        {**result, "planning_abstention": {**proof, "choice": "goal_0"}}):
                with self.assertRaises(JevError):
                    validate_actor_ownership(bad, ledger, ticks)
            with self.assertRaises(JevError):
                validate_actor_ownership(result, ledger, [dict(control=1, decision=0, action23=[0]*23)])
            with self.assertRaises(JevError):
                validate_actor_ownership(result, ledger[:-1], ticks)

    def test_runner_does_not_turn_transport_error_into_abstention(self):
        from run_v2 import select_initial_plan
        from unittest.mock import Mock
        policy = SimpleNamespace(plan=Mock(side_effect=JevError("network error")))
        with self.assertRaisesRegex(JevError, "network error"):
            select_initial_plan(policy, 0, "task", None)

    def test_first_action_abstention_is_an_outcome_not_control_integration(self):
        proof = dict(call=5, question="command", choice="abstain")
        row = dict(decision=0, accepted_before_motion=False, stop_reason="JEV_ABSTAINED",
                   selection_source="Jev_all_world_search", jev_abstention=proof)
        result = dict(stop_reason="JEV_ABSTAINED", decisions=[row], controls=1,
                      prefix_controls=0, diagnostic_replay_controls=0, jev_requests=5)
        ledger = [dict(call=5, event="validated", selections=dict(command="abstain"))]
        trace = [dict(control=1, safety_stop=True)]
        audit = validate_actor_ownership(result, ledger, trace)
        self.assertFalse(audit["control_integration_validated"])
        self.assertTrue(audit["command_abstained"])
        for changed in ({**result, "controls": 2}, {**result, "decisions": [{**row, "action": {}}]},
                        {**result, "decisions": [{**row, "jev_abstention": {**proof, "call": 4}}]}):
            with self.assertRaises(JevError): validate_actor_ownership(changed, ledger, trace)

    def test_probe_qualification_requires_all_canonical_trials_and_negative_controls(self):
        from probe_jev_planning import qualification
        rows = [dict(kind="exact_live_plan", status="complete", selected_goal_ids=["goal_0", "goal_1"])
                for _ in range(10)]
        rows += [dict(kind="contradictory_requirements_control", choice="abstain") for _ in range(2)]
        self.assertTrue(qualification(rows, 10))
        self.assertFalse(qualification(rows[:-1], 10))
        self.assertFalse(qualification([], 10))
        for bad_index, key, value in ((0, "status", "abstained_no_action"),
                                      (0, "selected_goal_ids", ["goal_1", "goal_0"]),
                                      (11, "choice", "goal_0")):
            bad = copy.deepcopy(rows); bad[bad_index][key] = value
            self.assertFalse(qualification(bad, 10))

    def test_probe_credential_validation_precedes_inputs_and_client(self):
        import os
        from unittest.mock import patch
        import probe_jev_planning as probe
        base = ["probe", "--baseline", "not-read", "--output", "not-created"]
        for extra, env in (([], {}), (["--dry-run", "--key-file", "not-read"], {}),
                           (["--dry-run"], {"TYPESAFE_API_KEY": "dummy-unit-test"})):
            with patch.dict(os.environ, env, clear=True), patch("sys.argv", base+extra), \
                    patch.object(probe, "baseline_request") as source, patch.object(probe, "JevClient") as api:
                with self.assertRaises(ValueError): probe.main()
                source.assert_not_called(); api.assert_not_called()

    def test_launcher_binds_contract_source_and_full_durable_validation_batch(self):
        from unittest.mock import patch
        import launch_jev_closedloop as launch
        from probe_jev_planning import digest
        goals, identity = load_task_plan(PLAN, 0)
        templates = [planning_request(goals, identity, identity["official_instruction"], p) for p in ([], [0])]
        choices = ["abstain"]*3 + ["goal_0", "goal_1"]*10 + ["abstain"]*2
        ledger = []
        for i, selected in enumerate(choices, 1):
            ledger += [dict(call=i, model="jev-1.13.0", event="attempt"),
                       dict(call=i, model="jev-1.13.0", event="validated", usage=dict(input_tokens=7, output_tokens=2),
                            selections=dict(next_goal=selected))]
        rows = [dict(kind="old_exact_second_question", index=i, choice="abstain") for i in range(3)]
        rows += [dict(kind="exact_live_plan", index=i, status="complete", selected_goal_ids=["goal_0", "goal_1"])
                 for i in range(10)]
        rows += [dict(kind="contradictory_requirements_control", index=i, choice="abstain") for i in range(2)]
        value = dict(schema="jev04-planning-validation-v1", model="jev-1.13.0", qualified=True, dry_run=False,
                     repetitions=10, rows=rows, contract_sha256=identity["sha256"],
                     official_instruction=identity["official_instruction"],
                     request_template_sha256=[digest(dict(state=s, questions=q)) for s, q in templates],
                     policy_sha256=launch.common.sha(launch.REPO/"src/semantic_robot/v2/jev_policy.py"),
                     client_sha256=launch.common.sha(launch.REPO/"src/semantic_robot/v2/jev_client.py"),
                     probe_sha256=launch.common.sha(launch.REPO/"scripts/semantic_robot/probe_jev_planning.py"),
                     api_requests=25, validated_responses=25, input_tokens=175, output_tokens=50,
                     new_controls=0, new_resets=0, training_updates=0)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); path = folder/"result.json"
            for name, records in (("calls", ledger), ("trials", [dict(summary=r) for r in rows])):
                raw = "".join(json.dumps(r)+"\n" for r in records).encode()
                (folder/(name+".jsonl")).write_bytes(raw)
                value[name+"_sha256"] = hashlib.sha256(raw).hexdigest()
            path.write_text(json.dumps(value))
            with patch.object(launch, "PLANNING_VALIDATION", path):
                self.assertEqual(launch.planning_qualification()["repetitions"], 10)
                for field, bad in (("dry_run", True), ("qualified", False), ("contract_sha256", "changed"),
                                   ("policy_sha256", "changed"), ("api_requests", 24), ("calls_sha256", "changed")):
                    path.write_text(json.dumps({**value, field: bad}))
                    with self.assertRaises(ValueError): launch.planning_qualification()
                changed = copy.deepcopy(ledger); changed[7]["selections"]["next_goal"] = "abstain"
                raw = "".join(json.dumps(r)+"\n" for r in changed).encode()
                (folder/"calls.jsonl").write_bytes(raw)
                path.write_text(json.dumps({**value, "calls_sha256": hashlib.sha256(raw).hexdigest()}))
                with self.assertRaisesRegex(ValueError, "durable model choices"):
                    launch.planning_qualification()


if __name__ == "__main__":
    unittest.main()
