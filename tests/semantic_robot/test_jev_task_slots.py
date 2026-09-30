"""Task coverage, model-selected hands/order, source binding and long plans."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from semantic_robot.v2.jev_client import JevError
from semantic_robot.v2.jev_policy import JevGroundedPolicy, load_task_plan, planning_request
from semantic_robot.v2.jev_task_slots import load_slots, slot_options, materialize_slots
from semantic_robot.v2.jev_plan_audit import validate_plan_proof, validate_goal_proof, validate_plan_contract
from semantic_robot.v2.jev_control import validate_actor_ownership
from test_jev import Opener, client, context

ROOT = Path(__file__).resolve().parents[2]
PLANS = {1: ROOT/"configs/semantic_robot/jev_task1_slots.json", 3: ROOT/"configs/semantic_robot/jev_task3_slots.json"}
TRASH = ["goal_2_right", "goal_3_right", "goal_0_left", "goal_1_left", "goal_4_left", "goal_5_left"]
FOOD = ["goal_5_left", "goal_6_left", "goal_7_right", "goal_8_right", "goal_0_right",
        "goal_3_both", "goal_4_both", "goal_1_both", "goal_2_both", "goal_9_left"]


def rehash(proof):
    body = {k: v for k, v in proof.items() if k != "plan_sha256"}
    proof["plan_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SlotTests(unittest.TestCase):
    def policy(self, task, selections):
        ledger = []
        opener = Opener(selections)
        p = JevGroundedPolicy(SimpleNamespace(identity={}, calls=0), client(opener, journal=ledger.append),
                              task_id=task, task_plan=PLANS[task])
        return p, ledger, opener

    def test_complete_task_outcomes_and_no_privileged_ids(self):
        goals, one = load_task_plan(PLANS[1], 1)
        self.assertEqual(len(goals), 6)
        self.assertEqual([s["binding"] for s in one["slots"] if s["kind"] == "pick"], ["can_a", "can_b", "can_c"])
        self.assertEqual([s["kind"] for s in one["slots"]].count("place"), 3)
        goals, three = load_task_plan(PLANS[3], 3)
        self.assertEqual(len(goals), 10)
        self.assertEqual([s["binding"] for s in three["slots"] if s["kind"] == "pick"],
                         ["pizza_plate_a", "pizza_plate_b", "bowl_a", "bowl_b"])
        self.assertEqual(three["slots"][0]["kind"], "open")
        self.assertEqual(three["slots"][9]["kind"], "close")
        self.assertEqual(set(three["dependencies"][9]), {2, 4})
        self.assertTrue(all(three["slots"][i]["level"] for i in (1, 2, 3, 4)))
        for path in PLANS.values():
            self.assertNotIn("bddl", path.read_text().lower())
            self.assertNotIn("instance_id", path.read_text())

    def test_offered_hands_and_non_scripted_order(self):
        _, one = load_task_plan(PLANS[1], 1)
        self.assertEqual(set(slot_options(one, [])), {f"goal_{i}_{hand}" for i in (0, 2, 4) for hand in ("left", "right")})
        self.assertEqual(set(slot_options(one, ["goal_2_right"])), {"goal_3_right"})
        for selected in (TRASH, ["goal_4_right", "goal_5_right", "goal_2_left", "goal_3_left", "goal_0_right", "goal_1_right"]):
            goals, indices = materialize_slots(one, selected)
            self.assertEqual(len(goals), 6)
            self.assertEqual(len(set(indices)), 6)
            for pick, place in zip(goals[::2], goals[1::2]):
                self.assertEqual(pick.hand, place.hand)
        _, three = load_task_plan(PLANS[3], 3)
        self.assertIn("goal_5_left", slot_options(three, []))
        self.assertNotIn("goal_1_both", slot_options(three, []))
        goals, indices = materialize_slots(three, FOOD)
        self.assertEqual(indices, [5, 6, 7, 8, 0, 3, 4, 1, 2, 9])
        self.assertEqual(goals[5].hand, "both")
        pizzas_first = ["goal_0_left", "goal_1_both", "goal_2_both", "goal_3_both", "goal_4_both"]
        self.assertTrue({"goal_9_left", "goal_9_right", "goal_5_left", "goal_7_right"} <= set(slot_options(three, pizzas_first)))
        materialize_slots(three, pizzas_first + ["goal_9_left", "goal_5_left", "goal_6_left", "goal_7_right", "goal_8_right"])

    def test_invalid_prefix_cannot_place_first_change_hand_or_double_load(self):
        _, identity = load_task_plan(PLANS[1], 1)
        for prefix in (["goal_1_left"], ["goal_0_left", "goal_1_right"],
                       ["goal_0_left", "goal_2_right"], ["goal_0_left", "goal_1_left", "goal_0_left"], [True]):
            with self.assertRaises(ValueError): slot_options(identity, prefix)
        with self.assertRaises(ValueError): materialize_slots(identity, TRASH[:2])

    def test_dag_pairing_and_resource_deadlock_rejected(self):
        original = json.loads(PLANS[1].read_text())
        candidates = []
        bad = copy.deepcopy(original); bad["dependencies"][1] = []; candidates.append(bad)
        bad = copy.deepcopy(original); bad["slots"][1]["hand_options"] = ["left"]; candidates.append(bad)
        bad = copy.deepcopy(original); bad["slots"][2]["binding"] = "can_a"; candidates.append(bad)
        bad = copy.deepcopy(original); bad["dependencies"][0] = [1]; candidates.append(bad)
        bad = copy.deepcopy(original); bad["dependencies"][1] = [0, 3]; candidates.append(bad)
        bad = copy.deepcopy(original); bad["target_references"][0] = "held_right"; candidates.append(bad)
        for value in candidates:
            with self.assertRaises(ValueError): load_slots(value, json.dumps(value).encode(), 1)

    def test_exact_instruction_checked_and_all_choices_materialized(self):
        policy, ledger, opener = self.policy(1, [{"next_goal": key} for key in TRASH])
        with self.assertRaises(JevError): policy.plan(1, "rewrite", None)
        self.assertEqual(len(opener.requests), 0)
        goals, receipt = policy.plan(1, policy.plan_identity["official_instruction"], None)
        self.assertEqual(receipt["result"]["selected_goal_ids"], TRASH)
        proof = policy.plan_ownership
        self.assertEqual([g.hand for g in goals], ["right", "right", "left", "left", "left", "left"])
        validate_plan_contract(proof, PLANS[1], 1, ledger)
        for slot, (request, _) in enumerate(opener.requests):
            state, questions = planning_request(policy.goals, policy.plan_identity,
                                                policy.plan_identity["official_instruction"], TRASH[:slot])
            actual = json.loads(request.data)
            self.assertEqual(actual["state"], state); self.assertEqual(actual["questions"], questions)
        with self.assertRaises(JevError): policy.plan(1, policy.plan_identity["official_instruction"], None)

    def test_plan_contract_rejects_rehashed_hand_order_source_or_ledger_tampering(self):
        policy, ledger, _ = self.policy(1, [{"next_goal": key} for key in TRASH])
        policy.plan(1, policy.plan_identity["official_instruction"], None)
        proof = policy.plan_ownership
        bad = copy.deepcopy(proof); bad["goals"][0]["hand"] = bad["goals"][1]["hand"] = "left"; rehash(bad)
        with self.assertRaises(JevError): validate_plan_contract(bad, PLANS[1], 1, ledger)
        bad = copy.deepcopy(proof); bad["selected_goal_ids"][0] = "goal_0_left"; rehash(bad)
        with self.assertRaises(JevError): validate_plan_proof(bad, ledger)
        bad = copy.deepcopy(proof); bad["planning_calls"][0] = 2; rehash(bad)
        with self.assertRaises(JevError): validate_plan_proof(bad, ledger)
        with self.assertRaises(JevError): validate_plan_contract(proof, PLANS[3], 3, ledger)
        with self.assertRaises(JevError): validate_plan_contract(proof, PLANS[1], 1, ledger[2:])
        row = {"jev_goal_authority": policy.goal_authority(0, policy.planned_goals[0])}
        validate_goal_proof(row, proof)
        row["jev_goal_authority"]["goal"]["hand"] = "left"
        with self.assertRaises(JevError): validate_goal_proof(row, proof)

    def test_first_or_mid_slot_abstention_is_normal_and_has_no_goal_plan(self):
        from run_v2 import select_initial_plan
        for selected in ([], TRASH[:2]):
            p, ledger, _ = self.policy(1, [{"next_goal": key} for key in selected] + [{"next_goal": "abstain"}])
            goals, receipt, proof = select_initial_plan(p, 1, p.plan_identity["official_instruction"], None)
            self.assertIsNone(goals); self.assertIsNone(p.plan_ownership)
            result = dict(stop_reason="JEV_PLAN_ABSTAINED", planning_abstention=proof, decisions=[], controls=1,
                          jev_requests=len(selected)+1, observer_calls=0, prefix_controls=0, diagnostic_replay_controls=0)
            audit = validate_actor_ownership(result, ledger, [dict(control=1, safety_stop=True)])
            self.assertFalse(audit["control_integration_validated"])

    def test_each_goal_in_long_plan_gets_one_reference_and_reorder_is_preserved(self):
        p, ledger, _ = self.policy(3, [{"next_goal": key} for key in FOOD] + [{"reference": "world"}]*10)
        goals, _ = p.plan(3, p.plan_identity["official_instruction"], None)
        h, _, _, _ = context()
        h.goals = goals; h.held_inspection_enabled = h.reference_from_planner = True
        for i in range(10):
            h.index = i
            self.assertIsNotNone(p.resolve_target_reference(h))
            self.assertIsNone(p.resolve_target_reference(h))  # cached, no additional call
            self.assertIsNone(h.stop_reason)
        self.assertEqual(p.reference_calls, 10)
        self.assertEqual(p.client.calls, 20)
        self.assertEqual(p.identity["semantic_reference_limit"], 10)
        h.target_references.clear(); h.index = 0
        self.assertIsNone(p.resolve_target_reference(h))
        self.assertEqual(h.stop_reason, "SEMANTIC_REFERENCE_CALL_BUDGET_REACHED")
        with self.assertRaises(JevError): p.goal_authority(0, replace(goals[0], hand="right"))


if __name__ == "__main__":
    unittest.main()
