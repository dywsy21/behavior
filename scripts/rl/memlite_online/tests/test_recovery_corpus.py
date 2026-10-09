from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from recovery_corpus import (anchor_candidate, group_key, outcome_candidate, split_group,
                             stable_arm, validate_source_episode)


def rows(held=False):
    return {step: dict(control_step=step, context=dict(parent_goal="goal",
                active_skills_semantic_json=json.dumps([dict(verb="GRASP", target="object", arm="UNSPECIFIED")])),
            physical_audit=dict(grasp_states={"object": dict(left="TRUE" if held else "FALSE", right="FALSE")}),
            terminated=False, truncated=False, policy_update=3, proprio_before=[float(step)] * 61)
            for step in range(64)}


class RecoveryCorpusTests(unittest.TestCase):
    def test_groups_ignore_display_spelling_and_all_seeds(self):
        self.assertEqual(group_key("picking_up_trash", 3), group_key("picking up trash", 3))
        self.assertEqual(split_group("picking_up_trash", 3), split_group("picking up trash", 3))
        self.assertEqual(split_group("t", 3, {"t:3"}), "protected")

    def test_only_pre_observation_physics_can_supply_label(self):
        data = rows()
        data[16]["physical_audit"]["grasp_states"]["object"]["left"] = "TRUE"
        skill = [dict(verb="GRASP", target="object")]
        self.assertIsNone(outcome_candidate(data, 16, skill, {})["value"])
        for step in range(10, 16):
            data[step]["physical_audit"]["grasp_states"]["object"]["left"] = "TRUE"
        self.assertEqual(outcome_candidate(data, 16, skill, {})["value"], "SUCCEEDED")

    def test_alternating_hands_not_a_stable_grasp(self):
        data = rows()
        for step in range(10, 16):
            data[step]["physical_audit"]["grasp_states"]["object"] = dict(
                left="TRUE" if step % 2 else "FALSE", right="FALSE" if step % 2 else "TRUE")
        self.assertFalse(stable_arm(data, 16, "object", "left"))
        self.assertIsNone(outcome_candidate(data, 16, [dict(verb="GRASP", target="object")], {})["value"])

    def test_no_guessing_entity_suffixes(self):
        proposal = outcome_candidate(rows(True), 16, [dict(verb="GRASP", target="object_123")], {})
        self.assertIsNone(proposal["value"])
        self.assertEqual(proposal["members"][0]["reason"], "target_binding_missing")

    def test_exact_name_binding_and_requested_hand(self):
        skill = [dict(verb="GRASP", target="asset_123", arm="RIGHT")]
        self.assertIsNone(outcome_candidate(rows(True), 16, skill, {"asset_123": "object"})["value"])
        skill[0]["arm"] = "LEFT"
        self.assertEqual(outcome_candidate(rows(True), 16, skill, {"asset_123": "object"})["value"], "SUCCEEDED")

    def test_source_rejects_holdout_split_commit_and_wrong_parent(self):
        source = dict(source_commit="code", model=dict(checkpoints={
            "high/p.pt": dict(sha256="high"), "low/p.pt": dict(sha256="low")}))
        contracts = {"task": dict(train_instances=[0, 2])}
        ep = dict(task="task", instance_id=0, split="train", source_commit="code",
                  high_checkpoint_sha256="high", low_checkpoint_sha256="low")
        validate_source_episode(ep, source, contracts)
        for field, value in (("instance_id", 1), ("split", "public_test"),
                             ("source_commit", "new"), ("high_checkpoint_sha256", "other")):
            with self.assertRaises(ValueError):
                validate_source_episode(dict(ep, **{field: value}), source, contracts)

    def test_partial_parallel_bundle_not_completed(self):
        skills = [dict(verb="GRASP", target="object"), dict(verb="GRASP", target="other")]
        self.assertIsNone(outcome_candidate(rows(True), 16, skills, {})["value"])

    def test_missing_or_unknown_physics_does_not_mean_failure(self):
        data = rows(True)
        del data[15]
        self.assertIsNone(outcome_candidate(data, 16, [dict(verb="GRASP", target="object")], {})["value"])
        self.assertFalse(stable_arm(data, 16, "object", "left"))

    def test_no_oracle_in_actor_and_no_automatic_bc(self):
        episode = dict(run="run", episode_id="e", task="t", instance_id=3)
        value = anchor_candidate(episode, rows(True), 16, {"archive": "clip"}, {}, "train")
        self.assertNotIn("physical_audit", value["actor_input"])
        self.assertNotIn("label_audit", value["actor_input"])
        self.assertFalse(any(value["eligibility"].values()))
        self.assertTrue(value["label_audit"]["full_executed_32_step_target_available"])

    def test_future_skill_change_or_short_tail_rejects_action_target(self):
        episode = dict(run="run", episode_id="e", task="t", instance_id=3)
        for kind in ("skill", "tail", "terminal"):
            data = deepcopy(rows(True))
            if kind == "skill":
                data[40]["context"]["active_skills_semantic_json"] = '[{"verb":"NAVIGATE","target":"object"}]'
            elif kind == "tail":
                del data[47]
            else:
                data[40]["terminated"] = True
            value = anchor_candidate(episode, data, 16, {}, {}, "train")
            self.assertFalse(value["label_audit"]["full_executed_32_step_target_available"])


if __name__ == "__main__":
    unittest.main()
