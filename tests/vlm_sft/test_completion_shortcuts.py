import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/vlm_sft"))
from audit_completion_shortcuts import (analyze, counts, fold_predictions, proprio_features,
                                       fit_threshold, trailing_up)


def row(name, group, positive, history=(), q=0.):
    return {"id": name, "provenance": {"group": [1, group]},
            "actor": {"protocol": "fixture", "task": "task", "active_instruction": "grasp",
                      "history": list(history), "proprio": {
                          "joint_positions_rad": {"torso": [q]*4, "left": [q]*7, "right": [q]*7},
                          "finger_opening_m": {"left": .05, "right": .03}}},
            "label": {"skill_status": "REQUEST_VERIFY" if positive else "CONTINUE"}}


class ShortcutAuditTests(unittest.TestCase):
    def fixture(self):
        return [row("a0", 1, False), row("a1", 1, True, ["RIGHT_UP"]*3, .1),
                row("b0", 2, False, ["RIGHT_UP"]*2, .01), row("b1", 2, True, ["RIGHT_UP"]*4, .11)]

    def test_group_split_never_uses_same_instance_or_self_as_neighbor(self):
        report = analyze(self.fixture())
        self.assertEqual(len(report["folds"]), 2)
        for fold in report["folds"]:
            for prediction in fold["predictions"]:
                self.assertNotIn(prediction["id"], fold["train_ids"])
                self.assertIn(prediction["nearest_train_id"], fold["train_ids"])
                self.assertNotEqual(prediction["nearest_train_group"], fold["held_out_TRAIN_group"])

    def test_private_fields_ids_and_images_are_not_features(self):
        source = self.fixture()[0]; changed = copy.deepcopy(source)
        changed.update(id="label-containing-id", images={"leak": "terminal"}, offline_label_evidence={"outcome": "SUCCEEDED"})
        changed["provenance"].update(group=[99, 777], tick=10000)
        changed["actor"]["current_rgb_sha256"] = {"head": "positive"}
        np.testing.assert_array_equal(proprio_features(source), proprio_features(changed))
        train = self.fixture()[:2]
        self.assertEqual(fold_predictions(train, [source])[1][0]["predictions"],
                         fold_predictions(train, [changed])[1][0]["predictions"])

    def test_threshold_is_fit_only_to_training_labels(self):
        train, test = self.fixture()[:2], self.fixture()[2:]
        old = fold_predictions(train, test)
        changed = copy.deepcopy(test)
        for r in changed: r["label"]["skill_status"] = "REQUEST_VERIFY" if r["label"]["skill_status"] == "CONTINUE" else "CONTINUE"
        new = fold_predictions(train, changed)
        self.assertEqual(old[0], new[0])
        self.assertEqual([r["predictions"] for r in old[1]], [r["predictions"] for r in new[1]])

    def test_suffix_not_total_up_count_and_deterministic_ties(self):
        self.assertEqual(trailing_up(row("x", 1, False, ["RIGHT_UP", "RIGHT_CLOSE", "RIGHT_UP"])), 1)
        self.assertEqual(fit_threshold(self.fixture()[:2]), 3)

    def test_minority_class_errors_are_not_hidden_by_accuracy(self):
        result = counts([False]*34+[True]*5, [False]*39)
        self.assertEqual(result["FN"], 5)
        self.assertEqual(result["request_recall"], 0)
        self.assertEqual(result["balanced_accuracy"], .5)

    def test_conflicting_history_and_full_nonvisual_input_are_reported_separately(self):
        rows = self.fixture()
        other = copy.deepcopy(rows[0]); other["id"] = "conflict"; other["provenance"]["group"] = [1, 2]
        other["label"]["skill_status"] = "REQUEST_VERIFY"
        report = analyze(rows+[other])
        self.assertIn(["a0", "conflict"], report["identical_nonvisual_input_conflicting_label_buckets"])
        other["actor"]["proprio"]["finger_opening_m"]["left"] = .02
        report = analyze(rows+[other])
        self.assertIn(["a0", "conflict"], report["same_history_conflicting_label_buckets"])
        self.assertEqual(report["identical_nonvisual_input_conflicting_label_buckets"], [])

    def test_degenerate_data_and_malformed_proprio_rejected(self):
        for rows in ([], self.fixture()[:2], self.fixture()+[self.fixture()[0]]):
            with self.assertRaises(ValueError): analyze(rows)
        r = self.fixture()[0]; r["actor"]["proprio"]["joint_positions_rad"]["left"][0] = float("nan")
        with self.assertRaises(ValueError): proprio_features(r)
        with self.assertRaises(ValueError): fit_threshold([self.fixture()[0]])


if __name__ == "__main__": unittest.main()
