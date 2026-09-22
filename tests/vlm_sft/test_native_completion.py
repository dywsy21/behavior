import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/vlm_sft"), str(ROOT / "src")]
import native_actor_protocol as protocol
from native_completion import VERSION, validate_output, loss_mask, validate_sidecar
from native_completion_dataset import (make_sidecar, state_label, terminal_hold_check, readable_tree,
                                       build, DATASET_SHA, ROWS_SHA, SOURCES_SHA)
from native_motion_codec import TOKENS, ACTOR_VERSION
import test_native_actor_protocol as actor_fixtures


class CompletionTests(unittest.TestCase):
    def test_all45_motions_remain_continuations_not_new_stop_tokens(self):
        self.assertEqual(len(TOKENS), 45)
        for motion in TOKENS:
            value = {"skill_status": "CONTINUE", "motion": motion}
            self.assertEqual(validate_output(value, version=VERSION), value)
            self.assertEqual(loss_mask(value), {"skill_status": True, "motion": True})
        value = {"skill_status": "REQUEST_VERIFY", "motion": None}
        self.assertEqual(validate_output(value, version=VERSION), value)
        self.assertEqual(loss_mask(value), {"skill_status": True, "motion": False})

    def test_unknown_extra_fields_and_verify_actions_fail_closed(self):
        bad = [{}, {"skill_status": "REQUEST_VERIFY"}, {"skill_status": "CONTINUE", "motion": None},
               {"skill_status": "CONTINUE", "motion": "STOP"}, {"skill_status": "SUCCEEDED", "motion": None},
               {"skill_status": "REQUEST_VERIFY", "motion": "HOLD"},
               {"skill_status": "REQUEST_VERIFY", "motion": []},
               {"skill_status": "REQUEST_VERIFY", "motion": None, "success": True},
               {"skill_status": True, "motion": "RIGHT_UP"}]
        for value in bad:
            with self.assertRaises(ValueError): validate_output(value, version=VERSION)
        with self.assertRaises(ValueError):
            validate_output({"skill_status": "CONTINUE", "motion": "HOLD"}, version="legacy")

    def test_unknown_failed_or_incomplete_stability_never_becomes_label(self):
        verdict = {"outcome": "IN_PROGRESS", "tick": 44, "stable_ticks": 0, "required_ticks": 12}
        self.assertEqual(state_label(verdict, "RIGHT_UP")["skill_status"], "CONTINUE")
        terminal = {**verdict, "outcome": "SUCCEEDED", "stable_ticks": 12}
        self.assertEqual(state_label(terminal, None)["skill_status"], "REQUEST_VERIFY")
        for value, motion in (({}, "RIGHT_UP"), ({**verdict, "outcome": "UNKNOWN"}, "RIGHT_UP"),
                              ({**verdict, "outcome": "FAILED"}, "RIGHT_UP"), (verdict, None),
                              (terminal, "HOLD"), ({**terminal, "stable_ticks": 11}, None),
                              ({**terminal, "required_ticks": 1}, None), ({**verdict, "tick": True}, "RIGHT_UP")):
            with self.assertRaises(ValueError): state_label(value, motion)

    def test_actual_single_hold_clock_q_and_latch_are_not_macro_labels(self):
        q = np.arange(18) / 100; command = np.zeros(23)
        command[3:14] = q[:11]; command[15:22] = q[11:]; command[[14,22]] = [1,-1]
        cap = {"clock": {"prefix_control": 396, "native_control": 327}, "q": q.tolist()}
        hold = {"completed": True, "prefix_controls": 396, "native_controls": 328, "action23": command.tolist()}
        final = {"frame": {"tick": 724}, "verdict": {"outcome": "SUCCEEDED", "tick": 724, "stable_ticks": 12, "required_ticks": 12}}
        terminal_hold_check(cap, hold, final, command)
        for field, value in (("completed", False), ("native_controls", 345), ("native_controls", True), ("prefix_controls", 395)):
            with self.assertRaises(ValueError): terminal_hold_check(cap, {**hold, field: value}, final, command)
        for index in (0, 1, 2, 3, 14, 22):
            wrong = copy.deepcopy(hold); wrong["action23"][index] += .001
            with self.assertRaises(ValueError): terminal_hold_check(cap, wrong, final, command)
        wrong = copy.deepcopy(final); wrong["frame"]["tick"] += 1
        with self.assertRaises(ValueError): terminal_hold_check(cap, hold, wrong, command)
        for tick in (723, 725, True):
            wrong = copy.deepcopy(final); wrong["verdict"]["tick"] = tick
            with self.assertRaises(ValueError): terminal_hold_check(cap, hold, wrong, command)

    def test_sidecar_keeps_public_actor_and_masks_terminal_motion(self):
        _, _, _, actor, raw = actor_fixtures.PoseProtocolTests().fixture()
        actor = {**actor, "protocol": ACTOR_VERSION}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = {}
            for name, data in raw.items():
                path = root / (name + ".png"); path.write_bytes(data); images[name] = str(path)
            original = {**protocol.training_row(actor, "RIGHT_UP"), "id": "original",
                        "images": images, "provenance": {"clock": {"prefix_control": 396, "native_control": 200}}}
            before = copy.deepcopy(original)
            value = make_sidecar(original, {"skill_status": "CONTINUE", "motion": "RIGHT_UP"}, {"offline": True})
            self.assertEqual(original, before); self.assertEqual(value["actor"], actor)
            terminal = make_sidecar(original, {"skill_status": "REQUEST_VERIFY", "motion": None}, {"offline": True}, terminal=True)
            self.assertIsNone(terminal["source_motion_row_id"]); self.assertFalse(terminal["loss_mask"]["motion"])
            self.assertNotIn("target", terminal); self.assertEqual(set(terminal["actor"]), protocol.ACTOR_KEYS)
            for mutate in (lambda x:x["actor"].update(held=True), lambda x:x["actor"].update(goal_status="SUCCEEDED"),
                           lambda x:x["loss_mask"].update(motion=True), lambda x:x["loss_mask"].update(skill_status=1),
                           lambda x:x.update(source_motion_row_id="old"), lambda x:x["label"].update(motion="HOLD"),
                           lambda x:x.update(text=x["text"]+" success")):
                wrong = copy.deepcopy(terminal); mutate(wrong)
                with self.assertRaises(ValueError): validate_sidecar(wrong)

    def test_unreadable_or_symlink_sources_rejected_even_when_privileged_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); hidden = root / "hidden"; hidden.mkdir(); hidden.chmod(0)
            try:
                with self.assertRaises(ValueError): readable_tree(root)
            finally: hidden.chmod(0o700)
            link = root / "alias"; link.symlink_to(hidden, target_is_directory=True)
            with self.assertRaises(ValueError): readable_tree(root)
        with patch("native_completion_dataset.os.scandir", side_effect=PermissionError("fixture")):
            with self.assertRaises(PermissionError): readable_tree(ROOT)

    def test_only_exact_frozen_dataset_and_new_output_admitted(self):
        for digest in (DATASET_SHA, ROWS_SHA, SOURCES_SHA): self.assertEqual(len(digest), 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError): build(root, root, root, root)
            with patch("native_completion_dataset.sha", return_value="0" * 64):
                with self.assertRaises(ValueError): build(root, root, root, root / "new")
            self.assertFalse((root / "new").exists())


if __name__ == "__main__": unittest.main()
