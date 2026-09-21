"""Synthetic negative cases only; no fixture is collected or released BC."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts/vlm_sft")]
from common import sha
from native_teacher_contract import digest
from native_teacher_seed import extract_seed, seed_identity, validate_seed_release
from native_teacher_pregrasp_seed import PROFILE, REVIEW_SCHEMA, build_sidecar, select_precontact
from native_teacher_collect import require_release
from test_native_teacher_automatic import frame


class PrecontactSeedTests(unittest.TestCase):
    def fixture(self):
        spec = {"schema": "h09t-private-teacher-v1", "verb": "GRASP", "hand": "right",
                "target": "target", "destination": "", "support_hand": None, "payloads": [],
                "goal_frame": "target", "pose_reviewer": "synthetic-parent"}
        reference = {"source": {"task": 1, "episode": 310, "instance": 192,
                                "extracted_arrays_and_labels_sha256": "a"*64},
                     "source_label": {"segment_start": 0, "segment_end": 34},
                     "private_original_semantic_json": json.dumps([
                         {"verb": "GRASP", "target": "target", "destination": "", "arm": "UNSPECIFIED"}])}
        rows = []
        for tick in range(48):
            f = frame(tick); action = [0.]*23; action[14] = 1.; action[22] = 1. if tick < 16 else -1.
            f["finger_opening"]["right"] = .05 if tick < 16 else .023 if tick < 19 else .005
            if tick >= 19: f["finger_contact"]["right"] = True
            if tick >= 22:
                f["held"]["right"] = True
                f["target_pose"] = np.eye(4).tolist()
                rotation = Rotation.from_euler("z", 24, degrees=True).as_matrix()
                for i in range(3): f["target_pose"][i][:3] = rotation[i].tolist()
            if tick >= 25:
                f["target_pose"][2][3] = f["hand_poses"]["right"][2][3] = .04
            rows.append({"frame": f, "actual_action23": None if tick == 0 else action})
        return spec, reference, rows

    def materialize(self, root):
        spec, reference, rows = self.fixture(); original = root/"original"; original.mkdir()
        (original/"PRIVATE_reference_trace.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        identity = seed_identity(reference, spec)
        result = {"status": "REFERENCE_LOCAL_SUCCEEDED", "identity": identity, "final_hold_completed": True,
                  "failure": None, "completed_controls": 47, "model_calls": 0, "reference_preparation_sha256": "b"*64}
        (original/"result.json").write_text(json.dumps(result)); captures = []
        for folder, tick in (("before", 0), ("segment_end", 34), ("after_stability", 46), ("after_final_hold", 47)):
            (original/folder).mkdir(); hashes = {}
            for name in ("head.png", "left_wrist.png", "right_wrist.png", "depth.npz", "robot_self_geometry.json",
                         "sensors.json", "proprio.json", "capture.json"):
                path = original/folder/name; path.write_bytes(b"SYNTHETIC ONLY"); hashes[folder+"/"+name] = sha(path)
            captures.append({"control": tick, "files_sha256": hashes})
        (original/"captures.json").write_text(json.dumps(captures))
        seed = {"schema": "h09u-measured-reference-seed-v1", "identity": identity, **extract_seed(spec, rows),
                "training_eligible": False, "reference_preparation_sha256": "b"*64,
                "evidence_sha256": {n: sha(original/n) for n in ("PRIVATE_reference_trace.jsonl", "result.json", "captures.json")}}
        terminal = original/"QUARANTINED_pose_seed.json"; terminal.write_text(json.dumps(seed))
        spec.update(goal_pose_local=seed["goal_pose_local"], pose_evidence_sha256=sha(terminal))
        legacy_review = {"seed_sha256": sha(terminal), "identity_sha256": digest(identity),
                         "goal_pose_sha256": digest(seed["goal_pose_local"]), "reviewer": spec["pose_reviewer"],
                         "decision": "approve", "reviewed_source_current_and_terminal_views": True,
                         "reviewed_contact_update_and_control_ledger": True, "local_outcome_and_pose_accepted": True,
                         "reason": "Synthetic unit-test receipt, not a real parent review."}
        old_review = root/"old-review.json"; old_review.write_text(json.dumps(legacy_review))
        source = root/"reference.json"; source.write_text(json.dumps(reference))
        return terminal, old_review, source, spec, reference

    def release(self, root):
        terminal, old_review, source, spec, reference = self.materialize(root)
        mocked = patch.dict("native_teacher_pregrasp_seed.TERMINAL_SHAS", {(1, 310, 192): sha(terminal)})
        mocked.start(); self.addCleanup(mocked.stop)
        seed = build_sidecar(terminal, old_review, source)
        path = root/"sidecar.json"; path.write_text(json.dumps(seed))
        spec.update(goal_pose_local=seed["goal_pose_local"], pose_evidence_sha256=sha(path))
        review = {"schema": REVIEW_SCHEMA, "seed_profile": PROFILE, "seed_sha256": sha(path),
                  "identity_sha256": digest(seed["identity"]), "goal_pose_sha256": digest(seed["goal_pose_local"]),
                  "selection_sha256": digest(seed["selection"]), "reviewer": spec["pose_reviewer"], "decision": "approve",
                  "reviewed_source_current_and_terminal_views": True, "reviewed_contact_update_and_control_ledger": True,
                  "reviewed_precontact_phase_and_half_closed_aperture": True, "local_outcome_and_pose_accepted": True,
                  "reason": "Synthetic unit-test receipt only; never real acquired data."}
        rp = root/"new-review.json"; rp.write_text(json.dumps(review))
        return path, rp, spec, reference, terminal

    def test_first_precontact_not_terminal_or_full_open_and_no_actor_aperture(self):
        spec, _, rows = self.fixture(); result = select_precontact(spec, rows)
        selection = result["selection"]
        self.assertEqual([selection[k] for k in ("first_close_tick", "selected_tick", "first_contact_tick", "first_held_tick")],
                         [16, 18, 19, 22])
        self.assertEqual(selection["measured_finger_aperture_m"], .023)
        self.assertFalse(selection["aperture_is_full_open_evidence"])
        self.assertFalse(selection["aperture_is_execution_target"])
        self.assertFalse(np.allclose(result["goal_pose_local"], result["terminal_goal_pose_local"]))
        self.assertEqual(result["outcome"]["outcome"], "SUCCEEDED")

    def test_missing_clock_close_contact_stability_and_retries_rejected(self):
        spec, _, original = self.fixture()
        cases = []
        rows = copy.deepcopy(original); rows.pop(18); cases.append(rows)
        rows = copy.deepcopy(original); rows[20]["actual_action23"][22] = 1.; cases.append(rows)
        rows = copy.deepcopy(original); rows[15]["actual_action23"][22] = 0.; cases.append(rows)
        rows = copy.deepcopy(original); rows[14]["frame"]["finger_contact"]["right"] = True; cases.append(rows)
        rows = copy.deepcopy(original); rows[18]["frame"]["finger_contact"]["left"] = True; cases.append(rows)
        rows = copy.deepcopy(original); rows[18]["frame"]["finger_contact"]["right"] = None; cases.append(rows)
        rows = copy.deepcopy(original); rows[18]["frame"]["target_pose"][0][3] = .006; cases.append(rows)
        rows = copy.deepcopy(original); rows[23]["frame"]["held"]["right"] = False; cases.append(rows)
        rows = copy.deepcopy(original); rows[18]["frame"]["finger_opening"]["right"] = float("nan"); cases.append(rows)
        rows = copy.deepcopy(original)
        for r in rows: r["frame"]["target_pose"][2][3] = 0.
        cases.append(rows)
        for number, rows in enumerate(cases):
            with self.subTest(case=number), self.assertRaises(ValueError): select_precontact(spec, rows)

    def test_full_reference_validation_sidecar_and_explicit_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            p, r, spec, ref, terminal = self.release(Path(folder))
            before = sha(terminal)
            self.assertEqual(validate_seed_release(p, r, spec, ref, "b"*64, profile=PROFILE)["selection"]["selected_tick"], 18)
            self.assertEqual(sha(terminal), before)
            for profile in (None, "unknown", False):
                with self.subTest(profile=profile), self.assertRaises(ValueError):
                    validate_seed_release(p, r, spec, ref, "b"*64, profile=profile)

    def test_selection_pose_review_source_and_complete_evidence_tamper_rejected(self):
        mutations = ("selected_tick", "goal_pose_local", "reference_source_sha256", "training_eligible")
        for kind in mutations:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                p, rp, spec, ref, _ = self.release(Path(folder)); seed = json.loads(p.read_text())
                if kind == "selected_tick": seed["selection"][kind] += 1
                elif kind == "goal_pose_local": seed[kind][0][3] += .00001
                elif kind == "reference_source_sha256": seed[kind] = "0"*64
                else: seed[kind] = True
                p.write_text(json.dumps(seed)); spec["pose_evidence_sha256"] = sha(p)
                with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, ref, "b"*64, profile=PROFILE)
        with tempfile.TemporaryDirectory() as folder:
            p, rp, spec, ref, terminal = self.release(Path(folder)); original = rp.read_text()
            for field in ("decision", "selection_sha256", "reviewed_precontact_phase_and_half_closed_aperture"):
                review = json.loads(original); review[field] = False; rp.write_text(json.dumps(review))
                with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, ref, "b"*64, profile=PROFILE)
            rp.write_text(original)
            (terminal.parent/"after_final_hold/head.png").write_bytes(b"corrupt")
            with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, ref, "b"*64, profile=PROFILE)

    def test_heldout_unregistered_source_missing_old_review_and_alias_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); t, r, source, spec, ref = self.materialize(root)
            # Synthetic TRAIN labels do not bypass the pinned real terminal hashes.
            with self.assertRaises(ValueError): build_sidecar(t, r, source)
            with patch.dict("native_teacher_pregrasp_seed.TERMINAL_SHAS", {(1, 310, 192): sha(t)}):
                alias = root/"alias.json"; alias.symlink_to(t)
                with self.assertRaises(ValueError): build_sidecar(alias, r, source)
                review = json.loads(r.read_text()); review["decision"] = "pending"; r.write_text(json.dumps(review))
                with self.assertRaises(ValueError): build_sidecar(t, r, source)
            seed = json.loads(t.read_text()); seed["identity"].update(instance=71, episode=247); t.write_text(json.dumps(seed))
            with self.assertRaises(ValueError): build_sidecar(t, r, source)

    def test_actual_collector_forwards_profile_and_rejects_mixed_release(self):
        tree = ast.parse((ROOT/"scripts/vlm_sft/native_teacher_collect.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "validate_seed_release"]
        self.assertEqual(len(calls), 1)
        self.assertEqual({k.arg: ast.unparse(k.value) for k in calls[0].keywords}, {"profile": "release.get('seed_profile')"})
        for value in (None, "unknown", PROFILE):
            with self.assertRaisesRegex(ValueError, "profile"):
                require_release({"seed_profile": value}, "irrelevant", "irrelevant")

    def test_exact_spec_pose_preparation_source_and_separate_sidecar_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p, rp, spec, ref, terminal = self.release(root)
            original_pose = copy.deepcopy(spec["goal_pose_local"])
            spec["goal_pose_local"][0][3] += 1e-15
            with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, ref, "b"*64, profile=PROFILE)
            spec["goal_pose_local"] = original_pose
            with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, ref, "c"*64, profile=PROFILE)
            wrong_ref = copy.deepcopy(ref); wrong_ref["source"]["instance"] = 71
            with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, wrong_ref, "b"*64, profile=PROFILE)
            inside = terminal.parent/"forbidden-sidecar.json"; inside.write_bytes(p.read_bytes())
            with self.assertRaisesRegex(ValueError, "separate"):
                validate_seed_release(inside, rp, spec, ref, "b"*64, profile=PROFILE)
            source = root/"reference.json"; source.write_text(source.read_text()+"\n")
            with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, ref, "b"*64, profile=PROFILE)

    def test_complete_success_and_final_hold_cannot_be_replaced_by_contact(self):
        spec, _, rows = self.fixture()
        # Contact and held alone still fail the original 30/25mm lift predicate.
        for row in rows:
            row["frame"]["target_pose"][2][3] *= .5
            row["frame"]["hand_poses"]["right"][2][3] *= .5
        with self.assertRaises(ValueError): select_precontact(spec, rows)
        with tempfile.TemporaryDirectory() as folder:
            p, rp, spec, ref, terminal = self.release(Path(folder))
            result_path = terminal.parent/"result.json"
            result = json.loads(result_path.read_text()); result["final_hold_completed"] = False
            result_path.write_text(json.dumps(result))
            with self.assertRaises(ValueError): validate_seed_release(p, rp, spec, ref, "b"*64, profile=PROFILE)


if __name__ == "__main__": unittest.main()
