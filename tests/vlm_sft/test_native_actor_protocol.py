import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts/vlm_sft"), str(ROOT/"src")]
import native_actor_protocol as p
from common import CAMERAS, sha, write_json
from live import runtime_proprio as legacy_proprio
from modeling import messages, encode, collate
from native_grasp_sources import reserve_groups
from semantic_robot.v2.kinematics import RobotModel


class PoseProtocolTests(unittest.TestCase):
    def fixture(self):
        jac = {arm: np.zeros((6, 18)) for arm in ("left", "right", "torso")}
        jac["right"][5, 17] = 1
        model = RobotModel.from_reference(np.zeros(18), -np.ones(18)*4, np.ones(18)*4,
            {arm: (np.zeros(3), [0, 0, 0, 1]) for arm in jac}, jac)
        state = model.state(np.zeros(18), np.array([.05, .05]), np.zeros(3))
        clock = {"prefix_control": 396, "native_control": 12}
        proprio = p.runtime_proprio(model, state, clock=clock, expected_clock=clock, calibration_sha256=model.sha)
        raw = {}
        for view in CAMERAS:
            n = 720 if view == "head" else 480
            b = BytesIO(); Image.new("RGB", (n, n), (10, 20, 30)).save(b, format="PNG"); raw[view] = b.getvalue()
        actor = p.actor_input("Move the bin", "verb=GRASP; target=trash can", proprio,
                              {v: hashlib.sha256(b).hexdigest() for v, b in raw.items()}, [])
        return model, state, clock, actor, raw

    def test_current_robot_yaw_is_observable_with_identical_legacy_position(self):
        model, state, clock, actor, _ = self.fixture()
        q = state.q.copy(); q[17] = np.deg2rad(3)
        moved = model.state(q, state.gripper, state.base_velocity)
        self.assertEqual(legacy_proprio(state), legacy_proprio(moved))
        changed = p.runtime_proprio(model, moved, clock=clock, expected_clock=clock, calibration_sha256=model.sha)
        self.assertNotEqual(actor["proprio"]["eef_rotation_base"], changed["eef_rotation_base"])
        self.assertAlmostEqual(changed["eef_rotation_base"]["right"][1][0], np.sin(np.deg2rad(3)), places=6)

    def test_robot_field_whitelist_bad_shapes_booleans_nan_and_se3(self):
        _, _, _, a, _ = self.fixture()
        for key in ("object_pose", "world_pose", "goal_status", "future_q", "seed_pose"):
            with self.assertRaises(ValueError): p.validate_proprio({**a["proprio"], key: 1})
            with self.assertRaises(ValueError): p.validate_actor({**a, key: 1})
        for matrix in (np.eye(3)*.1, np.diag([1, 1, -1]), np.full((3, 3), np.nan)):
            bad = copy.deepcopy(a["proprio"]); bad["eef_rotation_base"]["right"] = matrix.tolist()
            with self.assertRaises(ValueError): p.validate_proprio(bad)
        for values in ([True]*7, [0]*6, [float("inf")]*7):
            bad = copy.deepcopy(a["proprio"]); bad["joint_positions_rad"]["right"] = values
            with self.assertRaises(ValueError): p.validate_proprio(bad)

    def test_clock_and_calibration_mismatch_rejected_without_nearest_frame(self):
        model, state, clock, _, _ = self.fixture()
        for bad in ({**clock, "native_control": 13}, {**clock, "native_control": True}, {"native_control": 12}):
            with self.assertRaises(ValueError): p.runtime_proprio(model, state, clock=bad, expected_clock=clock, calibration_sha256=model.sha)
        with self.assertRaises(ValueError): p.runtime_proprio(model, state, clock=clock, expected_clock=clock, calibration_sha256="0"*64)
        state.poses["right"][0][0] += .001
        with self.assertRaises(ValueError): p.runtime_proprio(model, state, clock=clock, expected_clock=clock, calibration_sha256=model.sha)

    def test_same_protocol_base_adapter_training_and_live_prefix_images(self):
        _, _, _, actor, raw = self.fixture()
        registered = [actor["active_instruction"]]
        rows = []
        for variant in ("base", "finetuned"):
            payload = p.request_payload(actor, raw, variant)
            row, images = p.parse_request(payload, registered_instructions=registered); rows.append(row)
            train = p.training_row(actor, "RIGHT_YAW_PLUS")
            self.assertEqual(messages(train, images), messages(row, images))
            self.assertEqual(images["head"].size, (256, 256))
            with self.assertRaises(ValueError): p.parse_request(payload, registered_instructions=[])
            bad = copy.deepcopy(payload); bad["images"]["head"] = bad["images"]["left_wrist"]
            with self.assertRaises(ValueError): p.parse_request(bad, registered_instructions=registered)
        self.assertEqual(rows[0], rows[1])
        text = rows[0]["text"]
        for field in ("prefix_control", "native_control", "calibration_sha256", "current_rgb_sha256", "seed_pose"):
            self.assertNotIn(field, text)

    def test_training_prefix_is_inference_prefix_response_only_mask_real_tensors(self):
        import torch
        _, _, _, actor, raw = self.fixture()
        row, images = p.parse_request(p.request_payload(actor, raw, "base"), registered_instructions=[actor["active_instruction"]])
        class Tokenizer:
            unk_token_id=0; eos_token_id=300
            def convert_tokens_to_ids(self, token): return 300 if token == "<|im_end|>" else 0
            def convert_ids_to_tokens(self, value): return "<|im_end|>" if value == 300 else "?"
            def encode(self, text, **kwargs): return [20, 21, 22]
        class Processor:
            tokenizer=Tokenizer()
            def apply_chat_template(self, msg, **kwargs):
                text = msg[0]["content"] + msg[1]["content"][-1]["text"]
                ids = torch.tensor([[1]+[c+1 for c in text.encode()[::8]]])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        proc = Processor(); plain = encode(proc, row, images, supervised=False)
        training = encode(proc, p.training_row(actor, "RIGHT_UP"), images, supervised=True)
        n = plain["input_ids"].shape[1]
        self.assertTrue(torch.equal(training["input_ids"][:, :n], plain["input_ids"]))
        self.assertTrue((training["labels"][:, :n] == -100).all())
        self.assertEqual(training["labels"][0, -1].item(), 300)
        longer = copy.deepcopy(actor); longer["history"] = ["RIGHT_LEFT", "RIGHT_YAW_PLUS"]
        second = encode(proc, p.training_row(longer, "RIGHT_CLOSE"), images, supervised=True)
        batch = collate([training, second], 0)
        self.assertFalse(((batch["labels"] != -100) & (batch["attention_mask"] == 0)).any())
        broken = {**row, "text": row["text"]+" hidden target"}
        with self.assertRaises(ValueError): messages(broken, images)

    def test_capture_projection_clock_hash_and_legacy_proprio_consistency(self):
        model, state, clock, actor, raw = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for v, b in raw.items(): (root/(v+".png")).write_bytes(b)
            for name in ("depth.npz", "robot_self_geometry.json", "sensors.json"): (root/name).write_bytes(b"fixture")
            write_json(root/"proprio.json", legacy_proprio(state))
            c = {"clock": clock, "q": state.q.tolist(), "gripper": state.gripper.tolist(), "kinematic_model_sha256": model.sha,
                "source": "render_only_current_onboard_RGBD_and_robot_joint_FK", "scene_truth": False, "array_layout": {}, "depth_array_sha256": {},
                "files_sha256": {f.name: sha(f) for f in root.iterdir()}}
            write_json(root/"capture.json", c)
            result, hashes, binding = p.from_capture(root, model, capture_sha256=sha(root/"capture.json"), expected_clock=clock, calibration_sha256=model.sha)
            self.assertEqual(result, actor["proprio"]); self.assertEqual(hashes, actor["current_rgb_sha256"])
            self.assertTrue(binding["not_actor_input"])
            with self.assertRaises(ValueError): p.from_capture(root, model, capture_sha256="0"*64, expected_clock=clock, calibration_sha256=model.sha)
            (root/"proprio.json").write_bytes(b"{}")
            with self.assertRaises(ValueError): p.from_capture(root, model, capture_sha256=sha(root/"capture.json"), expected_clock=clock, calibration_sha256=model.sha)

    def test_split_reserved_before_actions_and_near_duplicates_stay_grouped(self):
        counts = json.loads((ROOT/"configs/vlm_sft/h09r_train_feasibility_counts.json").read_text())
        groups = reserve_groups(counts)
        self.assertEqual([s for s, _ in groups], ["train", "train", "heldout", "heldout"])
        self.assertEqual(len({r["instance"] for _, r in groups}), 4)
        self.assertEqual(groups[0][1]["instance"], 192)
        blocked = {tuple(x) for rows in counts["exclusions"].values() for x in rows}
        self.assertTrue(all((r["task"], r["instance"]) not in blocked for _, r in groups))
        duplicate = copy.deepcopy(counts); duplicate["sources"].append(groups[0][1])
        with self.assertRaises(ValueError): reserve_groups(duplicate)


if __name__ == "__main__": unittest.main()
