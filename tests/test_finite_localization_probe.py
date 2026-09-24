import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts/semantic_robot"), str(REPO / "tests/semantic_robot"), str(REPO / "src")]
import probe_finite_localization as probe
import probe_shared_vlm as shared
from semantic_robot.v2.affordance import SurfaceChoice
from semantic_robot.v2.finite_localization import region_request, surface_request, region_surfaces, region_boxes
from semantic_robot.v2.harness import Goal
from test_v2 import fixture


class FiniteProbeTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((REPO / "configs/semantic_robot/h50_finite_localization_probe.json").read_text())

    def test_frozen_spec_and_all_limits(self):
        probe.validate_spec(self.spec)
        for key, value in (("max_calls", 13), ("training_steps", 1), ("simulator_resets", 1),
                           ("allocator_limit_mib", 5000), ("reserve_mib", 2000),
                           ("non_torch_allowance_mib", 513), ("non_torch_allowance_mib", 2048),
                           ("gpu", 1), ("gpu_uuid", "another-gpu"), ("training_pids", [123]),
                           ("seed", 18), ("allocator_limit_mib", 4863), ("reserve_mib", 2049),
                           ("supervisor_seconds", 901), ("max_seconds", 601), ("image_max_side", 640),
                           ("quantization", None)):
            spec = copy.deepcopy(self.spec); spec[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): probe.validate_spec(spec)

    def test_model_manifest_and_inference_policy_are_frozen(self):
        for key, value in (("model", "/another/model"), ("revision", "another-revision"),
                           ("model_files", {"config.json": "0" * 64}), ("chat_stop_policy", None)):
            spec = copy.deepcopy(self.spec); spec[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): probe.validate_spec(spec)

    def test_case_contract_no_private_sources_free_uv_or_unpinned_fields(self):
        for key, value in (("view", "right_wrist"), ("format", "private"), ("capture", "../secret"),
                           ("source_run", "relative"), ("target_uv", [.4, .5])):
            spec = copy.deepcopy(self.spec); spec["cases"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): probe.validate_spec(spec)
        spec = copy.deepcopy(self.spec)
        spec["cases"][0]["files_sha256"]["PRIVILEGED_CONTACTS.jsonl"] = "0" * 64
        with self.assertRaises(ValueError): probe.validate_spec(spec)
        spec = copy.deepcopy(self.spec); spec["cases"][1]["id"] = spec["cases"][0]["id"]
        with self.assertRaises(ValueError): probe.validate_spec(spec)

    def make_capture(self, root):
        model, state = fixture()
        folder = root / "decision_000"; folder.mkdir()
        (root / "robot_calibration.json").write_text(json.dumps(model.spec))
        (folder / "proprio.json").write_text(json.dumps({"q": state.q.tolist(), "gripper": state.gripper.tolist()}))
        raw = {v: np.zeros((100, 100, 3), np.uint8) for v in probe.VIEWS}
        depth = {v: np.ones((100, 100), np.float32) for v in probe.VIEWS}
        for view in probe.VIEWS: Image.fromarray(raw[view]).save(folder / ("CURRENT_" + view.upper() + "_RAW.png"))
        np.savez_compressed(folder / "depth.npz", **depth)
        sensors = {v: dict(rgb_sha256=hashlib.sha256(raw[v].tobytes()).hexdigest(),
            depth_sha256=hashlib.sha256(depth[v].tobytes()).hexdigest(), modalities=["rgb", "depth_linear"],
            depth_units="metres", depth_convention="distance_to_image_plane", same_sensor_current_render=True,
            render_barrier_updates=4, control_steps_in_capture=0, snapshot_id=2) for v in probe.VIEWS}
        (folder / "depth_receipt.json").write_text(json.dumps(sensors))
        case = dict(id="test", source_run=str(root), capture="decision_000", format="run_v2", view="head",
                    goal=asdict(Goal("pick", "object", "right", "independent feedback required")))
        case["files_sha256"] = {name: shared.sha(root / name) for name in probe.expected_files(case)}
        return case

    def test_real_public_loader_exact_hashes_and_no_normalization(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); case = self.make_capture(root)
            arguments, binding = probe.load_case(case)
            self.assertEqual(len(binding), 64)
            self.assertEqual(arguments["views"], ("head",))
            self.assertEqual(arguments["raw"]["head"].dtype, np.uint8)
            np.testing.assert_array_equal(arguments["depths"]["head"], 1)
            spec = root / "decision_000/proprio.json"
            spec.write_text(spec.read_text() + " ")
            with self.assertRaises(ValueError): probe.load_case(case)

    def test_rehashed_stale_or_mixed_sensor_receipts_are_rejected(self):
        for key, value in (("render_barrier_updates", 3), ("control_steps_in_capture", 1),
                           ("same_sensor_current_render", False), ("snapshot_id", 3),
                           ("rgb_sha256", "0" * 64), ("depth_units", "millimetres")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as path:
                root = Path(path); case = self.make_capture(root)
                filename = "decision_000/depth_receipt.json"; target = root / filename
                sensors = json.loads(target.read_text()); sensors["head"][key] = value
                target.write_text(json.dumps(sensors)); case["files_sha256"][filename] = shared.sha(target)
                with self.assertRaises(ValueError): probe.load_case(case)

    def test_nonlocal_and_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); real = root / "data"; real.write_text("{}"); link = root / "link"; link.symlink_to(real)
            with self.assertRaises(ValueError): probe._read(root, "link", shared.sha(real))
            with self.assertRaises(ValueError): probe._read(root / "other", "../data", shared.sha(real))

    def test_baseline_is_separate_from_choices_and_uses_same_raw_target(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); arguments, binding = probe.load_case(self.make_capture(root))
            request = probe.baseline_request(arguments, binding)
            self.assertEqual(probe.validate_request(request), 320)
            self.assertEqual(request.allowed, ())
            self.assertEqual(json.loads(request.text), {"target": "object", "goal_kind": "pick"})
            np.testing.assert_array_equal(request.images[0], arguments["raw"]["head"])
            with self.assertRaises(ValueError): probe.validate_request(replace(request, allowed=(SurfaceChoice(),)))
            region = region_request(arguments["goal"], "head", arguments["raw"]["head"], binding)
            self.assertEqual(probe.validate_request(region), 32)
            with self.assertRaises(ValueError): probe.validate_request(replace(region, allowed=region.allowed[:-1]))
            box = region_boxes(100, 100)[4]
            candidates = region_surfaces("head", box, arguments["depths"], arguments["model"], arguments["state"])
            surface = surface_request(arguments["goal"], "head", arguments["raw"]["head"], binding, box, candidates)
            self.assertEqual(probe.validate_request(surface), 32)
            with self.assertRaises(ValueError): probe.validate_request(replace(surface, images=surface.images[:2]))
            with self.assertRaises(ValueError): probe.validate_request(replace(surface, allowed=surface.allowed[1:]))

    def test_shared_launcher_keeps_exact_new_entrypoint_and_exclusive_receipt(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); config = root / "spec.json"; config.write_text(json.dumps(self.spec))
            output = root / "run"
            child = Mock(pid=123)
            with patch.object(shared, "require_source_and_device"), patch.object(shared, "check_resources"), \
                 patch.object(shared, "gpu_snapshot", return_value={}), \
                 patch.object(shared.subprocess, "check_output", return_value="f" * 40), \
                 patch.object(shared.subprocess, "Popen", return_value=child) as popen, patch("builtins.print"):
                shared.launch(config, output, validator=probe.validate_spec, entrypoint=probe.__file__)
                argv = popen.call_args.args[0]
                self.assertEqual(argv[1], str(Path(probe.__file__).resolve()))
                self.assertEqual(argv[2], "--supervise")
                with self.assertRaises(FileExistsError):
                    shared.launch(config, output, validator=probe.validate_spec, entrypoint=probe.__file__)
                self.assertEqual(popen.call_count, 1)

    def test_shared_launcher_rejects_external_worker_before_process_creation(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); config = root / "spec.json"; config.write_text(json.dumps(self.spec))
            script = root / "fake.py"; script.write_text("raise AssertionError")
            with patch.object(shared.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    shared.launch(config, root / "run", validator=probe.validate_spec, entrypoint=script)
                popen.assert_not_called()


if __name__ == "__main__": unittest.main()
