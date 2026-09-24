"""CPU-only tests for the bounded shared-GPU saved-request diagnostic."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import subprocess
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.util.spec_from_file_location("probe_shared_vlm", ROOT / "scripts/semantic_robot/probe_shared_vlm.py")
probe = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(probe)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(gpu=2, gpu_uuid="gpu-test", training_pids=[40], reserve_mib=2048,
                         allocator_limit_mib=4864, non_torch_allowance_mib=512)
        self.snapshot = dict(index=2, uuid="gpu-test", free_mib=7489, apps=[dict(pid=40, used_mib=73644)])

    def check(self, before=True):
        probe.check_resources(self.snapshot, self.spec, own_pid=41, before_load=before)

    def test_exact_initial_budget(self):
        self.check()
        self.snapshot["free_mib"] = 7423
        with self.assertRaisesRegex(ValueError, "headroom"):
            self.check()

    def test_own_process_not_treated_as_peer(self):
        self.snapshot["apps"].append(dict(pid=41, used_mib=4500))
        self.snapshot["free_mib"] = 2048
        self.check(False)
        self.snapshot["free_mib"] = 2047
        with self.assertRaisesRegex(ValueError, "headroom"):
            self.check(False)

    def test_new_or_missing_peer_fails_closed(self):
        self.snapshot["apps"].append(dict(pid=42, used_mib=100))
        with self.assertRaisesRegex(ValueError, "process identity"):
            self.check()
        self.snapshot["apps"] = []
        with self.assertRaisesRegex(ValueError, "process identity"):
            self.check()

    def test_wrong_gpu_rejected(self):
        self.snapshot["uuid"] = "different"
        with self.assertRaisesRegex(ValueError, "GPU identity"):
            self.check()

    def test_recheck_accounts_for_only_own_existing_context(self):
        self.snapshot["apps"].append(dict(pid=41, used_mib=400))
        self.snapshot["free_mib"] = 7489 - 400
        self.check()
        self.snapshot["free_mib"] = 5000
        self.check(False)
        with self.assertRaisesRegex(ValueError, "headroom"):
            self.check(True)

    def test_non_torch_overhead_cannot_overrun_total_limit(self):
        self.snapshot["apps"].append(dict(pid=41, used_mib=5377))
        with self.assertRaisesRegex(ValueError, "total GPU memory budget"):
            self.check(False)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT / "configs/semantic_robot/h45_shared_small_vlm_probe.json").read_text())

    def test_frozen_spec(self):
        probe.validate_spec(self.spec)

    def test_limits_cannot_be_expanded(self):
        for key, value in [("allocator_limit_mib", 8000), ("max_seconds", 3600),
                           ("reserve_mib", 0), ("image_max_side", 640), ("training_steps", 1)]:
            with self.subTest(key=key):
                changed = dict(self.spec); changed[key] = value
                with self.assertRaises(ValueError):
                    probe.validate_spec(changed)

    def test_duplicate_or_extra_case_rejected(self):
        self.spec["cases"] = [self.spec["cases"][0]] * 4
        with self.assertRaises(ValueError):
            probe.validate_spec(self.spec)

    def test_paired_neutral_spec_and_rejections(self):
        paired = json.loads((ROOT / "configs/semantic_robot/h47_neutral_observation_probe.json").read_text())
        probe.validate_spec(paired)
        for key, value in [("request_profile", "arbitrary"), ("receipt", "planner.json"),
                           ("sha256", "changed")]:
            with self.subTest(key=key):
                changed = json.loads(json.dumps(paired))
                changed["cases"][1][key] = value
                with self.assertRaises(ValueError):
                    probe.validate_spec(changed)
        paired["cases"][1] = paired["cases"][0]
        with self.assertRaises(ValueError):
            probe.validate_spec(paired)

    def test_original_probe_cannot_silently_change_prompt(self):
        self.spec["cases"][0]["request_profile"] = probe.NEUTRAL_PROFILE
        with self.assertRaises(ValueError):
            probe.validate_spec(self.spec)

    def test_unknown_probe_kind_rejected(self):
        self.spec["probe_kind"] = "arbitrary"
        with self.assertRaises(ValueError):
            probe.validate_spec(self.spec)

    def test_static_paired_spec(self):
        spec = json.loads((ROOT / "configs/semantic_robot/h48_static_localization_probe.json").read_text())
        probe.validate_spec(spec)
        spec["cases"][0]["request_profile"] = "original"
        with self.assertRaises(ValueError):
            probe.validate_spec(spec)

    def test_stop_only_owned_process_and_escalate(self):
        child = Mock(); child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("owned", 15), 0]
        probe.stop_owned_child(child)
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)

    def test_exited_worker_not_signaled(self):
        child = Mock(); child.poll.return_value = 0
        probe.stop_owned_child(child)
        child.terminate.assert_not_called(); child.kill.assert_not_called()


class SavedRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.label = "CURRENT_HEAD_RAW"
        image = Image.new("RGB", (800, 400), (30, 40, 50))
        image.save(self.root / (self.label + ".png"))
        image.thumbnail((640, 640))
        self.record = {
            "request_without_pixel_duplicates": dict(kind="observe", system="public system", text="public text",
                                                       images=[dict(label=self.label)], allowed=[]),
            "result": {"images": [dict(label=self.label, size=list(image.size),
                                        pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest())]},
            "privileged_unused": {"target_world_pose": [999, 999, 999]},
        }

    def write(self):
        path = self.root / "receipt.json"
        path.write_text(json.dumps(self.record))
        return dict(receipt=path.name, sha256=probe.sha(path))

    def test_public_request_and_all_images_only(self):
        request, messages, receipts = probe.load_saved_request(self.root, self.write(), 320)
        self.assertEqual(request["text"], "public text")
        self.assertEqual(receipts[0]["size"], [320, 160])
        self.assertNotIn("target_world_pose", str(messages))
        self.assertEqual(messages[1]["content"][0]["text"], self.label)

    def test_changed_pixels_rejected(self):
        case = self.write()
        Image.new("RGB", (800, 400), (1, 2, 3)).save(self.root / (self.label + ".png"))
        with self.assertRaisesRegex(ValueError, "RGB pixels"):
            probe.load_saved_request(self.root, case, 320)

    def test_changed_receipt_rejected(self):
        case = self.write()
        self.record["request_without_pixel_duplicates"]["text"] = "changed"
        self.write()
        with self.assertRaisesRegex(ValueError, "SHA"):
            probe.load_saved_request(self.root, case, 320)

    def test_private_extra_request_field_rejected(self):
        self.record["request_without_pixel_duplicates"]["private_pose"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "public actor request"):
            probe.load_saved_request(self.root, self.write(), 320)

    def test_image_order_identity_rejected(self):
        self.record["result"]["images"][0]["label"] = "CURRENT_LEFT_WRIST_RAW"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            probe.load_saved_request(self.root, self.write(), 320)

    def test_path_escape_rejected(self):
        case = self.write()
        case["receipt"] = "../outside.json"
        with self.assertRaisesRegex(ValueError, "path or SHA"):
            probe.load_saved_request(self.root, case, 320)

    def test_neutral_profile_only_replaces_example_block(self):
        prefix, suffix = "Public context. ", "visible: choose from images.\nNo invented details."
        original_system = prefix + probe.EXAMPLE_MARKER + json.dumps(probe.NEGATIVE_EXAMPLE) + "\n" + suffix
        self.record["request_without_pixel_duplicates"]["system"] = original_system
        case = self.write()
        original, old_messages, old_images = probe.load_saved_request(self.root, case, 320)
        case["request_profile"] = probe.NEUTRAL_PROFILE
        neutral, new_messages, new_images = probe.load_saved_request(self.root, case, 320)
        self.assertEqual(new_images, old_images)
        self.assertEqual(new_messages[1], old_messages[1])
        self.assertEqual({k: v for k, v in neutral.items() if k != "system"},
                         {k: v for k, v in original.items() if k != "system"})
        self.assertTrue(neutral["system"].startswith(prefix))
        self.assertTrue(neutral["system"].endswith(suffix))
        self.assertNotIn("Target not identified", neutral["system"])
        self.assertEqual(original["system"], original_system)

    def test_neutral_missing_modified_or_duplicate_marker_rejected(self):
        original = self.record["request_without_pixel_duplicates"]
        valid = probe.EXAMPLE_MARKER + json.dumps(probe.NEGATIVE_EXAMPLE) + "\nunchanged tail"
        for system in ("no marker", valid + probe.EXAMPLE_MARKER,
                       valid.replace('"visible": false', '"visible": true')):
            with self.subTest(system=system):
                with self.assertRaises(ValueError):
                    probe.apply_request_profile({**original, "system": system}, probe.NEUTRAL_PROFILE)
        with self.assertRaises(ValueError):
            probe.apply_request_profile({**original, "system": valid, "kind": "plan"}, probe.NEUTRAL_PROFILE)
        with self.assertRaises(ValueError):
            probe.apply_request_profile(original, "unregistered")

    def add_other_view(self):
        label = "PREVIOUS_RIGHT_WRIST_RAW"
        image = Image.new("RGB", (320, 320), (99, 21, 15)); image.save(self.root / (label + ".png"))
        self.record["request_without_pixel_duplicates"]["images"].append(dict(label=label))
        self.record["result"]["images"].append(dict(label=label, size=list(image.size),
                                                   pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        return label

    def test_static_uses_only_original_goal_and_current_raw(self):
        self.add_other_view()
        self.record["request_without_pixel_duplicates"]["text"] = json.dumps({
            "current_goal": {"kind": "pick", "target": "public object", "done_when": "unused"},
            "current_robot": {"unused_coordinates": [1, 2, 3]}})
        case = self.write(); case["request_profile"] = "static_head_raw_v1"
        request, messages, receipts = probe.load_saved_request(self.root, case, 320)
        self.assertEqual(json.loads(request["text"]), dict(target="public object", goal_kind="pick"))
        self.assertEqual([r["label"] for r in receipts], [self.label])
        self.assertEqual(request["images"], [dict(label=self.label)])
        self.assertEqual(len(messages[1]["content"]), 3)
        self.assertNotIn("unused", str(messages))

    def test_static_still_checks_omitted_original_image(self):
        omitted = self.add_other_view()
        case = self.write(); case["request_profile"] = "static_head_raw_v1"
        Image.new("RGB", (320, 320), (0, 0, 0)).save(self.root / (omitted + ".png"))
        with self.assertRaisesRegex(ValueError, "RGB pixels"):
            probe.load_saved_request(self.root, case, 320)

    def test_static_missing_views_rejected(self):
        case = self.write(); case["request_profile"] = "static_three_raw_v1"
        with self.assertRaisesRegex(ValueError, "current RAW view absent"):
            probe.load_saved_request(self.root, case, 320)


class StaticLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.visible = dict(visible=True, view="head", target_uv=[0.4, 0.5], note="Public visible object.")

    def test_valid_visible_and_invisible(self):
        parsed = probe.parse_static_location(json.dumps(self.visible), "static_head_raw_v1")
        self.assertEqual(parsed, self.visible)
        self.assertNotIn("effect", parsed)
        probe.parse_static_location(json.dumps(dict(visible=False, view="none", target_uv=None,
                                                    note="Object not identified.")), "static_three_raw_v1")

    def test_bad_types_coordinates_unsupplied_view_and_extra_facts(self):
        for change in (dict(visible=1), dict(view="right_wrist"), dict(target_uv=[True, .4]),
                       dict(target_uv=[float("nan"), .4]), dict(target_uv=[1.1, .4]),
                       dict(effect=True), dict(note=""), dict(visible=False)):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    probe.parse_static_location(json.dumps({**self.visible, **change}), "static_head_raw_v1")


class ModelManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / "config.json"; self.file.write_text("{}")
        self.manifest = {self.file.name: probe.sha(self.file)}

    def test_complete_snapshot_with_download_cache(self):
        (self.root / ".cache").mkdir()
        probe.validate_model_files(self.root, self.manifest)

    def test_added_generation_configuration_rejected(self):
        (self.root / "generation_config.json").write_text('{"eos_token_id": 123}')
        with self.assertRaisesRegex(ValueError, "manifest"):
            probe.validate_model_files(self.root, self.manifest)

    def test_changed_config_rejected(self):
        self.file.write_text('{"eos_token_id": 123}')
        with self.assertRaisesRegex(ValueError, "file changed"):
            probe.validate_model_files(self.root, self.manifest)

    def test_symlink_and_extra_folder_rejected(self):
        self.file.rename(self.root / "original.json")
        self.file.symlink_to(self.root / "original.json")
        with self.assertRaises(ValueError):
            probe.validate_model_files(self.root, self.manifest)


class ChatStopTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleNamespace(eos_token="<|im_end|>", eos_token_id=248046,
                       convert_tokens_to_ids=lambda name: {"<|endoftext|>": 248044, "<|im_end|>": 248046}[name])
        self.generation = SimpleNamespace(eos_token_id=248044)
        self.policy = dict(model_default_eos_id=248044, tokenizer_eos_id=248046,
                           generation_eos_ids=[248044, 248046])

    def test_original_missing_chat_eos_still_rejected_without_opt_in(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            probe.configure_chat_stops(self.tokenizer, self.generation, None)

    def test_explicit_policy_preserves_text_end_and_adds_chat_end(self):
        receipt = probe.configure_chat_stops(self.tokenizer, self.generation, self.policy)
        self.assertEqual(receipt["original_generation_eos"], 248044)
        self.assertEqual(self.generation.eos_token_id, [248044, 248046])

    def test_matching_existing_model_unchanged(self):
        self.generation.eos_token_id = [248044, 248046]
        probe.configure_chat_stops(self.tokenizer, self.generation, None)
        self.assertEqual(self.generation.eos_token_id, [248044, 248046])

    def test_wrong_stop_identity_rejected(self):
        self.tokenizer.eos_token_id = 123
        with self.assertRaisesRegex(ValueError, "identity changed"):
            probe.configure_chat_stops(self.tokenizer, self.generation, self.policy)

    def test_arbitrary_early_stopping_override_rejected(self):
        self.policy["generation_eos_ids"] = [248044, 248046, 123]
        with self.assertRaisesRegex(ValueError, "unsupported"):
            probe.configure_chat_stops(self.tokenizer, self.generation, self.policy)


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        self.versions = lambda name: {"bitsandbytes": "0.49.2", "accelerate": "1.8.1"}[name]

    def test_default_preserves_bf16_loading(self):
        self.assertEqual(probe.quantization_options(None, None, None), {})

    def test_exact_nf4_single_visible_device_and_vision_exclusion(self):
        config = Mock()
        options = probe.quantization_options(probe.NF4_POLICY, SimpleNamespace(bfloat16="bf16"), config, self.versions)
        self.assertEqual(options["device_map"], {"": 0})
        config.assert_called_once_with(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bf16",
                                       bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head", "model.visual"])

    def test_unknown_policy_or_dependency_rejected(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            probe.quantization_options({"method": "int8"}, None, None, self.versions)
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            probe.quantization_options(probe.NF4_POLICY, None, None, lambda _: "changed")

    def test_actual_tensor_and_vision_checks(self):
        class Fake4bit:
            weight = SimpleNamespace(quant_state=SimpleNamespace(quant_type="nf4", nested=True))
            compute_dtype = "bf16"
        modules = [("model.language_model.linear", Fake4bit()), ("model.visual", object())]
        parameter = SimpleNamespace(device="cuda:0", dtype="bf16", is_floating_point=lambda: True)
        model = SimpleNamespace(is_loaded_in_4bit=True, named_modules=lambda: iter(modules),
                     parameters=lambda: iter([parameter]),
                     named_parameters=lambda: iter([("model.visual.linear.weight", parameter)]),
                     get_output_embeddings=lambda: SimpleNamespace(weight=parameter),
                     hf_device_map={"": 0}, get_memory_footprint=lambda: 123)
        self.assertEqual(probe.validate_nf4_model(model, Fake4bit, "bf16")["linear4bit_count"], 1)
        del model.hf_device_map  # Transformers may omit this optional attribute on a single GPU.
        receipt = probe.validate_nf4_model(model, Fake4bit, "bf16")
        self.assertIsNone(receipt["device_map_metadata"])
        self.assertEqual(receipt["actual_parameter_devices"], ["cuda:0"])
        modules[0][1].compute_dtype = "float32"
        with self.assertRaisesRegex(ValueError, "compute dtype"):
            probe.validate_nf4_model(model, Fake4bit, "bf16")
        modules[0][1].compute_dtype = "bf16"
        parameter.dtype = "float32"
        with self.assertRaisesRegex(ValueError, "visual dtype"):
            probe.validate_nf4_model(model, Fake4bit, "bf16")
        parameter.dtype = "bf16"
        output = SimpleNamespace(dtype="float32", is_floating_point=lambda: True)
        model.get_output_embeddings = lambda: SimpleNamespace(weight=output)
        with self.assertRaisesRegex(ValueError, "output/tied embedding"):
            probe.validate_nf4_model(model, Fake4bit, "bf16")
        model.get_output_embeddings = lambda: SimpleNamespace(weight=parameter)
        modules.append(("model.visual.attention", Fake4bit()))
        with self.assertRaisesRegex(ValueError, "unexpectedly quantized"):
            probe.validate_nf4_model(model, Fake4bit, "bf16")

    def test_cpu_offload_rejected(self):
        class Fake4bit:
            weight = SimpleNamespace(quant_state=SimpleNamespace(quant_type="nf4", nested=True))
            compute_dtype = "bf16"
        parameter = SimpleNamespace(dtype="bf16", is_floating_point=lambda: True, device="cpu")
        model = SimpleNamespace(is_loaded_in_4bit=True,
                     named_modules=lambda: iter([("model.language_model.linear", Fake4bit()), ("model.visual", object())]),
                     parameters=lambda: iter([parameter]),
                     named_parameters=lambda: iter([("model.visual.linear.weight", parameter)]),
                     get_output_embeddings=lambda: SimpleNamespace(weight=parameter))
        with self.assertRaisesRegex(ValueError, "placement"):
            probe.validate_nf4_model(model, Fake4bit, "bf16")


if __name__ == "__main__":
    unittest.main()
