"""H50/H51: four frozen target queries, <=12 calls, no controls.

Uses the H46/H49 pinned NF4 model and shared-training resource protections.
Only public RGB-D, robot calibration/proprioception and a preregistered target
description enter the locator. No scorer, privileged trace or old model answer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import probe_shared_vlm as shared
from semantic_robot.v2.finite_localization import SelectionRequest, SelectionReply, bind_frame, locate_target, choice_suffix
from semantic_robot.v2.grounding import validate_depth
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.native_grounding import locate_native, SYSTEM as NATIVE_SYSTEM
from semantic_robot.v2.protocol import VIEWS
from semantic_robot.v2.skill_completion import CaptureRef, read_capture

H50_RESOURCES = {
    "gpu": 2, "gpu_uuid": "GPU-3e4fda8c-536e-5899-e877-b8be97032fe0",
    "training_pids": [3294348], "allocator_limit_mib": 4864,
    "non_torch_allowance_mib": 512, "reserve_mib": 2048,
    "max_seconds": 600, "supervisor_seconds": 900, "image_max_side": 320, "seed": 17,
}
# Canonical JSON of these exact fields in the preregistered H50 configuration.
# This pins all model-file hashes and the EOS/quantization policy, not just a name.
H50_MODEL_FIELDS = ("model", "revision", "model_files", "quantization", "chat_stop_policy")
H50_MODEL_SHA256 = "4130b5766db85b08b80af4948b5aba272fb9e82405b654acc769467feb7b2e6f"
H50_CASES_SHA256 = "5846788db1516fd08b8a341947466dbcd4320c3041cb99937260bfc328a257ea"
H51_CASES_SHA256 = "5f6847b6ca2f38794e76268d6286b3f13bb4a3602053cddeee3b150a30b2279b"


def expected_files(case):
    capture = Path(case["capture"])
    if capture.is_absolute() or ".." in capture.parts or not capture.parts:
        raise ValueError("Relative public capture directory required")
    prefix = capture.as_posix() + "/"
    if case["format"] == "run_v2":
        return {"robot_calibration.json", *(prefix + name for name in
            ("proprio.json", "depth_receipt.json", "depth.npz",
             *("CURRENT_" + view.upper() + "_RAW.png" for view in VIEWS)))}
    if case["format"] == "capture_v1":
        return {"robot_calibration.json.gz", prefix + "capture.json"}
    raise ValueError("Unknown public capture format")


def validate_spec(spec):
    limits = {"H50-finite-localization-v1": 12, "H50b-explicit-choice-contract-v1": 8,
              "H51-native-protocol-v1": 12}
    if (spec.get("experiment") not in limits or spec.get("max_calls") != limits[spec["experiment"]] or
            spec.get("training_steps") != 0 or spec.get("simulator_resets") != 0 or
            spec.get("not_success_rate") is not True or spec.get("quantization") != shared.NF4_POLICY):
        raise ValueError("Exact preregistered no-physics experiment and call limit required")
    shared.validate_resource_bounds(spec)
    for key, expected in H50_RESOURCES.items():
        if type(spec.get(key)) is not type(expected) or spec[key] != expected:
            raise ValueError("Frozen H50 resource identity or budget changed: " + key)
    model_profile = {key: spec.get(key) for key in H50_MODEL_FIELDS}
    digest = hashlib.sha256(json.dumps(model_profile, sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != H50_MODEL_SHA256:
        raise ValueError("Frozen H50 model manifest or inference policy changed")
    cases = spec["cases"]
    case_locks = {"H50b-explicit-choice-contract-v1": H50_CASES_SHA256,
                  "H51-native-protocol-v1": H51_CASES_SHA256}
    if spec["experiment"] in case_locks:
        case_digest = hashlib.sha256(json.dumps(cases, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if case_digest != case_locks[spec["experiment"]]:
            raise ValueError("Experiment must retain its identical four preregistered queries")
    if not isinstance(cases, list) or len(cases) != 4 or len({c["id"] for c in cases}) != 4:
        raise ValueError("Exactly four uniquely named frozen cases required")
    for case in cases:
        if (set(case) != {"id", "source_run", "capture", "format", "view", "goal", "files_sha256"} or
                not isinstance(case["id"], str) or not case["id"].replace("_", "").isalnum() or
                case["view"] != "head" or not Path(case["source_run"]).is_absolute()):
            raise ValueError("Exact bounded public case contract required")
        Goal(**case["goal"])
        if set(case["files_sha256"]) != expected_files(case):
            raise ValueError("Only exact public sensor/robot files may be loaded")
        for digest in case["files_sha256"].values():
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("All public input bytes must be SHA256 pinned")


def _read(root, relative, digest):
    path = root / relative
    limit = (64 if relative in ("robot_calibration.json", "robot_calibration.json.gz") else 32) * 1024**2
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or
            path.stat().st_size > limit):
        raise ValueError("Bounded public file inside source run required")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Frozen public file bytes changed: " + relative)
    return data


def load_case(case, root_override=None):
    """CPU-only loading; an override is solely for hash-identical local copies."""
    root = Path(case["source_run"] if root_override is None else root_override)
    data = {name: _read(root, name, digest) for name, digest in case["files_sha256"].items()}
    goal = Goal(**case["goal"])
    prefix = case["capture"] + "/"
    if case["format"] == "capture_v1":
        calibration = gzip.decompress(data["robot_calibration.json.gz"])
        if len(calibration) > 64 * 1024**2:
            raise ValueError("Calibration size exceeded")
        model = RobotModel(json.loads(calibration))
        capture = json.loads(data[prefix + "capture.json"])
        frame = read_capture(CaptureRef(root / case["capture"],
            case["files_sha256"][prefix + "capture.json"], capture["clock"]), model)
        raw, depths, state = frame.images, frame.depths, frame.state
        capture_identity = case["files_sha256"][prefix + "capture.json"]
    else:
        model = RobotModel(json.loads(data["robot_calibration.json"]))
        proprio = json.loads(data[prefix + "proprio.json"])
        # Static localization does not use velocities or any task outcome.
        state = SimpleNamespace(q=np.asarray(proprio["q"], dtype=float),
                                gripper=np.asarray(proprio["gripper"], dtype=float))
        raw = {}
        for view in VIEWS:
            with Image.open(BytesIO(data[prefix + "CURRENT_" + view.upper() + "_RAW.png"])) as image:
                if image.mode != "RGB":
                    raise ValueError("Original RGB capture, no conversion permitted")
                raw[view] = np.asarray(image).copy()
        with np.load(BytesIO(data[prefix + "depth.npz"]), allow_pickle=False) as archive:
            if set(archive.files) != set(VIEWS):
                raise ValueError("Exactly three public depth views required")
            depths = {view: archive[view].copy() for view in VIEWS}
        sensors = json.loads(data[prefix + "depth_receipt.json"])
        if set(sensors) != set(VIEWS):
            raise ValueError("Exactly three public sensor receipts required")
        snapshots = set()
        for view in VIEWS:
            sensor = sensors[view]
            depth = validate_depth(depths[view], model.spec["metadata"]["cameras"][view])
            if (sensor.get("rgb_sha256") != hashlib.sha256(raw[view].tobytes()).hexdigest() or
                    sensor.get("depth_sha256") != hashlib.sha256(depth.tobytes()).hexdigest() or
                    sensor.get("modalities") != ["rgb", "depth_linear"] or sensor.get("depth_units") != "metres" or
                    sensor.get("depth_convention") != "distance_to_image_plane" or
                    sensor.get("same_sensor_current_render") is not True or
                    type(sensor.get("render_barrier_updates")) is not int or sensor["render_barrier_updates"] < 4 or
                    type(sensor.get("control_steps_in_capture")) is not int or sensor["control_steps_in_capture"] != 0 or
                    type(sensor.get("snapshot_id")) is not int or sensor["snapshot_id"] < 0):
                raise ValueError("Original sensor hashes and current render barrier required")
            snapshots.add(sensor["snapshot_id"])
        if len(snapshots) != 1:
            raise ValueError("Cameras must belong to one captured snapshot")
        capture_identity = case["files_sha256"][prefix + "depth_receipt.json"]
    frame_id = case["id"] + ":" + capture_identity
    binding = bind_frame(frame_id, goal, raw, depths, model, state)
    return dict(frame_id=frame_id, goal=goal, raw=raw, depths=depths, model=model, state=state,
                views=(case["view"],)), binding


def run(spec_path, output):
    spec = json.loads(spec_path.read_text()); validate_spec(spec)
    shared.require_source_and_device(spec)
    shared.check_resources(shared.gpu_snapshot(spec["gpu"]), spec, own_pid=os.getpid(), before_load=True)
    shared.validate_model_files(Path(spec["model"]), spec["model_files"])
    prepared = [load_case(case) for case in spec["cases"]]
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "experiment": spec["experiment"], "not_success_rate": True, "success_rate": None,
              "simulator_resets": 0, "training_steps": 0, "pid": os.getpid(), "calls": [], "cases": [],
              "source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
              "spec_sha256": shared.sha(spec_path)}
    def save(name):
        (output / name).write_text(json.dumps(result, indent=2, allow_nan=False))
    save("start.json")
    started = time.monotonic()
    deadline = started + spec["max_seconds"]
    def expired(_signal, _frame):
        raise TimeoutError("H50 total inference deadline reached")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(spec["max_seconds"])
    try:
        import torch
        import transformers
        import bitsandbytes as bnb
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        if transformers.__version__ != "5.7.0":
            raise ValueError("Pinned Transformers 5.7.0 overlay required")
        torch.set_num_threads(2); torch.manual_seed(spec["seed"])
        def resources(before_load=False):
            snapshot = shared.gpu_snapshot(spec["gpu"])
            shared.check_resources(snapshot, spec, own_pid=os.getpid(), before_load=before_load)
            return snapshot
        result["gpu_before_context"] = resources(True)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(spec["allocator_limit_mib"] * 1024**2 / total, 0)
        processor = AutoProcessor.from_pretrained(spec["model"], local_files_only=True)
        options = shared.quantization_options(spec["quantization"], torch, BitsAndBytesConfig)
        result["gpu_before_quantized_load"] = resources(True)
        model = AutoModelForImageTextToText.from_pretrained(spec["model"], local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa", **options).eval()
        result["chat_stops"] = shared.configure_chat_stops(processor.tokenizer, model.generation_config, spec["chat_stop_policy"])
        result["quantization"] = shared.validate_nf4_model(model, bnb.nn.Linear4bit, torch.bfloat16)
        result.update(load_seconds=time.monotonic()-started, model=spec["model"], revision=spec["revision"],
                      torch=torch.__version__, transformers=transformers.__version__)

        def choose(request):
            if len(result["calls"]) >= spec["max_calls"] or time.monotonic() >= deadline:
                raise TimeoutError("H50 model-call or wall budget exhausted")
            resources()
            cap = validate_request(request)
            content, hashes = [], []
            call_id = len(result["calls"]) + 1
            folder = output / f"call_{call_id:02d}"; folder.mkdir()
            for label, original in zip(request.labels, request.images):
                original.save(folder / (label + ".png"))
                served = original.copy(); served.thumbnail((spec["image_max_side"], spec["image_max_side"]))
                hashes.append({"label": label, "size": list(served.size),
                               "pixels_sha256": hashlib.sha256(served.tobytes()).hexdigest()})
                content += [{"type": "text", "text": label}, {"type": "image", "image": served}]
            content.append({"type": "text", "text": request.text})
            messages = [{"role": "system", "content": request.system}, {"role": "user", "content": content}]
            encoded = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                          return_dict=True, return_tensors="pt", enable_thinking=False)
            prefix = encoded["input_ids"].shape[1]
            if prefix > 2048:
                raise ValueError("H50 compact context budget exceeded")
            allowed = [choice.text() for choice in request.allowed]
            inputs = encoded.to("cuda")
            row = {"call": call_id, "case": result["cases"][-1]["id"], "phase": request.phase,
                   "status": "issued", "request_sha256": request.sha256, "request": request.receipt(),
                   "served_images": hashes, "input_tokens": prefix}
            result["calls"].append(row); save("progress.json")
            torch.cuda.synchronize(); generated_at = time.monotonic()
            grammar = ({"prefix_allowed_tokens_fn": shared.finite_choices(processor.tokenizer, model, allowed, prefix)}
                       if allowed else {})
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=cap, do_sample=False, use_cache=True, **grammar)[0, prefix:]
            torch.cuda.synchronize()
            raw_text = processor.decode(ids, skip_special_tokens=True).strip()
            row.update(status="completed", raw_text=raw_text, output_tokens=len(ids),
                       generation_seconds=time.monotonic()-generated_at,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2, gpu_after=resources(),
                       hit_token_cap=len(ids) >= cap)
            save("progress.json")
            (folder / "call.json").write_text(json.dumps(row, indent=2, allow_nan=False))
            if request.phase != "baseline" and (len(ids) >= cap or (allowed and raw_text not in allowed)):
                raise ValueError("Incomplete/noncanonical/outside-offered response; no repair or retry")
            del ids, inputs, encoded
            gc.collect(); torch.cuda.empty_cache()
            return SelectionReply(request.sha256, raw_text)

        for case, (arguments, binding) in zip(spec["cases"], prepared):
            row = {"id": case["id"], "status": "running", "binding": binding,
                   "source": case, "not_success_rate": True}
            result["cases"].append(row); save("progress.json")
            # A paired baseline sees the same raw head and target, not the
            # candidate grids. Its answer is recorded but never fed forward.
            if spec["experiment"] in ("H50-finite-localization-v1", "H51-native-protocol-v1"):
                request = baseline_request(arguments, binding)
                reply = choose(request)
                try:
                    if result["calls"][-1]["hit_token_cap"]:
                        raise ValueError("Truncated baseline output")
                    row["baseline"] = {"schema_valid": True,
                        "evidence": shared.parse_static_location(reply.text, "static_head_raw_v1")}
                except ValueError as error:
                    row["baseline"] = {"schema_valid": False, "error": str(error)}
            else:
                # Reuse H50's published baseline for analysis only. No old
                # answer is read by this worker or included in model inputs.
                row["baseline"] = {"status": "not_rerun", "reference": "H50-finite-localization-v1"}
            if spec["experiment"] == "H51-native-protocol-v1":
                native_args = {key: value for key, value in arguments.items() if key != "views"}
                for mode in ("point", "box"):
                    target = locate_native(**native_args, view=case["view"], mode=mode,
                                           choose=choose, deadline=deadline)
                    # This result contains no implicit box-centre evidence.
                    row["native_" + mode] = asdict(target.for_frame(**native_args))
                row["evidence"] = row["native_point"]["evidence"]
                row["status"] = "complete"
            else:
                target = locate_target(**arguments, choose=choose, deadline=deadline)
                row.update(status="complete", localization=target.receipt,
                           evidence=asdict(target.for_frame(arguments["frame_id"], arguments["goal"], arguments["raw"],
                                            arguments["depths"], arguments["model"], arguments["state"])))
            save("progress.json")
            print(json.dumps({"id": case["id"], "evidence": row["evidence"], "calls_so_far": len(result["calls"])}), flush=True)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        result["wall_seconds"] = time.monotonic()-started
        save("result.json")


def baseline_request(arguments, binding):
    goal = arguments["goal"]
    return SelectionRequest("baseline", binding, shared.STATIC_SYSTEM,
                            json.dumps({"target": goal.target, "goal_kind": goal.kind}),
                            (Image.fromarray(arguments["raw"]["head"]).copy(),), ("CURRENT_HEAD_RAW",), ())


def validate_request(request):
    from semantic_robot.v2.affordance import SurfaceChoice
    sizes = {"baseline": 1, "region": 2, "surface": 4, "native_point": 1, "native_box": 1}
    if (request.phase not in sizes or len(request.images) != sizes[request.phase] or
            len(request.labels) != len(request.images) or len(set(request.labels)) != len(request.labels)):
        raise ValueError("Exact registered phase image count required")
    if request.phase == "baseline":
        if request.allowed or request.system != shared.STATIC_SYSTEM:
            raise ValueError("Unchanged unconstrained minimal static baseline required")
        return 320
    if request.phase in ("native_point", "native_box"):
        if request.allowed or request.system != NATIVE_SYSTEM:
            raise ValueError("Native one-RAW grounding is unconstrained and uses its registered system")
        return 128
    if (not 1 <= len(request.allowed) <= 13 or len(set(request.allowed)) != len(request.allowed) or
            not all(type(c) is SurfaceChoice for c in request.allowed) or SurfaceChoice() not in request.allowed):
        raise ValueError("Bounded canonical choices including abstention required")
    if request.phase == "region" and request.allowed != (SurfaceChoice(), *(SurfaceChoice(i) for i in range(9))):
        raise ValueError("Exactly nine regions plus abstention required")
    if not request.text.endswith(choice_suffix(request.allowed)):
        raise ValueError("Model-visible choices differ from constrained-decoder choices")
    return 32


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-roots", help="JSON list of hash-identical local capture roots; prepare only")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("launch", "supervise", "worker", "prepare"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        spec = json.loads(args.spec.read_text()); validate_spec(spec)
        roots = json.loads(args.local_roots) if args.local_roots else [None] * 4
        if not isinstance(roots, list) or len(roots) != 4:
            raise ValueError("Four exact local source roots required")
        print(json.dumps({"prepared": [{"id": c["id"], "binding": load_case(c, root)[1]}
                            for c, root in zip(spec["cases"], roots)], "model_calls": 0, "simulator_resets": 0}, indent=2))
    else:
        if args.local_roots or args.output is None:
            raise ValueError("Runtime requires a new output and forbids local path overrides")
        if args.worker:
            run(args.spec.resolve(), args.output.resolve())
        else:
            entry = shared.launch if args.launch else shared.supervise
            entry(args.spec.resolve(), args.output.resolve(), validator=validate_spec, entrypoint=__file__)
