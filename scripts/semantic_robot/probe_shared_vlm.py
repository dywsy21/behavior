"""One bounded, no-physics small-VLM probe alongside an identified training job.

Uses previously saved public actor requests and their exact recorded images.
Explicit variants are image downsampling and a registered paired prompt test.
No labels, private traces, ground-truth correction, output selection, training,
or simulator.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
LABEL = re.compile(r"(?:CURRENT|PREVIOUS)_(?:HEAD|LEFT_WRIST|RIGHT_WRIST)_(?:RAW|ROBOT_GUIDE_NOT_OBJECT_LABELS)")
NEUTRAL_PROFILE = "neutral_observation_fields_v1"
EXAMPLE_MARKER = "Return only JSON with exactly these fields:\n"
NEGATIVE_EXAMPLE = dict(visible=False, view="none", target_uv=None, enclosed=None,
                        co_moving=None, supported=None, effect=None, hazard="none",
                        note="Target not identified in current views.", other_views=[],
                        target_reference="unknown")


def apply_request_profile(request, profile):
    """One opt-in prompt intervention; never alter task, images, or evidence."""
    if profile == "original":
        return request
    if profile != NEUTRAL_PROFILE or request["kind"] != "observe":
        raise ValueError("unregistered request profile or non-observation request")
    if request["system"].count(EXAMPLE_MARKER) != 1:
        raise ValueError("expected single observation example marker")
    before, rest = request["system"].split(EXAMPLE_MARKER)
    example, after = rest.split("\n", 1)
    if json.loads(example) != NEGATIVE_EXAMPLE:
        raise ValueError("pinned observation example changed")
    neutral = ("Return only JSON with exactly these keys (choose values from the images):\n"
               + ", ".join(NEGATIVE_EXAMPLE) + "\n")
    return {**request, "system": before + neutral + after}


def validate_spec(spec):
    if (spec.get("training_steps") != 0 or spec.get("simulator_resets") != 0 or
            spec.get("not_success_rate") is not True):
        raise ValueError("saved-request-only experiment required")
    cases = spec["cases"]
    if len(cases) != 4:
        raise ValueError("exactly four frozen calls required")
    kind = spec.get("probe_kind", "original_requests")
    if kind == "original_requests":
        if len({case["receipt"] for case in cases}) != 4 or any(
                case.get("request_profile", "original") != "original" for case in cases):
            raise ValueError("exactly four unique original requests required")
    elif kind == "paired_neutral_observation_v1":
        sources = {(case["receipt"], case["sha256"]) for case in cases}
        identities = {(case["receipt"], case["sha256"], case.get("request_profile", "original"))
                      for case in cases}
        expected = {(path, digest, profile) for path, digest in sources
                    for profile in ("original", NEUTRAL_PROFILE)}
        if len(sources) != 2 or len(identities) != 4 or identities != expected:
            raise ValueError("exactly two frozen original/neutral observation pairs required")
        if any(not case["receipt"].endswith("/observation.json") for case in cases):
            raise ValueError("paired probe only permits original observation receipts")
    else:
        raise ValueError("unregistered probe kind")
    bounds = {"allocator_limit_mib": (1, 4864), "reserve_mib": (2048, 8192),
              "non_torch_allowance_mib": (512, 2048), "max_seconds": (1, 600),
              "supervisor_seconds": (1, 900), "image_max_side": (1, 320)}
    for key, (low, high) in bounds.items():
        if type(spec[key]) is not int or not low <= spec[key] <= high:
            raise ValueError(f"outside probe resource bound: {key}")
    if spec["supervisor_seconds"] < spec["max_seconds"]:
        raise ValueError("supervisor deadline shorter than worker budget")
    if not spec["training_pids"] or any(type(pid) is not int or pid <= 0 for pid in spec["training_pids"]):
        raise ValueError("explicit positive training process identities required")


def require_source_and_device(spec):
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("clean pinned source required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != spec["gpu_uuid"]:
        raise ValueError("exact one-device UUID-based CUDA visibility required")


def stop_owned_child(child):
    """Signal only the Popen-owned direct worker, never peer training processes."""
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=15)


def supervise(spec_path, output):
    spec = json.loads(spec_path.read_text()); validate_spec(spec)
    require_source_and_device(spec)
    check_resources(gpu_snapshot(spec["gpu"]), spec, own_pid=os.getpid(), before_load=True)
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
                              "--spec", str(spec_path), "--output", str(output)], start_new_session=True)
    receipt = {"supervisor_pid": os.getpid(), "worker_pid": child.pid, "status": "running",
               "started_utc": datetime.now(timezone.utc).isoformat(), "resource_samples": []}
    terminal = output.with_suffix(".supervisor.json")
    def interrupted(signum, _frame):
        raise InterruptedError(f"supervisor signal {signum}")
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        terminal.write_text(json.dumps(receipt, indent=2))
        while child.poll() is None:
            if time.monotonic() - started >= spec["supervisor_seconds"]:
                raise TimeoutError("outer wall-clock deadline reached")
            snapshot = gpu_snapshot(spec["gpu"])
            check_resources(snapshot, spec, own_pid=child.pid, before_load=False)
            receipt["resource_samples"].append({"elapsed": time.monotonic()-started, **snapshot})
            terminal.write_text(json.dumps(receipt, indent=2))
            time.sleep(2)
        receipt.update(status="worker_exited", exit_code=child.returncode)
    except BaseException as error:
        receipt.update(status="stopped", error=repr(error))
        raise
    finally:
        stop_owned_child(child)
        signal.signal(signal.SIGTERM, previous_term)
        receipt.update(wall_seconds=time.monotonic()-started, exit_code=child.returncode)
        try:
            receipt["gpu_after_worker_exit"] = gpu_snapshot(spec["gpu"])
        except Exception as error:
            receipt["final_gpu_query_error"] = repr(error)
        terminal.write_text(json.dumps(receipt, indent=2))


def launch(spec_path, output):
    """Exclusive launch receipt prevents accidental retries after SSH interruption."""
    spec = json.loads(spec_path.read_text()); validate_spec(spec)
    require_source_and_device(spec)
    check_resources(gpu_snapshot(spec["gpu"]), spec, own_pid=os.getpid(), before_load=True)
    if output.exists():
        raise ValueError("probe output already exists; no retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".launch.json").open("x") as launch_file:
        receipt = {"status": "launch_reserved", "spec_sha256": sha(spec_path),
                   "source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                   "utc": datetime.now(timezone.utc).isoformat(), "output": str(output)}
        launch_file.write(json.dumps(receipt, indent=2)); launch_file.flush()
        with output.with_suffix(".log").open("x") as log:
            supervisor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise",
                         "--spec", str(spec_path), "--output", str(output)], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(status="supervisor_started", supervisor_pid=supervisor.pid)
        launch_file.seek(0); launch_file.truncate(); launch_file.write(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_files(model_path, expected_files):
    """Pin present files AND absent optional generation/adapter config files."""
    actual = {entry.name for entry in model_path.iterdir() if entry.name != ".cache"}
    if not expected_files or actual != set(expected_files):
        raise ValueError("pinned model root manifest changed")
    cache = model_path / ".cache"
    if cache.exists() and (cache.is_symlink() or not cache.is_dir()):
        raise ValueError("only an inert regular download-cache directory is allowed")
    for name, expected in expected_files.items():
        path = model_path / name
        if Path(name).name != name or path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError(f"pinned model file changed: {name}")


def check_resources(snapshot, spec, *, own_pid, before_load):
    if snapshot["uuid"] != spec["gpu_uuid"] or snapshot["index"] != spec["gpu"]:
        raise ValueError("GPU identity changed")
    expected = set(spec["training_pids"])
    peers = {row["pid"] for row in snapshot["apps"] if row["pid"] != own_pid}
    if peers != expected:
        raise ValueError("training/other GPU process identity changed")
    own_used = sum(row["used_mib"] for row in snapshot["apps"] if row["pid"] == own_pid)
    process_limit = spec["allocator_limit_mib"] + spec["non_torch_allowance_mib"]
    if own_used > process_limit:
        raise ValueError("worker total GPU memory budget exceeded")
    required = spec["reserve_mib"]
    if before_load:
        required += max(0, process_limit - own_used)
    if snapshot["free_mib"] < required:
        raise ValueError(f"insufficient shared GPU headroom: {snapshot['free_mib']} < {required}")


def gpu_snapshot(index):
    def query(*args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                       text=True, timeout=10)
    rows = [line.split(",") for line in query("--query-gpu=index,uuid,memory.total,memory.free").splitlines()]
    match = next([v.strip() for v in row] for row in rows if int(row[0]) == index)
    apps = []
    for line in query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory").splitlines():
        uuid, pid, used = [v.strip() for v in line.split(",")]
        if uuid == match[1]:
            apps.append({"pid": int(pid), "used_mib": int(used)})
    return {"index": int(match[0]), "uuid": match[1], "total_mib": int(match[2]),
            "free_mib": int(match[3]), "apps": apps}


def load_saved_request(root, case, max_side):
    """Restore only the actor-facing request, checking all historical RGB hashes."""
    from PIL import Image
    path = root / case["receipt"]
    if not path.resolve().is_relative_to(root.resolve()) or sha(path) != case["sha256"]:
        raise ValueError("saved request path or SHA changed")
    record = json.loads(path.read_text())
    request = record["request_without_pixel_duplicates"]
    required = {"kind", "system", "text", "images", "allowed"}
    if set(request) not in (required, required | {"response_schema"}):
        raise ValueError("only the original public actor request is permitted")
    labels = [row["label"] for row in request["images"]]
    original = record["result"]["images"]
    if labels != [row["label"] for row in original] or len(set(labels)) != len(labels):
        raise ValueError("image order/identity mismatch")
    content, image_receipts = [], []
    for label, expected in zip(labels, original):
        if not LABEL.fullmatch(label):
            raise ValueError("unknown/non-actor image label")
        image_path = path.parent / f"{label}.png"
        if not image_path.resolve().is_relative_to(root.resolve()):
            raise ValueError("image escapes saved run")
        with Image.open(image_path) as opened:
            original_image = opened.convert("RGB")
        old = original_image.copy(); old.thumbnail((640, 640))
        if (list(old.size) != expected["size"] or
                hashlib.sha256(old.tobytes()).hexdigest() != expected["pixels_sha256"]):
            raise ValueError("original served RGB pixels differ from saved source")
        small = original_image.copy(); small.thumbnail((max_side, max_side))
        image_receipts.append({"label": label, "original_served": expected, "size": list(small.size),
                               "pixels_sha256": hashlib.sha256(small.tobytes()).hexdigest()})
        content += [{"type": "text", "text": label}, {"type": "image", "image": small}]
    content.append({"type": "text", "text": request["text"]})
    request = apply_request_profile(request, case.get("request_profile", "original"))
    messages = [{"role": "system", "content": request["system"]}, {"role": "user", "content": content}]
    return request, messages, image_receipts


def finite_choices(tokenizer, model, lines, prefix):
    trie = {}
    for line in lines:
        node = trie
        for token in tokenizer.encode(line, add_special_tokens=False):
            node = node.setdefault(token, {})
        node[None] = {}
    def allowed(_batch, ids):
        node = trie
        for token in ids[prefix:].tolist():
            node = node[token]
        result = [token for token in node if token is not None]
        if None in node:
            eos = model.generation_config.eos_token_id
            result += eos if isinstance(eos, list) else [eos]
        return result
    return allowed


def configure_chat_stops(tokenizer, generation_config, policy):
    """Explicit pinned chat deployment override; never trim/repair model output."""
    original = generation_config.eos_token_id
    if policy is not None:
        expected = {"model_default_eos_id": 248044, "tokenizer_eos_id": 248046,
                    "generation_eos_ids": [248044, 248046]}
        if policy != expected:
            raise ValueError("unsupported chat stop policy")
        if (original != expected["model_default_eos_id"] or tokenizer.eos_token_id != expected["tokenizer_eos_id"]
                or tokenizer.eos_token != "<|im_end|>"
                or tokenizer.convert_tokens_to_ids("<|endoftext|>") != 248044
                or tokenizer.convert_tokens_to_ids("<|im_end|>") != 248046):
            raise ValueError("pinned chat stop-token identity changed")
        generation_config.eos_token_id = list(expected["generation_eos_ids"])
    eos = generation_config.eos_token_id
    if tokenizer.eos_token_id not in (eos if isinstance(eos, list) else [eos]):
        raise ValueError("structured tokenizer/model stop-token mismatch")
    return {"original_generation_eos": original, "tokenizer_eos": tokenizer.eos_token_id,
            "effective_generation_eos": eos, "explicit_policy": policy}


NF4_POLICY = {"method": "bitsandbytes-nf4", "compute_dtype": "bfloat16", "double_quant": True,
              "skip_modules": ["lm_head", "model.visual"],
              "bitsandbytes_version": "0.49.2", "accelerate_version": "1.8.1"}


def quantization_options(policy, torch_module, config_class, dependency_version=version):
    if policy is None:
        return {}
    if policy != NF4_POLICY:
        raise ValueError("unregistered quantization policy")
    for package in ("bitsandbytes", "accelerate"):
        if dependency_version(package) != policy[package + "_version"]:
            raise ValueError(f"pinned quantization dependency changed: {package}")
    return {"quantization_config": config_class(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_module.bfloat16, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=list(policy["skip_modules"])),
            "device_map": {"": 0}, "low_cpu_mem_usage": True}


def validate_nf4_model(model, linear_class, expected_dtype):
    quantized = [(name, module) for name, module in model.named_modules() if isinstance(module, linear_class)]
    if not quantized or not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("requested model did not load in 4bit")
    if any(name == "lm_head" or name.startswith("model.visual") for name, _ in quantized):
        raise ValueError("vision encoder or tied output unexpectedly quantized")
    if not any(name.startswith("model.visual") for name, _ in model.named_modules()):
        raise ValueError("expected vision module identity missing")
    if any(getattr(getattr(module.weight, "quant_state", None), "quant_type", None) != "nf4"
           for _, module in quantized):
        raise ValueError("NF4 tensors not actually materialized")
    if any(not getattr(module.weight.quant_state, "nested", False) for _, module in quantized):
        raise ValueError("registered double quantization not actually present")
    if any(module.compute_dtype != expected_dtype for _, module in quantized):
        raise ValueError("actual quantized compute dtype differs from BF16")
    visual = [parameter for name, parameter in model.named_parameters() if name.startswith("model.visual.")]
    if not visual or any(not parameter.is_floating_point() or parameter.dtype != expected_dtype for parameter in visual):
        raise ValueError("actual skipped visual dtype differs from BF16")
    # Read output embedding directly: named_parameters omits tied lm_head weights.
    output_weight = model.get_output_embeddings().weight
    if not output_weight.is_floating_point() or output_weight.dtype != expected_dtype:
        raise ValueError("actual output/tied embedding dtype differs from BF16")
    if any(str(parameter.device) != "cuda:0" for parameter in model.parameters()):
        raise ValueError("unexpected CPU/disk/other-device model placement")
    return {"linear4bit_count": len(quantized), "linear4bit_names": [name for name, _ in quantized],
            "vision_quantized": False, "device_map_metadata": getattr(model, "hf_device_map", None),
            "actual_parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
            "compute_dtypes": sorted({str(module.compute_dtype) for _, module in quantized}),
            "visual_parameter_dtypes": sorted({str(parameter.dtype) for parameter in visual}),
            "output_embedding_dtype": str(output_weight.dtype), "double_quantized": True,
            "footprint_bytes": model.get_memory_footprint()}


def run(spec_path, output):
    spec = json.loads(spec_path.read_text()); validate_spec(spec)
    require_source_and_device(spec)
    snapshot = gpu_snapshot(spec["gpu"])
    check_resources(snapshot, spec, own_pid=os.getpid(), before_load=True)
    model_path = Path(spec["model"])
    # Hash on CPU before creating a context; no model download or pip install.
    validate_model_files(model_path, spec["model_files"])
    requests = [load_saved_request(Path(spec["source_run"]), case, spec["image_max_side"])
                for case in spec["cases"]]
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def expired(_signal, _frame):
        raise TimeoutError("registered total probe wall-clock budget reached")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(spec["max_seconds"])
    result = {"status": "running", "not_success_rate": True, "simulator_resets": 0, "training_steps": 0,
              "source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
              "spec_sha256": sha(spec_path), "gpu_before": snapshot, "pid": os.getpid(), "calls": []}
    (output / "start.json").write_text(json.dumps(result, indent=2))
    try:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn
        from semantic_robot.v2.structured_planning import decoder_identity
        from semantic_robot.v2.harness import parse_plan
        from semantic_robot.v2.grounding import GroundedEvidence
        from semantic_robot.v2.protocol import Action
        from serve_v2 import normalize_json_transport, validate_response_schema, validate_scoped_choices, validate_image_count
        result["decoder"] = decoder_identity()
        if transformers.__version__ != "5.7.0":
            raise ValueError("original Transformers 5.7.0 overlay required")
        torch.set_num_threads(2); torch.manual_seed(spec["seed"])
        result["gpu_before_context"] = gpu_snapshot(spec["gpu"])
        check_resources(result["gpu_before_context"], spec, own_pid=os.getpid(), before_load=True)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(spec["allocator_limit_mib"] * 1024**2 / total, 0)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        tokenizer_data = build_token_enforcer_tokenizer_data(processor.tokenizer)
        quantized_options = quantization_options(spec.get("quantization"), torch, BitsAndBytesConfig)
        if quantized_options:
            # Quantized loader allocates during from_pretrained: recheck BEFORE it.
            result["gpu_before_quantized_load"] = gpu_snapshot(spec["gpu"])
            check_resources(result["gpu_before_quantized_load"], spec, own_pid=os.getpid(), before_load=True)
        model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa", **quantized_options)
        result["chat_stops"] = configure_chat_stops(processor.tokenizer, model.generation_config,
                                                     spec.get("chat_stop_policy"))
        if quantized_options:
            import bitsandbytes as bnb
            result["quantization"] = {"policy": spec["quantization"],
                                      "actual": validate_nf4_model(model, bnb.nn.Linear4bit, torch.bfloat16)}
        else:
            result["gpu_before_model_allocation"] = gpu_snapshot(spec["gpu"])
            check_resources(result["gpu_before_model_allocation"], spec, own_pid=os.getpid(), before_load=True)
            model = model.to("cuda")
        model.eval()
        result.update(load_seconds=time.monotonic()-started, transformers=transformers.__version__,
                      torch=torch.__version__, model=str(model_path), revision=spec["revision"],
                      dtype="nf4_weights_bfloat16_compute" if quantized_options else "bfloat16",
                      image_max_side=spec["image_max_side"])
        for case, (request, messages, images) in zip(spec["cases"], requests):
            check_resources(gpu_snapshot(spec["gpu"]), spec, own_pid=os.getpid(), before_load=False)
            schema = validate_response_schema(request, True)
            validate_scoped_choices(request["kind"], request["allowed"])
            validate_image_count(request["kind"], request["images"])
            encoded = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                           return_dict=True, return_tensors="pt", enable_thinking=False)
            prefix = encoded["input_ids"].shape[1]
            if prefix > 12000:
                raise ValueError("unchanged 12000-token context limit exceeded")
            inputs = encoded.to("cuda")
            options = {}
            if request["allowed"]:
                options["prefix_allowed_tokens_fn"] = finite_choices(processor.tokenizer, model, request["allowed"], prefix)
            if schema is not None:
                options["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(schema))
            cap = {"plan": 1024, "observe": 320, "act": 64}[request["kind"]]
            row = {"case": case["receipt"], "kind": request["kind"], "input_tokens": prefix,
                   "images": images, "request_sha256": case["sha256"], "status": "issued",
                   "request_profile": case.get("request_profile", "original"),
                   "effective_public_request": request}
            result["calls"].append(row)
            (output / "progress.json").write_text(json.dumps(result, indent=2))
            torch.cuda.synchronize(); before = time.monotonic()
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=cap, do_sample=False, use_cache=True, **options)[0, prefix:]
            torch.cuda.synchronize()
            raw = processor.decode(ids, skip_special_tokens=True).strip()
            row.update(raw_text=raw, output_tokens=len(ids), generation_seconds=time.monotonic()-before,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2, status="completed")
            try:
                normalized, wrapper = normalize_json_transport(raw)
                row.update(text=normalized, transport_wrapper_removed=wrapper)
                if len(ids) >= cap:
                    raise ValueError("truncated output")
                if request["kind"] == "plan":
                    parse_plan(normalized)
                elif request["kind"] == "observe":
                    GroundedEvidence.parse(normalized)
                else:
                    Action.parse(normalized)
                    if normalized not in request["allowed"]:
                        raise ValueError("action outside original feasible choices")
                row["schema_valid"] = True
            except ValueError as error:
                row.update(schema_valid=False, schema_error=str(error))
            row["gpu_after"] = gpu_snapshot(spec["gpu"])
            check_resources(row["gpu_after"], spec, own_pid=os.getpid(), before_load=False)
            (output / f"call_{len(result['calls']):02d}.json").write_text(json.dumps(row, indent=2))
            print(json.dumps({key: row.get(key) for key in ("case", "status", "schema_valid", "text", "generation_seconds", "peak_allocated_mib")}), flush=True)
            del ids, inputs, encoded
            gc.collect(); torch.cuda.empty_cache()
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        result["wall_seconds"] = time.monotonic()-started
        (output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    entry = launch if args.launch else supervise if args.supervise else run
    entry(args.spec.resolve(), args.output.resolve())
