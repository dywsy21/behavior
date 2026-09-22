"""Read-only real-environment TRAIN39 projection/tokenizer gates; no model load."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

from common import sha, write_json
from native_dataset import checked_images
from native_completion_training_data import load_reviewed, BalancedSampler
from native_completion_protocol import query_row
from native_completion_modeling import encode, check_mask
from native_completion_runtime import (clean_source, initial_checkpoint, BASE, CONFIG, SERVICE_CONFIG,
    validate_config, EXECUTOR_DIGEST, CORE_COMMIT)
from native_actor_protocol import training_row as old_row
from modeling import encode as old_encode


def main():
    p = argparse.ArgumentParser()
    for name in ("data", "motion-data", "data-review", "initial-training", "config", "service-config", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(); start = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("CPU-only/no source bytecode writes required")
    if a.output.exists(): raise FileExistsError(a.output)
    code = clean_source(); validate_config(json.loads(a.config.read_text()))
    validate_config(json.loads(a.service_config.read_text()), service=True)
    examples, data = load_reviewed(a.data, a.motion_data, a.data_review)
    initial, _ = initial_checkpoint(a.initial_training)
    import numpy, torch, transformers, peft
    from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    # Class lookup/import only; from_pretrained(model) and from_config(model)
    # are deliberately never invoked in this CPU preflight.
    model_class = AutoModelForImageTextToText._model_mapping[type(config)]
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True); processor.tokenizer.padding_side = "left"
    checks = []; motion_exact = 0
    for example in examples:
        images = checked_images(example["state"]); row = example["query"]
        x = encode(processor, row, images, supervised=True); check_mask(processor, row, x)
        y = encode(processor, query_row(row["actor"], row["query_kind"]), images, supervised=False)
        n = y["input_ids"].shape[1]
        if not torch.equal(x["input_ids"][:, :n], y["input_ids"]): raise RuntimeError("Real prefix mismatch")
        if example["category"] == "motion":
            old = old_encode(processor, old_row(row["actor"], row["target"]), images, supervised=True)
            if set(old) != set(x) or any(not torch.equal(old[k], x[k]) for k in x): raise RuntimeError("Original motion encoding changed")
            motion_exact += 1
        checks.append({"id": example["id"], "category": example["category"], "input_tokens": n,
            "response_tokens": x["labels"][x["labels"] != -100].tolist(),
            "input_ids_sha256": hashlib.sha256(y["input_ids"].numpy().tobytes()).hexdigest()})
    sampler = BalancedSampler(examples)
    draws = [[{k: e[k] for k in ("id", "category", "group")} for batch in sampler.update() for e in batch] for _ in range(120)]
    if torch.cuda.is_initialized(): raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    result = {"passed": True, "source_commit": code, "executor_digest": EXECUTOR_DIGEST,
        "carry_duration_core_commit": CORE_COMMIT, **data, "config_sha256": sha(a.config),
        "service_config_sha256": sha(a.service_config), "actual_python": sys.executable,
        "python": sys.version, "numpy": numpy.__version__, "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "model_class_import_only": model_class.__name__, "model_class_source": inspect.getfile(model_class),
        "initial_adapter": str(initial), "all73_masks_and_prefixes": checks,
        "original_motion_encodings_exact": motion_exact, "seed41_planned960_draws": draws,
        "no_weights_loaded": True, "new_training_updates": 0, "new_model_calls": 0,
        "new_resets": 0, "cuda_initialized": False, "seconds": time.monotonic()-start}
    write_json(a.output, result); print(json.dumps({k: v for k, v in result.items() if k not in ("all73_masks_and_prefixes", "seed41_planned960_draws")}, indent=2))


if __name__ == "__main__": main()
