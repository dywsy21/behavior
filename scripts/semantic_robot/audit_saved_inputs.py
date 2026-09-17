"""Bounded, no-actuation audit of saved observations; NOT a controller change.

Three existing states x six recipes. Never connects to the simulator, a policy
service, or an external API. Original prompts, weights and evidence stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from semantic_robot.actions import action_language, parse_action
from semantic_robot.prompts import SYSTEM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean fixed source worktree")
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    torch.set_num_threads(4)
    torch.manual_seed(17)
    model_path = root / "models/Qwen3.5-4B"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    manifest = {
        "kind": "saved_inputs_audit_not_robot_evaluation",
        "code_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "model": str(model_path), "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "transformers": transformers.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_calls": 18, "max_new_tokens": 192, "max_seconds_after_load": 900,
        "training_updates": 0, "actuations": 0, "generation_config": model.generation_config.to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    inputs_to_probe = [
        ("radio_start", root / "radio_4b_v2/decision_000", "radio"),
        ("radio_empty_lift", root / "radio_4b_v2/decision_016", "radio"),
        ("plates_start", root / "plates_4b_v2/decision_000", "plates"),
    ]
    recipes = ("original_grammar", "original_free", "visible_facts",
               "observation_then_action", "single_part", "higher_resolution")
    started = time.perf_counter()
    rows = []
    for case, source, task in inputs_to_probe:
        original_prompt = (source / "prompt.txt").read_text()
        raw_images = [Image.open(source / f"{name}.png").convert("RGB")
                      for name in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")]
        for recipe in recipes:
            if time.perf_counter() - started >= 900:
                raise TimeoutError("Static audit time budget reached; no automatic retry")
            system, text, size = SYSTEM, original_prompt, 320
            grammar = action_language() if recipe in ("original_grammar", "higher_resolution") else None
            cap = 48 if recipe in ("original_grammar", "original_free", "single_part", "higher_resolution") else 192
            if recipe == "visible_facts":
                system = "Describe only visible image evidence. Be concise. Do not output robot commands or infer success from an instruction."
                text = ("The images are HEAD, LEFT_WRIST, RIGHT_WRIST, not mirrored. "
                        "Report briefly: (1) what objects are visible in each view; "
                        "(2) whether either gripper visibly encloses an object, or this is unknown; "
                        "(3) the next immediate task stage, NOT a numeric action. "
                        + ("Task: pick up the radio, then press its power button with the other hand."
                           if task == "radio" else "Task: transport the pizzas on their plates into a refrigerator."))
            elif recipe == "observation_then_action":
                system = SYSTEM.replace(
                    "Output ONE action line only, no JSON, explanation, plan or markdown.",
                    "Output two lines: Observation: one short visible fact; Action: one allowed command. Do not output an extended explanation.")
            elif recipe == "single_part":
                grammar = action_language(include_pairs=False)
                if task == "radio":
                    grammar = tuple(x for x in grammar if x.startswith("R ") or x == "HOLD")
                    text += ("\nAudit-only human-selected scope: approach and align the radio with the RIGHT arm; "
                             "the left arm stays still. Choose one right-arm command or HOLD. "
                             "CLOSE only if the object is visibly between the fingers. This scope is not proof of a grasp.")
                else:
                    grammar = tuple(x for x in grammar if x.startswith(("BASE ", "TORSO ")) or x == "HOLD")
                    text += ("\nAudit-only human-selected scope: first look for the plate/food and prepare access. "
                             "Choose a single BASE or TORSO command to improve your view, or HOLD if unsafe. "
                             "Keep both arms still; do not manipulate a target before locating it.")
            elif recipe == "higher_resolution":
                size = 640
            content, image_receipts = [], []
            for name, raw in zip(("HEAD", "LEFT_WRIST", "RIGHT_WRIST"), raw_images):
                resized = raw.resize((size, size))
                image_receipts.append({"view": name, "source_size": list(raw.size),
                                       "model_size": [size, size],
                                       "pixels_sha256": hashlib.sha256(resized.tobytes()).hexdigest()})
                content.extend([{"type": "text", "text": name}, {"type": "image", "image": resized}])
            content.append({"type": "text", "text": text})
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
            prefix = inputs["input_ids"].shape[1]
            trie = {}
            if grammar is not None:
                for line in grammar:
                    node = trie
                    for token in processor.tokenizer.encode(line, add_special_tokens=False):
                        node = node.setdefault(token, {})
                    node[None] = {}

            def allowed_tokens(batch_id, input_ids):
                node = trie
                for token in input_ids[prefix:].tolist():
                    node = node[token]
                allowed = [token for token in node if token is not None]
                if None in node:
                    eos = model.generation_config.eos_token_id
                    allowed += eos if isinstance(eos, list) else [eos]
                return allowed

            options = {"prefix_allowed_tokens_fn": allowed_tokens} if grammar is not None else {}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=cap, do_sample=False, use_cache=True,
                                        return_dict_in_generate=True, output_logits=True, **options)
            torch.cuda.synchronize()
            ids = result.sequences[0, prefix:]
            output = processor.decode(ids, skip_special_tokens=True).strip()
            try:
                parse_action(output)
                parses = True
            except ValueError:
                parses = False
            probability = result.logits[0][0].float().softmax(-1)
            top = probability.topk(8)
            row = {
                "case": case, "source": str(source), "recipe": recipe,
                "system": system, "text": text, "images": image_receipts,
                "input_tokens": prefix, "image_grid_thw": inputs["image_grid_thw"].tolist(),
                "chat_prefix_tail": processor.tokenizer.decode(inputs["input_ids"][0, -32:]),
                "allowed_commands": len(grammar) if grammar else None,
                "output": output, "output_tokens": len(ids), "hit_token_cap": len(ids) == cap,
                "bare_action_parses": parses, "generation_s": time.perf_counter() - t0,
                "unconstrained_first_token_top8": [
                    {"token": int(token), "text": processor.tokenizer.decode([int(token)]), "probability": float(prob)}
                    for token, prob in zip(top.indices, top.values)],
                "grammar_root_probability_mass": float(probability[list(trie)].sum()) if trie else None,
                "acted": False,
            }
            rows.append(row)
            (out / f"{case}__{recipe}.json").write_text(json.dumps(row, indent=2))
            print(json.dumps({k: row[k] for k in ("case", "recipe", "output", "output_tokens", "generation_s", "grammar_root_probability_mass")}), flush=True)
            del result, inputs, probability
    (out / "result.json").write_text(json.dumps({"status": "complete", "calls": len(rows),
        "actuations": 0, "wall_s_after_load": time.perf_counter() - started, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
