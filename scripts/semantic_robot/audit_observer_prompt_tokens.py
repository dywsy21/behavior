"""Exact CPU processor budget check; never loads model weights or generates."""
import argparse
import json
from pathlib import Path
import time

from static_inspection_budget import restore, GroundedPolicy


class CapturePolicy(GroundedPolicy):
    def __init__(self):
        self.payload = None

    def _call(self, kind, system, text, bundle, allowed=()):
        content = []
        for label, image in zip(bundle.labels, bundle.images):
            image = image.convert("RGB").copy(); image.thumbnail((640, 640))
            content.extend([{"type": "text", "text": label}, {"type": "image", "image": image}])
        content.append({"type": "text", "text": text})
        self.payload = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        # This return is solely to complete prompt construction. No action is
        # generated, selected for deployment, or executed by this CPU script.
        return {"text": allowed[0].text()}, {"cpu_prompt_capture_only": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError("Preserve prior report")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    started = time.monotonic(); rows = []
    for index in (4, 20):
        h, state, bundle, allowed, _ = restore(a.run, index, multicamera=True)
        if h.stop_reason: raise ValueError(h.stop_reason)
        policy = CapturePolicy(); policy.act_feasible(h, state, bundle, allowed)
        inputs = processor.apply_chat_template(policy.payload, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
        tokens = inputs["input_ids"].shape[1]
        characters=len(policy.payload[0]["content"])+len(policy.payload[1]["content"][-1]["text"])
        rows.append({"decision": index, "choices": len(allowed), "input_tokens": tokens, "text_characters": characters,
                     "within_32000_characters": characters<=32000,"within_12000": tokens <= 12000})
    result = {"rows": rows, "model_calls": 0, "controls": 0, "wall_seconds": time.monotonic() - started,
              "within_existing_input_budget": all(r["within_12000"] and r["within_32000_characters"] for r in rows)}
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["within_existing_input_budget"]: raise SystemExit(1)


if __name__ == "__main__": main()
