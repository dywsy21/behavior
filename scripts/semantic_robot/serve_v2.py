"""Pinned local VLM, before/after images and per-stage finite action grammar."""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from semantic_robot.v2.protocol import Action, strict_json


def normalize_json_transport(raw):
    """Remove only ONE complete Markdown wrapper; never repair JSON/semantics.

    Original bytes remain in the ledger. Prose, multiple blocks, duplicate keys,
    truncated blocks and non-JSON content are still rejected, not guessed.
    """
    text = raw.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0] in ("```json", "```") and lines[-1] == "```":
        candidate = "\n".join(lines[1:-1]).strip()
        strict_json(candidate)
        return candidate, "single_markdown_json_fence"
    return text, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, default=8907)
    p.add_argument("--max-calls", type=int, default=320)
    args = p.parse_args()
    if not 1 <= args.max_calls <= 640:
        raise ValueError("Outside H-06 call budget")
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Clean pinned service source required")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.set_num_threads(4); torch.manual_seed(17)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    identity = {"protocol": "semantic-v2", "model": args.model, "revision": args.revision,
                "code_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "max_calls": args.max_calls, "calls": 0, "load_s": time.perf_counter()-start,
                "dtype": "bfloat16", "backend": "transformers-sdpa", "thinking": False,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "image_max_side": 640, "wrist_no_upsampling": True, "max_images": 9,
                "training_updates": 0, "action_grammar": "per-request scoped explicit JSON"}
    (out/"identity.json").write_text(json.dumps(identity, indent=2))
    ledger = (out/"calls.jsonl").open("x", buffering=1)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, data):
            raw = json.dumps(data).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            self.send(200, identity)

        def do_POST(self):
            started = time.perf_counter()
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 24_000_000 or identity["calls"] >= args.max_calls:
                    raise ValueError("Payload or global call budget exceeded")
                data = json.loads(self.rfile.read(n))
                if set(data) != {"kind", "system", "text", "images", "allowed"} or data["kind"] not in ("plan", "observe", "act"):
                    raise ValueError("Bad v2 request contract")
                if len(data["system"])+len(data["text"]) > 32000 or not 1 <= len(data["images"]) <= 9:
                    raise ValueError("Context/image count exceeded")
                content, hashes = [], []
                for entry in data["images"]:
                    if set(entry) != {"label", "png"} or len(entry["label"]) > 100:
                        raise ValueError("Bad image entry")
                    raw = base64.b64decode(entry["png"], validate=True)
                    img = Image.open(BytesIO(raw))
                    if max(img.size) > 1024:
                        raise ValueError("Unexpected image size")
                    img = img.convert("RGB"); img.thumbnail((640, 640))
                    hashes.append({"label": entry["label"], "size": img.size,
                                   "pixels_sha256": hashlib.sha256(img.tobytes()).hexdigest()})
                    content.extend([{"type": "text", "text": entry["label"]}, {"type": "image", "image": img}])
                content.append({"type": "text", "text": data["text"]})
                messages = [{"role": "system", "content": data["system"]}, {"role": "user", "content": content}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
                prefix = inputs["input_ids"].shape[1]
                if prefix > 12000:
                    raise ValueError("Token context budget exceeded")
                trie = {}
                if data["kind"] == "act":
                    if not 1 <= len(data["allowed"]) <= 240:
                        raise ValueError("Missing/bloated action palette")
                    for line in data["allowed"]:
                        if Action.parse(line).text() != line:
                            raise ValueError("Action grammar is not canonical")
                        node = trie
                        for token in processor.tokenizer.encode(line, add_special_tokens=False):
                            node = node.setdefault(token, {})
                        node[None] = {}
                elif data["allowed"]:
                    raise ValueError("Only action calls have a command grammar")
                def allowed_tokens(batch_id, ids):
                    node = trie
                    for token in ids[prefix:].tolist():
                        node = node[token]
                    allowed = [token for token in node if token is not None]
                    if None in node:
                        eos = model.generation_config.eos_token_id
                        allowed += eos if isinstance(eos, list) else [eos]
                    return allowed
                options = {"prefix_allowed_tokens_fn": allowed_tokens} if trie else {}
                cap = {"plan": 1024, "observe": 320, "act": 64}[data["kind"]]
                identity["calls"] += 1
                torch.cuda.synchronize(); generated_at = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=cap, do_sample=False, use_cache=True, **options)[0, prefix:]
                torch.cuda.synchronize()
                raw_text = processor.decode(ids, skip_special_tokens=True).strip()
                normalized, wrapper = normalize_json_transport(raw_text)
                row = {"call": identity["calls"], "kind": data["kind"], "text": normalized,
                       "raw_text": raw_text, "transport_wrapper_removed": wrapper,
                       "output_tokens": len(ids), "input_tokens": prefix, "hit_token_cap": len(ids) == cap,
                       "generation_s": time.perf_counter()-generated_at, "total_s": time.perf_counter()-started,
                       "images": hashes, "prompt_sha256": hashlib.sha256((data["system"]+data["text"]).encode()).hexdigest(),
                       "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2}
                ledger.write(json.dumps(row)+"\n"); self.send(200, row)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                ledger.write(json.dumps({"error": repr(exc), "calls": identity["calls"]})+"\n")
                self.send(400, {"error": repr(exc)})
    print("semantic-v2-ready", json.dumps(identity), flush=True)
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
