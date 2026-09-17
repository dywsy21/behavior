"""Bounded localhost-only Qwen3.5 service, in an isolated dependency overlay.

No training, remote API, privileged simulator input, or G05 import. Single worker.
"""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8897)
    parser.add_argument("--max-calls", type=int, default=160)
    parser.add_argument("--constrained", action="store_true", help="Constrain syntax only; never choose actions from simulator truth")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    torch.set_num_threads(4)
    torch.manual_seed(17)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    trie = {}
    if args.constrained:
        from semantic_robot.actions import action_language
        for line in action_language():
            node=trie
            for token in processor.tokenizer.encode(line,add_special_tokens=False):
                node=node.setdefault(token,{})
            node[None]={}
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    identity = dict(model_path=args.model, revision=args.revision, dtype="bfloat16", backend="transformers-sdpa",
                    transformers=transformers.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), load_seconds=time.perf_counter()-started,
                    training_updates=0, max_calls=args.max_calls, temperature=0, max_new_tokens=48,
                    thinking=False, batch_size=1, image_size=[320,320], calls=0,
                    constrained_action_grammar=args.constrained)
    (out / "identity.json").write_text(json.dumps(identity, indent=2))
    ledger = (out / "calls.jsonl").open("x", buffering=1)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, data):
            raw=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            self.send(200,identity)

        def do_POST(self):
            t0=time.perf_counter()
            try:
                n=int(self.headers.get("Content-Length","0"))
                if not 0 < n < 8_000_000 or identity["calls"] >= args.max_calls:
                    raise ValueError("Request size or global call budget exceeded")
                data=json.loads(self.rfile.read(n))
                if set(data) != {"system","text","images"} or len(data["images"]) != 3:
                    raise ValueError("Only explicit prompt and three RGB views are accepted")
                if len(data["text"])+len(data["system"]) > 16000:
                    raise ValueError("Context budget exceeded")
                content=[]; digests=[]
                for name,encoded in zip(("HEAD","LEFT_WRIST","RIGHT_WRIST"),data["images"]):
                    raw=base64.b64decode(encoded,validate=True)
                    img=Image.open(BytesIO(raw)).convert("RGB").resize((320,320))
                    digests.append(hashlib.sha256(img.tobytes()).hexdigest())
                    content += [{"type":"text","text":name}, {"type":"image","image":img}]
                content.append({"type":"text","text":data["text"]})
                messages=[{"role":"system","content":data["system"]},{"role":"user","content":content}]
                inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors="pt",enable_thinking=False).to("cuda")
                torch.cuda.synchronize(); t1=time.perf_counter()
                identity["calls"] += 1
                prefix_length=inputs["input_ids"].shape[1]
                def allowed_tokens(batch_id, input_ids):
                    node=trie
                    for token in input_ids[prefix_length:].tolist():
                        node=node[token]
                    allowed=[token for token in node if token is not None]
                    if None in node:
                        eos=model.generation_config.eos_token_id
                        allowed += eos if isinstance(eos,list) else [eos]
                    return allowed
                generation_options={"prefix_allowed_tokens_fn":allowed_tokens} if args.constrained else {}
                with torch.inference_mode():
                    outputs=model.generate(**inputs,max_new_tokens=48,do_sample=False,use_cache=True,**generation_options)
                torch.cuda.synchronize(); t2=time.perf_counter()
                token_ids=outputs[0,inputs["input_ids"].shape[1]:]
                text=processor.decode(token_ids,skip_special_tokens=True).strip()
                row={"call":identity["calls"], "action":text, "input_tokens":int(inputs["input_ids"].shape[1]),
                     "output_tokens":len(token_ids),"preprocess_s":t1-t0,"generation_s":t2-t1,
                     "server_total_s":time.perf_counter()-t0,"image_sha256":digests,
                     "prompt_sha256":hashlib.sha256((data["system"]+data["text"]).encode()).hexdigest(),
                     "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2}
                ledger.write(json.dumps(row)+"\n"); self.send(200,row)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                ledger.write(json.dumps({"error":repr(exc),"call_count":identity["calls"]})+"\n")
                self.send(400,{"error":repr(exc)})

    print("semantic VLM ready",json.dumps(identity),flush=True)
    try:
        HTTPServer(("127.0.0.1",args.port),Handler).serve_forever()
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
