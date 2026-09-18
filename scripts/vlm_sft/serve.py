"""One model, identical constrained decoder, adapter-on/off paired serving."""
import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import os
from pathlib import Path
import subprocess
import time

from common import TOKENS,VERSION,sha,write_json
from live import parse_request
from modeling import decode,encode,load_model


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--max-calls",type=int,default=160)
    args=p.parse_args();repo=Path(__file__).resolve().parents[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="1":raise RuntimeError("Owned GPU1 only")
    if not 1<=args.max_calls<=160:raise ValueError("Registered four-run call budget")
    if subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip():raise RuntimeError("Pinned clean source")
    args.output.mkdir(exist_ok=False)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(41)
    model,processor=load_model(args.model,adapter=args.adapter)
    identity={"protocol":VERSION,"code_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        "base_path":args.model,"base_weight_sha256":sha(Path(args.model)/"model.safetensors-00001-of-00001.safetensors"),
        "adapter":str(args.adapter),"adapter_sha256":sha(args.adapter/"adapter_model.safetensors"),"tokens":TOKENS,
        "physical_gpu":1,"port":8918,"max_calls":args.max_calls,"variants":["base","finetuned"],
        "same_processor_and_grammar":True,"fixed_local_skill_not_dynamic_planner":True}
    write_json(args.output/"identity.json",identity)
    ledger=(args.output/"calls.jsonl").open("x",buffering=1);calls=0
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value):
            data=json.dumps(value,allow_nan=False).encode();self.send_response(status)
            self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):self.send(200,{**identity,"calls":calls})
        def do_POST(self):
            nonlocal calls
            try:
                if self.path!="/motion":raise ValueError("Unknown endpoint")
                length=int(self.headers.get("Content-Length","0"))
                if not 1<=length<=1400000:raise ValueError("Request size cap")
                if calls>=args.max_calls:raise ValueError("Call budget exhausted")
                value=json.loads(self.rfile.read(length));row,images=parse_request(value)
                x=encode(processor,row,images,supervised=False);calls+=1
                with model.disable_adapter() if value["variant"]=="base" else nullcontext():
                    result=decode(model,processor,x)
                receipt={"call":calls,"variant":value["variant"],"unix":time.time(),"result":result}
                ledger.write(json.dumps(receipt)+"\n");write_json(args.output/"status.json",{"calls":calls,"last":receipt})
                self.send(200,{**result,"call":calls,"variant":value["variant"],"adapter_sha256":identity["adapter_sha256"],"protocol":VERSION})
            except Exception as exc:
                ledger.write(json.dumps({"call":calls,"error":repr(exc),"unix":time.time()})+"\n")
                self.send(400,{"error":str(exc),"calls":calls})
    print(json.dumps({"ready":True,**identity}),flush=True)
    HTTPServer(("127.0.0.1",8918),Handler).serve_forever()


if __name__=="__main__":main()
