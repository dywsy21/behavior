"""Explicit H09X-only base/adapter service; no legacy or old-adapter fallback."""
import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import os
from pathlib import Path
import subprocess
import time

from common import TOKENS,sha,write_json
from native_actor_protocol import VERSION,parse_request
from native_dataset import load_dataset
from native_train import BASE,base_identity,check_storage
from native_storage import activate as activate_storage
from native_execution import (require_pipeline_profile,require_same_pipeline,metadata as execution_metadata,
    TIMING_PROFILES,action_codec,actor_protocol,authorization_profile)
from native_motion_codec import tokens as motion_tokens
from modeling import load_model,encode,decode
from native_reference_profile import ROOT as EXPERIMENT_ROOT


def validate_checkpoint(training,data_sha,*,protocol=VERSION):
    training=Path(training);result=json.loads((training/"result.json").read_text());identity=json.loads((training/"identity.json").read_text())
    if (result.get("status")!="COMPLETE" or result.get("optimizer_updates")!=120 or result.get("protocol")!=protocol or
            result.get("dataset_sha256")!=data_sha or identity.get("dataset_sha256")!=data_sha or
            result.get("identity_sha256")!=sha(training/"identity.json") or
            identity.get("protocol")!=protocol or identity.get("old_adapter_loaded") is not False or identity.get("base_model")!=BASE):
        raise ValueError("Only complete fresh H09Y 120-update checkpoint")
    final=result["checkpoints"][-1];folder=training/"adapter_0120"
    if final["step"]!=120 or final["adapter_sha256"]!=sha(folder/"adapter_model.safetensors") or final["adapter_config_sha256"]!=sha(folder/"adapter_config.json"):
        raise ValueError("Final fixed checkpoint identity changed")
    if json.loads((training/"restore_gate.json").read_text()).get("passed") is not True:raise ValueError("Reload gate missing")
    return folder,identity,final


def main():
    p=argparse.ArgumentParser();p.add_argument("--training",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--authorization",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];code=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip():raise ValueError("Immutable source required")
    auth=json.loads(a.authorization.read_text());data_sha=sha(a.data/"dataset.json")
    expected_protocol=actor_protocol(authorization_profile(auth))
    if (auth.get("authorize_service") is not True or auth.get("code_commit")!=code or auth.get("dataset_sha256")!=data_sha or
            auth.get("training_result_sha256")!=sha(a.training/"result.json") or not auth.get("reviewer") or
            auth.get("protocol")!=expected_protocol or auth.get("physical_gpu")!=3 or auth.get("max_calls")!=48 or auth.get("port")!=8919):
        raise ValueError("Separate exact-model/data/protocol service authorization required")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="3":raise ValueError("Separately handed-over GPU3 only")
    if a.output.resolve()!=EXPERIMENT_ROOT/"service_v1":raise ValueError("Only the registered model service output")
    rows,dataset=load_dataset(a.data);adapter,training,checkpoint=validate_checkpoint(a.training,data_sha,protocol=expected_protocol)
    if dataset["protocol"]!=expected_protocol:raise ValueError("Service action protocol differs from TRAIN")
    execution_profile=require_pipeline_profile(auth,dataset);require_same_pipeline(auth,training)
    instructions=sorted({r["actor"]["active_instruction"] for r in rows})
    storage=activate_storage(auth,a.output);check_storage(storage);base_files=base_identity()
    if any(training.get(k)!=value for k,value in base_files.items()):raise ValueError("Base weights/config/tokenizer differ from training")
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(41)
    model,processor=load_model(BASE,adapter=adapter)
    check_storage(storage)
    identity={"protocol":expected_protocol,"code_commit":code,"dataset_sha256":data_sha,"adapter_sha256":checkpoint["adapter_sha256"],
        **execution_metadata(execution_profile),
        **base_files,"physical_gpu":3,"port":8919,"training_result_sha256":sha(a.training/"result.json"),
        "authorization_sha256":sha(a.authorization),
        "storage":None if storage is None else storage.check(),
        "max_calls":48,"tokens":list(motion_tokens(action_codec(execution_profile))),"instructions":instructions,"variants":["base","finetuned"],"old_adapter_loaded":False}
    write_json(a.output/"identity.json",identity);ledger=(a.output/"calls.jsonl").open("x",buffering=1);calls=0
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value):
            data=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):self.send(200,{**identity,"calls":calls})
        def do_POST(self):
            nonlocal calls
            try:
                size=int(self.headers.get("Content-Length","0"))
                if self.path!="/motion" or not 1<=size<=13*1024**2 or calls>=48:raise ValueError("Endpoint/size/call cap")
                value=json.loads(self.rfile.read(size));row,images=parse_request(value,registered_instructions=instructions)
                if row["protocol"]!=expected_protocol:raise ValueError("Request action protocol differs from this service")
                check_storage(storage)
                x=encode(processor,row,images,supervised=False);calls+=1
                with model.disable_adapter() if value["variant"]=="base" else nullcontext():
                    result=decode(model,processor,x,action_codec=action_codec(execution_profile))
                result.update(call=calls,variant=value["variant"],protocol=expected_protocol,adapter_sha256=identity["adapter_sha256"],dataset_sha256=data_sha,
                    code_commit=code,training_result_sha256=identity["training_result_sha256"])
                if execution_profile in TIMING_PROFILES:
                    result.update(**execution_metadata(execution_profile),storage_profile=identity["storage"]["profile"])
                ledger.write(json.dumps({"unix":time.time(),"actor":value["actor"],"result":result})+"\n");self.send(200,result)
            except Exception as exc:
                ledger.write(json.dumps({"unix":time.time(),"error":repr(exc),"calls":calls})+"\n");self.send(400,{"error":str(exc),"calls":calls})
    print(json.dumps({"ready":True,**identity}),flush=True)
    try:HTTPServer(("127.0.0.1",8919),Handler).serve_forever()
    finally:ledger.close()


if __name__=="__main__":main()
