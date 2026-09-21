"""Fresh, explicitly authorized H09X GRASP LoRA, 120 TOTAL updates incl gate.

No heldout CE or checkpoint selection. The two-update numerical gate is part
of the same optimizer run, not an extra training run or old-adapter warmstart.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
from common import sha,write_json
from native_actor_protocol import VERSION
from native_dataset import load_dataset,checked_images
from modeling import collate,encode,load_model,supervised_loss,eos_id
from native_reference_profile import ROOT as EXPERIMENT_ROOT
from native_storage import activate as activate_storage

REPO=Path(__file__).resolve().parents[2]
BASE="/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/models/Qwen3.5-2B"
BASE_WEIGHT_SHA="aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
CONFIG={"protocol":VERSION,"base_model":BASE,"physical_gpu":3,"seed":41,
        "lora_rank":16,"lora_alpha":32,"lora_dropout":.05,"learning_rate":5e-5,
        "effective_batch_size":8,"microbatch":2,"max_updates_including_gate":120,
        "max_wall_seconds":2700,"max_artifact_MiB":256,"old_adapter":None}


def validate_config(cfg):
    if set(cfg)!=set(CONFIG) or any(type(cfg[k]) is not type(v) or cfg[k]!=v for k,v in CONFIG.items()):
        raise ValueError("Exact prospective fresh H09Y training budget/protocol required")


def require_training(release,code,config_sha,dataset_sha):
    if (release.get("authorize_training") is not True or release.get("code_commit")!=code or
            release.get("config_sha256")!=config_sha or release.get("dataset_sha256")!=dataset_sha or
            not isinstance(release.get("reviewer"),str) or not release["reviewer"]):
        raise ValueError("Independent exact-code/data/config training authorization required")


def check_storage(storage=None):
    if storage is not None:storage.check()
    elif any(shutil.disk_usage(m).free<80*1024**3 for m in ("/mnt/sdc1","/mnt/nvme_tmp")):
        raise RuntimeError("Both-filesystem 80 GiB reserve")
    if sum(p.stat().st_size for p in EXPERIMENT_ROOT.rglob("*") if p.is_file())>=6144*1024**2:
        raise RuntimeError("Complete H09Y root cap")


def base_identity():
    result={"base_weight_sha256":sha(Path(BASE)/"model.safetensors-00001-of-00001.safetensors"),
        "base_config_sha256":sha(Path(BASE)/"config.json"),"tokenizer_sha256":sha(Path(BASE)/"tokenizer.json")}
    if result["base_weight_sha256"]!=BASE_WEIGHT_SHA:raise ValueError("Registered fresh 2B base weight changed")
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--authorization",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());validate_config(cfg)
    code=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(REPO),"status","--porcelain"],text=True).strip():raise ValueError("Clean immutable source")
    release=json.loads(a.authorization.read_text());require_training(release,code,sha(a.config),sha(a.data/"dataset.json"))
    if a.output.resolve()!=EXPERIMENT_ROOT/"training_v1":raise ValueError("Only the registered fresh training output")
    rows,dataset=load_dataset(a.data,require_gate=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="3":raise ValueError("Separately handed-over GPU3 only")
    storage=activate_storage(release,a.output);check_storage(storage);base_files=base_identity()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    write_json(a.output/"launch.json",{"pid":os.getpid(),"code_commit":code,"protocol":VERSION,
        "authorization_sha256":sha(a.authorization),"dataset_sha256":sha(a.data/"dataset.json"),
        "config_sha256":sha(a.config),"started_unix":time.time(),"status":"STARTING_NOT_COMPLETE"})
    try:return run_training(a,cfg,code,rows,dataset,storage,base_files,start)
    except BaseException as exc:
        if not (a.output/"failure.json").exists():
            write_json(a.output/"failure.json",{"error":repr(exc),"optimizer_updates":0,
                "phase":"BEFORE_OPTIMIZER_LOOP","wall_seconds":time.monotonic()-start,"no_resume_or_retry_authorized":True})
        raise


def run_training(a,cfg,code,rows,dataset,storage,base_files,start):
    def budget():
        if time.monotonic()-start>=2700:raise TimeoutError("Training wall budget; no implicit continuation")
        if sum(p.stat().st_size for p in a.output.rglob("*") if p.is_file())>=256*1024**2:raise RuntimeError("Adapter artifact cap")
        check_storage(storage)
    import torch,transformers,peft
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(41);np.random.seed(41);random.seed(41)
    budget();model,processor=load_model(BASE,train=True,cfg=cfg);eos=eos_id(processor);budget()
    params=[p for p in model.parameters() if p.requires_grad]
    identity={"protocol":VERSION,"code_commit":code,"config_sha256":sha(a.config),"dataset_sha256":sha(a.data/"dataset.json"),
        "authorization_sha256":sha(a.authorization),"base_model":BASE,**base_files,
        "physical_gpu":3,"config":cfg,"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,
        "rows":len(rows),"whole_runs":len(dataset["runs"]),"old_adapter_loaded":False,"eos":eos,"started_unix":time.time(),
        "trainable_parameters":sum(p.numel() for p in params),
        "storage":None if storage is None else storage.check(),
        "trainable_parameter_names":[n for n,p in model.named_parameters() if p.requires_grad]}
    write_json(a.output/"identity.json",identity)
    cache={r["id"]:encode(processor,r,checked_images(r),supervised=True) for r in rows};budget()
    def batch(batch_rows):return {k:v.to("cuda") for k,v in collate([cache[r["id"]] for r in batch_rows],processor.tokenizer.pad_token_id).items()}
    # Choose TRAIN examples by response length, not any heldout result.
    ordered=sorted(rows,key=lambda r:cache[r["id"]]["input_ids"].shape[1]);gate=[ordered[0],ordered[-1]]
    for r in rows:
        x=cache[r["id"]];labels=x["labels"][x["labels"]!=-100]
        if processor.tokenizer.decode(labels.tolist(),skip_special_tokens=True)!=r["target"] or int(labels[-1])!=eos:
            raise RuntimeError("Real response-only/EOS mask gate failed")
    model.eval();gb=batch(gate)
    with torch.no_grad():
        native=model(**gb,use_cache=False).loss;custom=supervised_loss(model,gb)
    if not torch.allclose(native,custom,rtol=1e-5,atol=1e-4):raise RuntimeError("Real native/custom CE disagreement")
    write_json(a.output/"native_loss_gate.json",{"native":float(native),"custom":float(custom),"atol":1e-4,"rtol":1e-5,
        "assistant_only_and_EOS":True,"mixed_length_left_padding":bool((gb["attention_mask"]==0).any()),"ids":[r["id"] for r in gate]})
    del gb,native,custom;model.train()
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(params,lr=cfg["learning_rate"],betas=(.9,.95),weight_decay=.01)
    # Balance whole trajectories, then sample within each; no token rarity
    # weighting or episode identity is supplied to the model.
    groups={run["inventory_sha256"]:[r for r in rows if r["provenance"]["run_inventory_sha256"]==run["inventory_sha256"]] for run in dataset["runs"]}
    sampler=random.Random(41);keys=sorted(groups);steps=0;checkpoints=[]
    def save(step):
        budget();bound=sum(p.numel()*max(4,p.element_size()) for p in params)+2*1024**2
        used=sum(p.stat().st_size for p in a.output.rglob("*") if p.is_file())
        if used+bound>256*1024**2:raise RuntimeError("Insufficient checkpoint capacity BEFORE writing")
        folder=a.output/f"adapter_{step:04d}";model.save_pretrained(folder,safe_serialization=True);budget()
        result={"step":step,"path":str(folder),"adapter_sha256":sha(folder/"adapter_model.safetensors"),"adapter_config_sha256":sha(folder/"adapter_config.json")}
        checkpoints.append(result);write_json(a.output/"checkpoints.json",checkpoints);return folder
    try:
        with (a.output/"steps.jsonl").open("x",buffering=1) as log:
            for step in range(1,121):
                budget();optimizer.zero_grad(set_to_none=True);losses=[];drawn=[];t=time.monotonic()
                for _ in range(4):
                    budget()
                    selected=[sampler.choice(groups[sampler.choice(keys)]) for _ in range(2)];drawn.extend(r["id"] for r in selected)
                    loss=supervised_loss(model,batch(selected))
                    if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
                    (loss/4).backward();losses.append(float(loss.detach()))
                grad=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(grad) or float(grad)==0:raise RuntimeError("Nonfinite/zero adapter gradient")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError("Frozen base got gradient")
                optimizer.step();steps=step
                log.write(json.dumps({"step":step,"loss":float(np.mean(losses)),"gradient_norm":float(grad),"seconds":time.monotonic()-t,
                    "elapsed":time.monotonic()-start,"ids":drawn})+"\n")
                if step<=2 or step%10==0:
                    print(json.dumps({"step":step,"loss":float(np.mean(losses)),"elapsed":time.monotonic()-start}),flush=True)
                if step==2:
                    changed=sum(not torch.equal(p,before[n]) for n,p in model.named_parameters() if p.requires_grad)
                    if not changed:raise RuntimeError("No adapter weights changed")
                    del before
                    folder=save(step);model.eval();probe=encode(processor,gate[0],checked_images(gate[0]),supervised=False)
                    inputs={k:v.to("cuda") for k,v in probe.items()}
                    with torch.no_grad():original=model(**inputs,use_cache=False,logits_to_keep=1).logits.float().cpu()
                    # Independent serving reload; original optimizer/state remains
                    # untouched and resumes at update THREE, not another 120.
                    restored,_=load_model(BASE,adapter=folder)
                    with torch.no_grad():reloaded=restored(**inputs,use_cache=False,logits_to_keep=1).logits.float().cpu()
                    if not torch.allclose(original,reloaded,rtol=1e-5,atol=1e-4):raise RuntimeError("Adapter reload numeric gate failed")
                    write_json(a.output/"restore_gate.json",{"passed":True,"changed_tensors":changed,"max_logit_error":float((original-reloaded).abs().max()),
                        "rtol":1e-5,"atol":1e-4,"updates_included_in_total":2})
                    del restored,original,reloaded,inputs;torch.cuda.empty_cache();model.train();budget()
            save(120)
        write_json(a.output/"result.json",{"status":"COMPLETE","optimizer_updates":steps,"wall_seconds":time.monotonic()-start,
            "protocol":VERSION,"dataset_sha256":sha(a.data/"dataset.json"),"checkpoints":checkpoints,
            "identity_sha256":sha(a.output/"identity.json"),"policy_effect_evaluated":False})
    except BaseException as exc:
        write_json(a.output/"failure.json",{"error":repr(exc),"optimizer_updates":steps,"wall_seconds":time.monotonic()-start,
            "no_resume_or_retry_authorized":True});raise


if __name__=="__main__":main()
