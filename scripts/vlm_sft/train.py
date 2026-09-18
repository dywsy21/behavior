"""One-GPU bounded LoRA SFT with explicit mask, gradient and restore gates."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from common import VERSION,sha,write_json
from modeling import collate,decode,encode,load_images,load_model,supervised_loss
from prepare import reserve


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--review",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--gate",action="store_true")
    p.add_argument("--microbatch",type=int,default=4)
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    repo=Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip():raise RuntimeError("Clean immutable source required")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="1":raise RuntimeError("H09 exclusively owns physical GPU1")
    args.output.mkdir(parents=True,exist_ok=False);reserve(args.output)
    review=json.loads(args.review.read_text());audit=json.loads((args.data/"frame_audit.json").read_text())
    if not review.get("approved") or review["image_manifest_sha256"]!=sha(args.data/"image_manifest.json") or not audit["passed"]:
        raise RuntimeError("Image/label/FK gates not passed for this exact data")
    import torch,transformers,peft
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"]);random.seed(cfg["seed"])
    train=[json.loads(x) for x in (args.data/"train_images.jsonl").read_text().splitlines()]
    valid=[json.loads(x) for x in (args.data/"validation_images.jsonl").read_text().splitlines()]
    if len(train)>cfg["max_training_samples"]:raise RuntimeError("Training sample cap exceeded")
    # Cache only train and a fixed validation subset as CPU processor tensors.
    model,processor=load_model(cfg["base_model"],train=True,cfg=cfg)
    params=[p for p in model.parameters() if p.requires_grad]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    identity={"code_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        "protocol":VERSION,"config":cfg,"config_sha256":sha(args.config),"review_sha256":sha(args.review),
        "data_image_manifest_sha256":sha(args.data/"image_manifest.json"),"train_rows":len(train),"validation_rows":len(valid),
        "base_model":cfg["base_model"],"base_weight_sha256":sha(Path(cfg["base_model"])/"model.safetensors-00001-of-00001.safetensors"),
        "base_config_sha256":sha(Path(cfg["base_model"])/"config.json"),"tokenizer_sha256":sha(Path(cfg["base_model"])/"tokenizer.json"),
        "torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,
        "gpu":torch.cuda.get_device_name(),"physical_gpu":1,"microbatch":args.microbatch,
        "trainable_parameters":sum(p.numel() for p in params),"total_parameters":sum(p.numel() for p in model.parameters()),
        "trainable_parameter_names":names,"gate_only":args.gate,"started_unix":time.time()}
    write_json(args.output/"identity.json",identity)
    encoded={}
    def item(row):
        if row["id"] not in encoded:
            encoded[row["id"]]=encode(processor,row,load_images(args.data,row),supervised=True)
        return encoded[row["id"]]
    def batch(rows):return {k:v.to("cuda") for k,v in collate([item(r) for r in rows],processor.tokenizer.pad_token_id).items()}
    # Actual image/token-prefix, response mask and mixed-length padding gate.
    gate_rows=[next(r for r in train if r["task_id"]==task) for task in cfg["tasks"]]
    mask_receipts=[]
    for row in gate_rows:
        x=item(row);n=int((x["labels"]!=-100).sum());labels=x["labels"][x["labels"]!=-100]
        actual=processor.tokenizer.decode(labels.tolist(),skip_special_tokens=True)
        if actual!=row["target"]:raise RuntimeError("Assistant-only target mask failed")
        mask_receipts.append({"id":row["id"],"tokens":x["input_ids"].shape[1],"supervised_tokens":n,
            "decoded_supervision":actual,"all_prompt_image_tokens_ignored":bool((x["labels"][:,:-n]==-100).all())})
    write_json(args.output/"mask_gate.json",mask_receipts)
    if args.gate:
        # Independent native-label loss comparison catches causal shifts, EOS
        # truncation, left-padding and logits-tail mistakes on real image input.
        model.eval();gate_batch=batch(gate_rows)
        with torch.no_grad():
            native_loss=model(**gate_batch,use_cache=False).loss
            custom_loss=supervised_loss(model,gate_batch)
        if not torch.allclose(native_loss,custom_loss,rtol=1e-5,atol=1e-4):
            raise RuntimeError(f"Native/custom supervised loss differs: {native_loss} vs {custom_loss}")
        write_json(args.output/"native_loss_gate.json",{"native":float(native_loss),"custom":float(custom_loss),
            "absolute_error":float((native_loss-custom_loss).abs()),"rtol":1e-5,"atol":1e-4,"batch_ids":[r["id"] for r in gate_rows],
            "includes_left_padding":bool((gate_batch["attention_mask"]==0).any()),"eos_explicitly_supervised":True})
        del gate_batch,native_loss,custom_loss;torch.cuda.empty_cache();model.train()
    optimizer=torch.optim.AdamW(params,lr=cfg["learning_rate"],betas=(.9,.95),weight_decay=.01)
    maximum=2 if args.gate else cfg["max_optimizer_updates"]
    accumulation=cfg["effective_batch_size"]//args.microbatch
    if accumulation*args.microbatch!=cfg["effective_batch_size"]:raise ValueError("Microbatch must divide effective batch")
    freq=Counter(r["target"] for r in train)
    # Cap rarity weighting at 5x: singleton labels do not become the dataset.
    largest=max(freq.values());weights=[min(5.,math.sqrt(largest/freq[r["target"]])) for r in train]
    sampler=random.Random(cfg["seed"])
    metrics=(args.output/"steps.jsonl").open("x",buffering=1)
    validation=sorted(valid,key=lambda r:r["source_fingerprint"])[:48]
    start=time.monotonic();checkpoints=[];stop="UPDATE_BUDGET"
    trainable_before={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad} if args.gate else None
    def save(step):
        reserve(args.output)
        folder=args.output/f"adapter_{step:04d}"
        model.save_pretrained(folder,safe_serialization=True)
        result={"step":step,"path":str(folder),"adapter_sha256":sha(folder/"adapter_model.safetensors"),
            "adapter_config_sha256":sha(folder/"adapter_config.json")}
        checkpoints.append(result);write_json(args.output/"checkpoints.json",checkpoints)
        return folder
    for step in range(1,maximum+1):
        if time.monotonic()-start>=cfg["max_training_wall_seconds"]:
            stop="WALL_TIME_BUDGET";break
        reserve(args.output)
        warm=max(1,int(cfg["max_optimizer_updates"]*cfg["warmup_ratio"]))
        multiplier=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/(cfg["max_optimizer_updates"]-warm)))
        for group in optimizer.param_groups:group["lr"]=cfg["learning_rate"]*multiplier
        optimizer.zero_grad(set_to_none=True)
        losses=[];drawn=[];step_start=time.monotonic()
        for _ in range(accumulation):
            rows=sampler.choices(train,weights=weights,k=args.microbatch);drawn.extend(r["id"] for r in rows)
            loss=supervised_loss(model,batch(rows))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            (loss/accumulation).backward();losses.append(float(loss.detach()))
        grad=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(grad) or float(grad)==0:raise RuntimeError("Nonfinite/zero LoRA gradient")
        if step==1:
            frozen_grad=[name for name,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            if frozen_grad:raise RuntimeError("Gradient reached frozen parameters")
            write_json(args.output/"gradient_gate.json",{"lora_gradient_norm":float(grad),"frozen_parameters_with_gradient":frozen_grad,
                "trainable_all_language_lora":True,"vision_and_projector_frozen":True})
        optimizer.step()
        record={"step":step,"loss":float(np.mean(losses)),"gradient_norm":float(grad),"learning_rate":optimizer.param_groups[0]["lr"],
                "step_s":time.monotonic()-step_start,"elapsed_s":time.monotonic()-start,
                "cuda_peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,"cuda_peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,
                "sample_ids":drawn}
        metrics.write(json.dumps(record)+"\n")
        if step%10==0 or step<=2:print(json.dumps({k:v for k,v in record.items() if k!="sample_ids"}),flush=True)
        if args.gate or step in cfg["checkpoints_at_updates"]:
            if not args.gate:
                model.eval();vals=[]
                with torch.inference_mode():
                    for i in range(0,len(validation),args.microbatch):vals.append(float(supervised_loss(model,batch(validation[i:i+args.microbatch]))))
                model.train();write_json(args.output/f"validation_{step:04d}.json",{"loss":float(np.mean(vals)),"sample_ids":[r["id"] for r in validation],"selection":"fixed first48 source hashes, validation only"})
            if step==maximum or not args.gate:save(step)
    final_step=step if stop=="UPDATE_BUDGET" else step-1
    if not checkpoints or checkpoints[-1]["step"]!=final_step:save(final_step)
    if args.gate:
        changed=[name for name,p in model.named_parameters() if p.requires_grad and not torch.equal(p,trainable_before[name])]
        if not changed:raise RuntimeError("Optimizer did not update LoRA")
        model.eval();probe=encode(processor,gate_rows[0],load_images(args.data,gate_rows[0]),supervised=False)
        before=decode(model,processor,probe)
        with torch.no_grad():before_logits=model(**{k:v.to("cuda") for k,v in probe.items()},use_cache=False,logits_to_keep=1).logits.detach().float().cpu()
        # Restore through the same API used for serving, no merged base copy.
        del optimizer,params,trainable_before,model;torch.cuda.empty_cache()
        restored,_=load_model(cfg["base_model"],adapter=checkpoints[-1]["path"])
        after=decode(restored,processor,encode(processor,gate_rows[0],load_images(args.data,gate_rows[0]),supervised=False))
        with torch.no_grad():after_logits=restored(**{k:v.to("cuda") for k,v in probe.items()},use_cache=False,logits_to_keep=1).logits.detach().float().cpu()
        maximum_error=float((before_logits-after_logits).abs().max())
        if not torch.allclose(before_logits,after_logits,rtol=1e-5,atol=1e-4):raise RuntimeError("Adapter restore logits differ")
        if before["prediction"]!=after["prediction"] or before["input_ids_sha256"]!=after["input_ids_sha256"]:
            raise RuntimeError("Adapter inference restore gate failed")
        write_json(args.output/"restore_gate.json",{"passed":True,"changed_lora_tensors":len(changed),"before":before,"restored":after,
            "max_logit_error":maximum_error,"rtol":1e-5,"atol":1e-4})
    else:
        # Final small optimizer state, useful for exact continuation; no frozen base.
        torch.save({"optimizer":optimizer.state_dict(),"python_sampler":sampler.getstate(),"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all(),"step":final_step,"code_commit":identity["code_commit"]},args.output/"optimizer_final.pt")
    write_json(args.output/"result.json",{"status":"complete","gate_only":args.gate,"optimizer_updates":final_step,"stop":stop,
        "training_elapsed_s":time.monotonic()-start,"checkpoints":checkpoints,"identity_sha256":sha(args.output/"identity.json"),
        "policy_effect_evaluated":False,"no_success_claim":True})
    print(json.dumps({"complete":True,"updates":final_step,"elapsed_s":time.monotonic()-start}),flush=True)


if __name__=="__main__":main()
