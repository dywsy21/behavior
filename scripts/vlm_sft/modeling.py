"""One chat-prefix implementation shared by SFT, holdout decoding and serving."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from common import CAMERAS, SYSTEM, TOKENS, VERSION


def messages(row, images):
    if set(images)!=set(CAMERAS):raise ValueError("Three onboard views required")
    content=[]
    for view in CAMERAS:
        img=images[view].convert("RGB")
        if img.size!=(256,256):img=img.resize((256,256),Image.Resampling.LANCZOS)
        content.extend([{"type":"text","text":view.upper()}, {"type":"image","image":img}])
    content.append({"type":"text","text":row["text"]})
    return [{"role":"system","content":SYSTEM},{"role":"user","content":content}]


def load_images(root,row):
    return {v:Image.open(Path(root)/row["images"][v]).convert("RGB") for v in CAMERAS}


def encode(processor,row,images,*,supervised):
    import torch
    msg=messages(row,images)
    prefix=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors="pt",enable_thinking=False)
    if prefix["input_ids"].shape[1]>1800:raise ValueError("Prefix exceeds registered 2048-token context")
    if not supervised:return prefix
    # Build the training input from the *identical inference prefix*. Tokenizing
    # assistant messages separately can silently introduce a different think block.
    target=row["target"]
    if target not in TOKENS:raise ValueError("Unknown target")
    response=processor.tokenizer.encode(target,add_special_tokens=False)+[processor.tokenizer.convert_tokens_to_ids("<|im_end|>")]
    if not 2<=len(response)<=16 or any(x is None or x<0 for x in response):raise ValueError("Unexpected response encoding")
    result={k:v.clone() for k,v in prefix.items()}
    old=prefix["input_ids"].shape[1]
    result["input_ids"]=torch.cat([prefix["input_ids"],torch.tensor([response],dtype=torch.long)],dim=1)
    for key in ("attention_mask","token_type_ids","mm_token_type_ids"):
        if key in result:
            value=1 if key=="attention_mask" else 0
            result[key]=torch.cat([result[key],torch.full((1,len(response)),value,dtype=result[key].dtype)],dim=1)
    labels=torch.full_like(result["input_ids"],-100);labels[:,old:]=result["input_ids"][:,old:]
    result["labels"]=labels
    if not torch.equal(result["input_ids"][:,:old],prefix["input_ids"]):raise RuntimeError("Train/infer prefix mismatch")
    return result


def collate(encoded,pad_token_id):
    import torch
    maximum=max(x["input_ids"].shape[1] for x in encoded)
    result={}
    seqkeys={"input_ids","attention_mask","token_type_ids","mm_token_type_ids","labels"}
    keys=set(encoded[0])
    if any(set(x)!=keys for x in encoded):raise ValueError("Processor returned inconsistent fields")
    for key in keys:
        if key in seqkeys:
            fill=pad_token_id if key=="input_ids" else (-100 if key=="labels" else 0)
            result[key]=torch.cat([torch.nn.functional.pad(x[key],(maximum-x[key].shape[1],0),value=fill) for x in encoded],dim=0)
        else:
            result[key]=torch.cat([x[key] for x in encoded],dim=0)
    if "labels" in result:
        selected=result["labels"]!=-100
        if not torch.equal(result["labels"][selected],result["input_ids"][selected]):raise RuntimeError("Target mask mismatch")
        if (selected & (result["attention_mask"]==0)).any():raise RuntimeError("Padding supervised")
    return result


def supervised_loss(model,batch):
    import torch
    labels=batch["labels"]
    counts=(labels!=-100).sum(1)
    keep=int(counts.max())+1
    # Only response-position logits are materialized. Full hidden-state training
    # still propagates through the entire legal image/text prefix.
    output=model(**{k:v for k,v in batch.items() if k!="labels"},use_cache=False,logits_to_keep=keep)
    logits=output.logits[:,:-1,:].float()
    targets=labels[:,-(keep-1):]
    if (targets!=-100).sum()!=counts.sum():raise RuntimeError("Response truncated from loss")
    return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100)


def load_model(model_path,*,adapter=None,train=False,cfg=None):
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from peft import LoraConfig,PeftModel,get_peft_model
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    processor.tokenizer.padding_side="left"
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda")
    if adapter:
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=train)
    elif train:
        target=[name for name,module in model.named_modules()
                if name.startswith("model.language_model.") and isinstance(module,torch.nn.Linear)]
        if not target:raise RuntimeError("Language-only linear targets not found; do not guess vision exclusions")
        model=get_peft_model(model,LoraConfig(r=cfg["lora_rank"],lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],target_modules=target,bias="none",task_type="CAUSAL_LM"))
    if train:
        model.train()
        params=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        if not params or any("lora_" not in name or "language_model" not in name for name,_ in params):
            raise RuntimeError("Unexpected trainable parameter outside language LoRA")
    else:model.eval()
    return model,processor


def decode(model,processor,encoded):
    import torch
    inputs={k:v.to("cuda") for k,v in encoded.items() if k!="labels"}
    prefix=inputs["input_ids"].shape[1]
    trie={}
    eos=processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for token in TOKENS:
        node=trie
        for value in processor.tokenizer.encode(token,add_special_tokens=False):node=node.setdefault(value,{})
        node[eos]={}
    def allowed(batch_id,ids):
        node=trie
        for value in ids[prefix:].tolist():node=node[value]
        return list(node) or [eos]
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        result=model.generate(**inputs,max_new_tokens=20,do_sample=False,use_cache=True,
            prefix_allowed_tokens_fn=allowed,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    answer=processor.tokenizer.decode(result[0,prefix:],skip_special_tokens=True).strip()
    if answer not in TOKENS:raise RuntimeError("Constrained decoder violated action vocabulary")
    return {"prediction":answer,"latency_s":elapsed,"input_tokens":prefix,
            "output_tokens":int(result.shape[1]-prefix),"input_ids_sha256":hashlib.sha256(inputs["input_ids"].cpu().numpy().tobytes()).hexdigest()}
