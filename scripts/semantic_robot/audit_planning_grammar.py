"""Real tokenizer + image prefix syntax gate, CPU only, no model weights/calls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
from semantic_robot.v2.harness import Goal, parse_plan
from semantic_robot.v2.grounded_harness import parse_recovery
from semantic_robot.v2.structured_planning import response_schema, decoder_identity, schema_digest
from semantic_robot.v2.policy import PLAN_SYSTEM, COMPLETE_PLAN_INSTRUCTION


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("Preserve earlier evidence")
    started=time.monotonic()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn
    identity=decoder_identity()
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    tokenizer=processor.tokenizer
    data=build_token_enforcer_tokenizer_data(tokenizer)
    eos=json.loads((Path(a.model)/"generation_config.json").read_text())["eos_token_id"]
    if not isinstance(eos,list):eos=[eos]
    if tokenizer.eos_token_id not in eos:raise ValueError("Grammar EOS not accepted by real generation config")
    # Images are exact saved initial inputs; instruction below is a labelled
    # synthetic grammar test, not reconstruction of a lost original request.
    content=[]
    for view in ("HEAD","LEFT_WRIST","RIGHT_WRIST"):
        for suffix in ("RAW","ROBOT_GUIDE_NOT_OBJECT_LABELS"):
            label=f"CURRENT_{view}_{suffix}"
            img=Image.open(a.run/(label+".png")).convert("RGB");img.thumbnail((640,640))
            content.extend([{"type":"text","text":label},{"type":"image","image":img}])
    content.append({"type":"text","text":"Synthetic CPU contract test: turn on a radio."})
    messages=[{"role":"system","content":PLAN_SYSTEM+"\n"+COMPLETE_PLAN_INSTRUCTION},{"role":"user","content":content}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors="pt",enable_thinking=False)
    prefix=inputs["input_ids"][0].tolist()
    if len(prefix)>12000:raise ValueError("Original input budget exceeded")
    plans=[("task_plan_v1",json.dumps([asdict(Goal("pick","radio","right","visible co-motion")),
                                     asdict(Goal("press","radio power control","left","indicator changes"))])),
           ("task_plan_v1",json.dumps([asdict(Goal("navigate","named destination","both","visibly reached"))]*16))]
    recoveries=[("recovery_v1",json.dumps({"strategy":s,"visible_reason":"Visible evidence, not assumed free space."}))
                for s in ("scan_left","scan_right","move_forward","move_left","move_right","retry_approach","hold")]
    rows=[]
    for name,text in plans+recoveries+plans[:1]:
        if time.monotonic()-started>300:raise TimeoutError("CPU grammar budget")
        fn=build_transformers_prefix_allowed_tokens_fn(data,JsonSchemaParser(response_schema(name,"plan")))
        ids=list(prefix);tokens=tokenizer.encode(text,add_special_tokens=False)
        for token in tokens:
            allowed=fn(0,torch.tensor(ids))
            if token not in allowed:raise ValueError(f"Legal example rejected: {name}, offset {len(ids)-len(prefix)}")
            if any(e in allowed for e in eos):raise ValueError("Premature EOS allowed before complete JSON")
            ids.append(token)
        if tokenizer.eos_token_id not in fn(0,torch.tensor(ids)):raise ValueError("Complete JSON cannot stop")
        (parse_plan if name=="task_plan_v1" else parse_recovery)(text)
        rows.append({"schema":name,"tokens":len(tokens),"all_tokens_allowed":True,"no_premature_eos":True})
    fn=build_transformers_prefix_allowed_tokens_fn(data,JsonSchemaParser(response_schema("task_plan_v1","plan")))
    first=fn(0,torch.tensor(prefix))
    for bad in ("The task cannot", "```json"):
        if tokenizer.encode(bad,add_special_tokens=False)[0] in first:raise ValueError("Prose/fence accepted at root")
    result={"status":"pass","decoder":identity,"schema_sha256":schema_digest(),"input_tokens":len(prefix),
            "generation_eos":eos,"tokenizer_eos":tokenizer.eos_token_id,"rows":rows,
            "task_semantic_completeness_not_certified":True,"model_calls":0,"controls":0,
            "prompt_is_synthetic_grammar_test":True,"wall_s":time.monotonic()-started}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=="__main__":main()
