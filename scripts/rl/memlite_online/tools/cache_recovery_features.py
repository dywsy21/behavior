"""Frozen high48045 member-specific causal feature cache. No optimization.

The default route requires per-sample admissions. A <=4-request diagnostic
mode is explicit and its cache is rejected by the formal H0/H1 data loader.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest,file_sha
from recovery_features import feature_requests
from recovery_sft_data import CandidateArchiveReader,raw_observation,require_training_pool


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','raw','audit','history','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--admission',type=Path)
    p.add_argument('--admission-sha256')
    p.add_argument('--diagnostic-limit',type=int,choices=range(1,5))
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise ValueError('Clean frozen code required')
    recipe=json.loads(a.recipe.read_text()); root=Path(recipe['root'])
    inventory=json.loads((a.audit/'inventory.json').read_text())
    anchors=[json.loads(x) for x in (a.audit/'anchors.jsonl').read_text().splitlines()]
    history=[json.loads(x) for x in (a.history/'contexts.jsonl').read_text().splitlines()]
    receipt=json.loads((a.history/'receipt.json').read_text())
    if (receipt['inventory_sha256']!=digest(inventory) or receipt['contexts_sha256']!=file_sha(a.history/'contexts.jsonl')
            or receipt['anchors_sha256']!=file_sha(a.audit/'anchors.jsonl')): raise ValueError('Changed causal source')
    selections=[]
    if a.diagnostic_limit:
        for row in anchors:
            if row['split']=='train' and 'binding_quarantine' not in row['label_audit']:
                selections.append((row['sample_id'],'observable',0))
                if len(selections)==a.diagnostic_limit: break
    else:
        if not a.admission or not a.admission_sha256: raise ValueError('Signed admissions required')
        _,outcomes=require_training_pool(a.admission,'outcome',a.admission_sha256,purpose='feature_extraction')
        # H0 can be prepared/trained before the planner pool is ready. Read
        # approved planner rows only if present; do not require its gate here.
        for row in outcomes:
            selections.append((row['candidate']['sample_id'],'observable',row['approval']['label']['member_index']))
        path=a.admission/'planner.jsonl'
        if path.exists():
            admitted=json.loads((a.admission/'admission.json').read_text())
            if file_sha(path)!=admitted['files']['planner.jsonl']: raise ValueError('Changed planner rows')
            histories={r['sample_id']:r for r in history}
            for line in path.read_text().splitlines():
                row=json.loads(line);sid=row['candidate']['sample_id'];prior=histories[sid]['predecision']
                if prior is None: raise ValueError('Planner target has no predecision context')
                for m in range(len(json.loads(prior['issued_skills_semantic_json']))): selections.append((sid,'predecision',m))
    requests=feature_requests(anchors,history,selections)
    if not requests: raise ValueError('Empty feature request set')
    parent=recipe['parents']['high']; path=root/parent['path']
    if file_sha(path)!=parent['sha256']: raise ValueError('Wrong high parent')
    # Explicit free GPU selection; never displace an existing process.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in tuple(str(x) for x in range(8)): raise ValueError('Select one verified idle A800')
    if subprocess.check_output(['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Selected GPU occupied')
    import torch
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.memlite_causal_feedback import single_frame_member_prefix
    from g05.models.g05.qwen35 import vision
    vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    torch.set_num_threads(2)
    names={}
    for line in (root/recipe['expert_release']/'episodes.jsonl').read_text().splitlines():
        ep=json.loads(line);names[ep['row']['task_index']]=ep['task_name']
    config=configuration(root,'high',names)
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    model,restoration=restore_model(config,'high',state=saved['model_state_dict'])
    del saved;gc.collect()
    model.requires_grad_(False).eval().cuda();processor=make_processor(config,False)
    reader=CandidateArchiveReader(a.raw,inventory);by_id={r['sample_id']:r for r in anchors}
    a.output.mkdir(parents=True)
    started=time.monotonic();memo={};features={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for req in requests:
            contexts,states,steps=[],[],[]
            for check in req['checks']:
                key=digest(check)
                if key not in memo:
                    raw=raw_observation(by_id[check['sample_id']],reader,config)
                    prepared=processor._process_tensors(raw)
                    prefix=single_frame_member_prefix(processor.samples_builder,prepared,
                        **{k:check[k] for k in ('task_name','parent_goal','issued_bundle','member_index','memory','served_controls')})
                    pixel={k:v.unsqueeze(0).cuda() for k,v in prepared['pixel_values'].items()}
                    context=model.outcome_context_from_prefix([prefix],pixel).detach().float().cpu()[0]
                    state=prefix['proprio']['value'].reshape(27).detach().float().cpu()
                    memo[key]=(context,state)
                context,state=memo[key];contexts.append(context);states.append(state);steps.append(check['control_step'])
            features[req['request_id']]=dict(context=torch.stack(contexts),proprio=torch.stack(states),steps=torch.tensor(steps))
    data=dict(schema='recovery_member_feature_cache_v1',features=features,requests=requests,
        initial_head={k:v.detach().cpu() for k,v in model.outcome_head.state_dict().items()})
    torch.save(data,a.output/'features.pt')
    result=dict(schema='recovery_member_feature_receipt_v1',high_sha256=parent['sha256'],
        admission_sha256=a.admission_sha256,inventory_sha256=digest(inventory),
        history_sha256=receipt['contexts_sha256'],requests_sha256=digest(requests),
        features_sha256=file_sha(a.output/'features.pt'),stats_sha256=file_sha(config['stats_path']),
        diagnostic_only=bool(a.diagnostic_limit),optimizer_steps=0,requests=len(requests),unique_prefills=len(memo),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        seconds=time.monotonic()-started,restoration=restoration)
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
