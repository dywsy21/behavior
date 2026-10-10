"""Audit real original-demonstration feedback clocks and the actual processor.

These are annotation-derived command traces, NOT physical outcome labels or
on-policy trajectories. Raw images remain audit artifacts outside Git.
"""
import argparse
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import file_sha,digest
from recovery_expert_feedback import with_expert_feedback,expert_unknown_feedback
from recovery_generation_metrics import select_original_indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();recipe=json.loads(a.recipe.read_text());root=Path(recipe['root'])
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Audit requires frozen clean code')
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    from g05.utils.training.stage1_model import configuration,make_processor
    from g05.utils.memlite_skill_protocol import planner_input_projection
    from g05.utils.memlite_causal_feedback import FeedbackIdentity,CausalFeedbackLedger,single_frame_planner_prefix
    torch.set_num_threads(2)
    release=root/recipe['expert_release'];names={}
    for line in (release/'episodes.jsonl').read_text().splitlines():
        ep=json.loads(line);names[ep['row']['task_index']]=ep['task_name']
    config=configuration(root,'high',names);processor=make_processor(config,False)
    repeat_stride=recipe['H1'].get('original_feedback_repeat_stride',16)
    wrapper=with_expert_feedback(Stage1Dataset,repeat_stride_controls=repeat_stride)
    datasets={s:wrapper(release,config,'high',s) for s in ('train','eval')}
    fixed=np.load(release/'fixed_eval_indices.npy').reshape(100,32).tolist()
    ev=datasets['eval']
    noninitial=select_original_indices(fixed,lambda i:ev.locate(i)[3],lambda i:int(ev.task_ids[i]),
        'noninitial_fixed_one_per_task_v1')
    # Broaden beyond the first noninitial anchor, using no model predictions.
    selected=[('eval',i) for ids in fixed for i in (ids[0],ids[15],ids[31])]
    selected += [('eval',i) for i in noninitial]
    for task in range(100):
        ids=np.flatnonzero(datasets['train'].task_ids==task)
        selected.extend(('train',int(ids[j])) for j in (len(ids)//4,len(ids)//2,3*len(ids)//4))
    selected=list(dict.fromkeys(selected));a.output.mkdir(parents=True)
    identity=FeedbackIdentity('expert-audit','annotation-trace',0,'only-local','a'*64,'b'*64)
    rows=[];counts=Counter();picture=0
    visual_tasks={0,3,17,40,60,99}
    for split,index in selected:
        ds=datasets[split];ep,frame,_,previous,parent,_=ds.locate(index)
        serial,anchor=(int(x) for x in ds.candidates[index]);value=expert_unknown_feedback(ep,anchor,repeat_stride_controls=repeat_stride)
        ledger=CausalFeedbackLedger(identity,uncertainty_protocol='unready_zero_confidence_v1')
        last_key=None;last_issued=-1
        for step,seg,*_ in ep['anchors'][:anchor]:
            segment=ep['segments'][seg];key=(segment['semantic'],segment['parent'])
            if key!=last_key or step-last_issued>=repeat_stride:
                ledger.issued(identity,step,segment['semantic'],segment['parent']);last_issued=step
            last_key=key
        if value!=ledger.projection(identity,frame):raise ValueError('Offline/runtime clock mismatch')
        feedback=None if value=='none' else json.loads(value)
        counts[(split,'initial' if feedback is None else 'noninitial')]+=1
        row=dict(split=split,candidate=index,task=int(ds.task_ids[index]),episode=int(ep['row']['episode_index']),
            frame=frame,anchor=anchor,previous_intent=previous,previous_parent_goal=parent,feedback=feedback,
            source_trace='annotation_derived_not_on_policy',repeat_stride_controls=repeat_stride)
        # Decode/process two temporally diverse real samples per selected task
        # and split (24 samples / 72 original RGB), without creating new labels.
        key=(split,row['task'],'visual')
        if row['task'] in visual_tasks and feedback is not None and counts[key]<2:
            raw,binding=ds.raw(index);projection=raw['model_projection']
            original=Stage1Dataset.raw(ds,index,images=False)[0]['model_projection']
            if {k:v for k,v in projection.items() if k!='execution_feedback'}!={k:v for k,v in original.items() if k!='execution_feedback'}:
                raise ValueError('Feedback wrapper changed an expert target')
            # Both processor entry points mutate their input dictionaries.
            # Give them independent copies and preserve native audit pixels.
            sample=processor.preprocess(deepcopy(raw))
            if sample['action'].shape!=(32,27) or tuple(torch.nonzero(sample['action_dim_is_pad']).flatten().tolist())!=(7,8,17,18):
                raise ValueError('Processor changed robot action mapping')
            prefix=single_frame_planner_prefix(processor.samples_builder,processor._process_tensors(deepcopy(raw)),planner_input_projection(projection))
            # Only whitelist planner inputs, not current target plan/decision.
            if not prefix:raise ValueError('Empty real model prefix')
            row.update(processor_passed=True,source_identity=binding,images={})
            panels=[]
            for camera,frames in raw['images'].items():
                pixels=frames[0].permute(1,2,0).numpy();path=a.output/f'{picture:03d}-{camera}.png'
                im=Image.fromarray(pixels);im.save(path);row['images'][camera]=dict(path=path.name,sha256=file_sha(path))
                panel=im.copy();panel.thumbnail((400,300));panels.append(panel)
            sheet=Image.new('RGB',(1200,340),'white');draw=ImageDraw.Draw(sheet)
            draw.text((4,4),f'{split} task {row["task"]} episode {row["episode"]} frame {frame}',fill='black')
            draw.text((4,18),f'prior controls={feedback["same_intent_controls"]} refreshes={feedback["same_intent_planner_refreshes"]} attempt={feedback["attempt_index"]}; UNKNOWN',fill='black')
            for j,panel in enumerate(panels):sheet.paste(panel,(400*j,40))
            path=a.output/f'sheet-{picture:03d}.png';sheet.save(path)
            row['sheet']=dict(path=path.name,sha256=file_sha(path));picture+=1;counts[key]+=1
        rows.append(row)
    if picture!=24 or len(noninitial)!=100:raise ValueError('Incomplete actual-processor audit')
    (a.output/'rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    receipt=dict(schema='expert_causal_feedback_audit_v1',status='AUTOMATED_PASSED',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        recipe_sha256=file_sha(a.recipe),release_manifest_sha256=file_sha(release/'manifest.json'),
        rows=len(rows),normal_noninitial_control_tasks=100,processed_raw_samples=picture,
        clocks_match_runtime=True,targets_unchanged=True,physical_outcomes_added=0,
        source_trace='annotation_derived_not_on_policy',original_feedback_repeat_stride=repeat_stride,
        rows_sha256=file_sha(a.output/'rows.jsonl'),
        noninitial_schedule_sha256=digest(noninitial),human_visual_review_required=True)
    (a.output/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
