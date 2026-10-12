"""Stratified original-RGB / real-estimate / unchanged-target human review.

Diagnostic contact sheets are not training images or outcome annotations.
Selecting extra rows by estimate is for error inspection, not a heldout metric.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound,read,write_new
from recovery_corpus import file_sha,digest
from recovery_handoff_inference import tree_sha,unique_checks
from recovery_handoff_heldout import generation_split


def validate_generation_receipt(result,cfg,config_sha256):
    if (result.get('status')!='frozen_handoff_predictions_completed_not_admitted' or result['optimizer_updates']!=0
            or result['requests_sha256']!=cfg['requests']['sha256'] or result['config_sha256']!=config_sha256
            or result['observer_before_sha256']!=result['observer_after_sha256']
            or not result['observer_head_unchanged'] or not result['all_rows_retained']
            or result['rows']!=cfg['rows'] or result['head_checks']!=cfg['head_checks']):
        raise ValueError('Complete unchanged actual observer predictions required')


def select_review_rows(rows, feedback, *, minimum=160):
    """Every covered task + every rare role-change + actual-estimate strata."""
    if len({r['sample_id'] for r in rows})!=len(rows) or set(feedback)!={r['sample_id'] for r in rows}:
        raise ValueError('Complete unique feedback universe required for review selection')
    ordered=sorted(rows,key=lambda r:digest(['human_review_v1',r['sample_id']]))
    selected={};tasks=set();strata=defaultdict(list)
    for r in ordered:
        if r['original_identity']['task'] not in tasks:
            selected[r['sample_id']]=r;tasks.add(r['original_identity']['task'])
        if r['target_change_class'] in ('explicit_bound_object_or_goal_change','unbound_relation_change_not_verified_roles'):
            selected[r['sample_id']]=r
        outcome=json.loads(feedback[r['sample_id']]['execution_feedback'])['estimated_bundle_outcome']
        change='continue' if r['target_change_class']=='identical_command' else 'change'
        strata[change,outcome].append(r)
    for group in strata.values():
        for r in group[:8]: selected[r['sample_id']]=r
    for r in ordered:
        if len(selected)>=min(minimum,len(rows)): break
        selected[r['sample_id']]=r
    return sorted(selected.values(),key=lambda r:(r['original_identity']['task'],r['sample_id']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--generation-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=read(a.config);root=Path(cfg['root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists(): raise ValueError('CPU-only fresh render')
    if file_sha(a.generation)!=a.generation_sha256: raise ValueError('Changed generation receipt')
    result=read(a.generation);run=a.generation.parent
    validate_generation_receipt(result,cfg,file_sha(a.config))
    for name,key in [('feedback.jsonl','feedback_sha256'),('checks.jsonl','checks_sha256'),('trace.jsonl','trace_sha256')]:
        if file_sha(run/name)!=result[key]: raise ValueError('Changed prediction evidence')
    rows=[json.loads(x) for x in bound(root,cfg['requests']).read_text().splitlines()]
    feedback_rows=[json.loads(x) for x in (run/'feedback.jsonl').read_text().splitlines()]
    feedback={r['sample_id']:r for r in feedback_rows}
    if len(feedback)!=len(feedback_rows): raise ValueError('Duplicate feedback')
    checks={(r['sample_id'],r['check_sha256']):r for r in
            (json.loads(x) for x in (run/'checks.jsonl').read_text().splitlines())}
    trace=defaultdict(list)
    for r in (json.loads(x) for x in (run/'trace.jsonl').read_text().splitlines()): trace[r['decision_sample_id']].append(r)
    selected=select_review_rows(rows,feedback)
    recipe=read(REPO/cfg['original_recipe'])
    from behavior_champion.release import AcceptedRelease
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    from g05.utils.training.stage1_model import configuration
    from recovery_calibrated_observer import calibrated_prediction
    from PIL import Image,ImageDraw,ImageFont
    import torch
    torch.set_num_threads(2)
    accepted=AcceptedRelease(root/recipe['expert_release'])
    model_config=configuration(root,'high',accepted.manifest['task_names'])
    split=generation_split(cfg)
    if result.get('data_split', 'train')!=split: raise ValueError('Wrong review split')
    data=Stage1Dataset(accepted.root,model_config,'high',split)
    calibration=read(bound(root,recipe['H1']['calibration']))
    if file_sha(bound(root,recipe['H1']['calibration']))!=result['calibration_sha256']: raise ValueError('Changed fixed calibration')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    a.output.mkdir(parents=True);evidence=[];pages=[];started=time.monotonic()
    for page,start in enumerate(range(0,len(selected),4)):
        subset=selected[start:start+4];canvas=Image.new('RGB',(1344,456*len(subset)),'white');draw=ImageDraw.Draw(canvas)
        for slot,r in enumerate(subset):
            sid=r['sample_id'];i=r['original_identity'];p=r['original_projection'];f=feedback[sid]
            if f['target_projection_sha256']!=digest(p) or f['original_identity']!=i or f['source_group']!=r['source_group']:
                raise ValueError('Changed teacher target or feedback source')
            all_checks=unique_checks(r);window=[all_checks[0],all_checks[-1]];top=slot*456
            estimates=[dict(control_step=t['control_step'],**calibrated_prediction(t['logits'],
                temperature=calibration['temperature'],verb='GRASP')) for t in trace[sid]]
            ef=json.loads(f['execution_feedback'])
            lines=[f"#{start+slot} task={i['task']} instance={i['instance']} episode={i['episode']} frame={i['frame']}",
                f"{r['target_change_class']} | estimated={ef['estimated_bundle_outcome']} | age={ef['same_intent_controls']} refresh={ef['same_intent_planner_refreshes']}"]
            for label,text in [('PRIOR (observer input)',p['previous_intent']),('TARGET (never observer input)',p['active_skills_text'])]:
                lines.extend(textwrap.wrap(label+': '+text,165))
            lines.append('Real checks: '+'; '.join(f"t={v['control_step']} {v['outcome']} {v['confidence']:.4f}" for v in estimates))
            lines.append('Left: T-64, head/L/R wrist. Right: T, head/L/R wrist. Estimates are NOT physical truth.')
            if len(lines)>13: raise ValueError('Review text would clip; do not silently truncate')
            for n,line in enumerate(lines): draw.text((4,top+n*16),line,font=font,fill='black')
            hashes={}
            for instant,check in enumerate(window):
                raw,identity=data.raw(check['original_candidate_index']);ep,frame,*_=data.locate(check['original_candidate_index'])
                observed=dict(images={k:v[0].numpy().copy() for k,v in raw['images'].items()},
                    proprio=data._raw_arrays(ep['row'])['observation.state'][frame].copy())
                audit=checks[sid,digest(check)]
                if tree_sha(observed)!=audit['observation_sha256'] or identity!=audit['source_identity']:
                    raise ValueError('Rendered pixels/state differ from actual observer observation')
                for camera,(key,rgb) in enumerate(observed['images'].items()):
                    hashes[f'{frame}:{key}']=tree_sha(rgb)
                    canvas.paste(Image.fromarray(rgb.transpose(1,2,0)).resize((224,224)),
                                 (instant*672+camera*224,top+224))
            evidence.append(dict(sample_id=sid,identity=i,projection=p,feedback=f,real_checks=estimates,
                source_group=r['source_group'],target_change_class=r['target_change_class'],
                original_rgb_sha256=hashes,sheet_row=slot,page=page))
        filename=f'page-{page:02d}.jpg';canvas.save(a.output/filename,quality=95,subsampling=0)
        pages.append(dict(path=filename,sha256=file_sha(a.output/filename),rows=list(range(start,start+len(subset)))))
        print(json.dumps(dict(pages=len(pages),cases=len(evidence))),flush=True)
    write_new(a.output/'review.json',dict(status='actual_feedback_rgb_rendered_requires_owner_review',
        generation_sha256=a.generation_sha256,requests_sha256=cfg['requests']['sha256'],
        render_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        cases=len(evidence),tasks=len({r['identity']['task'] for r in evidence}),pages=pages,rows=evidence,
        by_outcome=dict(Counter(json.loads(r['feedback']['execution_feedback'])['estimated_bundle_outcome'] for r in evidence)),
        by_change=dict(Counter(r['target_change_class'] for r in evidence)),human_review_complete=False,
        training_admitted=False,data_split=split,training_admission_forbidden=split=='eval',
        optimizer_updates=0,seconds=time.monotonic()-started,
        scope='All covered tasks, all rare role changes, >=8 per available estimate/change stratum, deterministic fill to160. Review selection, not independent test metrics.'))


if __name__=='__main__': main()
