"""Verify both ends of a completed champion episode and the original MP4.

No simulator/model calls. This authenticates execution, not a grasp, stage
semantics, general success rate or the effectiveness of fine-tuning.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8<<20):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),'Do not overwrite any prior audit')
    cfg=read(a.config);episode=a.root/'episode';service=a.root/'service'
    sim=read(episode/'status.json');model=read(service/'result.json')
    hello=read(episode/'service-hello.json');preflight=read(service/'preflight.json')
    require(sim['status']==model['status']=='completed','Both true terminal receipts required')
    require(sim['config_sha256']==model['config_sha256']==hello['config_sha256']==sha(a.config),'Mixed configuration')
    require(sim['source_commit']==model['source_commit'],'Mixed frozen integration sources')
    require(model['weights_unchanged'] and model['parameters_before']==model['parameters_after'],'Model was updated')
    require(model['optimizer_updates']==sim['optimizer_updates']==0,'Not a read-only probe')
    require(model['asset_receipt_sha256']==cfg['verified_sha256'],'Wrong public checkpoint')
    expected_identity=hashlib.sha256((sha(a.config)+cfg['verified_sha256']+model['parameters_before']).encode()).hexdigest()
    require(model['identity']==hello['identity']==expected_identity,'Wrong service identity')
    require(preflight['status']=='actual_model_interleaving_exact' and preflight['calls']==12 and
            preflight['physical_controls']==0 and preflight['parameters_before']==model['parameters_before'],
            'Missing actual-model isolation preflight')
    result=model['result'];steps=int(sim['controls'])
    require(result==sim['service_finish'] and result['control_step']==steps and result['task']==cfg['task_id'],
            'Incomplete or foreign final ACK')
    metric_path=episode/'json'/f"{cfg['task']}_{cfg['instance_id']}_0.json"
    metrics=read(metric_path)
    require(sha(metric_path)==sim['official_metrics_sha256'],'Official metrics changed')
    require(sim['official_result']=={key:metrics[key] for key in sim['official_result']} and
            metrics['steps']==steps and metrics['task']==cfg['task'] and metrics['instance_id']==cfg['instance_id'],
            'Official result and receipts disagree')
    chunks=lines(episode/'chunks.jsonl');controls=lines(episode/'controls.jsonl');events=lines(service/'events.jsonl')
    require(len(chunks)==len(events)==result['predictions'] and len(controls)==steps,'Record count mismatch')
    require(len(chunks)*20-steps==result['unused_offered_controls']<20,'Incorrect final unused action tail')
    offered=[];tokens=[];token_set=set();current_stage=0;transitions=[];stage_counts=Counter();seconds=[]
    for i,(chunk,event) in enumerate(zip(chunks,events,strict=True)):
        display=chunk['display'];token=chunk['token'];action=np.asarray(chunk['actions'],dtype=np.float32)
        require(event['kind']=='infer' and event['token']==token and event['display']==display,'Two ends differ')
        require(token not in token_set,'Reused inference token');token_set.add(token)
        require(display['control_step']==i*20 and display['task']==cfg['task_id'] and
                display['episode']==result['episode'] and display['input_stage']==current_stage,'Crossed stage/task/clock')
        require(display['inpainting_input_steps']==(0 if i==0 else 4),'Wrong continuation boundary')
        require(action.shape==(20,23) and np.isfinite(action).all() and
                hashlib.sha256(action.tobytes()).hexdigest()==event['action_sha256'],'Action bytes changed')
        identity=dict(episode=result['episode'],task=cfg['task_id'],prediction=i+1,start=i*20,
                      actions_sha256=event['action_sha256'])
        require(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==token,'Wrong command token')
        require(event['observation']==f'{i+1:06d}.npz','Wrong observation sequence')
        path=service/'observations'/event['observation']
        require(sha(path)==event['observation_sha256'],'Changed raw observation')
        with np.load(path,allow_pickle=False) as obs:
            require(set(obs.files)=={'proprio','head_rgb','left_wrist_rgb','right_wrist_rgb'},'Unexpected actor fields')
            require(obs['proprio'].shape==(61,) and np.isfinite(obs['proprio']).all(),'Bad real proprioception')
            for key in ('head_rgb','left_wrist_rgb','right_wrist_rgb'):
                image=obs[key]
                require(image.dtype==np.uint8 and image.ndim==3 and image.shape[0]==3,'Bad original camera')
        if display['next_stage']!=current_stage:
            transitions.append(dict(control=i*20,previous=current_stage,next=display['next_stage']))
        current_stage=display['next_stage'];stage_counts[display['input_stage']]+=1
        seconds.append(event['seconds']);offered.extend(action);tokens.extend([token]*20)
    for i,control in enumerate(controls):
        require(control['control_step']==i and control['token']==tokens[i] and control['simulator_apply_ack'] is True,
                'Missing, repeated or foreign actual control ACK')
        require(np.array_equal(np.asarray(control['action_executed_raw23'],dtype=np.float32),offered[i]),
                'Simulator did not execute the offered action')
        require(not (control['terminated'] or control['truncated']) or i==steps-1,'Post-terminal execution')
    require(controls[-1]['terminated'] or controls[-1]['truncated'],'No true official terminal')
    video=episode/'videos'/f"{cfg['task']}_{cfg['instance_id']}_0.mp4"
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(video)],text=True))['streams'][0]
    require(int(probe['nb_read_frames'])==steps+1==sim['video_frames_expected'] and probe['avg_frame_rate']=='30/1',
            'Not a complete original-rate episode video')
    report=dict(status='both_sides_all_controls_and_observations_verified',source_commit=model['source_commit'],
        config_sha256=sha(a.config),asset_receipt_sha256=cfg['verified_sha256'],parameters_sha256=model['parameters_before'],
        controls=steps,predictions=len(events),unused_offered_controls=result['unused_offered_controls'],
        official_result=sim['official_result'],official_metrics_sha256=sha(metric_path),video=str(video.resolve()),
        video_sha256=sha(video),video_probe=probe,stage_transitions=transitions,stage_prediction_counts=dict(stage_counts),
        service_seconds=dict(total=sum(seconds),mean=float(np.mean(seconds)),median=float(np.median(seconds))),
        elapsed_including_scene_and_finish_seconds=sim['finished']-sim['started'],
        source_files={str(path.relative_to(a.root)):sha(path) for path in
            (episode/'status.json',episode/'chunks.jsonl',episode/'controls.jsonl',service/'result.json',
             service/'events.jsonl',service/'preflight.json')},
        optimizer_updates=0,visual_review='Separate owner review required; no physical grasp inferred from stage logits',
        scope='One previously diagnosed301 DEV case; NOT an independent SR estimate or a trained2026 champion result')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
