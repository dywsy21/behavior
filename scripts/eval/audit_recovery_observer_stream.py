"""Fixed observer on real historical observations; predictions remain shadow.

The high commands and actual ACKs are replayed, NOT regenerated or rewritten.
This is TRAIN engineering diagnostics, never calibration or a new rollout.
"""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import file_sha
from skill_observation_archive import load_observation
from recovery_causal_handover import restore_teacher_prefix,AppliedActionClock
from recovery_observer_shadow import BoundObserverShadowInference
from g05.utils.memlite_causal_session import CausalModelIdentity,CausalSessionIdentity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit-config','fit','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--inputs-sha256',required=True)
    a=p.parse_args();cfg=json.loads(a.fit_config.read_text());root=Path(cfg['root'])
    fit=json.loads((a.fit/'result.json').read_text());fixture=json.loads((a.inputs/'manifest.json').read_text())
    if (a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
            or fit['status']!='observer_adapter_fit_complete_not_deployed' or fit['config_sha256']!=file_sha(a.fit_config)
            or fit['frozen_before_sha256']!=fit['frozen_after_sha256']
            or file_sha(a.inputs/'manifest.json')!=a.inputs_sha256
            or fixture['schema']!='audited_joint_observer_shadow_inputs_v1'
            or fixture['role']!='TRAIN_engineering_only_never_SFT_calibration_or_test'
            or len(fixture['rows'])!=4 or len({r['job_id'] for r in fixture['rows']})!=4):
        raise ValueError('Require complete frozen fit, explicit all-four TRAIN fixture and clean source')
    selected=a.fit/'checkpoints/selected.pt'
    if file_sha(selected)!=fit['selected_checkpoint_sha256'] or file_sha(root/cfg['high']['path'])!=cfg['high']['sha256']:
        raise ValueError('Changed fixed result observer or original backbone')
    for row in fixture['rows']:
        if Path(row['directory']).name!=row['directory']:raise ValueError('Unsafe fixture path')
        for record in row['records']:load_observation(a.inputs/row['directory'],record)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('One verified idle A800 required')
    import numpy as np
    import torch
    from g05.models.g05.qwen35 import vision
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.training.stage1_runtime import atomic_json
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    from g05.models.g05.helpers.vlm_lora import VLMloraConfig,inject_vlm_lora
    from g05.models.g05.helpers.observer_adapter import require_observer_adapter_only
    from train_memlite_recovery import parameter_digest
    torch.set_num_threads(2);vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    saved=torch.load(selected,map_location='cpu',weights_only=False)
    if (saved['schema']!='recovery_observer_adapter_checkpoint_v1' or saved['config_sha256']!=file_sha(a.fit_config)
            or saved['high_sha256']!=cfg['high']['sha256'] or saved['stats_sha256']!=cfg['stats_sha256']
            or saved['admission_sha256']!=cfg['admission_sha256'] or saved['source_commit']!=fit['source_commit']
            or saved['epoch']!=fit['selected_epoch'] or saved['adapter_config']!=cfg['adapter']
            or len(saved['optimizer']['state'])!=206
            or {float(v['step']) for v in saved['optimizer']['state'].values()}!={float(saved['step'])}):
        raise ValueError('Observer checkpoint does not reproduce selected completed fit')
    names=json.loads((root/cfg['expert_release']/'manifest.json').read_text())['task_names']
    config=configuration(root,'high',names)
    if file_sha(config['stats_path'])!=cfg['stats_sha256']:raise ValueError('Observer normalization drift')
    a.output.mkdir(parents=True)
    status=dict(status='loading',pid=os.getpid(),optimizer_updates=0,new_physical_controls=0,runtime_deployed=False,
        selected_observer_sha256=fit['selected_checkpoint_sha256'],inputs_sha256=a.inputs_sha256,
        fit_result_sha256=file_sha(a.fit/'result.json'),fit_config_sha256=file_sha(a.fit_config),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    atomic_json(a.output/'status.json',status)
    parent=torch.load(root/cfg['high']['path'],map_location='cpu',mmap=True,weights_only=False)
    policy,_=restore_model(config,'high',state=parent['model_state_dict']);del parent;gc.collect()
    policy.requires_grad_(False).eval().cuda()
    policy.model.vlm=inject_vlm_lora(policy.model.vlm,VLMloraConfig.from_mapping(cfg['adapter']))
    names=require_observer_adapter_only(policy);params=dict(policy.named_parameters())
    if len(names)!=192 or set(names)!=set(saved['adapter_state']):raise ValueError('Wrong observer adapter whitelist')
    with torch.no_grad():
        for name,tensor in saved['adapter_state'].items():
            if params[name].shape!=tensor.shape or not torch.isfinite(tensor).all():raise ValueError('Bad saved tensor')
            params[name].copy_(tensor)
            if not torch.equal(params[name].detach().cpu(),tensor):raise ValueError('Observer adapter reload differs')
    if parameter_digest(policy,frozen=True)!=fit['frozen_before_sha256']:raise ValueError('Wrong frozen observer base')
    policy.requires_grad_(False).eval()
    before=parameter_digest(policy,frozen=True)
    head=TemporalOutcomeObserver(saved['observer_state']['context_projection.0.weight'].numel(),
        include_absolute_proprio=True,include_served_controls=cfg.get('include_served_controls',False)).cuda()
    head.load_state_dict(saved['observer_state'],strict=True);head.requires_grad_(False).eval()
    if any(not torch.isfinite(t).all() for t in head.state_dict().values()):raise ValueError('Nonfinite observer head')
    processor=make_processor(config,False);results=[];started=time.monotonic()
    for row in fixture['rows']:
        old=CausalModelIdentity(**row['events'][0]['receipt']['destination_identity']['models'])
        if old.observer_backbone!=cfg['high']['sha256'] or old.observer_normalization!=cfg['stats_sha256']:
            raise ValueError('Fixture has another observer backbone or normalization')
        models=replace(old,observer_adapter=fit['selected_checkpoint_sha256'])
        identity=CausalSessionIdentity('observer-stream-shadow',row['task'],row['instance'],row['job_id'],models)
        session,_=restore_teacher_prefix(identity,row['issued_prefix'],source_task=row['task'],source_instance=row['instance'])
        clock=AppliedActionClock(session)
        observer=BoundObserverShadowInference(policy,processor,config,head,session,
            loaded_backbone_sha256=models.observer_backbone,loaded_adapter_sha256=models.observer_adapter)
        records={r['control_step']:r for r in row['records']};consumed=0;checks=[]
        def check(phase):
            step=session.feedback.control_step
            result=observer.check(identity,load_observation(a.inputs/row['directory'],records[step]))
            record=dict(case=row['case'],seed=row['seed'],phase=phase,issued_goal=session.low_goal(identity),**result)
            checks.append(record)
            with (a.output/'checks.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        for entry in row['events'][1:]:
            kind=entry['kind'];step=session.feedback.control_step
            if kind=='plan':
                check('before_recorded_plan')
                result=entry['result']
                if not result['reused']:
                    token,causal=session.begin_planning(identity,step)
                    if causal!=result['causal_input']:raise ValueError('Shadow result changed original planner input')
                    session.stage(identity,token,result['event']);session.commit(identity,token)
                if session.low_goal(identity)!=result['goal']:raise ValueError('Recorded command mismatch')
                check('after_recorded_plan')
            elif kind=='action':clock.offer(identity,np.asarray(entry['offered_actions_raw23'],dtype=np.float32))
            elif kind=='ack':
                n=entry['consumed_controls'];part=row['actual_acks'][consumed:consumed+n]
                ack=clock.acknowledge(identity,clock.pending['token'],part);consumed+=n
                if ack['control_step']!=entry['control_step']:raise ValueError('Wrong actual action clock')
            elif kind=='close':
                check('final_observation');session.close(identity,step)
            else:raise ValueError('Unknown historical event')
        if not session.closed or consumed!=len(row['actual_acks']):raise ValueError('Unclosed or omitted actual controls')
        results.append(dict(case=row['case'],seed=row['seed'],actual_source_controls=consumed,
            real_checks=sum(r['status']=='shadow_only_observer_check' for r in checks),
            max_history=max((len(m['context_steps']) for r in checks for m in r.get('members',[])),default=0),
            planner_inputs_exactly_reproduced_with_unknown_zero=True))
        atomic_json(a.output/'status.json',dict(status,status='streaming_shadow',completed=len(results)))
    if (parameter_digest(policy,frozen=True)!=before
            or any(not torch.equal(head.state_dict()[n].detach().cpu(),v) for n,v in saved['observer_state'].items())
            or any(p.grad is not None or p.requires_grad for m in (policy,head) for p in m.parameters())):
        raise ValueError('Read-only observer mutated parameters or gradients')
    atomic_json(a.output/'result.json',dict(status,status='completed_fixed_observer_stream_shadow_not_deployed',
        episodes=results,checks_sha256=file_sha(a.output/'checks.jsonl'),seconds=time.monotonic()-started,
        frozen_before_sha256=before,frozen_after_sha256=before,planner_predictions_entered=0,
        independent_accuracy_not_measured=True,calibration_not_performed=True,whole_task_sr=False))


if __name__=='__main__':main()
