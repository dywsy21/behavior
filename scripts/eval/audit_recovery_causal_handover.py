"""Actual GPU test of recorded-prefix handover and idempotent planning.

Two fresh local-branch TRAIN states, their past actually issued commands, no
command at the current boundary, no future target, and zero physical actions.
This is not online recovery/SR or an outcome observer deployment certificate.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts'),str(REPO/'scripts/rl/memlite_online/code'),
             str(REPO/'scripts/eval/memlite_sft100')]
from recovery_corpus import file_sha,digest
from recovery_causal_handover import restore_teacher_prefix
from recovery_causal_inference import CausalPlannerInference
from skill_observation_archive import load_observation
from skill_policy_adapter import SingleFrameSkillAdapter
from g05.utils.memlite_causal_session import CausalModelIdentity,CausalSessionIdentity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--inputs-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if (a.output.exists() or cfg['schema']!='causal_planner_shadow_runtime_qa_v1'
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Pinned engineering model recipe, new output and clean frozen source required')
    fixture=json.loads((a.inputs/'manifest.json').read_text())
    if (file_sha(a.inputs/'manifest.json')!=a.inputs_sha256
            or fixture['schema']!='actual_local_causal_handover_inputs_v1'
            or fixture['role']!='TRAIN_engineering_only_never_SFT_calibration_or_test'
            or len(fixture['rows'])!=2 or len({r['case'] for r in fixture['rows']})!=2):
        raise ValueError('Immutable real logged-command handover fixtures required')
    for row in fixture['rows']:
        if (digest(row['issued_prefix'])!=row['issued_prefix_sha256']
                or row['issued_prefix']['control_step']!=row['record']['control_step']):
            raise ValueError('Changed or wrong-clock history projection')
    for key in ('planner','low','observer_backbone','observer_adapter'):
        if file_sha(root/cfg[key]['path'])!=cfg[key]['sha256']:raise ValueError('Changed pinned '+key)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Use one verified idle A800 only')
    import torch
    from g05.models.g05.qwen35 import vision
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.training.stage1_runtime import atomic_json
    from g05.models.kv_cache import SparseKVCache
    from sparse_cache import indexed_cache_freeze
    from train_memlite_recovery import parameter_digest
    torch.set_num_threads(2);torch.manual_seed(17);vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    names=json.loads((root/cfg['expert_release']/'manifest.json').read_text())['task_names']
    high=configuration(root,'high',names);low_config=configuration(root,'low',names)
    norm=cfg['normalization_sha256']
    if (file_sha(high['stats_path'])!=norm['planner'] or file_sha(low_config['stats_path'])!=norm['low']
            or file_sha(root/cfg['observer_normalization_path'])!=norm['observer']):
        raise ValueError('Separate planner/low/observer statistics mismatch')
    ids=CausalModelIdentity(**{k:cfg[k]['sha256'] for k in ('planner','low','observer_backbone','observer_adapter')},
        **{k+'_normalization':v for k,v in norm.items()})
    a.output.mkdir(parents=True)
    status=dict(status='loading',pid=os.getpid(),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),config_sha256=file_sha(a.config),
        inputs_sha256=a.inputs_sha256,optimizer_updates=0,physical_controls=0,
        runtime_deployed=False,whole_task_sr=False,observer_predictions=0,planning_interval_controls=128)
    atomic_json(a.output/'status.json',status)
    saved=torch.load(root/cfg['planner']['path'],map_location='cpu',mmap=True,weights_only=False)
    model,restored=restore_model(high,'high',state=saved['model_state_dict']);del saved;gc.collect()
    model.requires_grad_(False).eval().cuda();before=parameter_digest(model,frozen=True)
    adapter=CausalPlannerInference(model,make_processor(high,False),high,models=ids,
        loaded_planner_sha256=cfg['planner']['sha256'],cache_context=lambda:indexed_cache_freeze(model.model.ar_helper,SparseKVCache))
    low=SingleFrameSkillAdapter(make_processor(low_config,False),low_config)
    observations=[load_observation(a.inputs,row['record']) for row in fixture['rows']]
    results={};started=time.monotonic();generations=reuses=0
    for name,schedule in [('grouped',[(i,j) for i in range(2) for j in range(2)]),
                          ('interleaved',[(i,j) for j in range(2) for i in range(2)])]:
        sessions=[];outputs={}
        for i,row in enumerate(fixture['rows']):
            identity=CausalSessionIdentity('handover-'+str(i),row['task'],row['instance'],row['source_episode'],ids)
            session,receipt=restore_teacher_prefix(identity,row['issued_prefix'],
                source_task=row['task'],source_instance=row['instance'])
            sessions.append(session)
            atomic_json(a.output/(name+'-history-'+str(i)+'.json'),receipt)
        for i,request in schedule:
            session=sessions[i];tick=time.monotonic()
            result=adapter.ensure_context(session,session.identity,observations[i],
                validate_low_goal=lambda goal:low.prepare(observations[i],goal),interval_controls=128)
            if result['reused']!=(request==1):raise ValueError('Repeated same-state request generated a new command')
            if request==0:
                feedback=json.loads(result['causal_input']['execution_feedback'])
                prior=fixture['rows'][i]['issued_prefix']
                if (feedback['same_intent_controls']!=prior['control_step']-prior['events'][-1]['control_step']
                        or feedback['estimated_bundle_outcome']!='UNKNOWN'):
                    raise ValueError('Serving prefix lost real command age or invented an outcome')
            result.update(case=fixture['rows'][i]['case'],request=request)
            outputs[str((i,request))]=result;generations+=not result['reused'];reuses+=result['reused']
            with (a.output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(order=name,seconds=time.monotonic()-tick,**result))+'\n')
            atomic_json(a.output/'status.json',dict(status,status='real_gpu_handover_qa',
                real_generations=generations,idempotent_reuses=reuses))
        for session in sessions:session.close(session.identity,session.feedback.control_step)
        results[name]=outputs
    if results['grouped']!=results['interleaved']:
        atomic_json(a.output/'mismatch.json',results);raise ValueError('Cross-session ordering changed outputs')
    after=parameter_digest(model,frozen=True)
    if before!=after or any(p.grad is not None for p in model.parameters()):raise ValueError('Read-only QA mutated model')
    result=dict(status,status='real_prefix_handover_engineering_passed_not_deployed',
        real_generations=generations,idempotent_reuses=reuses,order_invariant=True,
        frozen_before=before,frozen_after=after,restore=restored,seconds=time.monotonic()-started,
        unlogged_placement_excluded=True,semantic_correctness_not_certified=True)
    atomic_json(a.output/'result.json',result);atomic_json(a.output/'status.json',result);print(json.dumps(result))


if __name__=='__main__':main()
