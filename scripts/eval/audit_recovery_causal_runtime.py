"""Actual-GPU interleaving check, not a rollout or a recovery success test.

The same three real TRAIN RGB/proprio inputs are presented twice with NO
physical actions between requests. This checks zero-served-control refresh,
K3 memory recurrence, exact event invariance to other sessions, and downstream
low conditioning. It deliberately does not fabricate a post-command scene.
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
from recovery_corpus import file_sha
from skill_observation_archive import load_observation
from recovery_causal_inference import CausalPlannerInference
from skill_policy_adapter import SingleFrameSkillAdapter
from g05.utils.memlite_causal_session import CausalModelIdentity,CausalSessionIdentity,CausalPlannerSession


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--inputs-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    if (a.output.exists() or cfg['schema']!='causal_planner_shadow_runtime_qa_v1'
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Explicit engineering recipe, new output and frozen clean source required')
    fixture=json.loads((a.inputs/'manifest.json').read_text())
    if (file_sha(a.inputs/'manifest.json')!=a.inputs_sha256
            or fixture['schema']!='causal_runtime_initial_inputs_v1'
            or fixture['role']!='TRAIN_engineering_only_never_SFT_calibration_or_test'
            or len(fixture['rows'])!=3 or len({r['case'] for r in fixture['rows']})!=3):
        raise ValueError('Use the complete immutable three-case input export')
    for key in ('planner','low','observer_backbone','observer_adapter'):
        if file_sha(root/cfg[key]['path'])!=cfg[key]['sha256']:raise ValueError('Changed pinned '+key)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Only one verified idle A800; no teammate displacement')
    import torch
    from g05.models.g05.qwen35 import vision
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.utils.training.stage1_runtime import atomic_json
    from g05.models.kv_cache import SparseKVCache
    from sparse_cache import indexed_cache_freeze
    from train_memlite_recovery import parameter_digest
    torch.set_num_threads(2);torch.manual_seed(17);vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    names=json.loads((root/cfg['expert_release']/'manifest.json').read_text())['task_names']
    high_config=configuration(root,'high',names);low_config=configuration(root,'low',names)
    normalization=cfg['normalization_sha256']
    if (set(normalization)!={'planner','low','observer'}
            or file_sha(high_config['stats_path'])!=normalization['planner']
            or file_sha(low_config['stats_path'])!=normalization['low']):
        raise ValueError('Separately pinned high/low normalization drift')
    # This QA does not load the observer, but its identity still pins the
    # original observer statistics independently from the new serving high.
    if file_sha(root/cfg['observer_normalization_path'])!=normalization['observer']:
        raise ValueError('Original observer normalization drift')
    model_ids=CausalModelIdentity(**{k:cfg[k]['sha256'] for k in
        ('planner','low','observer_backbone','observer_adapter')},
        **{k+'_normalization':v for k,v in normalization.items()})
    a.output.mkdir(parents=True)
    status=dict(status='loading',pid=os.getpid(),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),config_sha256=file_sha(a.config),
        inputs_sha256=a.inputs_sha256,optimizer_updates=0,physical_controls=0,
        runtime_deployed=False,whole_task_sr=False,observer_predictions=0)
    atomic_json(a.output/'status.json',status)
    saved=torch.load(root/cfg['planner']['path'],map_location='cpu',mmap=True,weights_only=False)
    model,restore=restore_model(high_config,'high',state=saved['model_state_dict']);del saved;gc.collect()
    model.requires_grad_(False).eval().cuda();before=parameter_digest(model,frozen=True)
    adapter=CausalPlannerInference(model,make_processor(high_config,False),high_config,models=model_ids,
        loaded_planner_sha256=cfg['planner']['sha256'],
        cache_context=lambda:indexed_cache_freeze(model.model.ar_helper,SparseKVCache))
    low=SingleFrameSkillAdapter(make_processor(low_config,False),low_config)
    observations=[load_observation(a.inputs,r['record']) for r in fixture['rows']]
    results={};started=time.monotonic()
    for order,schedule in [('grouped',[(i,j) for i in range(3) for j in range(2)]),
                           ('interleaved',[(i,j) for j in range(2) for i in range(3)])]:
        sessions=[CausalPlannerSession(CausalSessionIdentity('runtime-qa-'+str(i),r['task'],r['instance'],
            r['source_episode'],model_ids),initial_control_step=r['record']['control_step'])
            for i,r in enumerate(fixture['rows'])]
        outputs={}
        for i,request in schedule:
            s=sessions[i];tick=time.monotonic()
            result=adapter.plan(s,s.identity,observations[i],
                validate_low_goal=lambda goal:low.prepare(observations[i],goal))
            result.update(case=fixture['rows'][i]['case'],request=request)
            outputs[str((i,request))]=result
            with (a.output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(order=order,seconds=time.monotonic()-tick,**result))+'\n')
            atomic_json(a.output/'status.json',dict(status,status='real_gpu_generation',order=order,
                generated=len(outputs),goal_is_engineering_only=True))
        for s in sessions:s.close(s.identity,s.feedback.control_step)
        results[order]=outputs
    if results['grouped']!=results['interleaved']:
        atomic_json(a.output/'mismatch.json',results)
        raise ValueError('Another task or request ordering changed a planner event/context')
    for value in results['grouped'].values():
        feedback=value['causal_input']['execution_feedback']
        if feedback!='none':
            feedback=json.loads(feedback)
            if feedback['same_intent_controls']!=0 or feedback['estimated_bundle_outcome']!='UNKNOWN':
                raise ValueError('Engineering fixture fabricated consumed actions or outcomes')
    after=parameter_digest(model,frozen=True)
    if before!=after or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Read-only serving check mutated weights or accumulated gradients')
    result=dict(status,status='gpu_interleaving_engineering_passed_not_deployed',real_high_generations=12,
        downstream_low_conditions=12,order_invariant=True,frozen_before=before,frozen_after=after,
        restore=restore,seconds=time.monotonic()-started)
    atomic_json(a.output/'result.json',result);atomic_json(a.output/'status.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
