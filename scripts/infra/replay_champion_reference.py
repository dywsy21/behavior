"""Read-only golden replay of the AUTHOR'S wrapper on one recorded DEV trace.

Extract the two original class ASTs to avoid importing the obsolete simulator.
Use the author's actual PiBehaviorPolicy.infer, transforms, corrections, stage
voting and inpainting; only the documented native61 state adapter is shared.
Every comparison is against the previously executed action SHA, not a target.
"""
import argparse
import ast
import dataclasses
from collections import deque
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from behavior_champion.inference import file_sha,checked_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--recorded-service',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recorded-chunks',type=Path,required=True)
    p.add_argument('--recorded-chunks-sha256',required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());original=json.loads((a.recorded_service/'result.json').read_text())
    if a.output.exists():raise ValueError('Fresh golden replay output required')
    if original['status']!='completed' or file_sha(a.config)!=original['config_sha256']:
        raise ValueError('Wrong/incomplete source episode')
    if file_sha(a.recorded_chunks)!=a.recorded_chunks_sha256:raise ValueError('Changed executed chunk journal')
    chunks=[json.loads(line) for line in a.recorded_chunks.read_text().splitlines()]
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    gpu=str(cfg['model_gpu'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Explicit idle single GPU required')
    root=Path(cfg['a800_root']);ref=root/cfg['reference'];openpi=root/cfg['openpi_reference']
    checked_sources(ref,openpi,REPO)
    receipt=root/cfg['verified']
    if file_sha(receipt)!=cfg['verified_sha256']:raise ValueError('Changed public asset receipt')
    for row in json.loads(receipt.read_text())['files']:
        path=root/cfg['asset_root']/row['path']
        if path.stat().st_size!=row['size'] or file_sha(path)!=row['sha256']:raise ValueError('Changed public asset')
    a.output.mkdir(parents=True)
    status=dict(status='loading',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                config_sha256=file_sha(a.config),recorded_result_sha256=file_sha(a.recorded_service/'result.json'),
                optimizer_updates=0,physical_controls=0,started=time.time())
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(a.output/'jax-cache'))
        sys.path[:0]=[str(ref/'src'),str(openpi/'src')]
        import numpy as np
        import jax
        import flax.nnx as nnx
        import torch
        from openpi_client.image_tools import resize_with_pad
        from b1k.shared.correction_rules import apply_correction_rules,check_gripper_variation
        from behavior_champion.inference import ChampionRuntime
        from behavior_champion.contracts import state23_from_proprio61
        from behavior_champion.session import ChampionSession
        layout=json.loads((REPO/cfg['layout']).read_text())['layout']
        runtime=ChampionRuntime(root/cfg['checkpoint'],layout['num_stages'][:cfg['model_task_count']])
        def digest():
            h=hashlib.sha256()
            for key,v in nnx.state(runtime.policy._model,nnx.Param).flat_state().items():
                array=np.asarray(v.value);h.update(repr((key,array.shape,str(array.dtype))).encode());h.update(array.tobytes())
            return h.hexdigest()
        before=digest()
        if before!=original['parameters_before']:raise ValueError('Not the exact previously executed model')
        runtime.policy._rng=jax.random.PRNGKey(cfg['policy_seed'])
        wrapper_file=ref/'src/b1k/shared/eval_b1k_wrapper.py'
        tree=ast.parse(wrapper_file.read_text())
        classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('B1KWrapperConfig','B1KPolicyWrapper')]
        if len(classes)!=2:raise ValueError('Missing fixed original wrapper classes')
        namespace=dict(__name__=__name__,dataclasses=dataclasses,np=np,torch=torch,deque=deque,BasePolicy=object,
            logger=logging.getLogger('golden_champion'),TASK_NUM_STAGES=tuple(layout['num_stages'][:50]),
            RESIZE_SIZE=224,resize_with_pad=resize_with_pad,extract_state_from_proprio=state23_from_proprio61,
            apply_correction_rules=apply_correction_rules,check_gripper_variation=check_gripper_variation)
        exec(compile(ast.Module(body=classes,type_ignores=[]),str(wrapper_file),'exec'),namespace)
        class RecordedPolicy:
            def reset(self):runtime.policy.reset()
            def infer(self,*args,**kwargs):
                self.last=runtime.policy.infer(*args,**kwargs)
                return self.last
        policy=RecordedPolicy()
        wrapper=namespace['B1KPolicyWrapper'](policy=policy,task_id=layout['dataset_to_model'][str(cfg['task_id'])])
        session=ChampionSession(episode=original['result']['episode'],task=cfg['task_id'],
            model_task=layout['dataset_to_model'][str(cfg['task_id'])],num_stages=layout['num_stages'][cfg['task_id']],
            seed=cfg['policy_seed'],predictor=runtime.infer,correct_actions=apply_correction_rules,
            gripper_variation=check_gripper_variation,apply_legacy_rules=cfg['apply_legacy_rules'])
        session.begin();wrapper.reset()
        correlation=np.asarray(runtime.policy._model.action_correlation_cholesky.value)
        status.update(status='replaying',parameters_before=before,wrapper_sha256=file_sha(wrapper_file),
            correlation_sha256=hashlib.sha256(correlation.tobytes()).hexdigest(),
            numeric_environment={k:os.environ.get(k) for k in ('XLA_FLAGS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','JAX_DEFAULT_MATMUL_PRECISION')},
            engineering_clock_replay_only=True,historical_exact_predictions=0);save()
        events=[json.loads(line) for line in (a.recorded_service/'events.jsonl').read_text().splitlines()]
        if len(chunks)!=len(events):raise ValueError('Different original control and observation journals')
        for i,event in enumerate(events):
            path=a.recorded_service/'observations'/event['observation']
            if file_sha(path)!=event['observation_sha256']:raise ValueError('Changed recorded observation')
            with np.load(path,allow_pickle=False) as obs:
                native={'robot_r1::proprio':obs['proprio'],'task_id':np.array([cfg['task_id']])}
                for key,sensor in [('head_rgb','zed_link'),('left_wrist_rgb','left_realsense_link'),
                                   ('right_wrist_rgb','right_realsense_link')]:
                    native[f'robot_r1::robot_r1:{sensor}:Camera:0::rgb']=obs[key].transpose(1,2,0)
                action=np.stack([wrapper.act(native).numpy() for _ in range(20)])
                fresh=session.infer(dict(proprio=obs['proprio'],images={name:obs[name] for name in
                    ('head_rgb','left_wrist_rgb','right_wrist_rgb')}))
            display=event['display'];actual_sha=hashlib.sha256(action.tobytes()).hexdigest()
            expected=np.asarray(chunks[i]['actions'],dtype=np.float32)
            if chunks[i]['token']!=event['token'] or hashlib.sha256(expected.tobytes()).hexdigest()!=event['action_sha256']:
                raise ValueError('Different original execution record')
            raw_grips=np.asarray(policy.last['actions'])[:,[14,22]]
            record=dict(prediction=i+1,control=i*20,action_sha256=actual_sha,
                action_exact=actual_sha==event['action_sha256'],next_stage=wrapper.current_stage,
                stage_exact=wrapper.current_stage==display['next_stage'] and list(wrapper.prediction_history)==display['stage_votes'],
                same_process_action_exact=np.array_equal(action,fresh['action_chunk']),
                same_process_stage_exact=wrapper.current_stage==fresh['display']['next_stage'] and
                    list(wrapper.prediction_history)==fresh['display']['stage_votes'],
                historical_max_abs_per_dim=np.max(np.abs(action-expected),axis=0).tolist(),
                same_process_max_abs_per_dim=np.max(np.abs(action-fresh['action_chunk']),axis=0).tolist(),
                raw_gripper_min=raw_grips.min(0).tolist(),raw_gripper_max=raw_grips.max(0).tolist())
            with (a.output/'checks.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if not record['same_process_action_exact'] or not record['same_process_stage_exact']:
                np.savez(a.output/'mismatch.npz',author=action,adapted=fresh['action_chunk'],historical=expected)
                raise ValueError(f'Same-process author/adapter mismatch at prediction {i+1}')
            # No physical execution occurs: advance only these diagnostic
            # histories by the offered chunk, including the unused final tail.
            session.ack(token=fresh['token'],controls=[dict(control_step=i*20+j,simulator_apply_ack=True,
                action_executed_raw23=x.tolist()) for j,x in enumerate(fresh['action_chunk'])])
            status.update(checked_predictions=i+1,
                historical_exact_predictions=status['historical_exact_predictions']+int(record['action_exact'] and record['stage_exact']));save()
        after=digest()
        if after!=before:raise ValueError('Read-only reference replay mutated weights')
        if len(events)!=original['result']['predictions']:raise ValueError('Incomplete recorded trace')
        exact_history=status['historical_exact_predictions']==len(events)
        status.update(status='same_process_author_adapter_exact',historical_replay_exact=exact_history,parameters_after=after,
            checked_predictions=len(events),engineering_offered_controls=len(events)*20,finished=time.time(),
            caveat='Same-process wrapper parity and cross-process historical reproducibility are separate gates. Any historical mismatch remains reported, never relaxed into an exact replay pass. Fixed recorded DEV observations; no new closed-loop robot controls or labels.')
        (a.output/'result.json').write_text(json.dumps(status,indent=2)+'\n');save();print(json.dumps(status),flush=True)
    except BaseException as e:
        status.update(status='failed_no_retry',error=repr(e),finished=time.time());save();raise


if __name__=='__main__':main()
