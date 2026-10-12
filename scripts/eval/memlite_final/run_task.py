"""Stock official evaluator plus a post-step execution-only ACK callback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parent/'memlite_sft100')]
from common import OFFICIAL,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--task-id',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--indices',type=int,nargs='+',required=True);p.add_argument('--num-envs',type=int,required=True)
    p.add_argument('--train-smoke',action='store_true');p.add_argument('--smoke-steps',type=int,default=256)
    a=p.parse_args()
    if (a.output.exists() or a.num_envs not in (1,2,4) or len(a.indices)%a.num_envs
            or len(set(a.indices))!=len(a.indices)
            or any(not (1<=i<=4 if a.train_smoke else 0<=i<=9) for i in a.indices)
            or (not a.train_smoke and a.smoke_steps!=256)):
        raise ValueError('Explicit fresh batch/indices; no public timeout override or retry')
    pinned=json.loads((HERE.parent/'memlite_sft100/official_manifest.json').read_text())
    for name,sha in pinned['files'].items():
        if hashlib.sha256((OFFICIAL/name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed official source '+name)
    a.output.mkdir(parents=True)
    for name in ('json','videos','attempts'):(a.output/name).mkdir()
    state=dict(status='loading',pid=os.getpid(),task=a.task,task_id=a.task_id,indices=a.indices,num_envs=a.num_envs,
        optimizer_updates=0,started=time.time(),official_commit=pinned['commit'],config_sha256=a.config_sha256,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    atomic_json(a.output/'status.json',state)
    import omnigibson as og
    if Path(og.__file__).resolve()!=OFFICIAL/'omnigibson/__init__.py':raise ValueError('Wrong simulator import')
    from omnigibson.eval.evaluator import BatchedEvaluator,resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED,EVAL_TIMEOUT_MULTIPLIER
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED)
    class AcknowledgedEvaluator(BatchedEvaluator):
        def _step_fn(self,active_env_indices):
            self.policy.set_active(active_env_indices)
            result=super()._step_fn(active_env_indices)
            # This callback cannot alter the applied action, physics, official
            # metrics, terminal flags or video. It runs only after success.
            self.policy.applied(active_env_indices)
            if max(self.policy.clock.steps)%128==0:
                state.update(status='running',control_steps=self.policy.clock.steps,updated=time.time())
                atomic_json(a.output/'status.json',state)
            return result
    mode='train' if a.train_smoke else 'public_test'
    config=OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='memlite_final_ack_batch',model=dict(_target_='client.AcknowledgedPolicy',port=a.port,
            task_id=a.task_id,config_sha256=a.config_sha256,metadata_path=str(a.output/'current_batch.json'),output=str(a.output)),
        headless=True,partial_scene_load=True,max_steps=a.smoke_steps if a.train_smoke else None,
        write_video=True,mode=mode,seed=DEFAULT_EVAL_SEED,num_envs=a.num_envs,task=dict(name=a.task),
        robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(a.output/'resolved_config.json',OmegaConf.to_container(config,resolve=True))
    try:
        with AcknowledgedEvaluator(config) as evaluator:
            (a.output/'scene.ready').touch()
            timeout=a.smoke_steps if a.train_smoke else int(evaluator.human_stats['length']*EVAL_TIMEOUT_MULTIPLIER)
            evaluator.policy.set_timeout(timeout)
            for begin in range(0,len(a.indices),a.num_envs):
                indices=a.indices[begin:begin+a.num_envs];ids=resolve_instance_ids(a.task,indices,mode=mode)
                tag='train_smoke' if a.train_smoke else 'public_test'
                metadata=[dict(task=a.task,instance_id=i,split=tag,episode=f'{a.run_id}-{a.task}-{i}-r0') for i in ids]
                atomic_json(a.output/'current_batch.json',dict(episodes=metadata,mode=tag))
                attempt=a.output/'attempts'/f'{indices[0]:02d}.json'
                if attempt.exists():raise ValueError('Never overwrite or reroll a public attempt')
                receipt=dict(status='running',indices=indices,instance_ids=ids,rollout=0,started=time.time(),max_steps=timeout,
                    simulator_seed=DEFAULT_EVAL_SEED,policy_seed=17)
                atomic_json(attempt,receipt);started=time.monotonic()
                results=evaluator.run(ids,write_video=True,video_path=str(a.output/'videos'),metrics_dir=str(a.output/'json'),rollout_id=0,video_fps=30)
                ack=evaluator.policy.finish(evaluator._batch_obs())
                if set(results)!=set(ids) or ack['steps']!=[results[i]['steps'] for i in ids]:raise ValueError('Metrics and real ACK clocks differ')
                atomic_json(attempt,dict(receipt,status='completed',finished=time.time(),seconds=time.monotonic()-started,
                    actual_applied_controls=ack['steps'],service_ack=ack))
            evaluator.policy.socket.close()
            state.update(status='completed',finished=time.time(),official_timeout=timeout,completed=len(a.indices))
            atomic_json(a.output/'status.json',state)
    except BaseException as e:
        if isinstance(e,SystemExit) and e.code in (None,0) and state['status']=='completed':return
        state.update(status='failed_no_retry',error=repr(e),finished=time.time());atomic_json(a.output/'status.json',state);raise


if __name__=='__main__':main()
