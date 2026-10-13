"""Stock official batched physics/scores/video, independent champion connections."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(REPO/'src'),str(HERE),str(HERE/'memlite_sft100')]
from common import OFFICIAL,atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--task-id',type=int,required=True)
    p.add_argument('--policy-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,choices=(1,2,4),required=True)
    p.add_argument('--indices',type=int,nargs='+',required=True);p.add_argument('--train-smoke',action='store_true')
    a=p.parse_args();cfg=json.loads(a.policy_config.read_text())
    if (a.output.exists() or not a.indices or len(a.indices)%a.num_envs
            or len(set(a.indices))!=len(a.indices)
            or any(not (1<=i<=4 if a.train_smoke else 0<=i<=9) for i in a.indices)
            or a.task_id not in cfg['allowed_task_ids'] or cfg['checkpoint_kind']!='formal_ema_v2'):
        raise ValueError('Explicit final-model scope, fresh run and prescribed indices required')
    pinned=json.loads((HERE/'memlite_sft100/official_manifest.json').read_text())
    for path,sha in pinned['files'].items():
        if sha256(OFFICIAL/path)!=sha:raise ValueError('Official simulator source changed')
    a.output.mkdir(parents=True)
    for name in ('json','videos','attempts'):(a.output/name).mkdir()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    state=dict(status='loading',pid=os.getpid(),task=a.task,task_id=a.task_id,indices=a.indices,
        source_commit=source,config_sha256=sha256(a.policy_config),started=time.time(),optimizer_updates=0)
    atomic_json(a.output/'status.json',state)
    import omnigibson as og
    if Path(og.__file__).resolve()!=OFFICIAL/'omnigibson/__init__.py':raise ValueError('Wrong simulator source')
    from omnigibson.eval.evaluator import BatchedEvaluator,resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED,EVAL_TIMEOUT_MULTIPLIER
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED)
    class Evaluator(BatchedEvaluator):
        def _step_fn(self,active_env_indices):
            self.policy.set_active(active_env_indices)
            value=super()._step_fn(active_env_indices)
            self.policy.applied(active_env_indices)
            if max(self.policy.steps)%100==0:
                state.update(status='running',control_steps=list(self.policy.steps),updated=time.time())
                atomic_json(a.output/'status.json',state)
            return value
    mode='train' if a.train_smoke else 'public_test'
    config=OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='champion_per_slot_official_protocol',model=dict(_target_='champion_batched_client.ChampionBatchedPolicy',
            port=cfg['port'],num_envs=a.num_envs,task_id=a.task_id,config_sha256=state['config_sha256'],source_commit=source),
        headless=True,partial_scene_load=True,max_steps=256 if a.train_smoke else None,
        write_video=True,mode=mode,seed=DEFAULT_EVAL_SEED,num_envs=a.num_envs,task=dict(name=a.task),
        robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(a.output/'resolved_config.json',OmegaConf.to_container(config,resolve=True))
    try:
        with Evaluator(config) as evaluator:
            (a.output/'scene.ready').touch()
            timeout=256 if a.train_smoke else int(evaluator.human_stats['length']*EVAL_TIMEOUT_MULTIPLIER)
            evaluator.policy.set_timeout(timeout)
            for begin in range(0,len(a.indices),a.num_envs):
                indices=a.indices[begin:begin+a.num_envs];ids=resolve_instance_ids(a.task,indices,mode=mode)
                attempt=a.output/'attempts'/f'{indices[0]:02d}.json'
                if attempt.exists():raise ValueError('No overwrite or implicit reroll')
                receipt=dict(status='running',task=a.task,indices=indices,instance_ids=ids,rollout=0,
                    started=time.time(),max_steps=timeout,policy_seed=17,simulator_seed=DEFAULT_EVAL_SEED)
                atomic_json(attempt,receipt);start=time.monotonic()
                results=evaluator.run(ids,write_video=True,video_path=str(a.output/'videos'),
                    metrics_dir=str(a.output/'json'),rollout_id=0,video_fps=30)
                physical=evaluator.policy.finish()
                if set(results)!=set(ids) or physical['actual_applied_controls']!=[results[i]['steps'] for i in ids]:
                    raise ValueError('Applied physical clocks differ from original metrics')
                atomic_json(attempt,dict(receipt,status='completed',finished=time.time(),seconds=time.monotonic()-start,**physical))
            evaluator.policy.close()
            state.update(status='completed',finished=time.time(),official_timeout=timeout,completed=len(a.indices))
            atomic_json(a.output/'status.json',state)
    except BaseException as e:
        if isinstance(e,SystemExit) and e.code in (None,0) and state['status']=='completed':return
        state.update(status='failed_no_retry',error=repr(e),finished=time.time());atomic_json(a.output/'status.json',state);raise


if __name__=='__main__':main()
