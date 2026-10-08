"""One task, five two-env batches; stock evaluator, metrics and video writers.

This only orchestrates official BatchedEvaluator.run calls. No reward adapter,
ground-truth planner feedback, physics patch, horizon override or best-of-N.
"""
import argparse
import json
import os
from pathlib import Path
import time
from common import atomic_json, OFFICIAL


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-run',type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    import omnigibson
    if Path(omnigibson.__file__).resolve()!=OFFICIAL/'omnigibson/__init__.py':
        raise RuntimeError('Not using the sealed official v3.9.3-post2 source')
    from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import seed_everything, DEFAULT_EVAL_SEED
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    gm.HEADLESS=True
    gm.RENDER_VIEWER_CAMERA=False  # Identical to the official headless CLI.
    seed_everything(DEFAULT_EVAL_SEED)
    args.output.mkdir(parents=True,exist_ok=True)
    for directory in ('json','videos','attempts'):
        (args.output/directory).mkdir(exist_ok=True)
    mode='train' if args.smoke else 'public_test'
    cfg=OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='websocket',model=dict(_target_='omnigibson.eval.policies.WebsocketPolicy',
        host='127.0.0.1',port=args.port,action_chunk_size=16),headless=True,
        partial_scene_load=True,max_steps=128 if args.smoke else None,
        write_video=True,mode=mode,seed=DEFAULT_EVAL_SEED,num_envs=2,
        task=dict(name=args.task),robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(args.output/'resolved_config.json',OmegaConf.to_container(cfg,resolve=True))
    atomic_json(args.output/'status.json',dict(status='loading',pid=os.getpid(),task=args.task,mode=mode))
    with BatchedEvaluator(cfg) as evaluator:
        (args.output/'scene.ready').touch()
        pairs=[[1,2]] if args.smoke else [list(range(start,start+2)) for start in range(0,10,2)]
        for indices in pairs:
            ids=resolve_instance_ids(args.task,indices,mode=mode)
            receipt=dict(task=args.task,instance_ids=ids,public_indices=None if args.smoke else indices,
                mode='train_smoke' if args.smoke else mode,rollout_id=0,policy_seed=17,
                simulator_seed=DEFAULT_EVAL_SEED,started=time.time(),max_steps=evaluator.env.task._max_steps
                if hasattr(evaluator.env.task,'_max_steps') else None)
            for instance in ids:
                name=f'{args.task}_{instance}_0'
                if (args.output/'json'/(name+'.json')).exists() or (args.output/'videos'/(name+'.mp4')).exists():
                    raise RuntimeError('Refusing to overwrite an already attempted rollout')
            attempt=args.output/'attempts'/f'batch_{indices[0]:02d}.json'
            if attempt.exists():raise RuntimeError('No implicit retry of an attempted public instance')
            atomic_json(attempt,dict(status='running',**receipt))
            atomic_json(args.policy_run/'current_batch.json',receipt)
            atomic_json(args.output/'status.json',dict(status='running',pid=os.getpid(),**receipt))
            try:
                results=evaluator.run(ids,write_video=True,video_path=str(args.output/'videos'),
                    metrics_dir=str(args.output/'json'),rollout_id=0,video_fps=30)
                if set(results)!=set(ids):raise RuntimeError('Incomplete official batch result')
                atomic_json(attempt,dict(status='completed',finished=time.time(),**receipt))
            except Exception as e:
                atomic_json(attempt,dict(status='infrastructure_failed',error=repr(e),finished=time.time(),**receipt))
                raise
    atomic_json(args.output/'status.json',dict(status='completed',task=args.task,mode=mode,finished=time.time()))


if __name__=='__main__':main()
