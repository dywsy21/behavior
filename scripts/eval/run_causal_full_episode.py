"""Fresh official full episode. No teacher prefix, timeout override or retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'memlite_sft100')]
from common import OFFICIAL,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());case=cfg['cases'][a.case]
    if a.output.exists() or cfg['schema']!='fresh_full_episode_readonly_v1':raise ValueError('Fresh output required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=HERE,text=True).strip():raise ValueError('Dirty source')
    if cfg['mode']!='public_test' or cfg['instance_id']!=301:raise ValueError('Use preregistered development instance')
    pinned=json.loads((HERE/'memlite_sft100/official_manifest.json').read_text())
    for name,sha in pinned['files'].items():
        if hashlib.sha256((OFFICIAL/name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed official source '+name)
    a.output.mkdir(parents=True)
    for name in ('videos','json'):(a.output/name).mkdir()
    status=dict(status='loading',case=a.case,task=case['task'],instance_id=301,pid=os.getpid(),started=time.time(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),official_commit=pinned['commit'])
    atomic_json(a.output/'status.json',status)
    import omnigibson as og
    if Path(og.__file__).resolve()!=OFFICIAL/'omnigibson/__init__.py':raise ValueError('Wrong simulator source')
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False
    seed_everything(DEFAULT_EVAL_SEED)
    class RecordedEvaluator(BatchedEvaluator):
        def load_batch(self,*args,**kwargs):
            super().load_batch(*args,**kwargs)
            if self.num_envs!=1:raise ValueError('Each film has one isolated task process')
            # One true initial-state frame, then one official post-action frame
            # for EVERY control including the last partial chunk's final action.
            self._write_video(self.instance_eval_states[0])
            status.update(status='running',official_max_steps=self.env.task._max_steps,
                simulator_seed=DEFAULT_EVAL_SEED,first_state_recorded=True)
            atomic_json(a.output/'status.json',status)
        def _step_fn(self,indices):
            if indices!=[0]:raise ValueError('Foreign environment slot')
            # Same official physics/metrics path, with ACK only AFTER success.
            actions=self.policy.forward(self._batch_obs())
            term,trunc,_=self._apply_actions(actions,indices)
            self.policy.applied(actions[0],term[0],trunc[0])
            if self.policy.elapsed%128==0:
                status.update(controls=self.policy.elapsed,updated=time.time())
                atomic_json(a.output/'status.json',status)
            return term,trunc
    resolved=OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='causal_full_episode',model=dict(_target_='full_episode_client.FullEpisodePolicy',
            port=case['port'],task_id=case['task_id'],case=a.case,config_sha256=status['config_sha256'],output=str(a.output)),
        headless=True,partial_scene_load=True,max_steps=None,write_video=True,mode='public_test',
        seed=DEFAULT_EVAL_SEED,num_envs=1,task=dict(name=case['task']),robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(a.output/'resolved_config.json',OmegaConf.to_container(resolved,resolve=True))
    try:
        with RecordedEvaluator(resolved) as evaluator:
            results=evaluator.run([301],write_video=True,video_path=str(a.output/'videos'),
                metrics_dir=str(a.output/'json'),rollout_id=0,video_fps=30)
            receipt=evaluator.policy.finish(evaluator._batch_obs())
            record=results[301]
            if receipt['control_step']!=record['steps']:raise ValueError('Official/ACK step mismatch')
            # Official normalized-distance diagnostics may be +inf when a
            # body part never moves. Preserve the original metric file, but
            # do not put nonfinite ancillary values into the strict receipt.
            summary={k:record[k] for k in ('task','instance_id','rollout_id','steps','success','q_score')}
            metric_path=a.output/'json'/f"{case['task']}_301_0.json"
            status.update(status='completed',finished=time.time(),controls=record['steps'],official_result=summary,
                official_metrics_sha256=hashlib.sha256(metric_path.read_bytes()).hexdigest(),
                service_finish=receipt,video_frames_expected=record['steps']+1,optimizer_updates=0)
            atomic_json(a.output/'status.json',status)
    except BaseException as e:
        if isinstance(e,SystemExit) and e.code in (None,0) and status.get('status')=='completed':
            return
        status.update(status='failed_no_retry',error=repr(e),finished=time.time())
        atomic_json(a.output/'status.json',status);raise


if __name__=='__main__':main()
