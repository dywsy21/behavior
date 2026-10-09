"""Cold simulator check of a shared skill sensor/reward on a pinned replay.

This executes recorded real controls, NOT a learned policy, RL update, or SR
evaluation. It validates the exact failure start and the physical reward clock.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'scripts/eval/memlite_sft100'),str(Path(__file__).resolve().parents[1]/'code'),str(REPO/'src')]
from common import atomic_json,OFFICIAL,OFFICIAL_COMMIT,sha256
from recovery_corpus import digest
from recovery_recorder import proprio61
from behavior_branch_state import restore_branch_metadata
from skill_aligned_reward import SkillIdentity,SkillReward
from skill_sim_measurements import OmniSkillMeasurements


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-directory',type=Path,required=True)
    p.add_argument('--branch',required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    official=json.loads((REPO/'scripts/eval/memlite_sft100/official_manifest.json').read_text())
    if official['commit']!=OFFICIAL_COMMIT or any(sha256(OFFICIAL/k)!=v for k,v in official['files'].items()):
        raise ValueError('Unpinned installed simulator')
    directory=a.case_directory/a.branch
    if sha256(directory/'manifest.json')!=a.manifest_sha256:raise ValueError('Not the pinned reviewed branch')
    m=json.loads((directory/'manifest.json').read_text());result=json.loads((a.case_directory/'result.json').read_text())
    source=json.loads((a.case_directory/'source.json').read_text())
    if (source['source_episode']['split']!='train' or m not in result['branches']
            or sha256(a.case_directory/'full_snapshot.pt')!=result['full_snapshot_sha256']
            or sha256(directory/'transitions.jsonl')!=m['transitions_sha256']
            or sha256(directory/'plans.json')!=m['plans_sha256']):
        raise ValueError('Unbound original state, commands, or physical evidence')
    rows=[json.loads(x) for x in (directory/'transitions.jsonl').read_text().splitlines()]
    plans=json.loads((directory/'plans.json').read_text())
    if (len(plans)!=2 or plans[1]['decision']!='RETRY' or plans[0]['control_step']!=0
            or [r['control_step'] for r in rows]!=list(range(len(rows))) or len(rows)!=m['controls']):
        raise ValueError('Need a single actually observed corrective attempt')
    retry=plans[1]['control_step'];bundle=json.loads(plans[1]['active_skills_semantic_json'])
    if len(bundle)!=1 or not 0<retry<len(rows):raise ValueError('Single-member start required')
    for row in rows:
        event=plans[int(row['control_step']>=retry)]
        if (row['simulator_apply_ack'] is not True or row['context']['context_id']!=event['event_sha256']
                or row['context']['active_skills_semantic_json']!=event['active_skills_semantic_json']):
            raise ValueError('Applied action or intent provenance changed')
    a.output.mkdir(parents=True);started=time.monotonic()
    receipt=dict(status='loading',pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        task=source['task'],instance_id=source['instance_id'],source_group=source['source_group'],split=result['split'],
        source_branch_sha256=a.manifest_sha256,full_snapshot_sha256=result['full_snapshot_sha256'],
        policy_evaluation=False,optimizer_steps=0,quota=None,actual_controls=0,skill_verb=bundle[0]['verb'])
    atomic_json(a.output/'status.json',receipt)
    import numpy as np
    import torch
    from PIL import Image
    import omnigibson as og
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED)
    class NoPolicy:
        def reset(self):pass
    class Evaluator(BatchedEvaluator):
        def load_policy(self):return NoPolicy()
        def __exit__(self,kind,error,traceback):
            if error is not None:
                atomic_json(a.output/'status.json',dict(receipt,status='failed',error=repr(error),seconds=time.monotonic()-started))
            return super().__exit__(kind,error,traceback)
    cfg=OmegaConf.create(json.loads((a.case_directory/'resolved_config.json').read_text()));cfg.write_video=False
    try:
        with Evaluator(cfg) as evaluator:
            evaluator.load_batch({0:source['instance_id']});inst=evaluator.instance_eval_states[0]
            saved=torch.load(a.case_directory/'full_snapshot.pt',weights_only=False,map_location='cpu')
            og.sim.load_state(deepcopy(saved['world']),serialized=False)
            restore_branch_metadata(evaluator,saved['metadata'])
            random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
            torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
            for _ in range(3):og.sim.render()
            obs,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(obs[0],inst)
            def capture(label):
                paths={}
                for camera,key in evaluator.robot_camera_names.items():
                    path=a.output/(label+'-'+camera+'.png')
                    rgb=inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]
                    Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR).save(path)
                    paths[path.name]=sha256(path)
                return paths
            error=float(np.max(np.abs(np.asarray(proprio61(inst.obs))-rows[0]['proprio_before'])))
            if error>1e-3:raise ValueError('Cold state restore mismatch: '+str(error))
            identity=SkillIdentity(str(a.output),source['task'],int(source['instance_id']),
                digest([result['full_snapshot_sha256'],a.branch]),plans[1]['event_sha256'],digest(bundle),0)
            sensor=OmniSkillMeasurements(inst.env_accessor,identity,bundle[0],bundle=bundle)
            reward=None;last=None;receipt.update(status='replaying_fault',initial_max_proprio_error=error,
                                               original_rgb=dict(restored_seed=capture('seed')))
            atomic_json(a.output/'status.json',receipt)
            with (a.output/'reward-ledger.jsonl').open('x',buffering=1) as stream:
                for t,row in enumerate(rows):
                    if t==retry:
                        initial,audit=sensor.read(identity)
                        target_pos=sensor.target.get_position_orientation()[0].detach().cpu().numpy()
                        pos_error=float(np.max(np.abs(target_pos-row['physical_before']['position'])))
                        if pos_error>.005:raise ValueError('Failure-start target position drift: '+str(pos_error))
                        if 'directed_open_fraction' in audit and abs(audit['directed_open_fraction']-row['physical_before']['directed_open_fraction'])>.005:
                            raise ValueError('Failure-start joint clearance drift')
                        reward=SkillReward(identity,initial,control_step=retry,protected_facts=())
                        receipt.update(status='checking_corrective_reward',failure_start_error_m=pos_error,
                            start_measurement=initial,reward_start_control=retry,protected_progress_contract_tested=False)
                        receipt['original_rgb']['reward_start']=capture('start')
                        atomic_json(a.output/'status.json',receipt)
                    command=np.asarray(row['action_executed_raw23'],dtype=np.float32)
                    if command.shape!=(23,) or not np.isfinite(command).all():raise ValueError('Malformed actual control')
                    term,trunc,_=evaluator._apply_actions(torch.as_tensor(command)[None],[0]);receipt['actual_controls']+=1
                    if bool(term[0]) or bool(trunc[0]):raise ValueError('Unexpected official terminal in local engineering replay')
                    if reward is not None:
                        measurement,audit=sensor.read(identity)
                        last=reward.advance(identity,t+1,measurement,protected_values={},time_limit=t==len(rows)-1)
                        data=dict(last);data['identity']=asdict(identity)
                        stream.write(json.dumps(dict(control_step=t+1,measurement=measurement,physical_evidence=audit,reward=data))+'\n')
                        if last['terminated'] or last['truncated']:break
            if last is None or not last['skill_success']:
                raise ValueError('Pinned corrective replay did not produce physically stable skill reward')
            receipt.update(status='passed_recorded_correction_sensor_reward',seconds=time.monotonic()-started,
                reward_end_control=last['control_step'],physical_skill_reward=True,
                learned_policy_success=False,protected_progress_contract_tested=False)
            receipt['original_rgb']['reward_end']=capture('end')
            atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt)
    except BaseException as error:
        atomic_json(a.output/'status.json',dict(receipt,status='failed',error=repr(error),seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
