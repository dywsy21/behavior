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
    p.add_argument('--policy-port',type=int)
    p.add_argument('--policy-config',type=Path)
    p.add_argument('--model',choices=('parent','candidate'))
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    policy_mode=a.policy_port is not None
    if policy_mode != (a.policy_config is not None and a.model is not None):
        raise ValueError('Policy mode requires port, pinned config and model together')
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
    policy_cfg=None
    if policy_mode:
        policy_cfg=json.loads(a.policy_config.read_text());case=policy_cfg['cases'][a.case_directory.name]
        if (case['manifest_sha256']!=a.manifest_sha256 or case['context_id']!=plans[1]['event_sha256']
                or case['semantic_bundle']!=plans[1]['active_skills_semantic_json']
                or case['parent_goal']!=plans[1]['parent_goal'] or case['start_control']!=retry
                or case['end_control']!=len(rows) or case['task']!=source['task']):
            raise ValueError('Policy comparison differs from frozen actual skill/horizon')
    a.output.mkdir(parents=True);started=time.monotonic()
    receipt=dict(status='loading',pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        task=source['task'],instance_id=source['instance_id'],source_group=source['source_group'],split=result['split'],
        source_branch_sha256=a.manifest_sha256,full_snapshot_sha256=result['full_snapshot_sha256'],
        policy_evaluation=policy_mode,optimizer_steps=0,quota=None,actual_controls=0,skill_verb=bundle[0]['verb'])
    if policy_mode:
        receipt.update(model=a.model,model_sha256=policy_cfg['models'][a.model]['sha256'],policy_seed=a.seed,
            policy_config_sha256=sha256(a.policy_config),high_layer_fixed_to_actual_issued_skill=True,
            horizon_kind='unchanged finite reference continuation',whole_task_success_evaluated=False)
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
            socket=None;video=None
            episode=digest([str(a.output),a.seed,a.model])
            if policy_mode:
                from websockets.sync.client import connect
                from wire import packb,unpackb
                import imageio.v2 as iio
                socket=connect('ws://127.0.0.1:'+str(a.policy_port),max_size=16<<20,compression=None,ping_timeout=None)
                hello=unpackb(socket.recv(timeout=60))
                if hello.get('protocol')!='fixed_skill_fm_v1' or hello['models']!=policy_cfg['models']:
                    raise ValueError('Wrong inference server/model identities')
                socket.send(packb(dict(op='begin',case=a.case_directory.name,model=a.model,episode=episode,
                    seed=a.seed,manifest_sha256=a.manifest_sha256)))
                ack=unpackb(socket.recv(timeout=60))
                if ack!=dict(model_sha256=receipt['model_sha256'],case=a.case_directory.name,episode=episode,
                             context_id=plans[1]['event_sha256'],control_step=retry):
                    raise ValueError('Inference session not exactly bound')
                video=iio.get_writer(str(a.output/'policy.mp4'),fps=15,codec='libx264',macro_block_size=None)
            reward=None;last=None;pending_actions=None
            receipt.update(status='replaying_fault',initial_max_proprio_error=error,
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
                    if policy_mode and t>=retry:
                        within=(t-retry)%16
                        if within==0:
                            images={camera+'_rgb':np.ascontiguousarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3].transpose(2,0,1))
                                    for camera,key in evaluator.robot_camera_names.items()}
                            socket.send(packb(dict(op='action',episode=episode,control_step=t,
                                context_id=plans[1]['event_sha256'],images=images,
                                proprio=np.asarray(proprio61(inst.obs),dtype=np.float32))))
                            response=unpackb(socket.recv(timeout=120))
                            if (set(response)!={'action_chunk','episode','context_id','control_step','model_sha256'}
                                    or response['episode']!=episode or response['context_id']!=plans[1]['event_sha256']
                                    or response['control_step']!=t or response['model_sha256']!=receipt['model_sha256']):
                                raise ValueError('Cross-model/context action or server error: '+repr(response)[:300])
                            pending_actions=response['action_chunk']
                            if pending_actions.shape!=(16,23) or not np.isfinite(pending_actions).all():
                                raise ValueError('Invalid returned raw23 chunk')
                            receipt['original_rgb'][f'policy_control_{t}']=capture(f'control-{t:06d}')
                        command=np.asarray(pending_actions[within],dtype=np.float32)
                    else:
                        command=np.asarray(row['action_executed_raw23'],dtype=np.float32)
                    if command.shape!=(23,) or not np.isfinite(command).all():raise ValueError('Malformed actual control')
                    term,trunc,_=evaluator._apply_actions(torch.as_tensor(command)[None],[0]);receipt['actual_controls']+=1
                    if not policy_mode and (bool(term[0]) or bool(trunc[0])):raise ValueError('Unexpected official terminal in local engineering replay')
                    if reward is not None:
                        measurement,audit=sensor.read(identity)
                        last=reward.advance(identity,t+1,measurement,protected_values={},
                            time_limit=t==len(rows)-1 or bool(trunc[0]),official_terminal=bool(term[0]))
                        data=dict(last);data['identity']=asdict(identity)
                        stream.write(json.dumps(dict(control_step=t+1,measurement=measurement,physical_evidence=audit,reward=data,
                            action_executed_raw23=command.tolist(),simulator_apply_ack=True,
                            action_source='learned_native_fm' if policy_mode else 'recorded_correction'))+'\n')
                        if video is not None and (t-retry)%2==0:
                            video.append_data(np.concatenate([np.asarray(Image.fromarray(
                                inst.obs[evaluator.robot_camera_names[k]+'::rgb'].detach().cpu().numpy()[...,:3]).resize((224,224)))
                                for k in ('head','left_wrist','right_wrist')],axis=1))
                        if last['terminated'] or last['truncated']:break
            if video is not None:video.close()
            if socket is not None:socket.close()
            if last is None or (not policy_mode and not last['skill_success']):
                raise ValueError('Pinned corrective replay did not produce physically stable skill reward')
            receipt.update(status='completed_fixed_skill_policy' if policy_mode else 'passed_recorded_correction_sensor_reward',seconds=time.monotonic()-started,
                reward_end_control=last['control_step'],physical_skill_reward=True,
                learned_policy_success=bool(policy_mode and last['skill_success']),final_outcome=last['outcome'],
                protected_progress_contract_tested=False)
            receipt['original_rgb']['reward_end']=capture('end')
            atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt)
    except BaseException as error:
        atomic_json(a.output/'status.json',dict(receipt,status='failed',error=repr(error),seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
