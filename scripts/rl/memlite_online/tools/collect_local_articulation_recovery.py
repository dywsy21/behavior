"""Cold TRAIN-only door/drawer recovery candidates, with actual Open evidence.

Replay a reference to an observed false->true stable goal transition. Reverse a
short measured robot path through real actuators until that goal is lost, then
issue RETRY and track forward using current-state joint/base feedback. Neither
reference frame boundaries nor teacher intent imply a label. Every output is
candidate-only; physical evidence must be independently inspected for admission.
"""
import argparse
from copy import deepcopy
from collections import deque
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'scripts/eval/memlite_sft100'),str(Path(__file__).resolve().parents[1]/'code'),str(REPO/'src')]
from common import atomic_json,OFFICIAL,OFFICIAL_COMMIT,sha256
from recovery_corpus import canonical,digest,group_key,split_group
from recovery_recorder import proprio61
from recovery_articulation_teacher import CausalArticulation,StationaryServo,servo,yaw_of
from behavior_branch_state import capture_branch_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    official=json.loads((REPO/'scripts/eval/memlite_sft100/official_manifest.json').read_text())
    if official['commit']!=OFFICIAL_COMMIT or any(sha256(OFFICIAL/k)!=v for k,v in official['files'].items()):
        raise ValueError('Changed official simulator')
    source=json.loads((a.proposal/'manifest.json').read_text());selected=source['selected_segment']
    skills=json.loads(selected['semantic'])
    if (source['schema']!='recovery_expert_skill_proposal_v1' or source['source_episode']['split']!='train'
            or source['source_group']!=group_key(source['task'],source['instance_id'])
            or source['recovery_split']!=split_group(source['task'],source['instance_id'])
            or len(skills)!=1 or skills[0]['verb'] not in ('OPEN_DOOR','CLOSE_DOOR','OPEN_DRAWER','CLOSE_DRAWER','OPEN_LID','CLOSE_LID')
            or not skills[0]['target'] or skills[0].get('unbound_relation') or skills[0].get('target_part')):
        raise ValueError('Need fully bound single whole-object articulation in original TRAIN')
    for name,value in source['files'].items():
        if Path(name).name!=name or sha256(a.proposal/name)!=value:raise ValueError('Changed source bytes')
    a.output.mkdir(parents=True);started=time.monotonic()
    status=dict(schema='local_articulation_collection_status_v1',status='loading',source_commit=commit,
        proposal_sha256=sha256(a.proposal/'manifest.json'),pid=os.getpid(),task=source['task'],
        instance_id=source['instance_id'],source_group=source['source_group'],split=source['recovery_split'],
        actual_controls=0,instance_loads=0,snapshot_restores=0,optimizer_steps=0,quota=None)
    atomic_json(a.output/'status.json',status);atomic_json(a.output/'source.json',source)
    import numpy as np
    import torch
    from PIL import Image
    import omnigibson as og
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED
    from omnigibson.eval.utils.obs_utils import create_video_writer
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    from g05.utils.memlite_skill_protocol import (semantic_active_skills_text,parse_active_skills_semantic_json,
        append_b_memory_idempotent,canonical_json)
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED)
    arrays=np.load(a.proposal/'prefix.npz',allow_pickle=False);actions=arrays['action']
    if actions.shape!=(source['controls'],23):raise ValueError('Wrong raw23 prefix')
    class NoPolicy:
        def reset(self):pass
    class Evaluator(BatchedEvaluator):
        def load_policy(self):return NoPolicy()
        def __exit__(self,kind,error,traceback):
            if error is not None:atomic_json(a.output/'status.json',dict(status,status='failed',error=repr(error),seconds=time.monotonic()-started))
            return super().__exit__(kind,error,traceback)
    cfg=OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='offline_local_articulation_teacher',headless=True,partial_scene_load=True,max_steps=None,
        write_video=True,mode='train',seed=DEFAULT_EVAL_SEED,num_envs=1,
        task=dict(name=source['task']),robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(a.output/'resolved_config.json',OmegaConf.to_container(cfg,resolve=True))
    with Evaluator(cfg) as evaluator:
        evaluator.load_batch({0:source['instance_id']});status['instance_loads']+=1
        inst=evaluator.instance_eval_states[0];robot=inst.env_accessor.robot
        for _ in range(3):og.sim.render()
        obs,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(obs[0],inst)
        pos_indices=list(range(3,10))+list(range(28,35))+list(range(53,57))
        initial=np.abs(np.asarray(proprio61(inst.obs))-arrays['state'][0])
        atomic_json(a.output/'initial_alignment.json',dict(absolute_error=initial.tolist(),fresh_render=True))
        if initial[pos_indices].max()>.05:raise ValueError('Wrong original initial joint configuration')
        scope={k:getattr(v,'wrapped_obj',v) for k,v in inst.env_accessor.object_scope.items()}
        matches=[(k,o) for k,o in scope.items() if o is not None and getattr(o,'name',None)==skills[0]['target']]
        if len(matches)!=1:raise ValueError('No exact native task-scope target; do not guess aliases')
        entity,target=matches[0]
        states=[v for k,v in target.states.items() if k.__name__=='Open']
        if len(states)!=1:raise ValueError('Missing unambiguous installed Open state')
        open_state=states[0];want_open=skills[0]['verb'].startswith('OPEN_')
        idx=lambda v:v.detach().cpu().numpy().astype(int)
        arms={arm:idx(robot.arm_action_idx[arm]) for arm in robot.arm_names}
        grips={arm:int(idx(robot.gripper_action_idx[arm])[0]) for arm in robot.arm_names}
        if arms['left'].tolist()!=list(range(7,14)) or arms['right'].tolist()!=list(range(15,22)) or grips!={'left':14,'right':22}:
            raise ValueError('Wrong R1Pro action mapping')
        def measured():
            joints=robot.get_joint_positions();q=np.zeros(23,dtype=np.float32)
            q[3:7]=joints[robot.trunk_control_idx].detach().cpu().numpy()
            for arm in robot.arm_names:
                q[arms[arm]]=joints[robot.arm_control_idx[arm]].detach().cpu().numpy()
                q[grips[arm]]=float(joints[robot.gripper_control_idx[arm]].mean())/.05*2-1
            q[[14,22]]=np.clip(q[[14,22]],-1,1)
            pos,quat=robot.get_position_orientation();base=np.array([float(pos[0]),float(pos[1]),yaw_of(quat)])
            return q,base
        def physical():
            raw=open_state.get_value();value=raw.item() if hasattr(raw,'item') else raw
            if type(value) is not bool:raise ValueError('Installed Open is not a Boolean measurement')
            pos,quat=target.get_position_orientation()
            return dict(target_name=target.name,entity=entity,open=value,goal_predicate=value==want_open,
                object_joints=target.get_joint_positions().detach().cpu().tolist(),position=pos.tolist(),orientation=quat.tolist(),
                grasp={arm:robot.is_grasping(arm=arm,candidate_obj=target).name for arm in robot.arm_names},
                predicate_contract='official Open whole-object state, not annotation endpoint',label_only=True)
        def step(command):
            if shutil.disk_usage(a.output).free<100*2**30:raise RuntimeError('Disk safety reserve reached')
            command=np.asarray(command,dtype=np.float32)
            if command.shape!=(23,) or not np.isfinite(command).all():raise ValueError('Invalid raw23 control')
            term,trunc,_=evaluator._apply_actions(torch.as_tensor(command[None]),[0]);status['actual_controls']+=1
            if bool(term[0]) or bool(trunc[0]):raise RuntimeError('Official terminal, not a local failure label')
            return physical()
        def video(path):evaluator._set_video_writer(inst,create_video_writer(fpath=str(path),resolution=(448,672),rate=30))
        path=[];history=CausalArticulation();seeded=False
        video(a.output/'prefix.mp4')
        with (a.output/'prefix.jsonl').open('x',buffering=1) as stream:
            for t,command in enumerate(actions):
                q,base=measured();before=physical();proprio=proprio61(inst.obs)
                # Measured joint/base waypoints, original gripper targets. A
                # contact-limited finger aperture is NOT the closing command.
                q[[14,22]]=command[[14,22]]
                if t>=selected['start']:
                    path.append(dict(q=q.tolist(),base=base.tolist(),source_frame=t,physical=before))
                evaluator._write_video(inst);after=step(command)
                outcome=history.update(t,after['goal_predicate'] if t>=selected['start'] else None,moving=False)
                stream.write(json.dumps(dict(control_step=t,action_executed_raw23=command.tolist(),proprio_before=proprio,
                    proprio_after=proprio61(inst.obs),physical_before=before,physical_audit=after,
                    outcome_candidate=outcome,source_label_kind='original_reference_not_corrective_BC'))+'\n')
                if outcome=='SUCCEEDED':seeded=True;break
                if t%16==0:atomic_json(a.output/'status.json',dict(status,status='replaying_prefix',source_frame=t))
        evaluator._set_video_writer(inst,None)
        if not seeded:
            receipt=dict(status,status='reference_articulation_not_reproduced',training_approved=False)
            atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt);return
        seed_q,seed_base=measured();seed_q[[14,22]]=actions[t,[14,22]]
        seed=dict(q=seed_q.tolist(),base=seed_base.tolist(),source_frame=t+1,physical=physical())
        # Keep only a physically local measured suffix. Missing/too distant
        # reverse targets are a coverage gap, not permission to teleport.
        recent=path[-192:];knots=recent[::4]
        if not knots or all(x['physical']['goal_predicate'] for x in knots):raise ValueError('No local physical false-goal path')
        if any(np.linalg.norm(np.asarray(x['base'])[:2]-seed_base[:2])>.25 for x in knots):
            raise ValueError('Reference suffix leaves local 25cm articulation domain')
        snapshot=dict(world=deepcopy(og.sim.dump_state(serialized=False)),metadata=capture_branch_metadata(evaluator),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
            seed=seed,waypoints=knots,source_manifest_sha256=status['proposal_sha256'],source_commit=commit,official_commit=OFFICIAL_COMMIT)
        torch.save(snapshot,a.output/'full_snapshot.pt');atomic_json(a.output/'seed.json',seed)
        directory=a.output/'reverse_reference_fault';directory.mkdir();(directory/'rgb').mkdir()
        semantic=canonical(skills);task=source['task'].replace('_',' ')
        text=semantic_active_skills_text(parse_active_skills_semantic_json(semantic))
        memory=canonical_json(dict(task_name=task,issued_command_history=[],verified_world_facts=[]))
        previous='None';plans=[];current=None
        def issue(tick,decision):
            nonlocal memory,previous,current
            updated=append_b_memory_idempotent(memory,previous,task_name=task)
            event=dict(control_step=tick,decision=decision,active_skills_semantic_json=semantic,active_skills_text=text,
                parent_goal=selected['parent'],memory_before=memory,memory_update=updated,previous_intent=previous,
                source='offline_measured_articulation_teacher_proposal',not_success_label=True)
            event['event_sha256']=digest(event);plans.append(event);current=event;memory=updated;previous=text
            atomic_json(directory/'plans.json',plans)
        issue(0,'EXECUTE');anchors={};outcomes=deepcopy(history);outcomes.last=-1
        reverse=list(reversed(knots));forward=knots+[seed];cursor=0;phase='clean_hold';count=0;retry_tick=None
        stable=0;stationary=StationaryServo();failure=None;label_counts={};video(directory/'rollout.mp4')
        with (directory/'transitions.jsonl').open('x',buffering=1) as stream:
            try:
                while True:
                    q,base=measured();before=physical();proprio=proprio61(inst.obs)
                    if count==16:phase='fault_reverse';cursor=0;stationary=StationaryServo()
                    if phase=='fault_reverse' and outcomes.false_streak>=12:
                        phase='corrective';cursor=max(0,len(knots)-1-cursor);retry_tick=count;issue(count,'RETRY');stationary=StationaryServo()
                    if count%4==0:
                        folder=directory/'rgb'/f'{count:08d}';folder.mkdir();meta=dict(control_step=count,sha256={},dimensions={})
                        for camera,key in evaluator.robot_camera_names.items():
                            im=Image.fromarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]).resize((224,224),Image.Resampling.BILINEAR)
                            dest=folder/(camera+'_rgb.jpg');im.save(dest,quality=90,subsampling=0)
                            meta['sha256'][camera+'_rgb']=sha256(dest);meta['dimensions'][camera+'_rgb']=[3,224,224]
                        anchors[str(count)]=meta
                    trajectory=reverse if phase=='fault_reverse' else forward
                    waypoint=seed if phase in ('clean_hold','confirmed_hold') else trajectory[min(cursor,len(trajectory)-1)]
                    command,info=servo(q,base,waypoint)
                    if info['reached'] and phase in ('fault_reverse','corrective') and cursor<len(trajectory)-1:
                        cursor+=1;stationary=StationaryServo()
                    evaluator._write_video(inst);after=step(command)
                    moved=np.max(np.abs(np.asarray(after['object_joints'])-before['object_joints']))>1e-5
                    outcome=outcomes.update(count,after['goal_predicate'],retry=retry_tick==count,moving=bool(moved))
                    label_counts[outcome]=label_counts.get(outcome,0)+1
                    row=dict(control_step=count,proprio_before=proprio,proprio_after=proprio61(inst.obs),
                        action_executed_raw23=command.tolist(),simulator_apply_ack=True,physical_before=before,physical_audit=after,
                        label_kind='injected_fault_not_BC' if phase=='fault_reverse' else 'same_state_articulation_teacher_candidate',
                        teacher=dict(kind='local_measured_reference_path_servo_v1',phase=phase,waypoint=cursor,**info),
                        outcome_candidate=outcome,outcome_approved=False,
                        context=dict(context_id=current['event_sha256'],active_skills_semantic_json=semantic,parent_goal=selected['parent']),
                        rgb_anchor_control_step=count//4*4,terminated=False,truncated=False)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');count+=1
                    if retry_tick is not None and outcome=='SUCCEEDED':phase='confirmed_hold';stable+=1
                    else:stable=0
                    if count%16==0:atomic_json(a.output/'status.json',dict(status,status='collecting',phase=phase,branch_controls=count))
                    if stable>=64:break
                    if phase!='clean_hold' and stationary.observe(q,base,before['object_joints'],command):
                        raise ValueError('Measured stationary control fixed point without completed physical recovery')
            except (ValueError,RuntimeError) as error:failure=str(error)
            finally:evaluator._set_video_writer(inst,None)
        summary=dict(kind='reverse_reference_fault',verb=skills[0]['verb'],controls=count,loss_observed=retry_tick is not None,
            retry_issued=retry_tick is not None,retry_control=retry_tick,final_stable_controls=stable,failure=failure,
            physical_recovery_candidate=bool(stable>=64 and failure is None),training_approved=False,bc_eligible=False,
            anchors=anchors,outcome_candidate_counts=label_counts,initial=seed['physical'],final=physical(),
            transitions_sha256=sha256(directory/'transitions.jsonl'),plans_sha256=sha256(directory/'plans.json'),
            teacher_kind='same-state reference-path servo; not original expert or learned actor')
        atomic_json(directory/'manifest.json',summary)
        receipt=dict(status,status='completed_candidates_only',branches=[summary],seconds=time.monotonic()-started,
            full_snapshot_sha256=sha256(a.output/'full_snapshot.pt'),training_approved=False,
            scope='local articulation goal restoration, not whole task or approved BC')
        atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',dict(status,status=receipt['status']))


if __name__=='__main__':main()
