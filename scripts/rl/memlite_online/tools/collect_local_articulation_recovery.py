"""Cold TRAIN-only door/drawer recovery candidates, with actual Open evidence.

Replay a reference to an observed false->true stable goal transition. Reverse a
short measured robot path through real actuators until that goal is lost, then
issue RETRY and track forward using current-state joint/base feedback. Neither
reference frame boundaries nor teacher intent imply a label. Every output is
candidate-only; physical evidence must be independently inspected for admission.
"""
import argparse
from copy import deepcopy
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
from recovery_articulation_teacher import CausalArticulation,StationaryServo,servo,yaw_of,functional_goal,validate_corridor,progressing
from recovery_reference_binding import ReferenceArticulationBinding
from behavior_branch_state import capture_branch_metadata,restore_branch_metadata
from behavior_light_state import capture_lights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-category-binding',action='store_true')
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
        atomic_json(a.output/'target-binding-audit.json',dict(requested=skills[0]['target'],
            source_proposal_sha256=status['proposal_sha256'],task=source['task'],instance_id=source['instance_id'],
            task_objects=[dict(entity=k,name=getattr(o,'name',None),category=getattr(o,'category',None))
                          for k,o in scope.items() if o is not None],
            scene_objects=[dict(name=o.name,category=getattr(o,'category',None)) for o in robot.scene.objects],
            guessed_alias_accepted=False,actor_input=False))
        matches=[(k,o) for k,o in scope.items() if o is not None and getattr(o,'name',None)==skills[0]['target']]
        if not matches:
            # A task's intermediate door need not be a BDDL goal object. Only
            # an exact, unique native scene name is accepted; no category guess.
            scene_matches=[o for o in robot.scene.objects if o.name==skills[0]['target']]
            if len(scene_matches)==1:matches=[('scene_object:'+scene_matches[0].name,scene_matches[0])]
        requested_name=skills[0]['target'];want_open=skills[0]['verb'].startswith('OPEN_')
        binder=None;candidate_objects={};candidate_inventory={}
        if len(matches)==1:
            entity,target=matches[0]
        elif not matches and a.reference_category_binding:
            entities={o.name:k for k,o in scope.items() if o is not None}
            candidate_objects={o.name:(entities.get(o.name,'scene_object:'+o.name),o)
                               for o in robot.scene.objects if getattr(o,'category',None)==requested_name}
            candidate_inventory={n:dict(entity=k,category=o.category) for n,(k,o) in candidate_objects.items()}
            binder=ReferenceArticulationBinding(candidate_inventory,requested_name,selected['start'],selected['end'])
            entity,target=None,None
        else:raise ValueError('No exact native task-scope target; do not guess aliases')
        from omnigibson.object_states.open_state import _get_relevant_joints,_compute_joint_threshold
        def contract(obj):
            states=[v for k,v in obj.states.items() if k.__name__=='Open']
            if len(states)!=1:raise ValueError('Missing unambiguous installed Open state')
            two_sided,relevant,directions=_get_relevant_joints(obj)
            if two_sided or len(relevant)!=1:
                raise ValueError('Whole-object clearance ambiguous; requires an exact part-level adapter')
            hinge=relevant[0];_,opened,closed=_compute_joint_threshold(hinge,directions[0])
            opened,closed=float(opened),float(closed)
            if not np.isfinite([opened,closed]).all() or abs(opened-closed)<1e-5:
                raise ValueError('Missing finite directed joint travel')
            return states[0],hinge,opened,closed
        contracts={n:contract(o) for n,(_k,o) in candidate_objects.items()} if binder else {target.name:contract(target)}
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
            def one(k,obj):
                state,hinge,open_end,closed_end=contracts[obj.name]
                raw=state.get_value();value=raw.item() if hasattr(raw,'item') else raw
                if type(value) is not bool:raise ValueError('Installed Open is not a Boolean measurement')
                pos,quat=obj.get_position_orientation()
                position=float(hinge.get_state()[0].item());fraction=(position-closed_end)/(open_end-closed_end)
                return dict(target_name=obj.name,entity=k,open=value,goal_predicate=functional_goal(value,fraction,want_open),
                    directed_open_fraction=fraction,open_end=open_end,closed_end=closed_end,
                    object_joints=obj.get_joint_positions().detach().cpu().tolist(),position=pos.tolist(),orientation=quat.tolist(),
                    grasp={arm:robot.is_grasping(arm=arm,candidate_obj=obj).name for arm in robot.arm_names},
                    predicate_contract='single exact hinge: open>=35% / lost<=10%; close<=2.5% + official not Open; middle UNKNOWN',label_only=True)
            if target is None:return dict(binding_candidates={n:one(k,o) for n,(k,o) in candidate_objects.items()})
            return one(entity,target)
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
                if binder:
                    bound=binder.observe(t,before['binding_candidates'],after['binding_candidates'])
                    outcome='UNKNOWN'  # no single target/label before physical resolution
                else:
                    bound=None;outcome=history.update(t,after['goal_predicate'] if t>=selected['start'] else None,moving=False)
                stream.write(json.dumps(dict(control_step=t,action_executed_raw23=command.tolist(),proprio_before=proprio,
                    proprio_after=proprio61(inst.obs),physical_before=before,physical_audit=after,
                    outcome_after_control_candidate=outcome,outcome_evidence_available_control_step=t+1,
                    source_label_kind='original_reference_not_corrective_BC'))+'\n')
                if bound is not None:
                    entity,target=candidate_objects[bound];history=binder.states[bound]
                    seeded=True;break
                if outcome=='SUCCEEDED':seeded=True;break
                if t%16==0:atomic_json(a.output/'status.json',dict(status,status='replaying_prefix',source_frame=t))
        evaluator._set_video_writer(inst,None)
        if not seeded:
            receipt=dict(status,status='reference_articulation_not_reproduced',training_approved=False)
            atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt);return
        if binder:
            raw_prefix=a.output/'binding-reference.jsonl';(a.output/'prefix.jsonl').rename(raw_prefix)
            raw_rows=[json.loads(line) for line in raw_prefix.read_text().splitlines()]
            replay=ReferenceArticulationBinding(candidate_inventory,requested_name,selected['start'],selected['end'])
            normalized=[]
            for i,row in enumerate(raw_rows):
                found=replay.observe(i,row['physical_before']['binding_candidates'],row['physical_audit']['binding_candidates'])
                if found is not None and (i!=len(raw_rows)-1 or found!=target.name):raise ValueError('Changed first reference articulation binding')
                row=deepcopy(row);row['physical_before']=row['physical_before']['binding_candidates'][target.name]
                row['physical_audit']=row['physical_audit']['binding_candidates'][target.name]
                row['outcome_after_control_candidate']='SUCCEEDED' if found else 'UNKNOWN'
                normalized.append(row)
            if found!=target.name:raise ValueError('Unreproduced reference articulation binding')
            with (a.output/'prefix.jsonl').open('x') as stream:
                for row in normalized:stream.write(json.dumps(row,allow_nan=False)+'\n')
            for knot in path:knot['physical']=knot['physical']['binding_candidates'][target.name]
            skills[0]['target']=target.name
            binding=dict(schema='original_reference_articulation_binding_v1',requested_category=requested_name,
                candidates=candidate_inventory,selected_target=target.name,selected_entity=entity,
                evidence_available_control_step=t+1,source_proposal_sha256=status['proposal_sha256'],
                raw_reference_sha256=sha256(raw_prefix),normalized_prefix_sha256=sha256(a.output/'prefix.jsonl'),
                resolved_skills=skills,training_approved=False,
                method='unique measured exact-category articulation false-to-true goal, all other candidates move <3.5% joint travel')
            atomic_json(a.output/'reference-binding.json',binding);status['reference_binding_sha256']=sha256(a.output/'reference-binding.json')
        seed_q,seed_base=measured();seed_q[[14,22]]=actions[t,[14,22]]
        seed=dict(q=seed_q.tolist(),base=seed_base.tolist(),source_frame=t+1,physical=physical())
        # Keep only a physically local measured suffix. Missing/too distant
        # reverse targets are a coverage gap, not permission to teleport.
        false_indices=[i for i,x in enumerate(path) if x['physical']['goal_predicate'] is False]
        if not false_indices:raise ValueError('No local physical false-goal path')
        recent=path[max(0,false_indices[-1]-16):];knots=recent[::4]
        validate_corridor(knots+[seed])
        snapshot=dict(world=deepcopy(og.sim.dump_state(serialized=False)),metadata=capture_branch_metadata(evaluator),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
            seed=seed,waypoints=knots,source_manifest_sha256=status['proposal_sha256'],source_commit=commit,official_commit=OFFICIAL_COMMIT)
        torch.save(snapshot,a.output/'full_snapshot.pt');atomic_json(a.output/'seed.json',seed)
        if snapshot['metadata'].get('light_synchronizer') is not None:
            # An explicit render-only engineering probe, outside all training
            # branches. Verify both cached edge history and actual USD lights;
            # no physics control, policy observation or outcome label is made.
            folder=a.output/'light-restore-probe';folder.mkdir()
            def lighting_frame(label):
                for _ in range(3):og.sim.render()
                observed,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(observed[0],inst)
                files={}
                for camera,key in evaluator.robot_camera_names.items():
                    image=Image.fromarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]).resize((224,224),Image.Resampling.BILINEAR)
                    path=folder/(label+'-'+camera+'.png');image.save(path);files[path.name]=sha256(path)
                return files
            original_light=capture_lights(inst);original_state=proprio61(inst.obs)
            media=dict(before=lighting_frame('before'))
            synchronizer=inst.light_synchronizer
            for name in original_light['target_visibility']:
                obj=robot.scene.object_registry('name',name,None)
                if obj is None:raise ValueError('Lost exact light target during round-trip probe')
                synchronizer._toggle_light(obj)
            changed_light=capture_lights(inst);media['perturbed']=lighting_frame('perturbed')
            if changed_light['target_visibility']==original_light['target_visibility']:
                raise ValueError('Light probe made no actual registered change')
            restore_branch_metadata(evaluator,snapshot['metadata'])
            restored_light=capture_lights(inst);media['restored']=lighting_frame('restored')
            if restored_light!=original_light or proprio61(inst.obs)!=original_state:
                raise ValueError('Light/robot render-only round-trip did not restore')
            atomic_json(folder/'receipt.json',dict(schema='light_visibility_metadata_roundtrip_v1',status='passed',
                actual_physics_controls=0,world_restore_tested=False,cold_restore_tested=False,training_sample=False,
                original=original_light,perturbed=changed_light,restored=restored_light,media=media,
                source_snapshot_sha256=sha256(a.output/'full_snapshot.pt')))
            status['light_visibility_roundtrip_sha256']=sha256(folder/'receipt.json')
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
        stable=0;stationary=StationaryServo();failure=None;label_counts={};prior_outcome='SUCCEEDED';last_imaged_outcome=None
        video(directory/'rollout.mp4')
        with (directory/'transitions.jsonl').open('x',buffering=1) as stream:
            try:
                while True:
                    q,base=measured();before=physical();proprio=proprio61(inst.obs)
                    if count==16:phase='fault_reverse';cursor=0;stationary=StationaryServo()
                    if phase=='fault_reverse' and outcomes.false_streak>=12:
                        phase='corrective';cursor=max(0,len(knots)-1-cursor);retry_tick=count;issue(count,'RETRY');stationary=StationaryServo()
                    if count%4==0 or retry_tick==count or prior_outcome!=last_imaged_outcome:
                        # A failure may be confirmed at an off-grid control,
                        # immediately before RETRY. Preserve THAT observation,
                        # not a nearby frame with unavailable future evidence.
                        folder=directory/'rgb'/f'{count:08d}';folder.mkdir();meta=dict(control_step=count,sha256={},dimensions={},
                            observation_context_id=current['event_sha256'],
                            predecision_context_id=plans[-2]['event_sha256'] if retry_tick==count else None,
                            causal_predecision_outcome_candidate=prior_outcome if retry_tick==count else None,
                            evidence_available_control_step=count)
                        for camera,key in evaluator.robot_camera_names.items():
                            im=Image.fromarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]).resize((224,224),Image.Resampling.BILINEAR)
                            dest=folder/(camera+'_rgb.jpg');im.save(dest,quality=90,subsampling=0)
                            meta['sha256'][camera+'_rgb']=sha256(dest);meta['dimensions'][camera+'_rgb']=[3,224,224]
                        anchors[str(count)]=meta
                        last_imaged_outcome=prior_outcome
                    trajectory=reverse if phase=='fault_reverse' else forward
                    waypoint=seed if phase in ('clean_hold','confirmed_hold') else trajectory[min(cursor,len(trajectory)-1)]
                    command,info=servo(q,base,waypoint)
                    if info['reached'] and phase in ('fault_reverse','corrective') and cursor<len(trajectory)-1:
                        cursor+=1;stationary=StationaryServo()
                    evaluator._write_video(inst);after=step(command)
                    moved=phase=='corrective' and progressing(before['directed_open_fraction'],after['directed_open_fraction'],want_open)
                    outcome=outcomes.update(count,after['goal_predicate'],retry=retry_tick==count,moving=bool(moved))
                    label_counts[outcome]=label_counts.get(outcome,0)+1
                    row=dict(control_step=count,proprio_before=proprio,proprio_after=proprio61(inst.obs),
                        action_executed_raw23=command.tolist(),simulator_apply_ack=True,physical_before=before,physical_audit=after,
                        label_kind='injected_fault_not_BC' if phase=='fault_reverse' else 'same_state_articulation_teacher_candidate',
                        teacher=dict(kind='local_measured_reference_path_servo_v1',phase=phase,waypoint=cursor,**info),
                        # RGB/proprio_before are at count, not count+1. Never
                        # attach the newly observed post-action label to them.
                        observation_outcome_candidate='UNKNOWN' if retry_tick==count else prior_outcome,
                        outcome_after_control_candidate=outcome,outcome_evidence_available_control_step=count+1,
                        outcome_approved=False,
                        context=dict(context_id=current['event_sha256'],active_skills_semantic_json=semantic,parent_goal=selected['parent']),
                        rgb_anchor_control_step=max(map(int,anchors)),terminated=False,truncated=False)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');count+=1;prior_outcome=outcome
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
