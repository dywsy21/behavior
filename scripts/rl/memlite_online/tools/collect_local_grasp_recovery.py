"""Collect physically executed local grasp corrections from original TRAIN.

No optimization, no deployment oracle. A real demonstrated stable grasp seeds
a restricted same-state feedback teacher. Perturbed actions are NEVER positive
BC targets. Every branch is saved, including failed or out-of-domain attempts.
The finite clean/intervention/confirmation segments define the experiment, not
a reset/time quota. New instances may be collected until data acceptance holds.
"""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO/'scripts/eval/memlite_sft100'),
               str(Path(__file__).resolve().parents[1]/'code'),str(REPO/'src')]
from common import atomic_json,OFFICIAL,OFFICIAL_COMMIT,sha256
from recovery_corpus import canonical,digest,group_key,split_group
from recovery_local_teacher import LocalGraspTeacher,perturb,NonGraspingFixedPoint,fault_duration,validate_terminal_observation
from recovery_recorder import proprio61
from recovery_reference_binding import ReferenceGraspBinding,resolve_prefix_rows
from behavior_branch_state import capture_branch_metadata,restore_branch_metadata


def main(argv=None, *, shared_session=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--diversify-fault-timing',action='store_true')
    p.add_argument('--reference-category-binding',action='store_true')
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Clean frozen source required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    official=json.loads((REPO/'scripts/eval/memlite_sft100/official_manifest.json').read_text())
    if official['commit']!=OFFICIAL_COMMIT or any(sha256(OFFICIAL/k)!=v for k,v in official['files'].items()):
        raise ValueError('Changed official simulator')
    source=json.loads((a.proposal/'manifest.json').read_text())
    if (source['schema']!='recovery_expert_grasp_proposal_v2' or source['source_episode']['split']!='train'
            or source['recovery_split'] not in ('train','dev')
            or source['source_group']!=group_key(source['task'],source['instance_id'])
            or source['recovery_split']!=split_group(source['task'],source['instance_id'])):
        raise ValueError('Invalid original TRAIN/split identity')
    for name,value in source['files'].items():
        if Path(name).name!=name or sha256(a.proposal/name)!=value:raise ValueError('Changed source bytes')
    a.output.mkdir(parents=True)
    status=dict(schema='local_recovery_collection_status_v1',status='loading',source_commit=commit,
        proposal_sha256=sha256(a.proposal/'manifest.json'),pid=os.getpid(),task=source['task'],
        instance_id=source['instance_id'],source_group=source['source_group'],split=source['recovery_split'],
        actual_controls=0,instance_loads=0,snapshot_restores=0,optimizer_steps=0,quota=None)
    atomic_json(a.output/'status.json',status)
    atomic_json(a.output/'source.json',source)
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
    # OmniGibson locks consumed macro fields after simulator construction.
    # Reusing the same configured session must not unlock/mutate them.
    if shared_session is None or not shared_session:
        gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False
    seed_everything(DEFAULT_EVAL_SEED)
    arrays=np.load(a.proposal/'prefix.npz',allow_pickle=False);actions=arrays['action']
    if actions.shape!=(source['controls'],23):raise ValueError('Wrong raw23 source')
    selected=source['selected_segment'];skills=json.loads(selected['semantic'])
    if len(skills)!=1 or skills[0]['verb']!='GRASP' or not skills[0]['target']:
        raise ValueError('Only a single fully bound grasp seed is supported')
    target_name=skills[0]['target']
    class NoPolicy:
        def reset(self):pass
    class Evaluator(BatchedEvaluator):
        def load_policy(self):return NoPolicy()
        def __exit__(self,kind,error,traceback):
            # Native shutdown can exit Python before the outer except executes.
            if error is not None:
                atomic_json(a.output/'status.json',dict(status,status='failed',error=repr(error),
                    seconds=time.monotonic()-started))
            return super().__exit__(kind,error,traceback)
    cfg=OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='offline_local_grasp_teacher',headless=True,partial_scene_load=True,max_steps=None,
        write_video=True,mode='train',seed=DEFAULT_EVAL_SEED,num_envs=1,
        task=dict(name=source['task']),robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(a.output/'resolved_config.json',OmegaConf.to_container(cfg,resolve=True))
    started=time.monotonic()
    try:
        if shared_session is not None:
            if shared_session and shared_session[0].cfg.task.name!=source['task']:
                raise ValueError('Warm session may not cross tasks')
            if not shared_session:shared_session.append(Evaluator(cfg))
            context=nullcontext(shared_session[0])
        else:
            context=Evaluator(cfg)
        with context as evaluator:
            evaluator.load_batch({0:source['instance_id']});status['instance_loads']+=1
            inst=evaluator.instance_eval_states[0];robot=inst.env_accessor.robot
            # Explicit render-only refresh also covers a reused simulator's
            # first frame after loading a different original instance.
            for _ in range(3):og.sim.render()
            fresh,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(fresh[0],inst)
            initial_error=np.abs(np.asarray(proprio61(inst.obs))-arrays['state'][0])
            atomic_json(a.output/'initial_alignment.json',dict(absolute_error_by_raw61_dimension=initial_error.tolist(),
                refreshed_without_control=True,policy_memory='fresh_per_branch',task=source['task'],instance=source['instance_id']))
            positions=list(range(3,10))+list(range(28,35))+list(range(53,57))
            if float(initial_error[positions].max())>.05:
                raise ValueError('Original initial joint configuration does not match reset instance')
            scope={k:getattr(v,'wrapped_obj',v) for k,v in inst.env_accessor.object_scope.items()}
            matches=[(k,o) for k,o in scope.items() if o is not None and getattr(o,'name',None)==target_name]
            requested_name=target_name;requested_arm=skills[0].get('arm','UNSPECIFIED').upper()
            if requested_arm not in ('LEFT','RIGHT','UNSPECIFIED'):raise ValueError('Unsupported original GRASP arm')
            allowed_arms=[requested_arm.lower()] if requested_arm!='UNSPECIFIED' else list(robot.arm_names)
            binder=None;candidate_objects={};candidate_inventory={};binding_receipt=None
            atomic_json(a.output/'target-binding-audit.json',dict(requested=requested_name,
                task_objects=[dict(entity=k,name=getattr(o,'name',None),category=getattr(o,'category',None))
                              for k,o in scope.items() if o is not None],
                proposal_sha256=status['proposal_sha256'],reference_binding_enabled=a.reference_category_binding,
                guessed_alias_accepted=False,actor_input=False))
            if len(matches)==1:
                entity,target=matches[0]
            elif not matches and a.reference_category_binding:
                candidate_objects={o.name:(k,o) for k,o in scope.items() if o is not None
                    and getattr(o,'category',None)==requested_name and hasattr(o,'links') and o is not robot}
                candidate_inventory={n:dict(entity=k,category=o.category) for n,(k,o) in candidate_objects.items()}
                binder=ReferenceGraspBinding(candidate_inventory,requested_name,requested_arm,selected['start'],selected['end'])
                entity,target=None,None  # No guessed target before the real contact evidence.
            else:raise ValueError('No exact task-scope target; do not guess aliases')
            def indices(v):return v.detach().cpu().numpy().astype(int)
            arm_action={arm:indices(robot.arm_action_idx[arm]) for arm in robot.arm_names}
            grip_action={arm:int(indices(robot.gripper_action_idx[arm])[0]) for arm in robot.arm_names}
            if (arm_action['left'].tolist()!=list(range(7,14)) or arm_action['right'].tolist()!=list(range(15,22))
                    or grip_action!={'left':14,'right':22}):raise ValueError('Unexpected R1Pro raw23 mapping')
            def native_q():
                q=robot.get_joint_positions();value=np.zeros(23,dtype=np.float32)
                value[3:7]=q[robot.trunk_control_idx].detach().cpu().numpy()
                for arm in robot.arm_names:
                    value[arm_action[arm]]=q[robot.arm_control_idx[arm]].detach().cpu().numpy()
                    value[grip_action[arm]]=float(q[robot.gripper_control_idx[arm]].mean())/.05*2-1
                value[[14,22]]=np.clip(value[[14,22]],-1,1)
                return value
            def physics():
                held={o.name:{arm:robot.is_grasping(arm=arm,candidate_obj=o).name for arm in robot.arm_names}
                      for o in scope.values() if o is not None and hasattr(o,'links') and o is not robot}
                aperture={arm:float(robot.get_joint_positions()[robot.gripper_control_idx[arm]].mean())
                          for arm in robot.arm_names}
                def one(k,obj):
                    pos,quat=obj.get_position_orientation()
                    return dict(target_name=obj.name,entity=k,position=pos.tolist(),orientation=quat.tolist(),
                                grasp=held[obj.name],gripper_aperture=aperture,all_held=held)
                if target is None:
                    return dict(binding_candidates={n:one(k,o) for n,(k,o) in candidate_objects.items()})
                return one(entity,target)
            def step(command):
                if shutil.disk_usage(a.output).free < 100*2**30:raise RuntimeError('Disk safety reserve reached')
                command=np.asarray(command,dtype=np.float32)
                if command.shape!=(23,) or not np.isfinite(command).all():raise ValueError('Invalid control')
                term,trunc,_=evaluator._apply_actions(torch.as_tensor(command[None]),[0])
                status['actual_controls']+=1
                if bool(term[0]) or bool(trunc[0]):raise RuntimeError('Official terminal: stop collection')
                return physics()
            def video(path):
                evaluator._set_video_writer(inst,create_video_writer(fpath=str(path),resolution=(448,672),rate=30))
            prefix=(a.output/'prefix.jsonl').open('x',buffering=1);video(a.output/'prefix.mp4')
            held_count={arm:0 for arm in allowed_arms};grasp_arm=None
            for t,command in enumerate(actions):
                before=proprio61(inst.obs);evaluator._write_video(inst);after=step(command)
                prefix.write(json.dumps(dict(control_step=t,action_executed_raw23=command.tolist(),
                    proprio_before=before,proprio_after=proprio61(inst.obs),physical_audit=after,
                    source_label_kind='original_reference_replay_not_corrective_BC'))+'\n')
                if binder is not None:
                    bound=binder.observe(t,after['binding_candidates'])
                    if bound is not None:
                        target_name,grasp_arm=bound;entity,target=candidate_objects[target_name];break
                else:
                    inside=selected['start']<=t<selected['end']
                    for arm in held_count:held_count[arm]=held_count[arm]+1 if inside and after['grasp'][arm]=='TRUE' else 0
                    if any(v>=6 for v in held_count.values()):
                        grasp_arm=next(arm for arm,v in held_count.items() if v>=6);break
                if t%16==0:atomic_json(a.output/'status.json',dict(status,status='replaying_prefix',source_frame=t))
            prefix.close();evaluator._set_video_writer(inst,None)
            if grasp_arm is None:
                atomic_json(a.output/'result.json',dict(status,status='reference_grasp_not_reproduced',
                    training_approved=False,reason='No target-identity stable grasp in original reference'))
                atomic_json(a.output/'status.json',dict(status,status='reference_grasp_not_reproduced'));return
            if binder is not None:
                # Keep the complete original all-competitor evidence. The
                # normalized prefix chooses an actually measured field; it
                # does not fabricate observations, commands or a new rollout.
                raw_prefix=a.output/'binding-reference.jsonl'
                (a.output/'prefix.jsonl').rename(raw_prefix)
                raw_rows=[json.loads(line) for line in raw_prefix.read_text().splitlines()]
                resolved=resolve_prefix_rows(raw_rows,candidate_inventory,requested_name,requested_arm,
                                             selected['start'],selected['end'],target_name,grasp_arm)
                with (a.output/'prefix.jsonl').open('x') as stream:
                    for row in resolved:stream.write(json.dumps(row,allow_nan=False)+'\n')
                # The new independent branch receives the explicitly resolved
                # intent. The original source annotation remains unchanged.
                skills[0]['target']=target_name
                binding_receipt=dict(schema='original_reference_grasp_binding_v1',requested_category=requested_name,
                    requested_arm=requested_arm,candidates=candidate_inventory,selected_target=target_name,
                    selected_entity=entity,selected_arm=grasp_arm,evidence_available_control_step=t+1,
                    source_proposal_sha256=status['proposal_sha256'],raw_reference_sha256=sha256(raw_prefix),
                    normalized_prefix_sha256=sha256(a.output/'prefix.jsonl'),resolved_skills=skills,
                    method='exact category + previously false + unique same-target same-arm grasp for six actual controls inside original skill interval',
                    actor_input_uses_physical_audit=False,training_approved=False)
                atomic_json(a.output/'reference-binding.json',binding_receipt)
                status['reference_binding_sha256']=sha256(a.output/'reference-binding.json')
            seed_frame=t+1;seed_q=native_q();seed_q[grip_action[grasp_arm]]=-1
            seed_physics=physics();seed_state=proprio61(inst.obs)
            snapshot=dict(world=deepcopy(og.sim.dump_state(serialized=False)),metadata=capture_branch_metadata(evaluator),
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all()),source_commit=commit,official_commit=OFFICIAL_COMMIT,
            source_manifest_sha256=sha256(a.proposal/'manifest.json'),source_frame=seed_frame,
                teacher_q=seed_q,seed_physics=seed_physics,seed_proprio=seed_state,policy_memory='fresh_per_branch')
            torch.save(snapshot,a.output/'full_snapshot.pt')
            atomic_json(a.output/'seed.json',dict(source_frame=seed_frame,arm=grasp_arm,physics=seed_physics,
                native_q=seed_q.tolist(),full_snapshot_sha256=sha256(a.output/'full_snapshot.pt')))
            # All branches stay in the original source-instance split. A new
            # branch creates a new policy ledger, NEVER inherits another task.
            branches=[]
            for ordinal,kind in enumerate(('clean','open_gripper','open_gripper_joint_jitter')):
                if ordinal:
                    og.sim.load_state(deepcopy(snapshot['world']),serialized=False)
                    restore_branch_metadata(evaluator,snapshot['metadata'])
                    random.setstate(snapshot['rng']['python']);np.random.set_state(snapshot['rng']['numpy'])
                    torch.set_rng_state(snapshot['rng']['torch']);torch.cuda.set_rng_state_all(snapshot['rng']['cuda'])
                    for _ in range(3):og.sim.render()
                    obs,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(obs[0],inst)
                    status['snapshot_restores']+=1
                directory=a.output/kind;directory.mkdir();(directory/'rgb').mkdir()
                teacher=LocalGraspTeacher(seed_q,seed_physics['position'],arm=grasp_arm,
                    arm_indices=arm_action[grasp_arm],gripper_index=grip_action[grasp_arm])
                teacher.stage='HOLD';rng=np.random.default_rng(a.seed+ordinal)
                fault_controls=fault_duration(source['source_group'],kind,a.seed,a.diversify_fault_timing)
                task=source['task'].replace('_',' ')
                semantic=canonical(skills);parent=selected['parent'];text=semantic_active_skills_text(parse_active_skills_semantic_json(semantic))
                memory=canonical_json(dict(task_name=task,issued_command_history=[],verified_world_facts=[]))
                previous='None';plans=[];current=None
                def issue(control,decision):
                    nonlocal memory,previous,current
                    next_memory=append_b_memory_idempotent(memory,previous,task_name=task)
                    event=dict(control_step=control,decision=decision,active_skills_semantic_json=semantic,
                        active_skills_text=text,parent_goal=parent,memory_before=memory,memory_update=next_memory,
                        previous_intent=previous,source='offline_verified_local_teacher_proposal',not_success_label=True)
                    event['event_sha256']=digest(event);plans.append(event)
                    memory=next_memory;previous=text;current=event
                    atomic_json(directory/'plans.json',plans)
                issue(0,'EXECUTE');stream=(directory/'transitions.jsonl').open('x',buffering=1)
                video(directory/'rollout.mp4');anchors={};count=0;confirmed=0;lost=0;loss_observed=False
                failure=None;retry_issued=False;last_vector=None;stagnant=0
                fixed_point=NonGraspingFixedPoint()
                initial=physics();start_error=float(np.max(np.abs(np.asarray(proprio61(inst.obs))-seed_state)))
                try:
                    while True:
                        measured=physics();q=native_q();before=proprio61(inst.obs)
                        # The intervention is a one-second actuation fault. The
                        # correction is issued only after its actual loss, not
                        # because a timed interval ended.
                        if kind!='clean' and count==fault_controls:
                            if not loss_observed:raise ValueError('Intervention did not produce verified grasp loss')
                            issue(count,'RETRY');retry_issued=True;teacher.stage='APPROACH_OPEN'
                        # 4-control images resolve opening/closing transitions;
                        # long outcome history still uses its own causal clock.
                        if count%4==0:
                            meta=dict(control_step=count,sha256={},dimensions={})
                            folder=directory/'rgb'/f'{count:08d}';folder.mkdir()
                            for camera,key in evaluator.robot_camera_names.items():
                                rgb=inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]
                                im=Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR)
                                path=folder/(camera+'_rgb.jpg');im.save(path,quality=90,subsampling=0)
                                meta['sha256'][camera+'_rgb']=sha256(path);meta['dimensions'][camera+'_rgb']=[3,224,224]
                            anchors[str(count)]=meta
                        intended,teacher_info=teacher.action(q,measured['position'],
                            measured['gripper_aperture'][grasp_arm],measured['grasp'][grasp_arm])
                        injected=kind!='clean' and count<fault_controls
                        noise,command=(perturb(intended,arm_indices=arm_action[grasp_arm],gripper_index=grip_action[grasp_arm],
                            rng=rng,kind=kind) if injected else (np.zeros(23,dtype=np.float32),intended.copy()))
                        evaluator._write_video(inst);after=step(command)
                        confirmed=confirmed+1 if after['grasp'][grasp_arm]=='TRUE' else 0
                        lost=lost+1 if all(v=='FALSE' for v in after['grasp'].values()) else 0
                        loss_observed=loss_observed or lost>=6
                        row=dict(control_step=count,proprio_before=before,proprio_after=proprio61(inst.obs),
                            a_intended=intended.tolist(),a_sampled_noise=noise.tolist(),a_executed_raw23=command.tolist(),
                            action_executed_raw23=command.tolist(),simulator_apply_ack=True,
                            label_kind='injected_fault_not_BC' if injected else 'same_state_local_teacher_candidate',
                            teacher=teacher_info,physical_before=measured,physical_audit=after,
                            context=dict(context_id=current['event_sha256'],active_skills_semantic_json=semantic,parent_goal=parent),
                            rgb_anchor_control_step=count//4*4,chunk_start_control_step=count//4*4,
                            experience_id=count//16,policy_update=0,terminated=False,truncated=False)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');count+=1
                        if count%16==0:atomic_json(a.output/'status.json',dict(status,status='collecting',branch=kind,branch_controls=count))
                        # Physical confirmation duration (not an experiment
                        # resource cap) gives genuine post-recovery supervision.
                        if (kind=='clean' and confirmed>=64) or (retry_issued and confirmed>=128):break
                        if (retry_issued and teacher.stage=='CLOSE' and lost>=6
                                and after['gripper_aperture'][grasp_arm]<.003
                                and teacher_info['joint_error_max_rad']<.012):
                            raise ValueError('Closed empty grasp at settled reference pose; correction failed')
                        if not injected and fixed_point.observe(command,q,measured['position'],
                                measured['gripper_aperture'][grasp_arm],measured['grasp']):
                            raise ValueError('Stationary commanded/measured non-grasping contact; local teacher cannot recover')
                        vector=np.r_[q,measured['position'],measured['gripper_aperture'][grasp_arm]]
                        stagnant=stagnant+1 if last_vector is not None and np.max(np.abs(vector-last_vector))<1e-5 else 0
                        last_vector=vector
                        if not injected and confirmed==0 and stagnant>=30:
                            raise ValueError('Corrective servo reached a non-grasping fixed point')
                except (ValueError,RuntimeError) as error:
                    failure=str(error)
                finally:
                    stream.close();evaluator._set_video_writer(inst,None)
                # A failed correction may end between the regular image
                # anchors. Preserve the actual final observation separately:
                # it has NO fabricated next action and is never positive BC.
                terminal_dir=directory/'terminal_rgb';terminal_dir.mkdir()
                terminal=dict(schema='recovery_terminal_observation_v1',control_step=count,
                    last_applied_control_step=count-1 if count else None,proprio=proprio61(inst.obs),
                    physical_audit=physics(),context=deepcopy(current),reason=failure,
                    images={},has_next_executed_action=False,training_approved=False,
                    actor_input_excludes=['physical_audit','reason'],
                    outcome_candidate=('FAILED' if failure=='Closed empty grasp at settled reference pose; correction failed' else None))
                for camera,key in evaluator.robot_camera_names.items():
                    im=Image.fromarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]).resize((224,224),Image.Resampling.BILINEAR)
                    path=terminal_dir/(camera+'_rgb.jpg');im.save(path,quality=90,subsampling=0)
                    terminal['images'][camera+'_rgb']=dict(path=str(path.relative_to(directory)),sha256=sha256(path))
                atomic_json(directory/'terminal-observation.json',terminal)
                validate_terminal_observation(terminal,[json.loads(line) for line in (directory/'transitions.jsonl').read_text().splitlines()],grasp_arm)
                summary=dict(kind=kind,controls=count,seed_frame=seed_frame,arm=grasp_arm,
                    fault_controls=fault_controls,diversified_fault_timing=a.diversify_fault_timing,
                    terminal_observation_sha256=sha256(directory/'terminal-observation.json'),
                    snapshot_start_max_proprio_error=start_error,initial=initial,final=physics(),
                    loss_observed=loss_observed,retry_issued=retry_issued,final_stable_controls=confirmed,
                    physical_recovery_candidate=bool(retry_issued and loss_observed and confirmed>=128 and failure is None),
                    failure=failure,training_approved=False,bc_eligible=False,anchors=anchors,
                    transitions_sha256=sha256(directory/'transitions.jsonl'),plans_sha256=sha256(directory/'plans.json'),
                    teacher_kind='DART-inspired local feedback; not original demonstrator or learned actor')
                atomic_json(directory/'manifest.json',summary);branches.append(summary)
            atomic_json(a.output/'result.json',dict(status,status='completed_candidates_only',branches=branches,
                full_snapshot_sha256=sha256(a.output/'full_snapshot.pt'),seconds=time.monotonic()-started,
                training_approved=False,scope='local GRASP correction, not full-task success or broad recovery coverage'))
            atomic_json(a.output/'status.json',dict(status,status='completed_candidates_only',seconds=time.monotonic()-started))
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code in (None,0) and (a.output/'result.json').exists():
            raise
        atomic_json(a.output/'status.json',dict(status,status='failed',error=repr(error),seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
