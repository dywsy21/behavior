"""Verify an unsolved held-object PLACE start by actual original TRAIN replay.

No reverse/teleport perturbation after release, invented recovery teacher, actor
or optimizer. An independent cold invocation must validate a saved start before
it becomes eligible for repeated short-skill learning.
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
from recovery_corpus import digest,group_key,split_group
from recovery_recorder import proprio61
from behavior_branch_state import capture_branch_metadata,restore_branch_metadata
from skill_aligned_reward import SkillIdentity,SkillReward,validate_placement_start
from skill_sim_measurements import OmniSkillMeasurements,vector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--cold-restore-from',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    official=json.loads((REPO/'scripts/eval/memlite_sft100/official_manifest.json').read_text())
    if official['commit']!=OFFICIAL_COMMIT or any(sha256(OFFICIAL/k)!=v for k,v in official['files'].items()):
        raise ValueError('Unpinned official simulator')
    source=json.loads((a.proposal/'manifest.json').read_text());segment=source['selected_segment']
    if sha256(a.proposal/'manifest.json')!=a.manifest_sha256:raise ValueError('Unpinned original source manifest')
    bundle=json.loads(segment['semantic'])
    if (source['schema']!='recovery_expert_skill_proposal_v1' or source['source_episode']['split']!='train'
            or source['source_group']!=group_key(source['task'],source['instance_id'])
            or source['recovery_split']!=split_group(source['task'],source['instance_id'])
            or len(bundle)!=1 or bundle[0]['verb'] not in ('PLACE_IN','PLACE_ON')
            or not bundle[0]['destination'] or bundle[0]['destination']==bundle[0]['target']
            or bundle[0]['unbound_relation'] or bundle[0]['target_part']):
        raise ValueError('Need exact original TRAIN single-target/destination PLACE')
    for name,sha in source['files'].items():
        if Path(name).name!=name or sha256(a.proposal/name)!=sha:raise ValueError('Changed original reference')
    a.output.mkdir(parents=True);started=time.monotonic()
    receipt=dict(status='loading',pid=os.getpid(),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),proposal_sha256=sha256(a.proposal/'manifest.json'),
        task=source['task'],instance_id=source['instance_id'],source_group=source['source_group'],
        split=source['recovery_split'],optimizer_steps=0,actor_used=False,actual_controls=0,
        training_approved=False,quota=None,cold_restore=a.cold_restore_from is not None)
    atomic_json(a.output/'status.json',receipt);atomic_json(a.output/'source.json',source)
    import numpy as np
    import torch
    from PIL import Image
    import omnigibson as og
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED)
    arrays=np.load(a.proposal/'prefix.npz',allow_pickle=False)
    actions=arrays['action'];begin=segment['start'];end=segment['end']
    if actions.shape!=(source['controls'],23) or not 0<begin<end<=len(actions):raise ValueError('Malformed finite source interval')
    class NoPolicy:
        def reset(self):pass
    class Evaluator(BatchedEvaluator):
        def load_policy(self):return NoPolicy()
        def __exit__(self,kind,error,tb):
            if error is not None:atomic_json(a.output/'status.json',dict(receipt,status='failed',error=repr(error),seconds=time.monotonic()-started))
            return super().__exit__(kind,error,tb)
    cfg=OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='reference_place_curriculum',headless=True,partial_scene_load=True,max_steps=None,
        write_video=False,mode='train',seed=DEFAULT_EVAL_SEED,num_envs=1,
        task=dict(name=source['task']),robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(a.output/'resolved_config.json',OmegaConf.to_container(cfg,resolve=True))
    with Evaluator(cfg) as evaluator:
        evaluator.load_batch({0:source['instance_id']});inst=evaluator.instance_eval_states[0]
        def refresh():
            for _ in range(3):og.sim.render()
            obs,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(obs[0],inst)
        def capture(label):
            refs={}
            for camera,key in evaluator.robot_camera_names.items():
                path=a.output/(label+'-'+camera+'.png')
                Image.fromarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]).resize((224,224)).save(path)
                refs[path.name]=sha256(path)
            return refs
        refresh();start_t=0
        if a.cold_restore_from is not None:
            original=json.loads((a.cold_restore_from/'result.json').read_text())
            if (original['proposal_sha256']!=receipt['proposal_sha256'] or original['status']!='reference_placement_completed'
                    or original['cold_restore'] or sha256(a.cold_restore_from/'start.pt')!=original['start_sha256']):
                raise ValueError('Not a pinned completed original reference start')
            saved=torch.load(a.cold_restore_from/'start.pt',weights_only=False,map_location='cpu')
            if saved['source_group']!=source['source_group'] or saved['control_step']!=begin:raise ValueError('Wrong cold source')
            og.sim.load_state(deepcopy(saved['world']),serialized=False);restore_branch_metadata(evaluator,saved['metadata'])
            random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
            torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
            refresh();error=float(np.max(np.abs(np.asarray(proprio61(inst.obs))-saved['proprio'])))
            if error>1e-3:raise ValueError('Cold placement joint-state mismatch')
            receipt.update(restored_max_proprio_error=error,start_sha256=original['start_sha256']);start_t=begin
        else:
            indices=list(range(3,10))+list(range(28,35))+list(range(53,57))
            error=float(np.max(np.abs(np.asarray(proprio61(inst.obs))-arrays['state'][0])[indices]))
            if error>.05:raise ValueError('Wrong original initial state')
            receipt['initial_joint_max_error']=error
        identity=SkillIdentity(str(a.output),source['task'],source['instance_id'],digest([receipt['proposal_sha256'],str(a.output)]),
            digest([source['source_group'],begin,segment['semantic']]),digest(bundle),0)
        sensor=OmniSkillMeasurements(inst.env_accessor,identity,bundle[0],bundle=bundle)
        if a.cold_restore_from is not None:
            for key,obj in (('target',sensor.target),('destination',sensor.destination)):
                if float(np.max(np.abs(vector(obj.get_position_orientation()[0])-saved['start_positions'][key])))>.001:
                    raise ValueError('Cold placement object pose mismatch')
            # The pinned official Simulator.load_state explicitly warns that
            # OnTop/Inside are stale until an actual simulator step; their
            # TensorizedRelativeState tables are NOT serialized. Rendering or
            # clearing per-object Python caches cannot repair PhysX contacts.
            # Apply exactly the first original reference control, account for
            # it, and only then query relations / initialize a skill reward.
            # This is a curriculum restore barrier, never an actor action or
            # a scored/BC-labelled transition. The RL start is now begin+1.
            before=proprio61(inst.obs);command=actions[begin]
            term,trunc,_=evaluator._apply_actions(torch.as_tensor(command)[None],[0])
            receipt['actual_controls']+=1
            after,physical=sensor.read(identity)
            barrier=dict(control_step=begin,action_executed_raw23=command.tolist(),
                simulator_apply_ack=True,proprio_before=before,proprio_after=proprio61(inst.obs),
                terminated=bool(term[0]),truncated=bool(trunc[0]),physical_after=physical,
                action_source='original_reference_restore_barrier_not_actor_not_BC',
                reward_assigned=False,actor_input=False)
            atomic_json(a.output/'restore-barrier.json',barrier)
            if bool(term[0]) or bool(trunc[0]):raise ValueError('Native terminal during recorded restore barrier')
            validate_placement_start(after,physical,[],cold=True)
            receipt.update(restore_barrier_controls=1,restore_barrier_sha256=sha256(a.output/'restore-barrier.json'),
                snapshot_control_step=begin,first_valid_relation_control_step=begin+1)
            start_t=begin=begin+1
        reward=None;last=None;rgb={};held_history=[]
        receipt['status']='replaying_reference';atomic_json(a.output/'status.json',receipt)
        with (a.output/'controls.jsonl').open('x',buffering=1) as log:
            for t in range(start_t,end):
                measurement,evidence=sensor.read(identity)
                if t==begin:
                    validate_placement_start(measurement,evidence,held_history,cold=a.cold_restore_from is not None)
                    reward=SkillReward(identity,measurement,control_step=t,protected_facts=())
                    rgb['start']=capture('start');receipt.update(start_measurement=measurement,start_physical_evidence=evidence,
                        start_control=begin,end_control=end,protected_progress_contract_tested=False)
                    if a.cold_restore_from is None:
                        saved=dict(world=deepcopy(og.sim.dump_state(serialized=False)),metadata=capture_branch_metadata(evaluator),
                            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
                            source_group=source['source_group'],control_step=t,proprio=proprio61(inst.obs),
                            start_positions={k:vector(obj.get_position_orientation()[0]).tolist()
                                             for k,obj in (('target',sensor.target),('destination',sensor.destination))},
                            actor_context=dict(task=source['task'].replace('_',' '),parent_goal=segment['parent'],semantic_bundle=segment['semantic']))
                        torch.save(saved,a.output/'start.pt');receipt['start_sha256']=sha256(a.output/'start.pt')
                    atomic_json(a.output/'status.json',receipt)
                if reward is not None and (t-begin)%16==0:rgb[str(t)]=capture(f'control-{t:06d}')
                before=proprio61(inst.obs);command=actions[t]
                term,trunc,_=evaluator._apply_actions(torch.as_tensor(command)[None],[0]);receipt['actual_controls']+=1
                after,physical=sensor.read(identity);held_history.append(not physical['released_from_all_hands'])
                item=dict(control_step=t,proprio_before=before,proprio_after=proprio61(inst.obs),
                    action_executed_raw23=command.tolist(),simulator_apply_ack=True,
                    physical_before=evidence,physical_after=physical,terminated=bool(term[0]),truncated=bool(trunc[0]),
                    action_source='original_demonstration_reference_not_recovery_teacher')
                if reward is not None:
                    last=reward.advance(identity,t+1,after,protected_values={},official_terminal=bool(term[0]),
                                        time_limit=t+1==end or bool(trunc[0]))
                    item['skill_reward']=dict(last,identity=asdict(identity))
                log.write(json.dumps(item,allow_nan=False)+'\n')
                if t%32==0:atomic_json(a.output/'status.json',dict(receipt,source_frame=t+1))
                if bool(term[0]) or bool(trunc[0]) or (last is not None and (last['terminated'] or last['truncated'])):break
        completed=bool(last is not None and last['skill_success'])
        rgb['end']=capture('end')
        receipt.update(status='reference_placement_completed' if completed else 'reference_placement_not_reproduced',
            physical_local_skill_completed=completed,whole_task_success_evaluated=False,
            local_end_control=None if last is None else last['control_step'],seconds=time.monotonic()-started,
            original_rgb=rgb,controls_sha256=sha256(a.output/'controls.jsonl'))
        atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt)


if __name__=='__main__':main()
