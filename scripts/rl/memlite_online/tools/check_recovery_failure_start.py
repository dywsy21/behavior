"""Verify a recorded corrective branch as a COLD-process failure curriculum.

Restore the full demonstrated world/controller/task/RNG snapshot, replay the
actually executed fault, and save the real failure state with OBSERVABLE prior
commands only. Replay the known continuation to check identity/physics; this
is not learned-actor evaluation, training, or a full-task success measurement.
"""
import argparse
from copy import deepcopy
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
from recovery_recorder import proprio61
from behavior_branch_state import capture_branch_metadata,restore_branch_metadata
from recovery_teacher_corpus import validate_branch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-directory',type=Path,required=True);p.add_argument('--branch',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    official=json.loads((REPO/'scripts/eval/memlite_sft100/official_manifest.json').read_text())
    if official['commit']!=OFFICIAL_COMMIT or any(sha256(OFFICIAL/k)!=v for k,v in official['files'].items()):
        raise ValueError('Unpinned simulator')
    source=json.loads((a.case_directory/'source.json').read_text());result=json.loads((a.case_directory/'result.json').read_text())
    directory=a.case_directory/a.branch
    m=json.loads((directory/'manifest.json').read_text())
    rows=[json.loads(x) for x in (directory/'transitions.jsonl').read_text().splitlines()]
    plans=json.loads((directory/'plans.json').read_text());validate_branch(rows,m,plans)
    if (sha256(a.case_directory/'full_snapshot.pt')!=result['full_snapshot_sha256']
            or sha256(directory/'transitions.jsonl')!=m['transitions_sha256']
            or not m['physical_recovery_candidate'] or m['failure'] is not None
            or len(plans)!=2 or plans[1]['control_step']!=32 or plans[1]['decision']!='RETRY'):
        raise ValueError('Not an exact physically recovered recorded branch')
    a.output.mkdir(parents=True);started=time.monotonic()
    receipt=dict(schema='recovery_failure_start_v1',status='loading',source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),official_commit=OFFICIAL_COMMIT,
        task=source['task'],instance_id=source['instance_id'],source_group=source['source_group'],split=result['split'],
        source_branch_sha256=sha256(directory/'manifest.json'),source_snapshot_sha256=result['full_snapshot_sha256'],
        actual_controls=0,snapshot_restores=0,optimizer_steps=0,pid=os.getpid(),quota=None)
    atomic_json(a.output/'status.json',receipt)
    import numpy as np
    import torch
    from PIL import Image
    import omnigibson as og
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.macros import gm
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED
    from omegaconf import OmegaConf
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED)
    class NoPolicy:
        def reset(self):pass
    class Evaluator(BatchedEvaluator):
        def load_policy(self):return NoPolicy()
    cfg=OmegaConf.create(json.loads((a.case_directory/'resolved_config.json').read_text()))
    try:
        with Evaluator(cfg) as evaluator:
            evaluator.load_batch({0:source['instance_id']});inst=evaluator.instance_eval_states[0]
            robot=inst.env_accessor.robot
            saved=torch.load(a.case_directory/'full_snapshot.pt',weights_only=False,map_location='cpu')
            target=getattr(inst.env_accessor.object_scope[m['initial']['entity']],'wrapped_obj',
                           inst.env_accessor.object_scope[m['initial']['entity']])
            if target.name!=m['initial']['target_name']:raise ValueError('Changed exact target identity')
            def refresh():
                for _ in range(3):og.sim.render()
                obs,_=evaluator.env.get_obs();inst.obs=evaluator._preprocess_obs(obs[0],inst)
            def restore(snapshot):
                og.sim.load_state(deepcopy(snapshot['world']),serialized=False)
                restore_branch_metadata(evaluator,snapshot['metadata'])
                random.setstate(snapshot['rng']['python']);np.random.set_state(snapshot['rng']['numpy'])
                torch.set_rng_state(snapshot['rng']['torch']);torch.cuda.set_rng_state_all(snapshot['rng']['cuda'])
                refresh();receipt['snapshot_restores']+=1
            def physical():
                pos,_=target.get_position_orientation()
                return dict(position=pos.tolist(),grasp={arm:robot.is_grasping(arm=arm,candidate_obj=target).name for arm in robot.arm_names})
            restore(saved)
            initial_error=float(np.max(np.abs(np.array(proprio61(inst.obs))-saved['seed_proprio'])))
            if initial_error>1e-3 or physical()['grasp'][m['arm']]!='TRUE':raise ValueError('Cold snapshot initial state mismatch')
            held=0;lost=0;stream=(a.output/'replay.jsonl').open('x',buffering=1)
            failure_snapshot=None
            for t,row in enumerate(rows):
                if t==32:
                    state=physical();expected=row['physical_before']
                    position_error=float(np.max(np.abs(np.array(state['position'])-expected['position'])))
                    if lost<6 or position_error>.005:raise ValueError('Replayed fault is not the same verified failure neighborhood')
                    # Prior issued context, not RETRY target or oracle feedback.
                    context=dict(task_name=source['task'].replace('_',' '),parent_goal=plans[0]['parent_goal'],
                        memory=plans[0]['memory_update'],previous_intent=plans[0]['active_skills_text'],
                        issued_skills_semantic_json=plans[0]['active_skills_semantic_json'],
                        served_controls=32,intent_started_control_step=0,attempt_number=1,known_previous_outcome='UNKNOWN')
                    failure_snapshot=dict(world=deepcopy(og.sim.dump_state(serialized=False)),metadata=capture_branch_metadata(evaluator),
                        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
                        actor_observable_context=context,source_group=source['source_group'],source_branch_sha256=receipt['source_branch_sha256'],
                        observation_control_step=32,proprio=proprio61(inst.obs),label_only_physical=state)
                    torch.save(failure_snapshot,a.output/'failure-start.pt')
                    for camera,key in evaluator.robot_camera_names.items():
                        Image.fromarray(inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]).save(a.output/(camera+'.png'))
                    receipt.update(status='replaying_continuation',initial_max_proprio_error=initial_error,
                        failed_target_position_error_m=position_error,failure_start_sha256=sha256(a.output/'failure-start.pt'))
                    atomic_json(a.output/'status.json',receipt)
                term,trunc,_=evaluator._apply_actions(torch.as_tensor(row['a_executed_raw23'])[None],[0])
                receipt['actual_controls']+=1
                if bool(term[0]) or bool(trunc[0]):raise ValueError('Unexpected official terminal during start validation')
                state=physical();held=held+1 if state['grasp'][m['arm']]=='TRUE' else 0
                lost=lost+1 if all(v=='FALSE' for v in state['grasp'].values()) else 0
                stream.write(json.dumps(dict(control_step=t,physical_audit=state))+'\n')
            stream.close()
            if failure_snapshot is None or held<64:raise ValueError('Recorded corrective continuation did not stably reproduce')
            restore(failure_snapshot)
            restored_error=float(np.max(np.abs(np.array(proprio61(inst.obs))-failure_snapshot['proprio'])))
            if restored_error>1e-3 or not all(v=='FALSE' for v in physical()['grasp'].values()):
                raise ValueError('Saved failure state did not restore')
            receipt.update(status='accepted_failure_start',restored_failure_max_proprio_error=restored_error,
                continued_stable_grasp_controls=held,cold_process=True,actor_oracle_inputs=False,
                memory_scope='isolated branch with only its actual prior issued command',seconds=time.monotonic()-started)
            atomic_json(a.output/'result.json',receipt);atomic_json(a.output/'status.json',receipt)
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),seconds=time.monotonic()-started)
        atomic_json(a.output/'status.json',receipt);raise


if __name__=='__main__':main()
