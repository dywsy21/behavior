"""Bounded TRAIN expert-prefix/reset/restore engineering check, no learning.

Run ONE proposal per fresh simulator process. Full scene/controller/RNG state
is retained; RGB and joint state alone are never advertised as a reset state.
No attempt is silently retried; a conservative reservation survives failures.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO/'scripts/eval/memlite_sft100'), str(Path(__file__).resolve().parents[1]/'code')]
from common import atomic_json, OFFICIAL, OFFICIAL_COMMIT, sha256


def reserve(root, case, commit):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'budget.json'
        value = json.loads(path.read_text()) if path.exists() else dict(
            schema='recovery_sim_preparation_budget_v1', maximum_resets=8,
            maximum_control_steps=4096, reservations=[])
        if any(r['case'] == case for r in value['reservations']):
            raise ValueError('No automatic retry of a reserved case')
        # Includes construction/reset, explicit instance load, restore, plus
        # one conservative spare reset. All 384 robot controls count.
        row = dict(case=case, source_commit=commit, reserved_resets=4, reserved_controls=384,
                   time=time.time(), status='reserved_even_if_process_fails')
        if (sum(r['reserved_resets'] for r in value['reservations'])+4 > 8
                or sum(r['reserved_controls'] for r in value['reservations'])+384 > 4096):
            raise ValueError('Approved simulator preparation budget exhausted')
        value['reservations'].append(row); atomic_json(path, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--budget', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Clean frozen Git source required')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=OFFICIAL, text=True).strip() != OFFICIAL_COMMIT:
        raise ValueError('Wrong simulator source')
    manifest = json.loads((a.proposal/'manifest.json').read_text())
    if (manifest['controls'] != 352 or manifest['prefix_is_verified'] is not False
            or manifest['source_episode']['split'] != 'train'
            or manifest['task'] not in ('preparing_lunch_box', 'make_pizza')
            or manifest['instance_id'] not in (241, 179)):
        raise ValueError('Unregistered expert proposal')
    for name, checksum in manifest['files'].items():
        if Path(name).name != name or sha256(a.proposal/name) != checksum:
            raise ValueError('Changed expert source')
    reserve(a.budget, a.proposal.name, commit)
    a.output.mkdir(parents=True)
    started = time.monotonic()
    status = dict(status='loading', source_commit=commit, proposal_sha256=sha256(a.proposal/'manifest.json'),
        actual_control_steps=0, explicit_instance_loads=0, explicit_snapshot_restores=0,
        optimizer_steps=0, pid=os.getpid(), task=manifest['task'], instance_id=manifest['instance_id'])
    atomic_json(a.output/'status.json', status)
    import numpy as np
    import torch
    from PIL import Image
    import omnigibson as og
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import seed_everything, DEFAULT_EVAL_SEED
    from omnigibson.macros import gm
    from omegaconf import OmegaConf
    from recovery_recorder import proprio61
    gm.HEADLESS = True; gm.RENDER_VIEWER_CAMERA = False
    seed_everything(DEFAULT_EVAL_SEED)
    arrays = np.load(a.proposal/'prefix.npz', allow_pickle=False)
    actions, reference = arrays['action'], arrays['state']
    if actions.shape != (352,23) or reference.shape != (353,61):
        raise ValueError('Expert source clock/dimension mismatch')
    class NoPolicy:
        def reset(self): pass
    class PrefixEvaluator(BatchedEvaluator):
        def load_policy(self): return NoPolicy()
    cfg = OmegaConf.create(dict(env_wrapper=dict(_target_='rgb_wrapper.RGBOnlyFullResWrapper'),
        policy_name='expert_prefix_diagnostic', headless=True, partial_scene_load=True,
        max_steps=4096, write_video=True, mode='train', seed=DEFAULT_EVAL_SEED, num_envs=1,
        task=dict(name=manifest['task']), robot=OmegaConf.load(OFFICIAL/'omnigibson/eval/r1pro.yaml')))
    atomic_json(a.output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
    with PrefixEvaluator(cfg) as evaluator:
        evaluator.load_batch({0:manifest['instance_id']})
        status['explicit_instance_loads'] += 1
        inst = evaluator.instance_eval_states[0]; robot = inst.env_accessor.robot
        scope = {k:getattr(v,'wrapped_obj',v) for k,v in inst.env_accessor.object_scope.items()}
        (a.output/'frames').mkdir(); (a.output/'videos').mkdir()
        from omnigibson.eval.utils.obs_utils import create_video_writer
        evaluator._set_video_writer(inst, create_video_writer(fpath=str(a.output/'videos/prefix.mp4'),
                                                              resolution=(448,672), rate=30))
        stream = (a.output/'transitions.jsonl').open('x')
        def physics():
            result = {}
            for key,obj in scope.items():
                if obj is None or not hasattr(obj,'get_position_orientation'): continue
                pos,quat = obj.get_position_orientation()
                grasp = {}
                if hasattr(obj,'links') and obj is not robot:
                    for arm in robot.arm_names:
                        grasp[arm] = robot.is_grasping(arm=arm,candidate_obj=obj).name
                result[key] = dict(name=obj.name, position=pos.tolist(), orientation=quat.tolist(), grasp=grasp)
            return result
        def sample(label, media=False):
            data = dict(label=label, proprio=proprio61(inst.obs), physics=physics())
            if media:
                for camera,key in evaluator.robot_camera_names.items():
                    value = inst.obs[key+'::rgb'].detach().cpu().numpy()[...,:3]
                    Image.fromarray(value).save(a.output/'frames'/f'{label}-{camera}.png')
            return data
        errors = []
        def advance(action, label):
            if status['actual_control_steps'] >= 384 or time.monotonic()-started > 3600:
                raise RuntimeError('Per-case engineering budget reached')
            before = sample(label)
            term,trunc,_ = evaluator._apply_actions(torch.as_tensor(action[None], dtype=torch.float32), [0])
            status['actual_control_steps'] += 1
            after = sample(label+'-after')
            stream.write(json.dumps(dict(before=before, action_raw23=action.tolist(), after=after,
                terminated=bool(term[0]), truncated=bool(trunc[0])))+'\n')
            if bool(term[0]) or bool(trunc[0]): raise RuntimeError('Unexpected terminal in short prefix check')
            return after
        for t in range(353):
            if t % 16 == 0:
                state = np.asarray(proprio61(inst.obs))
                # Different units kept as per-dimension errors, not one fake
                # universal robot-position tolerance or an automatic approval.
                errors.append(dict(frame=t, abs_error=np.abs(state-reference[t]).tolist()))
                sample(f'{t:06d}', media=(t%64==0 or t==352))
                atomic_json(a.output/'status.json', dict(status, status='replaying_prefix', source_frame=t))
            evaluator._write_video(inst)
            if t < 352: advance(actions[t], f'expert-{t}')
        evaluator._set_video_writer(inst, None)
        # Save full world plus controller filters/goals and host RNG. Never
        # infer these from the saved RGB/proprio files.
        controllers = {k:deepcopy(c.dump_state(serialized=False)) for k,c in robot.controllers.items()}
        world = deepcopy(og.sim.dump_state(serialized=False))
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        at_snapshot = sample('snapshot', True)
        torch.save(dict(world=world, controllers=controllers, rng=rng, observation=at_snapshot,
                        source=manifest, source_commit=commit, official_commit=OFFICIAL_COMMIT),
                   a.output/'full_snapshot.pt')
        branches = []
        # Same real action sequence on both branches. A replay check is NOT a
        # recovery attempt and must not generate automatic positive BC labels.
        for branch in range(2):
            if branch:
                og.sim.load_state(deepcopy(world), serialized=False)
                for key,state in controllers.items(): robot.controllers[key].load_state(deepcopy(state), serialized=False)
                random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
                if rng['cuda'] is not None: torch.cuda.set_rng_state_all(rng['cuda'])
                # Refresh sensor output WITHOUT advancing physics/control.
                for _ in range(3): og.sim.render()
                obs,_ = evaluator.env.get_obs(); inst.obs=evaluator._preprocess_obs(obs[0],inst)
                status['explicit_snapshot_restores'] += 1
            start_state = sample(f'branch{branch}-start', True)
            sequence = []
            for offset in range(16): sequence.append(advance(actions[336+offset], f'branch{branch}-{offset}'))
            branches.append(dict(start=start_state, states=sequence))
        stream.flush(); os.fsync(stream.fileno()); stream.close()
        diffs = [float(np.max(np.abs(np.asarray(x['proprio'])-np.asarray(y['proprio']))))
                 for x,y in zip(branches[0]['states'], branches[1]['states'])]
        atomic_json(a.output/'result.json', dict(status, status='completed_diagnostic_only',
            prefix_error_by_dimension=errors, snapshot=at_snapshot, branches=branches,
            replay_max_proprio_abs_error=max(diffs), full_snapshot_sha256=sha256(a.output/'full_snapshot.pt'),
            seconds=time.monotonic()-started, training_labels_approved=False,
            legal_start_approved=False, pending='Original RGB and real physical alignment must be reviewed; no automatic recovery label'))
        atomic_json(a.output/'status.json', dict(status, status='completed_diagnostic_only', seconds=time.monotonic()-started))


if __name__ == '__main__': main()
