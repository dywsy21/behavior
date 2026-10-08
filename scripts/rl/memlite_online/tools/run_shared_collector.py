"""One TRAIN task queue per rank; the model update remains globally shared."""
# ruff: noqa: E402 -- pin dependency paths before importing runtime modules.
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time

from bootstrap import bootstrap
SOURCE = bootstrap()
from checkpoint_io import atomic_json
from probe_task_catalog import anonymous_gib, stop_owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.job / 'manifest.json').read_text())
    run = args.job / 'collectors' / f'gpu_{args.rank}'
    run.mkdir(parents=True, exist_ok=False)
    tasks = manifest['groups'][args.rank]['tasks']
    policy_run = args.job / 'policies' / f'gpu_{args.rank}'
    state = dict(status='starting', rank=args.rank, pid=os.getpid(), cycles=[], started=time.time())
    env = os.environ | dict(RL_SYNCHRONOUS='1', RL_GPU=str(args.rank),
        RL_POLICY_PORT=str(manifest['port_base'] + args.rank), RL_JOB_ROOT=str(args.job),
        RL_REWARD_PROTOCOL='shared_terminal_q_v1', RL_EVALUATION_ONLY='0',
        RL_CHUNKS_PER_UPDATE=str(manifest['training']['chunks_per_rank']),
        RL_SOURCE_COMMIT=manifest['source_commit'], RL_TRAINING_CONFIG=str(SOURCE/'training.json'),
        RL_RECOVERY_ROOT=str(args.job/'recovery'/f'gpu_{args.rank}'),
        RL_SIM_DATA_ROOT='/run/ti/rl_memlite_stage1_20261006/assets_readiness/sim_data',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for key in ('RL_PREP_BARRIER_DIR', 'RL_RESUME_CHECKPOINT'):
        env.pop(key, None)
    cycle = 0
    if manifest.get('resume_cursors'):
        cycle = manifest['resume_cursors'][str(args.rank)]
    sim, slot = None, None
    try:
        while time.time() < manifest['absolute_deadline'] and not (args.job / 'ABORT').exists():
            if (policy_run / 'save_ack.json').exists():
                state['status'] = 'finished'
                break
            task = tasks[cycle % len(tasks)]
            sweep = cycle // len(tasks)
            pool = list(task['train_instances'])
            random.Random(20261008 + 1000 * task['task_index']).shuffle(pool)
            offset = 2 * sweep % len(pool)
            instances = [pool[offset], pool[(offset + 1) % len(pool)]]
            seed = 261008 + 1000 * task['task_index'] + sweep + manifest.get('resume_seed_offset', 0)
            sub = run / f'cycle_{cycle:05d}_{task["task"]}'
            sub.mkdir()
            # User stop is acted on by a shared optimizer barrier, not by one
            # rank dropping out while peers wait inside an all-reduce.
            import fcntl
            while slot is None:
                if (args.job / 'ABORT').exists() or time.time() >= manifest['absolute_deadline']:
                    raise RuntimeError('Aborted while waiting for scene load')
                if anonymous_gib() < 160:
                    for number in range(2):
                        candidate = (args.job / f'scene_load_{number}.lock').open('a')
                        try:
                            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            slot = candidate
                            break
                        except BlockingIOError:
                            candidate.close()
                if slot is None:
                    time.sleep(1)
            metadata = [dict(task=task['task'], task_index=task['task_index'], instance_id=i,
                split='train', source_commit=manifest['source_commit'],
                resume_checkpoint_sha256=manifest.get('resume_sha256') or manifest['model']['checkpoints'][
                    'low/step_00098414_save_0021.pt']['sha256'],
                high_checkpoint_sha256=manifest['model']['checkpoints'][
                    'high/step_00048045_save_0027.pt']['sha256'],
                low_checkpoint_sha256=manifest['model']['checkpoints'][
                    'low/step_00098414_save_0021.pt']['sha256'],
                run=str(sub), gpu=args.rank, cycle=cycle, seed_base=seed,
                model_parent='fduTristin/memlite-stage1') for i in instances]
            cycle_env = env | dict(RL_EPISODE_STEPS=str(task['official_horizon_steps']),
                RL_SEED_BASE=str(seed), RL_EPISODE_METADATA=json.dumps(metadata),
                RL_TASK_FULL_SCENE='1' if task['template_mode'] == 'full_scene' else '0')
            info = dict(cycle=cycle, task=task['task'], instances=instances, seed=seed,
                        horizon=task['official_horizon_steps'], started=time.time(), run=str(sub))
            with (sub / 'sim.log').open('w') as output:
                sim = subprocess.Popen(['bash', str(SOURCE/'tools/run_pilot_sim.sh'), str(sub),
                    task['task'], *map(str, instances)], env=cycle_env, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='collecting', current=info, sim_pid=sim.pid, updated=time.time())
            atomic_json(run / 'status.json', state)
            while sim.poll() is None:
                if slot is not None and (sub / 'scene.ready').exists():
                    slot.close()
                    slot = None
                if (args.job / 'ABORT').exists() or time.time() >= manifest['absolute_deadline']:
                    raise RuntimeError('Global abort/deadline during collection')
                if anonymous_gib() > 210:
                    raise RuntimeError('Anonymous memory reserve exceeded')
                time.sleep(2)
            if slot is not None:
                slot.close()
                slot = None
            if (sub / 'shared_training_stopped.json').exists():
                state.update(status='finished', stop=json.loads((sub/'shared_training_stopped.json').read_text()))
                break
            if sim.returncode != 0 or not (sub / (task['task'] + '.completed')).exists():
                raise RuntimeError(f'Collector exited abnormally: {task["task"]}, code {sim.returncode}')
            records = [json.loads(path.read_text()) for path in sub.glob('eval_*/json/*.json')]
            if len(records) != 2 or {r['instance_id'] for r in records} != set(instances):
                raise RuntimeError('Official TRAIN outcome identity mismatch')
            info.update(finished=time.time(), official_episodes=records)
            state['cycles'].append(info)
            cycle += 1
            state['next_cycle'] = cycle
            atomic_json(run/'status.json', state)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        (args.job / 'ABORT').touch()
        raise
    finally:
        if slot is not None:
            slot.close()
        if sim is not None:
            stop_owned(sim)
        state['finished'] = time.time()
        atomic_json(run / 'status.json', state)


if __name__ == '__main__':
    main()
