"""Register one E3 continuation; old runs and checkpoints are read-only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq

from common import OUT, RUNTIME, PREVIOUS_RUNTIME, DATA, PARENT, PARENT_SHA, commit, save, sha
from g05.rl.dense_recipe import REWARD, validate_recipe
from g05.rl.protocol import validate_worker, paired_summary
from g05.rl.recovery import validate_baseline
from prepare_final_eval import PRIOR, DELTA, DELTA_SHA

COMPLETED = PRIOR.parent/'e2_final_recovery_v1'


def main():
    if (OUT != PRIOR.parent/'e3_dense_v1' or OUT.exists() or RUNTIME.exists()
            or RUNTIME != Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e3_dense_v1')
            or PREVIOUS_RUNTIME != Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e2_final_recovery_v1/final')):
        raise ValueError('A new registered E3 run/runtime is required')
    if shutil.disk_usage(OUT.parent).free < 400*(1<<30):
        raise ValueError('Insufficient reserved space for bounded checkpoints and traces')
    inputs = {}
    def read(path):
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())
    old = read(PRIOR/'manifest.json')
    for run in (PRIOR, COMPLETED):
        supervisor = read(run/'supervisor.json')
        pids = read(run/'pids.json')
        owned = {supervisor['supervisor'], supervisor['learner']}
        for group in pids['history']:
            owned.update(group['simulators']); owned.add(group['learner'])
        for pid in owned:
            file = Path(f'/proc/{pid}/stat')
            if file.exists() and file.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                raise ValueError('Old owned process is still alive')
        if run == COMPLETED and (supervisor['status'] != 'completed' or supervisor['exit_code'] != 0):
            raise ValueError('Previous final evaluation incomplete')
    receipt = read(DELTA.with_suffix('.json'))
    if (sha(PARENT) != PARENT_SHA or sha(DELTA) != DELTA_SHA or receipt['sha256'] != DELTA_SHA
            or receipt['actor_updates'] != 94 or receipt['critic_updates'] != 16):
        raise ValueError('Original parent or selected resume delta changed')
    final_result = read(COMPLETED/'result.json')
    if final_result['checkpoint_sha256'] != DELTA_SHA:
        raise ValueError('Prior final evaluation used a different checkpoint')
    def rows(path):
        inputs[str(path)] = sha(path)
        return [json.loads(s) for s in path.read_text().splitlines()]
    baselines = rows(PRIOR/'evaluations.jsonl')
    validate_baseline(baselines)
    previous_results = rows(COMPLETED/'evaluations.jsonl')
    paired_summary(previous_results)
    resume_evaluations = [r for r in previous_results if r['variant'] == 'rl_fp32']
    if any(r['actor_updates'] != 94 or r['ae_precision'] != 'float32' for r in resume_evaluations):
        raise ValueError('Prior delta precision or update count changed')
    last = read(PRIOR/'train_batch_005.json')
    workers = deepcopy(old['workers'])
    metadata = pq.read_table(DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    heldout = {(r['task_index'], r['task_instance_id']) for r in metadata if r['episode_index'] % 200 >= 190}
    for spec, episode in zip(workers, last['episodes']):
        if spec['instance'] != episode['instance']:
            raise ValueError('Curriculum continuation identity mismatch')
        spec['prefix_controls'] = episode['next_prefix']
        validate_worker(spec, evaluation=False)
        if (spec['task_id'], spec['instance']) in heldout or sha(spec['actions']) != spec['actions_sha256']:
            raise ValueError('TRAIN-only source changed')
        actions = np.load(spec['actions'], allow_pickle=False)
        if actions.ndim != 2 or actions.shape[1] != 23 or not np.isfinite(actions).all():
            raise ValueError('Invalid demo controls')
    references = {}
    evaluation = deepcopy(old['sim_pools']['final'])
    for spec in evaluation:
        validate_worker(spec, evaluation=True)
        if sha(spec['instance_file']) != spec['instance_file_sha256']:
            raise ValueError('Official evaluation instance changed')
        references[str(spec['instance'])] = read(PRIOR/'baseline'/f'worker_{spec["worker"]}/reset_state_000.json')
    manifest = deepcopy(old)
    manifest.update(source_commit=commit(), entry='method_dense', experiment='RL-G05-50K-E3',
        resume_checkpoint=str(DELTA), resume_sha256=DELTA_SHA, continued_from=str(PRIOR),
        workers=workers, sim_pools=dict(training=workers, final=evaluation),
        max_controls=100000, max_active_wall_seconds=43200, max_training_controls=80000,
        max_training_seconds=28800, max_batches=64, max_new_actor_updates=2000, max_actor_updates=2094,
        final_eval_reserved_controls=19344, final_eval_reserved_seconds=10800,
        learning_rates=dict(action_expert=1e-7, noise=1e-6, critic=1e-4), clip=.1,
        target_path_kl=.1, target_mean_path_kl=.02, ppo_epochs=4, critic_steps_per_batch=4,
        reset_candidate_lr_each_minibatch=True, curriculum_admission='automatic', reward=deepcopy(REWARD),
        previous_runtime=str(PREVIOUS_RUNTIME), runtime=str(RUNTIME),
        prior_evaluations=baselines, resume_evaluations=resume_evaluations,
        reset_references=references, audit_input_sha256=inputs,
        hypothesis='more effective PPO updates and bounded goal potential with unattended curriculum improve full-reset development success',
        training_curriculum='resume 1076/1096; three official successes move prefix96 earlier, original gap768 floor; automatic checks only')
    validate_recipe(manifest)
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'manifest.json', manifest)
    print(json.dumps(dict(run=str(OUT), source=manifest['source_commit'],
                         prefixes=[w['prefix_controls'] for w in workers],
                         human_review_required=False, new_gpu_processes=0)))


if __name__ == '__main__': main()
