"""Read-only audit of stopped E2, then claim one evaluation-only continuation."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from common import OUT, RUNTIME, PREVIOUS_RUNTIME, REPO, PARENT, PARENT_SHA, commit, save, sha
from g05.rl.protocol import validate_worker
from g05.rl.recovery import remaining_evaluation_budget, validate_baseline, count_physical_steps

PRIOR = Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e2_v1')
DELTA = PRIOR / 'rl_batch_007_updates_0094.pt'
DELTA_SHA = 'aee508c41fa03b044e8c21d2c12b763b96d643f2b864d50495e1ab3bc554444f'


def main():
    if (OUT != PRIOR.parent/'e2_final_recovery_v1' or OUT.exists() or RUNTIME.exists()
            or RUNTIME != Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e2_final_recovery_v1')
            or PREVIOUS_RUNTIME != Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e2_v1/training')):
        raise ValueError('Only a fresh registered final-evaluation continuation is allowed')
    inputs = {}

    def read(path):
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())

    old = read(PRIOR/'manifest.json')
    status = read(PRIOR/'status.json')
    supervisor = read(PRIOR/'supervisor.json')
    train = read(PRIOR/'training_result.json')
    pids = read(PRIOR/'pids.json')
    if ((PRIOR/'final').exists() or (PRIOR/'result.json').exists()
            or train['reason'] != 'registered_training_deadline'
            or train['checkpoint'] != str(DELTA) or train['actor_updates'] != 94
            or train['new_actor_updates'] != 84):
        raise ValueError('Not the registered stopped E2; do not select a different checkpoint')
    owned = {supervisor['supervisor'], supervisor['learner']}
    for pool in pids['history']:
        owned.update(pool['simulators'])
        owned.add(pool['learner'])
    for pid in owned:
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
            raise ValueError(f'Previous owned process still alive: {pid}')
    receipt = read(DELTA.with_suffix('.json'))
    if (sha(PARENT) != PARENT_SHA or sha(DELTA) != DELTA_SHA or receipt['sha256'] != DELTA_SHA
            or receipt['parent_sha256'] != PARENT_SHA or receipt['actor_updates'] != 94
            or receipt['critic_updates'] != 16 or receipt['ae_precision'] != 'float32'):
        raise ValueError('Selected saved checkpoint identity changed')
    baseline_file = PRIOR/'evaluations.jsonl'
    rows = [json.loads(line) for line in baseline_file.read_text().splitlines()]
    inputs[str(baseline_file)] = sha(baseline_file)
    baseline_controls = validate_baseline(rows)
    totals = {}
    for pool, phases in [('baseline', {'eval_parent_fp32', 'eval_parent_bf16'}),
                         ('training', {'policy', 'expert_prefix'})]:
        counts = []
        for w in (0, 1):
            file = PRIOR/pool/f'worker_{w}/steps.jsonl'
            with file.open() as stream:
                n = count_physical_steps((json.loads(line) for line in stream), allowed_phases=phases)
            inputs[str(file)] = sha(file)
            counts.append(n)
            if pool == 'training':
                failure = read(file.parent/'primary_failure.json')
                if failure['error'] != "TimeoutError('Simulator IPC timeout')" or failure['controls'] != n:
                    raise ValueError('Unresolved physical control or different failure')
        closed = read(PRIOR/pool/'closed.json')
        if closed['exits'] != [0, 0] or closed['pending_controls'] or closed['ledger_controls'] != sum(counts):
            raise ValueError('Prior physical ledger/process exit mismatch')
        if pool == 'baseline' and (not closed['clean'] or sum(counts) != baseline_controls):
            raise ValueError('Original baseline did not close cleanly')
        totals[pool] = counts
    if sum(totals['training']) != train['controls']:
        raise ValueError('Training control ledger mismatch')
    budget = remaining_evaluation_budget(old, status, supervisor, sum(totals['training']), baseline_controls)
    source = commit()
    # No change to inference, observation, reset, physical or renderer code.
    subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code', old['source_commit'], source, '--',
        'scripts/rl/common.py', 'scripts/rl/sim_worker.py', 'scripts/rl/method.py', 'scripts/rl/learner.py',
        'scripts/semantic_robot', 'src/semantic_robot', 'src/g05/rl/g05_adapter.py',
        'src/g05/rl/flow_ppo.py', 'scripts/experiments/eval_g05_100k.py'], check=True)
    reset_refs = {}
    specs = deepcopy(old['sim_pools']['final'])
    for spec in specs:
        validate_worker(spec, evaluation=True)
        if sha(spec['instance_file']) != spec['instance_file_sha256']:
            raise ValueError('Official public instance changed')
        ref = PRIOR/'baseline'/f'worker_{spec["worker"]}/reset_state_000.json'
        reset_refs[str(spec['instance'])] = read(ref)
    manifest = deepcopy(old)
    manifest.update(budget, source_commit=source, entry='final_eval', continued_from=str(PRIOR),
        runtime=str(RUNTIME), previous_runtime=str(PREVIOUS_RUNTIME),
        final_checkpoint=str(DELTA), final_checkpoint_sha256=DELTA_SHA,
        max_training_controls=0, max_training_seconds=0, max_new_actor_updates=0, max_batches=0,
        sim_pools={'final': specs}, audit_input_sha256=inputs, reset_references=reset_refs,
        prior_evaluations=rows, resumed_training=False)
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'manifest.json', manifest)
    save(OUT/'prior_control_audit.json', dict(counts=totals, budget=budget, inputs=inputs,
        prior_pool_failure='manual review silence exceeded worker IPC timeout; both exited',
        all_prior_processes_dead=True, prior_evaluation_unmodified=True))
    save(OUT/'frozen_final_selection.json', dict(checkpoint=str(DELTA), sha256=DELTA_SHA,
        selection='last saved legal E2 checkpoint; selected before any final evaluation',
        eval_results_used=False, actor_updates=94, new_actor_updates=84))
    print(json.dumps(dict(run=str(OUT), source=source, **budget)))


if __name__ == '__main__':
    main()
