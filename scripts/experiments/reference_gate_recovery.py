"""Explicit recovery of four zero-formal-update jobs after full reference proof.

This never loosens the evaluator or retries a completed trial. FM reuses its
already verified five-update gate, not its weights; formal training remains a
fresh original-A4 500. The other three jobs have not run their smoke yet.
"""
import hashlib
from pathlib import Path

from method_queue_recovery import assert_exited, digest, read

BASE = Path('/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913')
DECLARED = {
    'fm_action_control_v3': ('fm_action_control_v2', '09b3bb2a385b7b45266238f097327bcbf49c95c0670ac589da889692bb398c0c'),
    'joint_a4_fulltrain_v3': ('joint_a4_fulltrain_v2', '02a01661ad86667291195e6b631c5d68c1c0d3a624d3945204f4c9ea425abb93'),
    'ki_a4_fulltrain_v3': ('ki_a4_fulltrain_v2', '1850c78e2d729c476871c67a3e5bfa13a73934adfc591374f8c70d1f123bc285'),
    'ar_a4_marker_fulltrain_v3': ('ar_a4_marker_fulltrain_v2', 'eb2036b66c3045ac25f2764d2104a6c8c3cbd4f35052857eb80385dc145223c2'),
}
CANARY_COMMIT = '90deb969012fd5ff82d699f81711a87a340e9cc2'
FAILED_EVAL_SHA = '41dfc82e0e21da3010242e1b83241cff8018ba114d905e81f12a9ad9b353e40c'
OLD_SMOKE_SHA = 'c79ddfb32cb254fa9255effe90418225ffe3324356d04bb1abc1142827a97103'
OLD_SMOKE_CHECKPOINT_SHA = '676d1a8622d0748ec4207159ffaca4313f3db9f61361845ebdd69fcfaf444f4a'


def checkpoint_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def full_reference_identity():
    root = BASE / 'fm_full_reference_probe_v1'
    result = read(root / 'result.json')
    if (result.get('complete') is not True or result.get('actual_forwards') != 160
            or result.get('actual_windows') != 80 or result.get('optimizer_updates') != 0
            or result.get('actual_train_draws') != 0 or result.get('simulator_controls') != 0
            or result.get('original_reference_gate_passed') is not True
            or result.get('original_threshold_unchanged') is not True
            or result.get('original_absolute_tolerance') != 2e-6
            or result.get('per_window_differences') != []
            or result.get('same_input_paired_forward_differences') != []):
        raise RuntimeError('All80 original references and160 paired forward traces must pass before recovery')
    files, window_ids = {}, []
    for rank in range(4):
        identity_path, trace_path = root / f'identity_rank{rank}.json', root / f'trace_rank{rank}.json'
        identity, trace = read(identity_path), read(trace_path)
        if (identity['commit'] != CANARY_COMMIT or identity['rank'] != rank
                or trace['rank'] != rank or trace['model_forwards'] != 40
                or len(trace['windows']) != 20
                or any(len(row['calls']) != 2 or row['calls'][0] != row['calls'][1] for row in trace['windows'])):
            raise RuntimeError('Missing exact four-rank no-update reference proof')
        files[str(identity_path)] = digest(identity_path)
        files[str(trace_path)] = digest(trace_path)
        window_ids.extend(row['window_id'] for row in trace['windows'])
    if len(window_ids) != 80 or len(set(window_ids)) != 80:
        raise RuntimeError('The full reference proof must cover80 different windows')
    return dict(result_path=str(root / 'result.json'), result_sha256=digest(root / 'result.json'),
                source_commit=CANARY_COMMIT, evidence_sha256=files, threshold_unchanged=True)


def recovery_identity(previous, output):
    if previous is None:
        raise ValueError('An explicit failed-v2 predecessor is required')
    previous, output = Path(previous), Path(output)
    declared = DECLARED.get(output.name)
    if (declared is None or output.parent != BASE or previous.parent != BASE or previous.name != declared[0]
            or previous.is_symlink() or output.is_symlink()):
        raise ValueError('Only the four declared v2-to-v3 recoveries are permitted')
    if digest(previous / 'method_spec.json') != declared[1]:
        raise RuntimeError('Original zero-update recipe changed')
    status, launch = read(previous / 'status.json'), read(previous / 'launch.json')
    if status.get('state') != 'failed' or status.get('automatic_retry') is not False:
        raise RuntimeError('Recovery requires an already failed, non-retrying v2')
    assert_exited(launch['supervisor_pid'])
    smoke = None
    if output.name == 'fm_action_control_v3':
        formal = previous / 'formal'
        expected_files = {'config.yaml', 'dataset_stats.json', 'eval_step_0.json', 'train_source_spec.json'}
        expected_files.update(f'{kind}_rank{rank}.json' for kind in ('identity', 'restoration') for rank in range(4))
        if (not formal.is_dir() or {str(p.relative_to(formal)) for p in formal.rglob('*') if p.is_file()} != expected_files
                or any(p.is_symlink() for p in formal.rglob('*'))
                or digest(formal / 'eval_step_0.json') != FAILED_EVAL_SHA
                or (previous / 'formal.log').read_text().count(
                    'RuntimeError: A4 prefix-only fixed80 metric differs from the original reference') != 4):
            raise RuntimeError('FM v2 is not the exact audited initial-eval-only failure')
        gate_path = previous / 'smoke/checkpoint_inspection.json'
        gate = read(gate_path)
        if (digest(gate_path) != OLD_SMOKE_SHA or not gate['passed'] or gate['actual_updates'] != 5
                or gate['actual_adam_states'] != 504 or not gate['frozen_unchanged']
                or not gate['full_model_optimizer_rng_roundtrip'] or gate['actual_train_rows'] != 80
                or Path(gate['checkpoint']) != previous / 'smoke/checkpoints/step_5.pt'
                or gate['checkpoint_sha256'] != OLD_SMOKE_CHECKPOINT_SHA
                or checkpoint_sha(gate['checkpoint']) != OLD_SMOKE_CHECKPOINT_SHA):
            raise RuntimeError('The previous five-update FM save/read gate is no longer verified')
        smoke = dict(inspection_path=str(gate_path), inspection_sha256=OLD_SMOKE_SHA,
                     checkpoint_sha256=OLD_SMOKE_CHECKPOINT_SHA, prior_updates=5, new_smoke_updates=0,
                     formal_initialization_is_original_a4=True)
    elif (previous / 'formal').exists() or (previous / 'smoke').exists():
        raise RuntimeError('Downstream v2 has already entered training; cannot repeat it')
    return dict(previous_run=str(previous), previous_method_sha256=declared[1],
        previous_status_sha256=digest(previous / 'status.json'), previous_launch_sha256=digest(previous / 'launch.json'),
        previous_supervisor_pid=launch['supervisor_pid'], zero_formal_updates_verified=True,
        full_reference=full_reference_identity(), reused_smoke=smoke,
        mathematical_training_changes=False, reference_threshold_changed=False,
        automatic_retry=False, preserves_previous_runs=True)


def validate_recovery(spec):
    receipt = spec.get('reference_gate_recovery')
    if Path(spec['output']).name not in DECLARED or not receipt or spec.get('recovery_from'):
        raise RuntimeError('A declared reference recovery must have exactly its own evidence')
    actual = recovery_identity(receipt['previous_run'], spec['output'])
    if receipt != actual:
        raise RuntimeError('Reference recovery evidence changed')
    old = read(Path(receipt['previous_run']) / 'method_spec.json')
    fields = ('route', 'recipe', 'initialization', 'conditioning', 'parent_path', 'parent_sha256',
              'marker_rows', 'marker_recipe', 'fresh_adam', 'high_unchanged', 'train_split', 'eval_split')
    if any(spec.get(key) != old.get(key) for key in fields):
        raise RuntimeError('Reference recovery must preserve the original recipe and finite500 budget')


def smoke_gate_path(spec):
    receipt = spec.get('reference_gate_recovery') or {}
    reused = receipt.get('reused_smoke')
    if reused:
        validate_recovery(spec)
        return Path(reused['inspection_path'])
    return Path(spec['output']) / 'smoke/checkpoint_inspection.json'


def training_phases(spec):
    receipt = spec.get('reference_gate_recovery') or {}
    if receipt.get('reused_smoke'):
        validate_recovery(spec)
        return ('formal',)
    return ('smoke', 'formal')
