from copy import deepcopy
import json
from pathlib import Path

import pytest

import reference_gate_recovery as recovery


def fixture(tmp_path, monkeypatch, name):
    previous_name, expected_sha = recovery.DECLARED[name]
    previous = tmp_path / previous_name
    previous.mkdir()
    monkeypatch.setattr(recovery, 'BASE', tmp_path)
    monkeypatch.setattr(recovery, 'assert_exited', lambda pid: None)
    monkeypatch.setattr(recovery, 'full_reference_identity', lambda: {'verified_canary': 'identity'})
    original_digest = recovery.digest
    def digest(path):
        path = Path(path)
        return {'method_spec.json': expected_sha, 'eval_step_0.json': recovery.FAILED_EVAL_SHA,
                'checkpoint_inspection.json': recovery.OLD_SMOKE_SHA}.get(path.name, original_digest(path))
    monkeypatch.setattr(recovery, 'digest', digest)
    monkeypatch.setattr(recovery, 'checkpoint_sha', lambda path: recovery.OLD_SMOKE_CHECKPOINT_SHA)
    old = dict(route='fm' if name.startswith('fm_') else name.split('_')[0], recipe={'max_updates': 500},
        initialization='a4', conditioning='skills', parent_path='/original/a4.pt', parent_sha256='original-a4',
        marker_rows='marker' in name, marker_recipe={'rows': 8} if 'marker' in name else None,
        fresh_adam=True, high_unchanged=True, train_split='original950', eval_split='original50')
    for file, contents in {'method_spec.json': old, 'status.json': {'state': 'failed', 'automatic_retry': False},
                           'launch.json': {'supervisor_pid': 42}}.items():
        (previous / file).write_text(json.dumps(contents))
    if name == 'fm_action_control_v3':
        formal = previous / 'formal'
        formal.mkdir()
        for file in ['config.yaml', 'dataset_stats.json', 'eval_step_0.json', 'train_source_spec.json'] + [
                f'{kind}_rank{rank}.json' for kind in ('identity', 'restoration') for rank in range(4)]:
            (formal / file).write_text('{}')
        (previous / 'formal.log').write_text('RuntimeError: A4 prefix-only fixed80 metric differs from the original reference\n' * 4)
        (previous / 'smoke').mkdir()
        (previous / 'smoke/checkpoint_inspection.json').write_text(json.dumps(dict(passed=True, actual_updates=5,
            actual_adam_states=504, frozen_unchanged=True, full_model_optimizer_rng_roundtrip=True, actual_train_rows=80,
            checkpoint=str(previous / 'smoke/checkpoints/step_5.pt'), checkpoint_sha256=recovery.OLD_SMOKE_CHECKPOINT_SHA)))
    spec = {**deepcopy(old), 'output': str(tmp_path / name), 'recovery_from': None}
    spec['reference_gate_recovery'] = recovery.recovery_identity(previous, Path(spec['output']))
    return previous, spec


@pytest.mark.parametrize('name', tuple(recovery.DECLARED))
def test_exact_original_recipe_is_preserved_and_only_completed_fm_smoke_is_reused(tmp_path, monkeypatch, name):
    previous, spec = fixture(tmp_path, monkeypatch, name)
    recovery.validate_recovery(spec)
    if name == 'fm_action_control_v3':
        assert recovery.training_phases(spec) == ('formal',)
        assert recovery.smoke_gate_path(spec) == previous / 'smoke/checkpoint_inspection.json'
    else:
        assert recovery.training_phases(spec) == ('smoke', 'formal')
        assert recovery.smoke_gate_path(spec) == Path(spec['output']) / 'smoke/checkpoint_inspection.json'
    spec['recipe']['max_updates'] = 5000
    with pytest.raises(RuntimeError, match='original recipe'):
        recovery.validate_recovery(spec)


@pytest.mark.parametrize('damage', ['new_update', 'live', 'completed', 'canary', 'dual_receipt', 'smoke', 'changed_parent'])
def test_recovery_never_hides_new_training_or_unverified_evidence(tmp_path, monkeypatch, damage):
    previous, spec = fixture(tmp_path, monkeypatch, 'fm_action_control_v3')
    if damage == 'new_update':
        (previous / 'formal/train_metrics.jsonl').write_text('update')
    elif damage == 'live':
        def fail(_):
            raise RuntimeError('live')
        monkeypatch.setattr(recovery, 'assert_exited', fail)
    elif damage == 'completed':
        (previous / 'status.json').write_text(json.dumps({'state': 'complete', 'automatic_retry': False}))
    elif damage == 'canary':
        monkeypatch.setattr(recovery, 'full_reference_identity', lambda: {'different': True})
    elif damage == 'dual_receipt':
        spec['recovery_from'] = {'unexpected': True}
    elif damage == 'smoke':
        monkeypatch.setattr(recovery, 'checkpoint_sha', lambda _: 'changed')
    else:
        spec['parent_sha256'] = 'changed'
    with pytest.raises(RuntimeError):
        recovery.validate_recovery(spec)


def test_downstream_that_really_started_is_not_repeated(tmp_path, monkeypatch):
    previous, spec = fixture(tmp_path, monkeypatch, 'ki_a4_fulltrain_v3')
    (previous / 'smoke').mkdir()
    with pytest.raises(RuntimeError, match='already entered'):
        recovery.validate_recovery(spec)
    with pytest.raises(ValueError, match='four declared'):
        recovery.recovery_identity(previous, tmp_path / 'ki_a4_fulltrain_v4')


def test_unfinished_full_reference_probe_cannot_release_training(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'BASE', tmp_path)
    root = tmp_path / 'fm_full_reference_probe_v1'
    root.mkdir()
    (root / 'result.json').write_text(json.dumps({'complete': True, 'original_reference_gate_passed': False}))
    with pytest.raises(RuntimeError, match='All80'):
        recovery.full_reference_identity()
