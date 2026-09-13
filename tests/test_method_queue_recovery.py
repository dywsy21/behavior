"""Recovery is explicit, identity-bound, and cannot repeat completed training."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import method_queue_recovery as recovery


def failed_fixture(tmp_path, monkeypatch, name='fm_beta_stratified_v2'):
    previous_name, kind, expected_sha = recovery.DECLARED[name]
    previous, output = tmp_path / previous_name, tmp_path / name
    previous.mkdir()
    monkeypatch.setattr(recovery, 'BASE', tmp_path)
    monkeypatch.setattr(recovery, 'assert_exited', lambda pid: None)
    original_digest = recovery.digest
    monkeypatch.setattr(recovery, 'digest', lambda p: expected_sha if Path(p).name == 'method_spec.json'
                        else original_digest(p))
    common = dict(parent_path='/pinned/a4.pt', parent_sha256='a4-sha', initialization='a4')
    if kind == 'fm':
        method = dict(**common, trial='beta_stratified' if 'beta' in name else 'exec_weight2',
                      settings={'lr': 1e-5}, max_updates=500, smoke_updates=5, seed=41)
    else:
        route = 'fm' if name.startswith('fm_') else name.split('_')[0]
        method = dict(**common, route=route, recipe={'max_updates': 500}, conditioning='skills',
                      marker_rows='marker' in name, marker_recipe={'rows': 8} if 'marker' in name else None)
    method['output'] = str(previous)
    for filename, content in {
        'method_spec.json': method, 'launch.json': {'supervisor_pid': 42},
        'status.json': {'state': 'failed', 'automatic_retry': False},
    }.items():
        (previous / filename).write_text(json.dumps(content))
    if previous_name == 'fm_beta_stratified_v1':
        smoke = previous / 'smoke'
        (smoke / '.hydra').mkdir(parents=True)
        (smoke / 'coordination_run_receipt.json').write_text(json.dumps({'pid': 43, 'state': 'failed'}))
        (smoke / 'train.log').write_text('')
        for filename in ('config.yaml', 'hydra.yaml', 'overrides.yaml'):
            (smoke / '.hydra' / filename).write_text('fixture')
        (previous / 'smoke.log').write_text(
            'AssertionError: Fine-tuning assumes at least one GPU is available!\n' * 4)
    spec = {**deepcopy(method), 'output': str(output)}
    spec['recovery_from'] = recovery.recovery_identity(previous, output, kind)
    return previous, spec, kind


@pytest.mark.parametrize('name', tuple(recovery.DECLARED))
def test_each_declared_zero_update_recovery_is_audited_and_recipe_unchanged(tmp_path, monkeypatch, name):
    previous, spec, kind = failed_fixture(tmp_path, monkeypatch, name)
    recovery.validate_recovery(spec, kind)
    assert spec['recovery_from']['previous_run'] == str(previous)
    assert not spec['recovery_from']['automatic_retry']
    spec['parent_sha256'] = 'changed'
    with pytest.raises(RuntimeError, match='original method'):
        recovery.validate_recovery(spec, kind)


@pytest.mark.parametrize('damage', ['formal', 'status', 'launch', 'live', 'log', 'extra', 'hash', 'budget'])
def test_recovery_refuses_new_work_changed_evidence_or_live_process(tmp_path, monkeypatch, damage):
    previous, spec, kind = failed_fixture(tmp_path, monkeypatch)
    if damage == 'formal':
        (previous / 'formal').mkdir()
    elif damage == 'status':
        (previous / 'status.json').write_text(json.dumps({'state': 'complete', 'automatic_retry': False}))
    elif damage == 'launch':
        (previous / 'launch.json').write_text(json.dumps({'supervisor_pid': 99}))
    elif damage == 'live':
        def fail(_):
            raise RuntimeError('live')
        monkeypatch.setattr(recovery, 'assert_exited', fail)
    elif damage == 'log':
        (previous / 'smoke' / 'train.log').write_text('a real update')
    elif damage == 'extra':
        (previous / 'smoke' / 'batch_sources.jsonl').write_text('data already consumed')
    elif damage == 'hash':
        monkeypatch.setattr(recovery, 'digest', lambda _: 'changed')
    else:
        spec['max_updates'] = 5000
    with pytest.raises(RuntimeError):
        recovery.validate_recovery(spec, kind)


def test_declared_v2_needs_receipt_and_cannot_be_extended_to_v3(tmp_path, monkeypatch):
    previous, spec, kind = failed_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='explicit'):
        recovery.validate_recovery({**spec, 'recovery_from': None}, kind)
    with pytest.raises(ValueError, match='six predeclared'):
        recovery.recovery_identity(previous, tmp_path / 'fm_beta_stratified_v3', kind)


def test_downstream_started_smoke_is_not_a_zero_update_recovery(tmp_path, monkeypatch):
    previous, spec, kind = failed_fixture(tmp_path, monkeypatch, 'ki_a4_fulltrain_v2')
    (previous / 'smoke').mkdir()
    with pytest.raises(RuntimeError, match='entered smoke'):
        recovery.validate_recovery(spec, kind)


def test_real_current_process_is_refused_without_signalling_it():
    import os
    with pytest.raises(RuntimeError, match='live or reused'):
        recovery.assert_exited(os.getpid())
