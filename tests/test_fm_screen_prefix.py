from dataclasses import asdict
import json
import sys

import numpy as np
import pytest

import run_fm_screen_prefix as pilot


def response():
    return dict(actions=np.zeros((16, 23), dtype=np.float32), inference_alignment_admission=dict(
        padded_dimensions=[7, 8, 17, 18], real_control_dimensions=23, image_history_frames=6,
        action_history_steps=0, action_execution_start_index=0, expert_actions_used=False))


@pytest.mark.parametrize('damage', [None, 'shape', 'nan', 'dtype', 'padding', 'start', 'teacher'])
def test_actual_response_is_validated_before_physics(damage):
    value = response()
    if damage == 'shape':
        value['actions'] = np.zeros((16, 27), dtype=np.float32)
    elif damage == 'nan':
        value['actions'][0, 0] = np.nan
    elif damage == 'dtype':
        value['actions'] = value['actions'].astype(np.float64)
    elif damage == 'padding':
        value['inference_alignment_admission']['padded_dimensions'] = [7, 8, 17, 18, 22]
    elif damage == 'start':
        value['inference_alignment_admission']['action_execution_start_index'] = 5
    elif damage == 'teacher':
        value['inference_alignment_admission']['expert_actions_used'] = True
    if damage:
        with pytest.raises(ValueError):
            pilot.validate_response(value)
    else:
        assert pilot.validate_response(value) is value


@pytest.mark.parametrize('name', tuple(pilot.SCREEN))
@pytest.mark.parametrize('damage', [None, 'checkpoint', 'step', 'source', 'load', 'ar'])
def test_socket_identity_requires_actual_full_500_weights(name, damage):
    identity = dict(kind='formal_a2_native_skill_fm_prefix_history6_service', checkpoint_sha256=pilot.SCREEN[name],
        checkpoint_step=500, service_entry_sha256=pilot.SERVICE_SHA, initial_action_count=448,
        prefix_window_sha256='window', num_obs_steps=6, history_context_kind='actual_demo_prefix_then_policy',
        checkpoint_load=dict(full_base_and_adapter_bitwise_equal=True))
    hello = dict(mode='native_skill_fm_history6', components=['low'], identity=identity)
    if damage == 'checkpoint':
        identity['checkpoint_sha256'] = 'parent-or-another-screen'
    elif damage == 'step':
        identity['checkpoint_step'] = 5
    elif damage == 'source':
        identity['service_entry_sha256'] = 'unreviewed-source'
    elif damage == 'load':
        identity['checkpoint_load']['full_base_and_adapter_bitwise_equal'] = False
    elif damage == 'ar':
        hello['mode'] = 'native_task_ar_history6'
    if damage:
        with pytest.raises(ValueError):
            pilot.validate_hello(hello, dict(window_sha256='window'), name)
    else:
        assert pilot.validate_hello(hello, dict(window_sha256='window'), name) == identity


def test_screen_weights_match_completed_generation_comparison():
    from probe_fm_screen_actions import SCREEN
    assert pilot.SCREEN == SCREEN and pilot.RECIPE['total_model_controls'] == 3 * 32 * 16


def test_actual_window_only_changes_the_predeclared_budget(monkeypatch):
    monkeypatch.setattr(sys, 'path', [str(pilot.RUNTIME), str(pilot.WORK / 'c1_v2'), str(pilot.ADAPTER), *sys.path])
    from c1_feedback.official_factory_c1 import load_c1_window_and_context
    root = pilot.WORK / 'c1_windows_v2_matched/c1v2-matched-t0-train-e121-f448-grasp'
    original, context = load_c1_window_and_context(root / 'window.json', root / 'context.json')
    bounded = pilot.bound_window(original, context)
    before, after = asdict(original), asdict(bounded)
    assert before['max_chunks'] == 80 and after.pop('max_chunks') == 32
    before.pop('max_chunks')
    assert before == after
    assert all(np.array_equal(a, b) for a, b in zip(original.frozen_window().prefix_actions,
                                                   bounded.frozen_window().prefix_actions))
    with pytest.raises(ValueError):
        pilot.bound_window(bounded, context)


def test_recovery_ports_exclude_completed_control_and_never_reuse_its_port(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'OUTPUT', tmp_path)
    assert pilot.active_ports() == dict.fromkeys(pilot.SCREEN, 8786)
    (tmp_path / 'recovery.json').write_text('{}')
    ports = pilot.active_ports()
    assert ports == {'fm_beta_stratified_v2': 8787, 'fm_exec_weight2_v2': 8788}
    assert len(set(ports.values())) == 2 and 8786 not in ports.values()


@pytest.mark.parametrize('damage', [None, 'started_arm', 'already_recovered', 'completed_comparison'])
def test_only_previously_unused_arms_may_be_recovered_once(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(pilot, 'OUTPUT', tmp_path)
    for name in pilot.REMAINING_PORTS:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'event_context.json').write_text('{}')
    if damage == 'started_arm':
        (tmp_path / 'fm_beta_stratified_v2/service.launch.json').write_text('{}')
    elif damage == 'already_recovered':
        (tmp_path / 'recovery.json').write_text('{}')
    elif damage == 'completed_comparison':
        (tmp_path / 'completion.json').write_text('{}')
    monkeypatch.setattr(pilot, 'recovery_identity', lambda: {'audited': True})
    validations = []
    monkeypatch.setattr(pilot, 'validate', lambda: validations.append(True))
    if damage:
        with pytest.raises((RuntimeError, FileExistsError)):
            pilot.prepare_recovery()
        assert not validations
    else:
        pilot.prepare_recovery()
        assert json.loads((tmp_path / 'recovery.json').read_text()) == {'audited': True}
        assert validations == [True]
        with pytest.raises(FileExistsError):
            pilot.prepare_recovery()


@pytest.mark.parametrize('damage', [None, 'changed_failure', 'live_process'])
def test_recovery_identity_binds_the_actual_original_failure_and_dead_processes(monkeypatch, damage):
    identities = {
        pilot.OUTPUT / 'manifest.json': pilot.ORIGINAL_MANIFEST_SHA,
        pilot.OUTPUT / 'supervise_failure.json': pilot.PORT_FAILURE_SHA,
        pilot.OUTPUT / 'fm_control_v1/completion.json': pilot.CONTROL_COMPLETION_SHA,
        pilot.OUTPUT / 'fm_control_v1/actual_rollout/collection_result.json': pilot.CONTROL_RESULT_SHA,
    }
    if damage == 'changed_failure':
        identities[pilot.OUTPUT / 'supervise_failure.json'] = 'unrelated-failure'
    monkeypatch.setattr(pilot, 'sha', lambda path: identities.get(path, 'other-sha'))
    monkeypatch.setattr(pilot, 'clean_commit', lambda: 'new-pinned-source')
    monkeypatch.setattr(pilot, 'read', lambda path: dict(pid=4177433, returncode=-15, other_jobs_stopped=False)
                        if path.name == 'private_service_exit.json' else {'pid': 42})
    exits = []
    def assert_exited(pid):
        exits.append(pid)
        if damage == 'live_process':
            raise RuntimeError('Original physics process must have exited')
    monkeypatch.setattr(pilot, 'assert_exited', assert_exited)
    if damage:
        with pytest.raises(RuntimeError):
            pilot.recovery_identity()
    else:
        receipt = pilot.recovery_identity()
        assert exits == [42, 42, 42]
        assert receipt['remaining_model_controls'] == 1024
        assert receipt['skipped_completed_arms'] == ['fm_control_v1']
