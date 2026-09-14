from dataclasses import asdict
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
