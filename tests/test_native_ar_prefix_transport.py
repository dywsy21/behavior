from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from native_ar_prefix_client import (OBSERVABLE_FIELDS, task_only_request,
                                     validate_native_identity, validate_response)
from serve_native_ar_prefix import NativeARSession


def observation():
    return {key: 'observable-only-test-value' for key in OBSERVABLE_FIELDS}


def hello():
    return dict(mode='native_task_ar_history6', components=['native_task_actor'], identity=dict(
        kind='native_task_ar500_prefix_history6_service', checkpoint_sha256='native-sha',
        initial_action_count=448, prefix_window_sha256='window-sha', num_obs_steps=6,
        execution_start=0, max_chunks=16, static_format_forced=True, teacher_or_planner_in_actor=False))


def test_wire_request_does_not_require_or_send_an_evaluator_subgoal():
    request = task_only_request(observation(), 16)
    assert set(request) == {'kind', 'observation', 'execute_steps'}
    assert request['kind'] == 'native_task_chunk'
    for field in ('installed_subgoal', 'action', 'outcome', 'source_episode', 'success'):
        with pytest.raises(ValueError):
            task_only_request({**observation(), field: 'forbidden'}, 16)


@pytest.mark.parametrize('damage', [None, 'fm', 'parent', 'prefix', 'schema', 'horizon', 'budget', 'teacher'])
def test_native_endpoint_is_not_a_fake_fm_or_unbounded_route(damage):
    message = hello()
    if damage == 'fm':
        message['mode'] = 'native_skill_fm_history6'
    elif damage == 'parent':
        message['identity']['checkpoint_sha256'] = 'a4-sha'
    elif damage == 'prefix':
        message['identity']['initial_action_count'] = 0
    elif damage == 'schema':
        message['identity']['static_format_forced'] = False
    elif damage == 'horizon':
        message['identity']['execution_start'] = 5
    elif damage == 'budget':
        message['identity']['max_chunks'] = 80
    elif damage == 'teacher':
        message['identity']['teacher_or_planner_in_actor'] = True
    args = dict(checkpoint_sha='native-sha', prefix_actions=448, window_sha='window-sha')
    if damage:
        with pytest.raises(ValueError):
            validate_native_identity(message, **args)
    else:
        assert validate_native_identity(message, **args) == message['identity']


def session_fixture():
    commits, calls, seeds, records = [], [], [], []
    admission = dict(consumed_actions=448, execute_steps=16, anchor_hashes={448: 'sha'})
    ingress = SimpleNamespace(prepare=lambda raw, **kwargs: (raw, admission), commit=commits.append)
    def infer(observed):
        calls.append(observed)
        return dict(actions=np.zeros((16, 23), dtype=np.float32),
                    schema=dict(complete=True, rule_safe_clamp=False), timing={})
    session = NativeARSession(identity=hello()['identity'], infer=infer, ingress=ingress,
        seed_rng=seeds.append, publish_chunk=lambda *args: records.append(args))
    return session, commits, calls, seeds, records


def test_session_has_one_rng_begin_and_sixteen_total_neural_attempts():
    session, commits, calls, seeds, records = session_fixture()
    owner = object()
    session.begin(owner, dict(kind='begin', seed=17))
    for _ in range(16):
        response = session.chunk(owner, task_only_request(observation(), 16))
    assert len(calls) == len(commits) == len(records) == 16 and seeds == [17]
    validate_response(response, checkpoint_sha='native-sha', count=448, anchor_hashes={'448': 'sha'})
    with pytest.raises(ValueError):
        session.chunk(owner, task_only_request(observation(), 16))
    with pytest.raises(ValueError):
        session.begin(object(), dict(kind='begin', seed=17))


def test_other_connection_and_teacher_fields_fail_before_neural_call():
    session, commits, calls, _, _ = session_fixture()
    owner = object()
    session.begin(owner, dict(kind='begin', seed=17))
    with pytest.raises(ValueError):
        session.chunk(object(), task_only_request(observation(), 16))
    with pytest.raises(ValueError):
        session.chunk(owner, {**task_only_request(observation(), 16), 'installed_subgoal': 'oracle'})
    assert not calls and not commits


def test_failed_neural_output_consumes_attempt_but_never_commits_history():
    session, commits, _, _, records = session_fixture()
    owner = object()
    session.begin(owner, dict(kind='begin', seed=17))
    session.infer = lambda _: dict(actions=np.zeros((16, 27), dtype=np.float32), schema={}, timing={})
    with pytest.raises(ValueError):
        session.chunk(owner, task_only_request(observation(), 16))
    assert session.calls == 1 and not commits and not records


@pytest.mark.parametrize('field,value', [('execution_start', 5), ('teacher_or_planner_in_actor', True),
                                       ('actions', np.zeros((16, 23), dtype=np.float64))])
def test_client_rejects_changed_actual_control_protocol(field, value):
    session, _, _, _, _ = session_fixture()
    owner = object()
    session.begin(owner, dict(kind='begin', seed=17))
    result = session.chunk(owner, task_only_request(observation(), 16))
    result[field] = value
    with pytest.raises(ValueError):
        validate_response(result, checkpoint_sha='native-sha', count=448, anchor_hashes={'448': 'sha'})


def test_actual_frozen_window_can_reduce_budget_without_changing_prefix_or_semantics(monkeypatch):
    from run_native_ar_prefix_pilot import WORK, HISTORY, ADAPTER
    monkeypatch.setattr(sys, 'path', [str(HISTORY), str(WORK / 'c1_v2'), str(ADAPTER), *sys.path])
    from c1_feedback.official_factory_c1 import load_c1_window_and_context, _verify_window_context
    root = WORK / 'c1_windows_v2_matched/c1v2-matched-t0-train-e121-f448-grasp'
    window, context = load_c1_window_and_context(root / 'window.json', root / 'context.json')
    _verify_window_context(window, context)
    original = window.frozen_window()
    bounded = replace(original, max_chunks=16)
    assert original.max_chunks == 80 and bounded.max_chunks == 16
    assert bounded.execute_steps == 16 and len(bounded.prefix_actions) == 448 and window.seed == 0
    assert all(np.array_equal(x, y) for x, y in zip(original.prefix_actions, bounded.prefix_actions))
    assert context.split == 'train' and context.instance_id == 138
