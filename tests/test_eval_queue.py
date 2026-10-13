from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
from queue_champion_after_memlite import dependency_ready


def test_only_full_successful_completion_releases_next_owned_gpu_job():
    assert not dependency_ready(dict(status='running'),dict(completed=999))
    assert dependency_ready(dict(status='completed'),dict(status='complete',completed=1000,missing=0))
    with pytest.raises(ValueError):dependency_ready(dict(status='completed'),dict(status='complete',completed=999,missing=1))


@pytest.mark.parametrize('state',['needs_diagnosis','failed_no_retry','failed'])
def test_failure_is_not_permission_to_reroll_or_take_over(state):
    with pytest.raises(RuntimeError):dependency_ready(dict(status=state),{})


def test_explicit_continuation_may_finish_with_disclosed_execution_errors():
    state=dict(status='completed_with_execution_errors')
    summary=dict(status=state['status'],completed=996,execution_errors=4,coverage=1000,
        unaccounted=0,missing=4,policy_changed=False,optimizer_updates=0,
        official_full_q_score=None,official_full_sr=None)
    assert dependency_ready(state,summary,continuation=True)
    # No opt-in, no interpretation of an error-bearing run as original1000.
    assert not dependency_ready(state,summary)
    for changes in ({'completed':992},{'coverage':999},{'unaccounted':1},{'missing':0},
                    {'policy_changed':True},{'optimizer_updates':1},{'official_full_q_score':0.},
                    {'official_full_sr':0.},{'execution_errors':True}):
        with pytest.raises(ValueError):dependency_ready(state,summary|changes,continuation=True)
    with pytest.raises(RuntimeError):dependency_ready(dict(status='needs_diagnosis'),summary,continuation=True)


def test_preflight_flag_never_accepts_missing_or_failed_gates(tmp_path):
    from queue_champion_after_memlite import existing_preflight
    import json
    with pytest.raises(ValueError):existing_preflight(tmp_path,tmp_path)
    p=tmp_path/'preflight/batch2/status.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(status='failed_no_retry')))
    with pytest.raises(ValueError):existing_preflight(tmp_path,tmp_path)
