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


def replacement_fixture(tmp_path):
    import json
    from queue_champion_after_memlite import sha
    job=tmp_path/'champion';mem=tmp_path/'mem-next';prior=tmp_path/'mem-old';queue=job/'queue'
    for p in (queue,mem,prior):p.mkdir(parents=True)
    (job/'manifest.json').write_text('{}');(prior/'manifest.json').write_text('{}')
    (prior/'status.json').write_text(json.dumps(dict(status='needs_diagnosis')))
    (mem/'manifest.json').write_text(json.dumps(dict(prior=str(prior),
        prior_manifest=dict(path=str(prior/'manifest.json'),sha256=sha(prior/'manifest.json')))))
    old=dict(status='needs_diagnosis',finished=10,pid=98765000,phases=[],champion_job=str(job),mem_job=str(prior),
        error="RuntimeError('MEM evaluation needs diagnosis; no automatic reroll or GPU takeover')",
        bindings={str(p/'manifest.json'):sha(p/'manifest.json') for p in (job,prior)})
    (queue/'status.json').write_text(json.dumps(old))
    return job,mem,queue


def test_replace_only_dead_dependency_queue_and_preserve_evidence(tmp_path):
    from queue_champion_after_memlite import queue_location,sha
    job,mem,previous=replacement_fixture(tmp_path)
    queue,pin=queue_location(job,mem,'queue-resume-v3',previous,completed_preflight=True,continuation=True)
    assert queue==job/'queue-resume-v3' and not queue.exists()
    assert pin==dict(path=str(previous/'status.json'),sha256=sha(previous/'status.json'))
    assert previous.is_dir()


@pytest.mark.parametrize('bad',['live','phase_started','other_failure','changed_parent','default','escape','no_preflight'])
def test_queue_replacement_rejects_implicit_reroll_or_live_predecessor(tmp_path,bad):
    import json,os
    from queue_champion_after_memlite import queue_location
    job,mem,previous=replacement_fixture(tmp_path);status=previous/'status.json'
    row=json.loads(status.read_text());name='queue-next';completed=True
    if bad=='live':row['pid']=os.getpid()
    elif bad=='phase_started':row['phases']=[dict(name='public1000',status='running')]
    elif bad=='other_failure':row['error']='out of memory'
    elif bad=='changed_parent':(tmp_path/'mem-old/manifest.json').write_text('{"changed":true}')
    elif bad=='default':name='queue'
    elif bad=='escape':name='../queue-next'
    else:completed=False
    status.write_text(json.dumps(row))
    with pytest.raises(ValueError):queue_location(job,mem,name,previous,completed_preflight=completed,continuation=True)
