import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
import queue_memlite_after_champion_continuation as m


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    mem, root, champion, prior = [tmp_path / n for n in ('mem', 'champion-root', 'champion-resume', 'mem-prior')]
    auth = tmp_path / 'authorization.json'; ms, cs = tmp_path / 'mem-source', tmp_path / 'champion-source'
    original = dict(automatic_retries=False); save(root / 'manifest.json', original); save(prior / 'manifest.json', {})
    save(auth, dict(schema='user_authorized_eval_priority_stop_v1', mem_job=str(prior), champion_job=str(root),
        mem_manifest_sha256=m.c.sha256(prior/'manifest.json'), champion_manifest_sha256=m.c.sha256(root/'manifest.json')))
    seal = tmp_path/'mem-seal.json'; save(seal, dict(priority_authorization=m.c.binding(auth)))
    mm = dict(prior=str(prior), prior_manifest=m.c.binding(prior/'manifest.json'), seal=m.c.binding(seal),
        automatic_retries=False, policy_changed=False, optimizer_updates=0, source_commit='original-mem-coordinator')
    save(mem/'manifest.json', mm)
    champ_seal = tmp_path/'champion-seal.json'; save(champ_seal, {})
    cm = dict(seal=m.c.binding(champ_seal), source_commit='actual-champion-coordinator'); save(champion/'manifest.json', cm)
    old = mem/'queue-after-champion'
    save(old/'status.json', dict(status='needs_diagnosis', finished=1, phases=[], pid=99999000,
        mem_job=str(mem), champion_job=str(root), error="RuntimeError('Champion needs diagnosis; do not launch MEM or retry champion')",
        bindings=[m.c.binding(mem/'manifest.json'), m.c.binding(root/'manifest.json'), m.c.binding(auth), mm['seal']]))
    def mem_registration(job, *, coordinator_source):
        assert Path(job)==mem and Path(coordinator_source)==ms
        return mem, m.c.read(mem/'manifest.json')
    def champ_registration(job, *, coordinator_source):
        assert Path(job)==champion and Path(coordinator_source)==cs
        return champion, m.c.read(champion/'manifest.json')
    monkeypatch.setattr(m.c.shared, 'registration', mem_registration)
    monkeypatch.setattr(m.c, 'registration', champ_registration)
    monkeypatch.setattr(m.c.shared, 'root_registration', lambda _: (prior, {}, m.c.binding(prior/'manifest.json')))
    monkeypatch.setattr(m.c, 'root_registration', lambda _: (root, original, m.c.binding(root/'manifest.json')))
    return mem, champion, auth, ms, cs, old, root


def test_same_authorization_binds_root_and_new_continuation_without_repinning_mem(tmp_path, monkeypatch):
    mem, champ, auth, ms, cs, old, root = fixture(tmp_path, monkeypatch)
    assert m.validate_order(mem, champ, auth, ms, cs)[5] == root
    queue, binding = m.replacement_queue(mem, root, old, 'queue-resume-v1', auth)
    assert queue == mem/'queue-resume-v1' and binding == m.c.binding(old/'status.json')
    assert m.mem_command(mem, ms)[1] == str(ms/'scripts/eval/continue_memlite_final.py')


@pytest.mark.parametrize('bad', ['auth_champion', 'auth_mem', 'seal', 'trained', 'policy_changed', 'retry'])
def test_reject_unrelated_authorization_and_modified_policy(tmp_path, monkeypatch, bad):
    mem, champ, auth, ms, cs, old, root = fixture(tmp_path, monkeypatch)
    path = auth if bad.startswith('auth') else mem/'manifest.json'; value=m.c.read(path)
    if bad=='auth_champion':value['champion_job']=str(champ)
    elif bad=='auth_mem':value['mem_manifest_sha256']='wrong'
    elif bad=='seal':
        path=Path(value['seal']['path']);value={'priority_authorization':{'path':str(auth),'sha256':'wrong'}}
    elif bad=='trained':value['optimizer_updates']=1
    elif bad=='policy_changed':value['policy_changed']=True
    else:value['automatic_retries']=True
    save(path,value)
    with pytest.raises(ValueError):m.validate_order(mem,champ,auth,ms,cs)


@pytest.mark.parametrize('bad', ['live', 'launched', 'unknown', 'binding', 'different_mem', 'existing', 'mem_started', 'stop', 'path'])
def test_never_replace_live_started_or_unclassified_queue(tmp_path, monkeypatch, bad):
    mem, champ, auth, ms, cs, old, root = fixture(tmp_path, monkeypatch)
    value=m.c.read(old/'status.json');name='queue-resume-v1'
    if bad=='live':value['pid']=os.getpid()
    elif bad=='launched':value['phases']=[{'pid':99999}]
    elif bad=='unknown':value['error']='Unknown'
    elif bad=='binding':value['bindings'][0]['sha256']='changed'
    elif bad=='different_mem':value['mem_job']=str(tmp_path/'other')
    elif bad=='existing':(mem/name).mkdir()
    elif bad=='mem_started':save(mem/'status.json', {'status':'running'})
    elif bad=='stop':(mem/'STOP').touch()
    else:name='../outside'
    save(old/'status.json',value)
    with pytest.raises(ValueError):m.replacement_queue(mem,root,old,name,auth)


def finished():
    state=dict(status='completed_with_execution_errors', finished=20, pid=99999003,
        worker_pids={}, completed_gpus=list(range(8)), failed_gpus={})
    summary=dict(status=state['status'], completed=996, execution_errors=4, coverage=1000, unaccounted=0,
        missing=4, policy_changed=False, optimizer_updates=0, official_full_q_score=None, official_full_sr=None,
        original_outputs_unmodified=True, fabricated_official_metrics=0, automatic_rerolls=0)
    return state, summary


@pytest.mark.parametrize('bad', [None, 'failed_champion', 'unaccounted', 'live_gpu', 'audit', 'mem_failure', 'wait_then_finish'])
def test_queue_waits_audits_then_launches_only_original_mem_source(tmp_path, monkeypatch, bad):
    mem, champ, auth, ms, cs, old, root = fixture(tmp_path, monkeypatch)
    state, summary=finished()
    if bad=='failed_champion':state['status']='needs_diagnosis'
    if bad=='unaccounted':summary['unaccounted']=1
    save(champ/'status.json',state);save(champ/'summary.json',summary)
    if bad=='wait_then_finish':
        save(champ/'status.json',state|dict(status='running',pid=os.getpid()))
        def sleep(seconds):
            assert seconds==30
            save(champ/'status.json',state)
        monkeypatch.setattr(m.time,'sleep',sleep)
    outputs=tmp_path/'outputs';outputs.mkdir();calls=[]
    monkeypatch.setattr(m.c, 'frozen_source', lambda *a:None)
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:
        ('queue-commit\n' if args[0]=='git' else '777\n' if bad=='live_gpu' else ''))
    def audit(*args):
        calls.append('audit_champion')
        if bad=='audit':raise ValueError('tampered original')
        return {'originals_modified':False}
    monkeypatch.setattr(m.pack,'package',audit)
    monkeypatch.setattr(m.mem_pack,'package',lambda *a:calls.append('audit_mem') or {'submitted':False})
    class Process:
        pid=99999010
        def __init__(self,argv,**kw):
            assert calls==['audit_champion']
            assert argv==m.mem_command(mem,ms) and kw['cwd']==ms
            calls.append('launch_original_mem')
        def wait(self):
            st,sm=finished();save(mem/'status.json',st);save(mem/'summary.json',sm)
            return 1 if bad=='mem_failure' else 0
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    before=m.c.binding(old/'status.json')
    args=(mem,champ,auth,ms,cs,old,'queue-resume-v1',outputs)
    if bad not in (None,'wait_then_finish'):
        with pytest.raises((ValueError,RuntimeError)):m.run(*args)
    else:m.run(*args)
    q=m.c.read(mem/'queue-resume-v1/status.json')
    assert before==m.c.binding(old/'status.json')
    if bad in ('failed_champion','unaccounted'):assert calls==[] and q['phases']==[]
    elif bad in ('live_gpu','audit'):assert calls==['audit_champion'] and q['phases']==[]
    elif bad=='mem_failure':assert calls==['audit_champion','launch_original_mem']
    else:assert calls==['audit_champion','launch_original_mem','audit_mem'] and q['status'].startswith('both_internal')
    if bad not in (None,'wait_then_finish'):assert q['status']=='needs_diagnosis'


def test_terminal_gate_rejects_only_seven_completed_workers():
    state, summary=finished();state['completed_gpus'].pop()
    with pytest.raises(ValueError):m.pack.terminal_ready(state,summary)


@pytest.mark.parametrize('bad',[None,'prior','phase','ancestor','live'])
def test_replace_only_exact_dead_continuation_queue(tmp_path,monkeypatch,bad):
    mem,champ,auth,ms,cs,old,root=fixture(tmp_path,monkeypatch)
    previous=mem/'queue-after-resume';newchamp=tmp_path/'new-champ'
    save(newchamp/'manifest.json',dict(prior=str(champ)))
    row=dict(status='needs_diagnosis',finished=2,phases=[],pid=99999001,
        mem_job=str(mem),champion_job=str(champ),champion_root=str(root),
        error="RuntimeError('Champion continuation needs diagnosis; do not start MEM or retry')",
        bindings=[m.c.binding(mem/'manifest.json'),m.c.binding(champ/'manifest.json'),m.c.binding(auth),
            m.c.read(mem/'manifest.json')['seal'],m.c.read(champ/'manifest.json')['seal'],
            m.c.binding(root/'manifest.json'),m.c.binding(old/'status.json')])
    if bad=='prior':save(newchamp/'manifest.json',dict(prior=str(root)))
    elif bad=='phase':row['phases']=[{'name':'remaining_mem'}]
    elif bad=='ancestor':row['bindings'][-1]['sha256']='bad'
    elif bad=='live':row['pid']=os.getpid()
    save(previous/'status.json',row)
    args=(mem,root,previous,'queue-resume-v2',auth,newchamp)
    if bad:
        with pytest.raises(ValueError):m.replacement_queue(*args)
    else:
        q,pin=m.replacement_queue(*args)
        assert q==mem/'queue-resume-v2' and pin==m.c.binding(previous/'status.json')
