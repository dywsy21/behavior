from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
import queue_memlite_after_champion as m


def complete():
    return (dict(status='completed', failed_gpus={}, completed_gpus=list(range(8)), finished=10,
                 q_score=.1, success_rate=.02),
            dict(status='complete', completed=1000, missing=0, optimizer_updates=0,
                 original_outputs_unmodified=True, official_q_score=.1, official_sr=.02))


def test_wait_for_whole_champion_instead_of_first_done_worker():
    state, summary = complete()
    assert not m.champion_ready(state | dict(status='running'), summary)
    assert m.champion_ready(state, summary)


@pytest.mark.parametrize('status', ['needs_diagnosis', 'failed', 'failed_no_retry'])
def test_champion_failure_cannot_release_mem(status):
    with pytest.raises(RuntimeError):
        m.champion_ready(dict(status=status), {})


@pytest.mark.parametrize('bad', ['partial', 'modified', 'updated', 'wrong_q', 'missing',
                                 'live_worker', 'failed_worker', 'not_finished'])
def test_incomplete_or_changed_evidence_cannot_release_mem(bad):
    state, summary = complete()
    if bad == 'partial': summary['completed'] = 999
    elif bad == 'modified': summary['original_outputs_unmodified'] = False
    elif bad == 'updated': summary['optimizer_updates'] = 1
    elif bad == 'wrong_q': summary['official_q_score'] = .9
    elif bad == 'missing': summary['missing'] = 1
    elif bad == 'live_worker': state['completed_gpus'].pop()
    elif bad == 'failed_worker': state['failed_gpus'] = {'7': {}}
    else: state.pop('finished')
    with pytest.raises(ValueError):
        m.champion_ready(state, summary)


def test_queue_rechecks_real_terminal_evidence_before_gpu_start(tmp_path):
    import json
    state, summary = complete()
    (tmp_path/'status.json').write_text(json.dumps(state | dict(status='running')))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='not complete'):
        m.validate_completed_champion(tmp_path, {})


def sealed_fixture(tmp_path):
    import importlib.util
    import json
    import zipfile
    spec = importlib.util.spec_from_file_location('queue_pack_fixture',
        Path(__file__).with_name('test_package_final_public_eval.py'))
    pack = importlib.util.module_from_spec(spec); spec.loader.exec_module(pack)
    job, _, _ = pack.prepared(tmp_path, 'champion_final_public100_run_v1')
    manifest = m.read(job/'manifest.json')
    manifest.update(export_sha256='export',configs=[dict(sha256=f'config{i}') for i in range(8)])
    pack.save(job/'manifest.json',manifest)
    summary = m.read(job/'summary.json')
    summary.update(optimizer_updates=0,original_outputs_unmodified=True)
    pack.save(job/'summary.json',summary)
    state, _ = complete()
    state.update(pid=98767000,worker_pids={str(i):98767001+i for i in range(8)},
                 q_score=summary['official_q_score'],success_rate=summary['official_sr'])
    pack.save(job/'status.json',state)
    for i in range(8):
        pack.save(job/f'workers/gpu_{i}/status.json',dict(status='completed',finished=10,pid=98767001+i,
            source_commit='frozen',export_sha256='export',verification=dict(weights_unchanged=True,
            optimizer_updates=0,source_commit='frozen',config_sha256=f'config{i}')))
    archive=job/'submission/evaluation_outputs.zip'
    with zipfile.ZipFile(archive) as stream:
        members={name:stream.read(name) for name in stream.namelist()}
    for name in ('manifest.json','summary.json'):members[name]=(job/name).read_bytes()
    with zipfile.ZipFile(archive,'w') as stream:
        for name,content in members.items():stream.writestr(name,content)
    receipt=m.read(job/'submission/package-status.json');receipt['zip_sha256']=m.sha256(archive)
    pack.save(job/'submission/package-status.json',receipt)
    return job,manifest


def test_full_original_terminal_gate_passes_without_rewriting_or_reroll(tmp_path):
    job,manifest=sealed_fixture(tmp_path)
    before=m.binding(job/'summary.json')
    result=m.validate_completed_champion(job,manifest)
    assert result['commands']==100 and result['originals']==1000
    assert result['gpu_takeover'] is False and result['videos_rehashed_by_this_queue'] is False
    assert before==m.binding(job/'summary.json')


@pytest.mark.parametrize('bad',['live_supervisor','live_worker','bad_weights','changed_metric',
                               'missing_video','tampered_archive'])
def test_terminal_gate_rejects_live_or_tampered_original_evidence(tmp_path,bad):
    import json, os
    job,manifest=sealed_fixture(tmp_path)
    if bad=='live_supervisor':
        path=job/'status.json';row=m.read(path);row['pid']=os.getpid();path.write_text(json.dumps(row))
    elif bad in ('live_worker','bad_weights'):
        path=job/'workers/gpu_7/status.json';row=m.read(path)
        if bad=='live_worker':row['pid']=os.getpid()
        else:row['verification']['weights_unchanged']=False
        path.write_text(json.dumps(row))
    elif bad=='changed_metric':
        path=job/'parts/task_0/json/task_0_301_0.json';row=m.read(path)
        row['q_score']['final']=.5;path.write_text(json.dumps(row))
    elif bad=='missing_video':(job/'parts/task_0/videos/task_0_301_0.mp4').unlink()
    else:(job/'submission/evaluation_outputs.zip').write_bytes(b'not-original')
    with pytest.raises(ValueError):m.validate_completed_champion(job,manifest)


def test_reverse_queue_rejects_unrelated_authorization_or_seal(tmp_path,monkeypatch):
    import json
    mem=tmp_path/'mem';champion=tmp_path/'champion';mem.mkdir();champion.mkdir()
    prior=tmp_path/'prior';prior.mkdir();(prior/'manifest.json').write_text('{}')
    (champion/'manifest.json').write_text(json.dumps(dict(schema='champion_final_public100_run_v1',
        optimizer_updates=0,automatic_retries=False)))
    auth=tmp_path/'authorization.json'
    auth.write_text(json.dumps(dict(schema='user_authorized_eval_priority_stop_v1',mem_job=str(prior),
        champion_job=str(champion),mem_manifest_sha256=m.sha256(prior/'manifest.json'),
        champion_manifest_sha256=m.sha256(champion/'manifest.json'))))
    seal=tmp_path/'seal.json';seal.write_text(json.dumps(dict(priority_authorization=m.binding(auth))))
    mm=dict(prior=str(prior),prior_manifest=m.binding(prior/'manifest.json'),seal=m.binding(seal),
            automatic_retries=False,optimizer_updates=0,policy_changed=False)
    monkeypatch.setattr(m,'registration',lambda path:(mem,mm))
    assert m.validate_order(mem,champion,auth)[0]==mem
    seal.write_text(json.dumps(dict(priority_authorization=dict(path=str(auth),sha256='wrong'))))
    with pytest.raises(ValueError):m.validate_order(mem,champion,auth)
