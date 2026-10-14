import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

HERE = Path(__file__).resolve().parents[1] / 'scripts/eval'
SPEC = importlib.util.spec_from_file_location('continuation_draft', HERE / 'package_memlite_continuation.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def rows():
    tasks = [f'task_{i}' for i in range(100)]
    cases = [dict(task=t, instance_id=i, rollout_id=0) for t in tasks for i in range(301, 311)]
    records = [r | dict(success=False, steps=2, q_score=dict(final=.25)) for r in cases[:-4]]
    errors = [r | dict(rerun_allowed=False, diagnosis=dict(kind='planner_memory_recurrence_rejected',
        official_score_available=False)) for r in cases[-4:]]
    return tasks, records, errors


def test_missing_four_keeps_1000_denominator_and_does_not_fabricate_json():
    tasks, records, errors = rows();s = m.accounted_summary(tasks, records, errors)
    assert s['completed']==996 and s['execution_errors']==4 and s['coverage']==1000
    assert s['conservative_q_errors_and_missing_as_zero']==pytest.approx(.249)
    assert s['completed_only_q']==.25 and s['official_full_q_score'] is None
    assert s['fabricated_official_metrics']==0


def test_diagnosed_articulation_failure_remains_a_disclosed_missing_result():
    tasks,records,errors=rows()
    for error in errors:error['diagnosis']['kind']='simulator_articulation_unavailable'
    s=m.accounted_summary(tasks,records,errors)
    assert s['execution_errors']==4 and s['coverage']==1000
    assert s['official_full_q_score'] is None and s['fabricated_official_metrics']==0


@pytest.mark.parametrize('bad', ['incomplete','duplicate','overlap','foreign','unknown','retry','claimed_score'])
def test_diagnosed_missing_is_not_a_loophole_for_early_stopping_or_cherry_picking(bad):
    tasks, records, errors = rows()
    if bad=='incomplete': records.pop()
    elif bad=='duplicate': errors.append(errors[0])
    elif bad=='overlap':errors[0].update({k:records[0][k] for k in ('task','instance_id','rollout_id')})
    elif bad=='foreign':errors[0]['task']='foreign'
    elif bad=='unknown':errors[0]['diagnosis']['kind']='unclassified'
    elif bad=='retry':errors[0]['rerun_allowed']=True
    else:errors[0]['diagnosis']['official_score_available']=True
    with pytest.raises(ValueError):m.accounted_summary(tasks,records,errors)


def fixture(tmp_path, monkeypatch):
    """Synthetic raw-byte ZIP fixture; never an actual physics acceptance."""
    tasks,records,errors=rows();job=tmp_path/'job';job.mkdir()
    originals=[]
    for row in records:
        name=f'{row["task"]}_{row["instance_id"]}_0.json'
        originals.append((name,json.dumps(row,indent=3).encode()+b'\n'))
    value=dict(originals=originals, files={'rgb_wrapper.py':b'# original wrapper','r1pro.yaml':b'model: r1pro'},
        inventory=[],errors=errors,commands=[dict(gpu='0',argv=['bash','/frozen/launch_sim.sh','/frozen/run_task.py'])],
        summary=m.accounted_summary(tasks,records,errors),manifest=dict(runtime_commit='actor',config_sha256='config',
        pack_manifest_sha256='assets'),root_manifest=dict(path='/original/manifest.json',sha256='root'),lineage=[job])
    monkeypatch.setattr(m,'audited_materials',lambda *a:value)
    return job,tmp_path/'draft',value


def test_archive_contains_only_actual_original_json_and_discloses_missing(tmp_path,monkeypatch):
    job,out,data=fixture(tmp_path,monkeypatch)
    receipt=m.package(job,'manifest-sha',out)
    assert receipt['metrics']==996 and receipt['missing']==4 and not receipt['model_delivery_ready']
    assert receipt['submission_q_missing_as_zero']==pytest.approx(.249)
    with zipfile.ZipFile(receipt['output']) as z:
        assert len([n for n in z.namelist() if n.endswith('.json')])==996
        assert 'execution-errors.json' not in z.namelist() and 'summary.json' not in z.namelist()
        for name,raw in data['originals']:assert z.read('json/'+name)==raw
        assert b'4 missing official results' in z.read('README.md')
    assert json.loads((out/'execution-errors.json').read_text())==data['errors']
    assert receipt['submitted'] is False and receipt['fabricated_metrics']==0
    with pytest.raises(ValueError):m.package(job,'manifest-sha',out)


def test_reject_output_inside_original_and_duplicate_names(tmp_path,monkeypatch):
    job,out,data=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):m.package(job,'manifest-sha',job/'draft')
    data['originals'].append(data['originals'][0])
    with pytest.raises(ValueError):m.package(job,'manifest-sha',out)
    assert not out.exists()


def test_real_entry_refuses_running_state_before_any_draft(tmp_path):
    job=tmp_path/'running';job.mkdir()
    (job/'manifest.json').write_text(json.dumps(dict(schema='memlite_final_explicit_continuation_v1')))
    (job/'status.json').write_text(json.dumps(dict(status='running')))
    with pytest.raises(ValueError,match='not terminal'):
        m.package(job,m.c.sha256(job/'manifest.json'),tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


@pytest.mark.parametrize('partial',[False,True])
def test_full_materials_verifies_original_lineage_physics_config_and_source(tmp_path,monkeypatch,partial):
    # Reuse synthetic original 1000-case fixture; no GPU or fake real-video claim.
    spec=importlib.util.spec_from_file_location('base_fixture',Path(__file__).with_name('test_package_final_public_eval.py'))
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    root,_,_=base.prepared(tmp_path)
    source=tmp_path/'frozen';source.mkdir()
    actor=source/'scripts/eval/memlite_final';actor.mkdir(parents=True)
    other=source/'scripts/eval/memlite_sft100';other.mkdir()
    for name in ('client.py','lifecycle.py','run_task.py'):(actor/name).write_text('# frozen '+name)
    (source/'scripts/eval/full_episode_client.py').write_text('# frozen client')
    (other/'wire.py').write_text('# frozen wire')
    (other/'rgb_wrapper.py').write_bytes((root/'submission/rgb_wrapper.py').read_bytes())
    base.save(other/'official_manifest.json',dict(files={'omnigibson/eval/r1pro.yaml':m.c.sha256(root/'submission/r1pro.yaml')}))
    config=tmp_path/'config.json';config.write_text('{}')
    original=m.c.read(root/'manifest.json');original.update(config_path=str(config),config_sha256=m.c.sha256(config),pack='assets',pack_manifest_sha256='pack')
    base.save(root/'manifest.json',original)
    inventory=[];errors=[]
    for part in (root/'parts').iterdir():
        ps=m.c.read(part/'status.json');i=int(ps['task'].split('_')[-1]);ps.update(source_commit='frozen',official_commit='official',
            config_sha256=original['config_sha256'],task_id=i,indices=list(range(10)),num_envs=2)
        base.save(part/'status.json',ps)
        cmd=root/'workers/gpu_0'/(part.name+'-command.json');cv=m.c.read(cmd)
        cv['argv'][1:3]=[str(other/'launch_sim.sh'),str(actor/'run_task.py')]
        cv['argv'][3:3]=['--config-sha256',original['config_sha256']];base.save(cmd,cv)
        if partial and i==99:
            # Synthetic original interrupted last two-instance batch: there
            # never were valid JSON results for those two cases.
            base.save(part/'attempts/00.json',dict(status='completed',instance_ids=list(range(301,309)),rollout=0,actual_applied_controls=[2]*8))
            base.save(part/'attempts/08.json',dict(status='running',instance_ids=[309,310],rollout=0))
            log=root/'workers/gpu_0/task_99.log';log.write_text('synthetic diagnosed rejection')
            for instance in (309,310):
                (part/'json'/f'task_99_{instance}_0.json').unlink()
                errors.append(dict(task='task_99',instance_id=instance,rollout_id=0,rerun_allowed=False,
                    diagnosis=dict(kind='planner_memory_recurrence_rejected',official_score_available=False),
                    original_attempt=m.c.binding(part/'attempts/08.json'),log=m.c.binding(log)))
        for path,row in m.c.completed_metrics(part):
            video=part/'videos'/(path.stem+'.mp4')
            inventory.append({k:row[k] for k in ('task','instance_id','rollout_id','steps')}|
                dict(metrics=m.c.binding(path),video=m.c.binding(video),video_probe=dict(nb_frames='2',r_frame_rate='30/1',width=672,height=448)))
    seal=tmp_path/'seal.json';base.save(seal,dict(records=inventory,execution_errors=errors,evidence=[]))
    job=tmp_path/'continued';(job/'parts').mkdir(parents=True)
    manifest={k:original[k] for k in ('tasks','cases','config_path','config_sha256','pack','pack_manifest_sha256')}
    manifest.update(schema='memlite_final_explicit_continuation_v1',runtime_source=str(source),runtime_commit='frozen',prior=str(root),
        prior_manifest=m.c.binding(root/'manifest.json'),root_manifest=m.c.binding(root/'manifest.json'),seal=m.c.binding(seal),
        policy_changed=False,optimizer_updates=0)
    base.save(job/'manifest.json',manifest)
    base.save(job/'status.json',dict(status='completed_with_execution_errors' if partial else 'completed',
        finished=1,pid=None,worker_pids={},completed_gpus=list(range(8)),failed_gpus={}))
    for i in range(8):base.save(job/f'workers/gpu_{i}/status.json',dict(status='completed',finished=1,pid=None,model_generations=[]))
    base.save(job/'inventory.json',inventory);base.save(job/'execution-errors.json',errors)
    records=[m.c.read(v['metrics']['path']) for v in inventory]
    base.save(job/'summary.json',m.c.summary_from_rows(original['tasks'],records,errors))
    monkeypatch.setattr(m.c,'frozen_source',lambda *a:None)
    monkeypatch.setattr(m.c.subprocess,'check_output',lambda *a,**kw:json.dumps({'streams':[dict(nb_frames='2',r_frame_rate='30/1',width=672,height=448)]}))
    data=m.audited_materials(job,m.c.sha256(job/'manifest.json'))
    assert len(data['originals'])==(998 if partial else 1000) and len(data['commands'])==100
    assert len(data['errors'])==(2 if partial else 0)
    metric=Path(inventory[0]['metrics']['path']);metric.write_text(metric.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):m.audited_materials(job,m.c.sha256(job/'manifest.json'))
