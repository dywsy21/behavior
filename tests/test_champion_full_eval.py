import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('champion_final_suite',Path(__file__).resolve().parents[1]/'scripts/eval/champion_full_eval.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def receipt(job,size,speed):
    manifest=dict(source_commit='commit',export_sha256='ema',configs=[dict(sha256='config')])
    r=dict(status='completed',source_commit='commit',export_sha256='ema',config_sha256='config',num_envs=size,
        preflight=True,control_steps_per_second=speed,verification=dict(source_commit='commit',
            config_sha256='config',weights_unchanged=True,optimizer_updates=0),
        completed_parts=[f'/run/{task}-n{size}-i1' for task in ('turning_on_radio','clean_a_keyboard')])
    path=job/'preflight'/f'batch{size}'/'status.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(r))
    return manifest,path,r


@pytest.mark.parametrize('speed,chosen',[(20,4),(5,2)])
def test_choose_measured_speed_not_environment_count(tmp_path,speed,chosen):
    manifest,_,_=receipt(tmp_path,2,10);receipt(tmp_path,4,speed)
    assert m.select_preflight(tmp_path,manifest)['num_envs']==chosen


@pytest.mark.parametrize('bad',['weight','source','parts','config'])
def test_foreign_or_incomplete_preflight_cannot_launch(tmp_path,bad):
    manifest,path,r=receipt(tmp_path,2,10)
    if bad=='weight':r['verification']['weights_unchanged']=False
    elif bad=='source':r['source_commit']='other'
    elif bad=='parts':r['completed_parts']*=2
    else:r['verification']['config_sha256']='other'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.select_preflight(tmp_path,manifest)
