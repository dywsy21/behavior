import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO=Path(__file__).resolve().parents[1]
HERE=REPO/'scripts/eval/memlite_final'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('final_supervise',HERE/'supervise.py')
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def receipt(job,manifest,size,throughput):
    parts=[f'/run/preflight/{task}-n{n}-i{indices[0]}'
        for task in ('turning_on_radio','clean_a_keyboard')
        for n,indices in s.episode_portions(size,preflight=True)]
    value=dict(manifest,status='completed',num_envs=size,preflight=True,
        verification=dict(config_sha256=manifest['config_sha256'],weights_unchanged=True,optimizer_updates=0),
        completed_parts=parts,control_steps_per_second=throughput)
    path=job/'preflight'/f'batch{size}'/'status.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    return path,value


@pytest.mark.parametrize('speed4,chosen',[(12.,4),(8.,2)])
def test_equal_eight_train_cases_are_batched_inside_two_task_parts(tmp_path,speed4,chosen):
    manifest=dict(source_commit='source',config_sha256='config',pack_manifest_sha256='pack')
    _,two=receipt(tmp_path,manifest,2,10.)
    _,four=receipt(tmp_path,manifest,4,speed4)
    assert len(two['completed_parts'])==len(four['completed_parts'])==2
    assert s.episode_portions(2,preflight=True)==[(2,[1,2,3,4])]
    assert s.episode_portions(4,preflight=True)==[(4,[1,2,3,4])]
    assert s.select_preflight(tmp_path,manifest)['num_envs']==chosen


def test_incomplete_or_duplicate_workload_is_not_a_speed_result(tmp_path):
    manifest=dict(source_commit='source',config_sha256='config',pack_manifest_sha256='pack')
    path,value=receipt(tmp_path,manifest,2,10.)
    value['completed_parts'][1]=value['completed_parts'][0]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):s.select_preflight(tmp_path,manifest)


def test_submission_readme_pins_identity_without_claiming_delivery():
    from report import reproduction_readme
    value=dict(source_commit='commit123',official_commit='official123',
        config_sha256='config123',pack_manifest_sha256='pack123')
    text=reproduction_readme(value)
    for identity in value.values():assert identity in text
    assert 'NOT a completed model-delivery submission' in text
    assert 'UNKNOWN' in text and 'public indices 0–9' in text
