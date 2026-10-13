import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
from completed_metrics import completed_metrics


def fixture(path):
    (path/'json').mkdir();(path/'attempts').mkdir()
    (path/'status.json').write_text(json.dumps(dict(task='radio',status='running')))
    a=path/'attempts/00.json';m=path/'json/radio_301_0.json'
    row=dict(status='running',instance_ids=[301],rollout=0,actual_applied_controls=[17])
    a.write_text(json.dumps(row));m.write_text('{')
    return a,m,row


def test_mid_write_is_ignored_until_original_physical_batch_completes(tmp_path):
    a,m,row=fixture(tmp_path)
    assert completed_metrics(tmp_path)==[]
    row['status']='completed';a.write_text(json.dumps(row))
    with pytest.raises(json.JSONDecodeError):completed_metrics(tmp_path)
    value=dict(task='radio',instance_id=301,rollout_id=0,steps=17)
    m.write_text(json.dumps(value))
    assert completed_metrics(tmp_path)==[(m,value)]
    value['steps']=18;m.write_text(json.dumps(value))
    with pytest.raises(ValueError):completed_metrics(tmp_path)


def test_final_never_silently_drops_incomplete_or_extra_results(tmp_path):
    a,m,row=fixture(tmp_path)
    with pytest.raises(ValueError):completed_metrics(tmp_path,final=True)
    row['status']='completed';a.write_text(json.dumps(row))
    m.write_text(json.dumps(dict(task='radio',instance_id=301,rollout_id=0,steps=17)))
    (tmp_path/'status.json').write_text(json.dumps(dict(task='radio',status='completed')))
    assert len(completed_metrics(tmp_path,final=True))==1
    (tmp_path/'json/extra.json').write_text('{}')
    with pytest.raises(ValueError):completed_metrics(tmp_path,final=True)
