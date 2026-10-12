import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/rl/memlite_online/code'),str(ROOT/'scripts/eval/memlite_final')]
from assets import relocate_config


def test_recursive_relocation_never_changes_model_hyperparameters():
    source={'arch':{'path':'/registered/models/a','n':10,'relative':'assets/b'},'rows':['/registered/stats.json']}
    exported=relocate_config(source,source_root='/registered')
    assert source['arch']['path']=='/registered/models/a'
    target=relocate_config(exported,target_root='/new')
    assert target=={'arch':{'path':'/new/models/a','n':10,'relative':'assets/b'},'rows':['/new/stats.json']}
    with pytest.raises(ValueError):relocate_config({'path':'/foreign/model'},source_root='/registered')
    with pytest.raises(ValueError):relocate_config('@PACK@/../escape',target_root='/new')


def test_new_model_binding_matches_prior_registered_final_checkpoints():
    base=ROOT/'configs/recovery_sft'
    c=json.loads((base/'memlite_final_full_eval_v1.json').read_text())
    prior=json.loads((base/'periodic_planner_calibrated_pilot_models_v1.json').read_text())
    low=json.loads((base/'a800_native_learner_candidate_probes_heartbeat_v2.json').read_text())
    assert c['normalization_sha256']==prior['normalization_sha256']
    for key in ('observer_backbone','observer_adapter'):
        assert c['models'][key]['sha256']==prior[key]['sha256']
        assert c['models'][key]['path']==prior[key]['path']
    assert c['calibration']==prior['pilot_receipts']['calibration']
    assert c['models']['low']['sha256']==low['model']['sha256']
    assert c['models']['planner']['sha256']=='047152f1ced65db0ecdf9348ced13980fbae3c512e914e14fccfe871e30ed38d'
    assert c['mode']=='public_test' and c['indices']==list(range(10)) and c['rollouts']==1
