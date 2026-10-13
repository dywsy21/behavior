from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/eval'))
from prepare_champion_full_eval import require_terminal_export,policy_config,ROOT


def fixture():
    terminal=dict(status='completed_target_updates',completed_updates=8000,checkpoint=dict(manifest_sha256='sha',completed_updates=8000))
    export=dict(schema='champion-ema-export-v2',weights='EMA',completed_updates=8000,
        source_checkpoint=deepcopy(terminal['checkpoint']),purpose='CHAMPION100_FORMAL_SFT',
        usable_for_deployment=True,restored_by_author_loader_exact=True)
    return terminal,export


def test_only_final_exact_training_checkpoint_can_be_registered():
    t,e=fixture();require_terminal_export(t,e)
    e['completed_updates']=7000
    with pytest.raises(ValueError):require_terminal_export(t,e)
    t,e=fixture();t['status']='failed_no_retry'
    with pytest.raises(ValueError):require_terminal_export(t,e)
    t,e=fixture();e['source_checkpoint']['manifest_sha256']='different'
    with pytest.raises(ValueError):require_terminal_export(t,e)


def test_per_gpu_configs_change_only_gpu_and_port():
    a=policy_config(ROOT/'runs/final/ema','sha',8000,gpu=0)
    b=policy_config(ROOT/'runs/final/ema','sha',8000,gpu=7)
    assert {k for k in a if a[k]!=b[k]}=={'port','model_gpu'}
    assert a['allowed_task_ids']==list(range(100)) and a['num_inference_steps']==20
