import copy
import json
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'scripts/rl/memlite_online/code')]
from recovery_probe_restart import validate_probe_restart


class RestartTests(unittest.TestCase):
    def test_all_three_exact_scientific_configs_and_changed_seed_rejected(self):
        for arm in ('parent','control','candidate'):
            cfg=json.loads((REPO/f'configs/recovery_sft/a800_native_learner_{arm}_probes_heartbeat_v2.json').read_text())
            self.assertEqual(validate_probe_restart(cfg,REPO)['model_gpu'],2)
            wrong=copy.deepcopy(cfg);wrong['evaluation_seeds'][0]+=1
            with self.assertRaises(ValueError): validate_probe_restart(wrong,REPO)
            wrong=copy.deepcopy(cfg);wrong['learning_rounds']=1
            with self.assertRaises(ValueError): validate_probe_restart(wrong,REPO)
            wrong=copy.deepcopy(cfg);next(iter(wrong['cases'].values()))['end_control']+=16
            with self.assertRaises(ValueError): validate_probe_restart(wrong,REPO)
        self.assertIsNone(validate_probe_restart({},REPO))


if __name__=='__main__': unittest.main()
