import importlib.util
from pathlib import Path
import sys
import unittest

DIRECTORY=Path(__file__).resolve().parents[1]/'scripts/experiments'
sys.path.insert(0,str(DIRECTORY))
spec=importlib.util.spec_from_file_location('bounded_best_eval',DIRECTORY/'eval_g05_best.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class SelectionTests(unittest.TestCase):
    def test_primary_is_fm_not_ce(self):
        rows=[dict(step=10000,fm_loss=.1,ce_loss=3),dict(step=20000,fm_loss=.11,ce_loss=1)]
        self.assertEqual(evaluation.ranked(rows)[0]['step'],10000)

    def test_ties_are_deterministic(self):
        rows=[dict(step=s,fm_loss=.1,ce_loss=c) for s,c in [(30000,2),(20000,1),(10000,1)]]
        self.assertEqual([r['step'] for r in evaluation.ranked(rows)],[10000,20000,30000])

    def test_nonfinite_and_duplicates_fail(self):
        with self.assertRaises(ValueError):evaluation.ranked([dict(step=10000,fm_loss=float('nan'),ce_loss=1)])
        row=dict(step=10000,fm_loss=.1,ce_loss=1)
        with self.assertRaises(ValueError):evaluation.ranked([row,row])

    def test_all_ten_saved_checkpoints_only(self):
        self.assertEqual(len(evaluation.STEPS),10)
        for s in evaluation.STEPS:self.assertEqual(evaluation.core.checkpoint(s).name,f'step_{s}.pt')
        for s in (False,True,10000.0,5000,110000):
            with self.assertRaises(ValueError):evaluation.core.checkpoint(s)

    def test_evidence_reuse_and_fresh_paths(self):
        self.assertEqual(evaluation.folder(40000),evaluation.PREVIOUS)
        self.assertNotEqual(evaluation.folder(10000),evaluation.PREVIOUS)
        self.assertEqual(set(evaluation.STEPS)-evaluation.REUSE.keys(),{10000,20000,30000,50000,60000,70000,80000,90000})

    def test_registered_budget(self):
        self.assertEqual(sum(evaluation.core.LIMITS[t] for t in evaluation.SIM_TASKS),4248)
        self.assertEqual(sum((evaluation.core.LIMITS[t]+15)//16 for t in evaluation.SIM_TASKS),266)

    def test_locator_identity_includes_episode_and_task(self):
        row=dict(task='a',episode=190,index=12,fraction=.25)
        self.assertNotEqual(evaluation.locators([row]),evaluation.locators([dict(row,episode=191)]))


if __name__=='__main__':unittest.main()
