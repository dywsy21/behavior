import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_action_diagnostic import summarize_action_losses


class ActionLossTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(sample_id=sample,noise_seed=seed,source_group='group',event_id=event,
            split='train',control_step=step,mechanism='OPEN_DOOR',numerator=loss*10,denominator=10)
            for sample,event,step,loss in [('a','one',0,1),('b','one',32,3),('c','two',0,10)] for seed in (1,2)]
    def test_equal_events_not_window_inflation(self):
        summary=summarize_action_losses(self.rows,['a','b','c'],[1,2])[0]
        self.assertEqual(summary['event_balanced_loss'],6)
        self.assertEqual(summary['events'],2)
        self.assertEqual(summary['anchors'],3)
        self.assertAlmostEqual(summary['anchor_mean_loss'],14/3)
    def test_no_missing_duplicate_or_foreign_noise(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],self.rows+[{**self.rows[0],'noise_seed':4}]):
            with self.assertRaises(ValueError):summarize_action_losses(rows,['a','b','c'],[1,2])
    def test_no_bad_objective_or_identity(self):
        for change in (dict(denominator=0),dict(numerator=float('nan')),dict(split='test'),dict(event_id='mixed')):
            rows=copy.deepcopy(self.rows); rows[0].update(change)
            with self.assertRaises(ValueError):summarize_action_losses(rows,['a','b','c'],[1,2])


if __name__=='__main__':unittest.main()
