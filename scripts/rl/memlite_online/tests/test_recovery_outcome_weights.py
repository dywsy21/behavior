from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_convergence import event_weights,observer_training_weights


class OutcomeWeightsTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(candidate=dict(source_group=group),approval=dict(event_id=event,label=dict(value=label)))
            for group,event,label in [('g1','e1','FAILED'),('g1','e1','SUCCEEDED'),('g2','e1','FAILED'),
                ('g2','e1','IN_PROGRESS'),('g2','e1','UNKNOWN'),('g3','e2','SUCCEEDED')]]
    def test_default_is_exactly_old_objective(self):
        self.assertEqual(observer_training_weights(self.rows),event_weights(self.rows))
    def test_equal_class_then_events_not_reviewed_frame_counts(self):
        weights=observer_training_weights(self.rows,'outcome_then_event_v1')
        self.assertAlmostEqual(sum(weights),1.)
        for label in ('FAILED','SUCCEEDED','IN_PROGRESS','UNKNOWN'):
            self.assertAlmostEqual(sum(w for r,w in zip(self.rows,weights) if r['approval']['label']['value']==label),.25)
        self.assertEqual(weights[0],weights[2])
        expanded=self.rows+[deepcopy(self.rows[0])]*3
        new=observer_training_weights(expanded,'outcome_then_event_v1')
        self.assertAlmostEqual(new[0]+sum(new[-3:]),weights[0])
        self.assertEqual(new[1:6],weights[1:])
    def test_no_invented_missing_class_or_unknown_protocol(self):
        for rows,protocol in (([], 'outcome_then_event_v1'),(self.rows[:4],'outcome_then_event_v1'),
                              (self.rows,'automatic')):
            with self.assertRaises(ValueError):observer_training_weights(rows,protocol)


if __name__=='__main__':unittest.main()
