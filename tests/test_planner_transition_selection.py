from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_generation_metrics import select_transition_indices


class TransitionSelectionTests(unittest.TestCase):
    def test_no_initial_or_repeated_skill_counts_as_transition(self):
        previous={0:'None',1:'NAV',2:'NAV',3:'GRASP',4:'None',5:'NAV',6:'NAV'}
        target={0:'NAV',1:'NAV',2:'GRASP',3:'GRASP',4:'NAV',5:'NAV',6:'OPEN'}
        tasks={i:0 if i<4 else 1 for i in range(7)}
        result=select_transition_indices([[0,1,2,3],[4,5,6]],previous.__getitem__,target.__getitem__,tasks.__getitem__)
        self.assertEqual(result,[2,6])

    def test_missing_boundary_is_not_replaced_by_easy_continuation(self):
        with self.assertRaisesRegex(ValueError,'No actual heldout'):
            select_transition_indices([[0,1]],lambda i:'None' if i==0 else 'NAV',lambda i:'NAV',lambda i:0)

    def test_wrong_task_and_empty_schedule_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'binding'):
            select_transition_indices([[0]],lambda i:'NAV',lambda i:'OPEN',lambda i:2)
        with self.assertRaisesRegex(ValueError,'Empty'):
            select_transition_indices([],lambda i:None,lambda i:None,lambda i:None)


if __name__=='__main__':unittest.main()
