from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_action_coverage import later_action_windows


class Tests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(control_step=t,label_kind='clean' if t>=32 else 'fault',
            simulator_apply_ack=True,terminated=False,truncated=False,context={'issued':'command'}) for t in range(384)]
        self.available=list(range(0,384,4))

    def run_selector(self,rows=None):
        return later_action_windows(self.rows if rows is None else rows,self.available,
            retry_control=32,first_success=320,clean_label='clean')

    def test_phase_spread_complete_targets_not_three_independent_events(self):
        selected=self.run_selector()
        self.assertEqual(selected['candidate_action_steps'],[128,224,304])
        self.assertEqual(selected['independent_new_events'],0);self.assertFalse(selected['human_approved'])
        for t in selected['candidate_action_steps']:
            self.assertTrue(set(range(t,t+33,4))<=set(selected['frames']))

    def test_injected_or_mixed_command_or_unacked_action_not_admitted(self):
        for field,value in [('label_kind','fault'),('context',{'issued':'other'}),('simulator_apply_ack',False),('terminated',True)]:
            rows=deepcopy(self.rows);rows[130][field]=value
            selected=self.run_selector(rows)
            self.assertTrue(all(not t<=130<t+32 for t in selected['candidate_action_steps']))

    def test_no_clean_later_windows_fail_closed(self):
        for row in self.rows:row['label_kind']='fault'
        with self.assertRaises(ValueError):self.run_selector()

    def test_no_padding_to_fabricate_32_actions(self):
        for first in (-1,400,True):
            with self.assertRaises(ValueError):later_action_windows(self.rows,self.available,
                retry_control=32,first_success=first,clean_label='clean')
        selected=self.run_selector()
        self.assertTrue(all(t+32<len(self.rows) for t in selected['candidate_action_steps']))


if __name__=='__main__':unittest.main()
