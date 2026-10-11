import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('audit_cadence',Path(__file__).resolve().parents[1]/'scripts/infra/audit_memlite_transition_cadence.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CadenceTests(unittest.TestCase):
    def test_coarse_previous_is_not_stride16_oracle_at_a_boundary(self):
        record=dict(phase=0,segments=[dict(start=0,end=32,text='NAV',parent='A'),
            dict(start=32,end=160,text='GRASP',parent='A'),dict(start=160,end=257,text='PLACE',parent='B')])
        result=module.analyze_episode(record)
        self.assertEqual(result['anchors'],17)
        self.assertEqual(result['initial'],1)
        self.assertEqual(result['bundle_transition'],2)
        self.assertEqual(result['periodic128_eligible_calls'],3)
        self.assertEqual(result['periodic128_prior_differs_from_stride16'],2)
        self.assertEqual(result['periodic128_bundle_transition'],2)

    def test_same_skill_and_parent_only_change_are_not_bundle_transitions(self):
        record=dict(phase=5,segments=[dict(start=0,end=64,text='NAV',parent='A'),
            dict(start=64,end=200,text='NAV',parent='B')])
        result=module.analyze_episode(record)
        self.assertEqual(result['bundle_transition'],0)
        self.assertEqual(result['parent_only_transition'],1)
        self.assertEqual(result['periodic128_prior_differs_from_stride16'],0)
        self.assertEqual(result['periodic128_parent_differs_from_stride16'],1)


if __name__=='__main__':unittest.main()
