from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_independent_cohort import cohort_sources


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(schema='recovery_independent_cohort_v1',declared_before_any_model_predictions=True,
            training_forbidden=True,model_selection_forbidden=True,calibration_must_not_use_frozen_test=True,
            calibration_groups=[dict(source_group='a:1',case='a_1',arm='left')],
            frozen_test_groups=[dict(source_group='b:2',case='b_2',arm='right')])
        self.queue=[dict(entry,split='dev',branch='clean') for kind in ('calibration','frozen_test')
                    for entry in self.spec[kind+'_groups']]

    def test_roles_are_fixed_and_no_success_filter(self):
        self.queue[0]['failure']='unsuccessful recovery'
        selected=cohort_sources(self.queue,self.spec,'calibration')
        self.assertEqual([e['source_group'] for e,_ in selected],['a:1'])
        self.assertEqual(cohort_sources(self.queue,self.spec,'frozen_test')[0][0]['source_group'],'b:2')

    def test_overlap_missing_identity_and_train_fail_closed(self):
        bad=deepcopy(self.spec);bad['frozen_test_groups']=bad['calibration_groups']
        with self.assertRaises(ValueError):cohort_sources(self.queue,bad,'calibration')
        for queue in (self.queue[:1],self.queue+self.queue[:1]):
            with self.assertRaises(ValueError):cohort_sources(queue,self.spec,'calibration')
        for field,value in [('arm','right'),('case','other'),('split','train')]:
            bad=deepcopy(self.queue);bad[0][field]=value
            with self.assertRaises(ValueError):cohort_sources(bad,self.spec,'calibration')
        for field in ('training_forbidden','model_selection_forbidden','declared_before_any_model_predictions'):
            bad=deepcopy(self.spec);bad[field]=False
            with self.assertRaises(ValueError):cohort_sources(self.queue,bad,'calibration')


if __name__=='__main__':unittest.main()
