from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_independent_cohort import cohort_sources, proposed_phase_points


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

    def test_insufficient_history_is_not_fabricated_as_failure_or_unknown(self):
        points, missing = proposed_phase_points({0:None,4:None},7)
        self.assertEqual(points,[(4,'UNLABELLED')])
        self.assertEqual(set(missing),{'FAILED','IN_PROGRESS','SUCCEEDED'})
        points, missing = proposed_phase_points({8:'FAILED',36:'IN_PROGRESS',40:'SUCCEEDED'},32)
        self.assertEqual(points,[(8,'FAILED'),(36,'IN_PROGRESS'),(40,'SUCCEEDED')])
        self.assertEqual(missing,[])

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

    def test_new_calibration_keeps_old_reserved_test_external_and_unread(self):
        spec=deepcopy(self.spec);spec['schema']='recovery_independent_calibration_only_cohort_v1'
        spec.pop('frozen_test_groups')
        spec.update(external_frozen_test_source_groups=['b:2'],external_frozen_test_cohort_sha256='a'*64,
            prospective_config_sha256='b'*64,prospective_source_audit_sha256='c'*64,selected_observer_sha256='d'*64)
        selected=cohort_sources(self.queue[:1],spec,'calibration')
        self.assertEqual(selected[0][0]['source_group'],'a:1')
        with self.assertRaises(ValueError):cohort_sources(self.queue[:1],spec,'frozen_test')
        with self.assertRaises(ValueError):cohort_sources(self.queue,spec,'calibration')
        for key,value in [('external_frozen_test_source_groups',['a:1']),
                          ('external_frozen_test_source_groups',[]),('prospective_config_sha256','bad'),
                          ('selected_observer_sha256',None)]:
            bad=deepcopy(spec);bad[key]=value
            with self.assertRaises(ValueError):cohort_sources(self.queue[:1],bad,'calibration')
        with self.assertRaises(ValueError):cohort_sources([],spec,'calibration')
        missing=deepcopy(spec)
        missing['calibration_groups'][0]['unavailable_collection']=dict(
            receipt='case/result.json',receipt_sha256='e'*64,reason='reference_grasp_not_reproduced')
        # Retain the declared group with no labels; never turn it into FAILED.
        self.assertEqual(cohort_sources([],missing,'calibration')[0][1],{})
        self.assertEqual(cohort_sources([],missing,'calibration')[0][0]['source_group'],'a:1')
        with self.assertRaises(ValueError):cohort_sources(self.queue[:1],missing,'calibration')
        missing['calibration_groups'][0]['unavailable_collection']['receipt_sha256']='bad'
        with self.assertRaises(ValueError):cohort_sources([],missing,'calibration')


if __name__=='__main__':unittest.main()
