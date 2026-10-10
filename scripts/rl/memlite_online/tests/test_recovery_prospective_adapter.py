from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_prospective_adapter import pool_sources,make_pool_selection,require_prospective_selection
from recovery_evaluation_partition import build_partition


class ProspectiveAdapterTests(unittest.TestCase):
    def fixture(self):
        pool=json.loads((REPO/'configs/recovery_sft/observer_calibration_pool_v2.json').read_text())
        cohorts=[];rows=[];cursor=0
        for n,sha in ((90,'a'*64),(97,'b'*64)):
            groups=[];missing=[]
            for i in range(n):
                group=f'g:{cursor}';cursor+=1;entry=dict(source_group=group)
                if i<3:
                    entry['unavailable_collection']=dict(reason='reference_grasp_not_reproduced',receipt_sha256='e'*64)
                    missing.append(group)
                else:
                    for label in ('FAILED','IN_PROGRESS','SUCCEEDED'):
                        rows.append(dict(candidate=dict(source_group=group,split='dev',sample_id=f'{group}:{label}'),
                            approval=dict(pool='outcome',usage_role='calibration',cohort_sha256=sha,
                                label=dict(value=label,member_index=0))))
                groups.append(entry)
            cohorts.append((dict(schema='recovery_independent_calibration_only_cohort_v1',
                model_predictions_read=False,declared_before_any_model_predictions=True,
                training_forbidden=True,model_selection_forbidden=True,
                selected_observer_sha256=pool['selected_observer_sha256'],prospective_config_sha256='c'*64,
                calibration_groups=groups,declared_sources=n,unavailable_source_groups=missing,
                external_frozen_test_source_groups=[f'test:{i}' for i in range(20)]),sha,'c'*64))
        provenance=pool_sources(pool,cohorts)
        selection=make_pool_selection(pool,provenance,rows,pool_sha256='d'*64,fit_result_sha256='f'*64)
        spec=dict(schema='recovery_evaluation_partition_spec_v3',base_unit_count=0,
            no_added_training_rows_expected=True,calibration_may_not_select_checkpoints=True,
            cohort_sha256='d'*64,calibration_groups=sorted(provenance['source_cohort_by_group']),
            frozen_test_groups=provenance['reserved_frozen_test_groups'],
            reserved_frozen_test_groups=provenance['reserved_frozen_test_groups'],
            source_cohort_by_group=provenance['source_cohort_by_group'])
        partition=build_partition(spec,[rows])
        cfg=dict(schema='prospective_observer_adapter_calibration_v1',source_pool_sha256='d'*64,
            fit_result_sha256='f'*64,selected_observer_sha256=pool['selected_observer_sha256'],
            high_sha256=pool['high_sha256'])
        return pool,cohorts,rows,provenance,selection,partition,cfg

    def test_all_attempts_and_original_signatures_preserved(self):
        pool,_,rows,p,sel,part,cfg=self.fixture()
        self.assertEqual(len(p['declared_groups']),187)
        self.assertEqual(sel['unavailable_source_count'],6)
        self.assertEqual(sel['reviewed_source_count'],181)
        self.assertEqual(len(sel['reserved_groups']),91)
        self.assertEqual(len(require_prospective_selection(cfg,pool,sel,part,rows,{'old:train'})),90)
        self.assertFalse(sel['certifies_all_declared_sources'])

    def test_no_prediction_exposure_changed_binding_or_missing_attempt(self):
        pool,cohorts,*_=self.fixture()
        for field,value in (('model_predictions_read',True),('training_forbidden',False),
                            ('selected_observer_sha256','f'*64),('prospective_config_sha256','f'*64),
                            ('unavailable_source_groups',[])):
            bad=deepcopy(cohorts);bad[0][0][field]=value
            with self.assertRaises(ValueError):pool_sources(pool,bad)
        bad=deepcopy(cohorts);bad[1][0]['calibration_groups'][4]['source_group']='g:4'
        with self.assertRaises(ValueError):pool_sources(pool,bad)
        bad_pool=deepcopy(pool);bad_pool['unchanged_gates']['minimum_precision_wilson95_lower']=.7
        with self.assertRaises(ValueError):pool_sources(bad_pool,cohorts)
        bad=deepcopy(cohorts);bad[1][0]['calibration_groups'].pop()
        with self.assertRaises(ValueError):pool_sources(pool,bad)

    def test_no_model_switch_training_overlap_threshold_or_signature_override(self):
        pool,_,rows,p,sel,part,cfg=self.fixture()
        for change in (dict(selected_observer_sha256='a'*64),dict(high_sha256='a'*64),
                       dict(fit_result_sha256='a'*64),dict(source_pool_sha256='a'*64)):
            with self.assertRaises(ValueError):require_prospective_selection(dict(cfg,**change),pool,sel,part,rows,set())
        with self.assertRaises(ValueError):require_prospective_selection(cfg,pool,sel,part,rows,{'g:10'})
        with self.assertRaises(ValueError):require_prospective_selection(cfg,pool,dict(sel,per_class=20),part,rows,set())
        bad=deepcopy(rows);bad[0]['approval']['cohort_sha256']='d'*64
        with self.assertRaises(ValueError):make_pool_selection(pool,p,bad,pool_sha256='d'*64,fit_result_sha256='f'*64)
        with self.assertRaises(ValueError):make_pool_selection(pool,p,rows[3:],pool_sha256='d'*64,fit_result_sha256='f'*64)

    def test_v3_quarantines_fit_exposure_without_calling_it_physics_failure(self):
        pool,cohorts,rows,_,_,_,cfg=self.fixture()
        exposed={'g:0','g:10','g:100'}
        pool.update(schema='prospective_calibration_source_pool_v3',
            previously_exposed_declared_groups=sorted(exposed),exposure_quarantined_before_any_prediction=True,
            fit_admission_sha256='9'*64)
        p=pool_sources(pool,cohorts,exposed_groups=exposed)
        self.assertEqual(len(p['unavailable_sources']),6)
        self.assertEqual({r['source_group'] for r in p['ineligible_sources']},{'g:10','g:100'})
        eligible=[r for r in rows if r['candidate']['source_group'] not in exposed]
        selection=make_pool_selection(pool,p,eligible,pool_sha256='d'*64,fit_result_sha256='f'*64)
        self.assertEqual((selection['declared_source_count'],selection['reviewed_source_count'],
            selection['unavailable_source_count'],selection['ineligible_source_count']),(187,179,6,2))
        self.assertEqual(len(selection['reserved_groups']),89)
        spec=dict(schema='recovery_evaluation_partition_spec_v3',base_unit_count=0,
            no_added_training_rows_expected=True,calibration_may_not_select_checkpoints=True,
            cohort_sha256='d'*64,calibration_groups=sorted(p['source_cohort_by_group']),
            source_cohort_by_group=p['source_cohort_by_group'],
            frozen_test_groups=p['reserved_frozen_test_groups'],reserved_frozen_test_groups=p['reserved_frozen_test_groups'])
        partition=build_partition(spec,[eligible])
        chosen=require_prospective_selection(cfg,pool,selection,partition,eligible,exposed)
        self.assertEqual(len(chosen),90)
        self.assertFalse({r['candidate']['source_group'] for r in chosen}&exposed)
        with self.assertRaises(ValueError):pool_sources(pool,cohorts)
        with self.assertRaises(ValueError):pool_sources(pool,cohorts,exposed_groups=exposed|{'g:11'})
        with self.assertRaises(ValueError):make_pool_selection(pool,p,rows,pool_sha256='d'*64,fit_result_sha256='f'*64)
        with self.assertRaises(ValueError):require_prospective_selection(cfg,pool,selection,partition,eligible,exposed|{'g:11'})

    def test_v2_preselection_now_rejects_fit_overlap_before_any_gpu_call(self):
        pool,cohorts,*_=self.fixture()
        with self.assertRaises(ValueError):pool_sources(pool,cohorts,exposed_groups={'g:10'})


if __name__=='__main__':unittest.main()
