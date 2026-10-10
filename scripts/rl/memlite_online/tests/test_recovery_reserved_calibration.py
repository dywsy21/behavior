"""Synthetic provenance only: no scientific success implied by these tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from recovery_reserved_calibration import prediction_exposure,reserve_pool_sources,load_prediction_exposure
from recovery_corpus import file_sha
from recovery_prospective_adapter import make_pool_selection,require_prospective_selection
from recovery_evaluation_partition import build_partition


class ReservedCalibrationTests(unittest.TestCase):
    def fixture(self):
        old_pool=json.loads((BASE.parents[2]/'configs/recovery_sft/observer_calibration_pool_v3.json').read_text())
        groups=[f'old:{i}' for i in range(187)]
        missing=[dict(source_group=g,reason='reference_grasp_not_reproduced',receipt_sha256='a'*64) for g in groups[:23]]
        ineligible=[dict(source_group=g,reason='previous_training_or_model_selection_source',
            exposure_admission_sha256=old_pool['fit_admission_sha256']) for g in groups[23:27]]
        test=[f'test:{i}' for i in range(20)]
        provenance=dict(declared_groups=sorted(groups),unavailable_sources=missing,ineligible_sources=ineligible,
            source_cohort_by_group={g:'b'*64 for g in groups[27:]},reserved_frozen_test_groups=test,
            previously_exposed_declared_groups=sorted(groups[23:27]))
        def rows(group,sha):
            return [dict(candidate=dict(source_group=group,split='dev',sample_id=group+':'+label),
                approval=dict(pool='outcome',usage_role='calibration',cohort_sha256=sha,
                    label=dict(value=label,member_index=0))) for label in ('IN_PROGRESS','SUCCEEDED','FAILED')]
        old_rows=[r for g in groups[27:] for r in rows(g,'b'*64)]
        selection=make_pool_selection(old_pool,provenance,old_rows,pool_sha256='c'*64,fit_result_sha256='d'*64)
        predictions=[dict(source_group=r['source_group'],sample_id=r['sample_id'],label=r['value'],logits=[0,0,0,0])
            for r in selection['selected_anchors']]
        result=dict(status='completed_prospective_adapter_calibration_not_deployed',optimizer_updates=0,
            frozen_test_not_read=True,anchor_selection_sha256='e'*64,predictions_sha256='f'*64,
            selected_observer_sha256=old_pool['selected_observer_sha256'],source_pool_sha256='c'*64,
            reserved_groups_not_predicted=selection['reserved_groups'])
        predicted=prediction_exposure(selection,result,predictions,selection_sha256='e'*64,predictions_sha256='f'*64)
        pool=deepcopy(old_pool);pool.update(schema='prospective_calibration_source_pool_v4',
            selected_observer_sha256='9'*64,expected_unpredicted_reserves=70,expected_new_sources=30,declared_sources=217,
            prior_prediction_exposure=dict(selection=dict(sha256='e'*64),predictions=dict(sha256='f'*64)))
        new_groups=[dict(source_group=f'new:{i}') for i in range(30)]
        for r in new_groups[:2]:r['unavailable_collection']=dict(reason='reference_not_reproduced',receipt_sha256='4'*64)
        cohort=dict(schema='recovery_independent_calibration_only_cohort_v1',model_predictions_read=False,
            declared_before_any_model_predictions=True,training_forbidden=True,model_selection_forbidden=True,
            prospective_config_sha256='5'*64,selected_observer_sha256='9'*64,
            external_frozen_test_source_groups=test,declared_sources=30,calibration_groups=new_groups,
            unavailable_source_groups=['new:0','new:1'])
        args=(pool,old_pool,provenance,selection,predicted,(cohort,'6'*64,'5'*64))
        kept=[r for r in old_rows if r['candidate']['source_group'] in selection['reserved_groups']]
        kept += [r for group in new_groups[2:] for r in rows(group['source_group'],'6'*64)]
        return args,kept,predictions,result

    def test_217_denominator_90_exposed_70_reserves_new30_and_signatures_preserved(self):
        args,rows,_,_=self.fixture();pool=args[0]
        p=reserve_pool_sources(*args,exposed={'old:23','old:24','old:25','old:26'})
        sel=make_pool_selection(pool,p,rows,pool_sha256='7'*64,fit_result_sha256='8'*64)
        self.assertEqual(sel['schema'],'independent_calibration_pool_selection_v4')
        self.assertEqual((sel['declared_source_count'],sel['unavailable_source_count'],sel['ineligible_source_count'],
            sel['previously_predicted_source_count'],sel['reviewed_source_count']),(217,25,4,90,98))
        self.assertEqual(len(sel['selected_groups']),90);self.assertEqual(len(sel['reserved_groups']),8)
        spec=dict(schema='recovery_evaluation_partition_spec_v3',base_unit_count=0,no_added_training_rows_expected=True,
            calibration_may_not_select_checkpoints=True,cohort_sha256='7'*64,
            calibration_groups=sorted(p['source_cohort_by_group']),source_cohort_by_group=p['source_cohort_by_group'],
            frozen_test_groups=p['reserved_frozen_test_groups'],reserved_frozen_test_groups=p['reserved_frozen_test_groups'])
        part=build_partition(spec,[rows])
        cfg=dict(schema='prospective_observer_adapter_calibration_v1',source_pool_sha256='7'*64,
            fit_result_sha256='8'*64,selected_observer_sha256=pool['selected_observer_sha256'],high_sha256=pool['high_sha256'])
        chosen=require_prospective_selection(cfg,pool,sel,part,rows,{'old:23','old:24','old:25','old:26'})
        self.assertEqual(len(chosen),90)
        self.assertFalse({r['candidate']['source_group'] for r in chosen}&args[4])
        self.assertEqual({r['approval']['cohort_sha256'] for r in rows},{'b'*64,'6'*64})

    def test_all_previous_predictions_excluded_independent_of_logits(self):
        args,_,predictions,result=self.fixture();selection=args[3]
        original=prediction_exposure(selection,result,predictions,selection_sha256='e'*64,predictions_sha256='f'*64)
        changed=deepcopy(predictions)
        for row in changed:row['logits']=[999,-999,0,0]
        self.assertEqual(prediction_exposure(selection,result,changed,selection_sha256='e'*64,predictions_sha256='f'*64),original)
        for bad in (predictions[:-1],predictions[:-1]+[predictions[0]]):
            with self.assertRaises(ValueError):prediction_exposure(selection,result,bad,selection_sha256='e'*64,predictions_sha256='f'*64)

    def test_cannot_relabel_prior_predictions_as_unseen(self):
        args,rows,_,_=self.fixture();p=reserve_pool_sources(*args,exposed=set())
        bad=deepcopy(rows[0]);bad['candidate']['source_group']=next(iter(args[4]))
        with self.assertRaises(ValueError):make_pool_selection(args[0],p,rows+[bad],pool_sha256='7'*64,fit_result_sha256='8'*64)
        selection=deepcopy(args[3]);selection['reserved_groups'][0]=selection['selected_groups'][0]
        with self.assertRaises(ValueError):reserve_pool_sources(*args[:3],selection,*args[4:],exposed=set())

    def test_gate_and_new_source_substitution_rejected(self):
        args,*_=self.fixture()
        pool=deepcopy(args[0]);pool['unchanged_gates']['minimum_precision_wilson95_lower']=.7
        with self.assertRaises(ValueError):reserve_pool_sources(pool,*args[1:],exposed=set())
        cohort=deepcopy(args[5][0]);cohort['calibration_groups'][2]['source_group']='old:100'
        with self.assertRaises(ValueError):reserve_pool_sources(*args[:5],(cohort,'6'*64,'5'*64),exposed=set())
        with self.assertRaises(ValueError):reserve_pool_sources(*args,exposed={'new:2'})

    def test_missing_new_attempt_and_model_or_signature_drift_rejected(self):
        args,*_=self.fixture()
        for key,value in [('selected_observer_sha256','0'*64),('model_predictions_read',True),
                          ('unavailable_source_groups',[]),('calibration_groups',args[5][0]['calibration_groups'][:-1])]:
            cohort=deepcopy(args[5][0]);cohort[key]=value
            with self.assertRaises(ValueError):reserve_pool_sources(*args[:5],(cohort,'6'*64,'5'*64),exposed=set())

    def test_full_hash_bound_exposure_reader_rejects_tampered_ledger(self):
        args,_,predictions,result=self.fixture()
        with TemporaryDirectory() as directory:
            root=Path(directory);spec={}
            for key,value in [('selection',args[3]),('predictions',predictions)]:
                path=root/(key+'.json');path.write_text(json.dumps(value))
                spec[key]=dict(path=path.name,sha256=file_sha(path))
            result.update(anchor_selection_sha256=spec['selection']['sha256'],predictions_sha256=spec['predictions']['sha256'])
            path=root/'result.json';path.write_text(json.dumps(result));spec['result']=dict(path=path.name,sha256=file_sha(path))
            groups,_=load_prediction_exposure(root,spec);self.assertEqual(groups,args[4])
            (root/'predictions.json').write_text('[]')
            with self.assertRaises(ValueError):load_prediction_exposure(root,spec)


if __name__=='__main__':unittest.main()
