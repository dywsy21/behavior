from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_calibration_selection import preselect_anchors, preselect_pool_anchors, selected_rows


def rows():
    result=[]
    for i in range(60):
        # Eight actual short-fault sources supply only clean success.
        classes=['SUCCEEDED'] if i < 8 else ['IN_PROGRESS','SUCCEEDED','FAILED']
        for label in classes:
            result.append(dict(candidate=dict(sample_id=f'{i}-{label}',source_group=f'g:{i}',split='dev'),
                approval=dict(pool='outcome',usage_role='calibration',
                    label=dict(value=label,member_index=0))))
    return result


class SelectionTests(unittest.TestCase):
    def pool(self):
        data=rows()
        for i in range(60,125):
            for label in ('IN_PROGRESS','SUCCEEDED','FAILED'):
                data.append(dict(candidate=dict(sample_id=f'{i}-{label}',source_group=f'g:{i}',split='dev'),
                    approval=dict(pool='outcome',usage_role='calibration',label=dict(value=label,member_index=0))))
        groups=[f'g:{i}' for i in range(130)]
        missing=[dict(source_group=f'g:{i}',reason='reference_grasp_not_reproduced',receipt_sha256='a'*64)
            for i in range(125,130)]
        return data,groups,missing

    def test_pool_keeps_failed_and_reserved_denominators_without_predictions(self):
        data,groups,missing=self.pool()
        result=preselect_pool_anchors(data,groups,missing)
        self.assertEqual(result['declared_source_count'],130)
        self.assertEqual(result['reviewed_source_count'],125)
        self.assertEqual(result['unavailable_source_count'],5)
        self.assertEqual(result['calibration_source_count'],90)
        self.assertEqual(len(result['reserved_groups']),35)
        self.assertEqual({c:sum(r['value']==c for r in result['selected_anchors'])
            for c in ('IN_PROGRESS','SUCCEEDED','FAILED')},dict(IN_PROGRESS=30,SUCCEEDED=30,FAILED=30))
        self.assertEqual(result,preselect_pool_anchors(list(reversed(data)),list(reversed(groups)),list(reversed(missing))))
        manifest=dict(result,schema='independent_calibration_pool_selection_v2',
            declared_before_model_predictions=True,per_class=30,seed=17)
        self.assertEqual(len(selected_rows(data,manifest)),90)
        changed=deepcopy(manifest);changed['reserved_groups'].pop()
        with self.assertRaises(ValueError):selected_rows(data,changed)
        changed=deepcopy(manifest);changed['declared_source_count']=125
        with self.assertRaises(ValueError):selected_rows(data,changed)

    def test_pool_rejects_silent_drops_false_missing_duplicates_and_lower_quota(self):
        data,groups,missing=self.pool()
        for bad,bad_missing in ((data,missing[:-1]),(data[3:],missing),(data+data[:1],missing),
                (data,missing+[dict(source_group='g:0',reason='model_wrong',receipt_sha256='b'*64)]),
                (data,[dict(m,receipt_sha256='bad') for m in missing])):
            with self.assertRaises(ValueError):preselect_pool_anchors(bad,groups,bad_missing)
        for quota in (20,29,True):
            with self.assertRaises(ValueError):preselect_pool_anchors(data,groups,missing,per_class=quota)
        with self.assertRaises(ValueError):
            preselect_pool_anchors([r for r in data if r['approval']['label']['value']!='FAILED'],groups,missing)
        bad=deepcopy(data);bad[0]['approval']['usage_role']='frozen_test'
        with self.assertRaises(ValueError):preselect_pool_anchors(bad,groups,missing)

    def test_exact_unique_balanced_sources_not_frames_or_predicted_success(self):
        data=rows();groups=[f'g:{i}' for i in range(60)]
        selection=preselect_anchors(data,groups)
        self.assertEqual(len(selection),60)
        self.assertEqual({r['source_group'] for r in selection},set(groups))
        self.assertEqual({c:sum(r['value']==c for r in selection)
            for c in ('IN_PROGRESS','SUCCEEDED','FAILED')},dict(IN_PROGRESS=20,SUCCEEDED=20,FAILED=20))
        self.assertEqual(selection,preselect_anchors(list(reversed(data)),list(reversed(groups))))
        manifest=dict(schema='independent_calibration_anchor_selection_v1',declared_before_model_predictions=True,
            expected_groups=groups,per_class=20,seed=17,selected_anchors=selection)
        self.assertEqual(len(selected_rows(data,manifest)),60)
        manifest=deepcopy(manifest);manifest['selected_anchors'][0]['sample_id']='cherry-picked'
        with self.assertRaises(ValueError):selected_rows(data,manifest)

    def test_missing_support_test_or_training_never_substitute(self):
        data=rows();groups=[f'g:{i}' for i in range(60)]
        for bad in ([r for r in data if r['approval']['label']['value']!='FAILED'],
                    [r for r in data if r['candidate']['source_group']!='g:0']):
            with self.assertRaises(ValueError):preselect_anchors(bad,groups)
        for field,value in [('split','train')]:
            bad=deepcopy(data);bad[0]['candidate'][field]=value
            with self.assertRaises(ValueError):preselect_anchors(bad,groups)
        bad=deepcopy(data);bad[0]['approval']['usage_role']='frozen_test'
        with self.assertRaises(ValueError):preselect_anchors(bad,groups)
        with self.assertRaises(ValueError):preselect_anchors(data,groups,per_class=19)


if __name__=='__main__':unittest.main()
