from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_calibration_selection import preselect_anchors, selected_rows


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
