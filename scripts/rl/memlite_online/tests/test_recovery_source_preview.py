import importlib.util
from pathlib import Path
import unittest
from copy import deepcopy

spec=importlib.util.spec_from_file_location('source_preview_verify',Path(__file__).resolve().parents[1]/'tools/verify_prospective_calibration_sources.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SourcePreviewTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(source_group=f'task:{i}',controls=i+1) for i in range(30)]
        self.preview=dict(status='metadata_only_prospective30_no_export_no_labels_no_predictions',
            model_not_selected_by_this_preview=True,optimizer_updates=0,physical_controls=0,rows=self.rows,
            expected_reference_controls=465,fit_config_sha256='a'*64,fit_admission_sha256='b'*64,
            protected_sha256='c'*64,exclusions=[dict(path='prior',manifest_sha256='d'*64)])
        self.args=dict(groups={r['source_group'] for r in self.rows},controls=465,
            fit=dict(config_sha256='a'*64,admission_sha256='b'*64),protected_sha256='c'*64,
            exclusions=[dict(path='prior',sha256='d'*64)])

    def test_exact_fixed_preview_passes(self):
        module.require_metadata_preview(self.preview,**self.args)

    def test_missing_extra_or_duplicate_source_rejected(self):
        for groups in (set(list(self.args['groups'])[:-1]),self.args['groups']|{'extra:1'}):
            with self.assertRaises(ValueError):module.require_metadata_preview(self.preview,**dict(self.args,groups=groups))
        changed=deepcopy(self.preview);changed['rows'][-1]=changed['rows'][0]
        with self.assertRaises(ValueError):module.require_metadata_preview(changed,**self.args)

    def test_fit_control_and_inventory_drift_rejected(self):
        for key,value in [('controls',464),('fit',dict(config_sha256='f'*64,admission_sha256='b'*64)),
                          ('protected_sha256','e'*64),('exclusions',[])]:
            with self.assertRaises(ValueError):module.require_metadata_preview(self.preview,**dict(self.args,**{key:value}))

    def test_prediction_dependent_or_physical_preview_rejected(self):
        for key,value in [('model_not_selected_by_this_preview',False),('optimizer_updates',1),('physical_controls',1)]:
            with self.assertRaises(ValueError):module.require_metadata_preview(dict(self.preview,**{key:value}),**self.args)


if __name__=='__main__':unittest.main()
