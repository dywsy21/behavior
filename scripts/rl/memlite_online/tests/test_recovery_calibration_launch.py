"""Orchestration tests use synthetic metadata, not a model/accuracy result."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/'code'))
from recovery_calibration_launch import preflight, prepare, check_transport
from recovery_corpus import file_sha


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root/'src/frozen'
        self.repo.mkdir(parents=True)
        self.calls = []
        self.bad_selection = False
        self.fail_child = False
        self.commit = 'a'*40
        def put(root, path, value):
            file = root/path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(value))
            return dict(path=path, sha256=file_sha(file))
        self.put = put
        high = put(self.root, 'models/base.pt', {'synthetic_weight': 1})
        selected = put(self.root, 'models/observer.pt', {'synthetic_weight': 2})
        train = put(self.root, 'datasets/train/admission.json', {})
        fit_cfg = dict(root=str(self.root), high=high, admission='datasets/train',
            admission_sha256=train['sha256'])
        fit_config = put(self.root, 'src/old/fit-config.json', fit_cfg)
        fit = dict(status='observer_adapter_fit_complete_not_deployed',
            config_sha256=fit_config['sha256'], frozen_before_sha256='b'*64,
            frozen_after_sha256='b'*64, selected_checkpoint_sha256=selected['sha256'])
        fit_result = put(self.root, 'runs/fit/result.json', fit)
        pool = dict(schema='prospective_calibration_source_pool_v4',
            selected_observer_sha256=selected['sha256'], fit_result_sha256=fit_result['sha256'],
            fit_admission_sha256=train['sha256'], high_sha256=high['sha256'],
            prior_prediction_exposure=dict(selection={'path':'old-selection','sha256':'e'*64},
                result={'path':'old-result','sha256':'f'*64},
                predictions={'path':'old-predictions','sha256':'0'*64}))
        pool_ref = put(self.repo, 'configs/pool.json', pool)
        cohort_refs = [put(self.root, f'runs/cohort-{i}.json', {'synthetic':i}) for i in range(3)]
        unit = self.root/'datasets/transport'
        admission = put(unit, 'unit/admission/admission.json', {'synthetic_calibration':True})
        media = put(unit, 'media.json', {'synthetic_pixels':True})
        manifest = dict(schema='recovery_release_transfer_v1', partial_reviewed_unit=True,
            grants_training_launch_permission=False, all_three_pools_ready=False,
            contains_weights=False, contains_credentials=False, admission_sha256=admission['sha256'],
            files=[dict(**r, bytes=(unit/r['path']).stat().st_size) for r in (admission,media)])
        manifest_ref = put(unit, 'transfer-manifest.json', manifest)
        union = dict(evidence_root=str(self.root),
            evaluation_partition=dict(base_unit_count=0, cohort_sha256=pool_ref['sha256']),
            prediction_exposure_quarantine=pool['prior_prediction_exposure'],
            units=[dict(corpus=str(unit/'unit'))])
        union_ref = put(self.repo, 'configs/union.json', union)
        self.declaration = dict(schema='observer_reserved_calibration_launch_v1',
            owner='synthetic unittest only', root=str(self.root), union_spec=union_ref,
            source_pool=pool_ref, fit_result=fit_result, fit_config=fit_config,
            selected_observer=selected, cohorts=cohort_refs,
            new_unit=dict(directory='datasets/transport', manifest_sha256=manifest_ref['sha256'],
                admission='datasets/transport/unit/admission/admission.json', admission_sha256=admission['sha256']),
            run_output='runs/new-launch', corpus_output='datasets/new-union')

    def runner(self, argv, *, check, cwd):
        self.assertTrue(check)
        self.assertEqual(cwd, self.repo)
        self.calls.append(argv)
        if self.fail_child:
            raise RuntimeError('Synthetic full-audit failure')
        target = Path(argv[argv.index('--output')+1])
        if Path(argv[1]).name == 'merge_reviewed_recovery.py':
            self.put(target, 'admission/admission.json', {'synthetic_union':True})
            self.put(target, 'receipt.json', dict(status='passed',
                admission_sha256=file_sha(target/'admission/admission.json')))
        elif Path(argv[1]).name == 'prepare_observer_calibration_pool.py':
            config = json.loads(Path(argv[argv.index('--config')+1]).read_text())
            groups = [f'g:{i}' for i in range(90)]
            anchors = [dict(source_group=g, sample_id=f'row-{i}',
                value=('SUCCEEDED' if self.bad_selection else
                       ('IN_PROGRESS','SUCCEEDED','FAILED')[i//30])) for i,g in enumerate(groups)]
            self.put(target.parent, target.name, dict(schema='independent_calibration_pool_selection_v4',
                declared_before_model_predictions=True, admission_sha256=config['admission_sha256'],
                source_pool_sha256=config['source_pool_sha256'],
                selected_observer_sha256=config['selected_observer_sha256'],
                fit_result_sha256=config['fit_result_sha256'], source_commit=self.commit,
                selected_groups=groups, selected_anchors=anchors, declared_source_count=217,
                unavailable_source_count=27, previously_predicted_source_count=90))
        else:
            self.fail('Preparation attempted a GPU/model or unknown child')

    def launch(self):
        return prepare(self.declaration, self.repo, source_commit=self.commit,
            declaration_sha256='c'*64, runner=self.runner)

    def test_pins_final_configuration_and_does_not_launch_gpu(self):
        result = self.launch()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result['status'], 'calibration_inputs_prepared_no_gpu_or_deployment')
        self.assertEqual(result['model_forwards'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        config = json.loads(Path(result['config']).read_text())
        self.assertEqual(config['anchor_selection_sha256'], file_sha(self.root/config['anchor_selection']))
        self.assertEqual(config['admission_sha256'], file_sha(self.root/config['admission']/'admission.json'))
        self.assertEqual(config['source_pool'], 'src/frozen/configs/pool.json')
        self.assertFalse((self.root/'runs/new-launch/calibration').exists())
        with self.assertRaisesRegex(ValueError, 'new run/corpus'):
            self.launch()

    def test_changed_weight_or_transport_fails_before_any_write(self):
        for path in ('models/observer.pt', 'datasets/transport/media.json'):
            original = (self.root/path).read_bytes()
            (self.root/path).write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'pinned dependency'):
                self.launch()
            self.assertFalse((self.root/'runs/new-launch').exists())
            self.assertFalse(self.calls)
            (self.root/path).write_bytes(original)

    def test_transport_cannot_grant_training_even_with_updated_manifest_hash(self):
        path = self.root/'datasets/transport/transfer-manifest.json'
        value = json.loads(path.read_text()); value['grants_training_launch_permission'] = True
        path.write_text(json.dumps(value))
        self.declaration['new_unit']['manifest_sha256'] = file_sha(path)
        with self.assertRaisesRegex(ValueError, 'calibration-only'):
            self.launch()

    def test_duplicate_transport_entry_rejected(self):
        path = self.root/'datasets/transport/transfer-manifest.json'
        value = json.loads(path.read_text()); value['files'].append(deepcopy(value['files'][0]))
        path.write_text(json.dumps(value))
        self.declaration['new_unit']['manifest_sha256'] = file_sha(path)
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            self.launch()

    def test_changed_prior_prediction_quarantine_rejected(self):
        path = self.repo/'configs/union.json'
        value = json.loads(path.read_text()); value['prediction_exposure_quarantine'] = None
        path.write_text(json.dumps(value))
        self.declaration['union_spec']['sha256'] = file_sha(path)
        with self.assertRaisesRegex(ValueError, 'quarantine mismatch'):
            self.launch()

    def test_escape_duplicate_cohort_or_wrong_fit_root_rejected(self):
        original = deepcopy(self.declaration)
        self.declaration['corpus_output'] = '../escaped'
        with self.assertRaises(ValueError):
            self.launch()
        self.declaration = deepcopy(original)
        self.declaration['cohorts'][2] = self.declaration['cohorts'][0]
        with self.assertRaisesRegex(ValueError, 'two old signed cohorts'):
            self.launch()
        self.declaration = deepcopy(original)
        self.declaration['root'] = str(self.repo)
        with self.assertRaisesRegex(ValueError, 'inside the shared'):
            self.launch()

    def test_child_failure_retains_evidence_and_never_advances_to_selection(self):
        self.fail_child = True
        with self.assertRaisesRegex(RuntimeError, 'full-audit'):
            self.launch()
        self.assertEqual(len(self.calls), 1)
        run = self.root/'runs/new-launch'
        self.assertTrue((run/'preflight.json').exists())
        failure = json.loads((run/'failure.json').read_text())
        self.assertTrue(failure['partial_run_and_corpus_preserved'])
        self.assertFalse((run/'result.json').exists())
        self.assertFalse((run/'calibration-config.json').exists())

    def test_ninety_unique_rows_with_wrong_phase_quota_are_not_ready(self):
        self.bad_selection = True
        with self.assertRaisesRegex(ValueError, 'exact independent90'):
            self.launch()
        run = self.root/'runs/new-launch'
        self.assertTrue((run/'anchor-selection.json').exists())
        self.assertTrue((run/'failure.json').exists())
        self.assertFalse((run/'calibration-config.json').exists())

    def test_real_transport_if_present_is_all_byte_checked(self):
        directory = BASE.parents[2]/'artifacts/recovery-observer-fresh-calibration-20261010/grasp-v11-new30-transport-v1'
        if not directory.exists():
            self.skipTest('Real non-Git data not installed in this test environment')
        result = check_transport(directory,
            'fe222f88abc4cde12bd4a93336349dbbb4b2f601f9d48f02dda11cc4f8063c04',
            '9f409965c7129f43f3668aaa48c712fe081e696bbeaf54dca68e7a61ec4ff71e')
        self.assertEqual((result['files'], result['bytes']), (140, 114748003))


if __name__ == '__main__':
    unittest.main()
