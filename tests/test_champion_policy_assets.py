import json
from pathlib import Path
import tempfile
import unittest

from behavior_champion.policy_assets import verify_policy_assets
from behavior_champion.release import sha256
from behavior_champion.sampling import digest_json


class PolicyAssetTests(unittest.TestCase):
    def make(self, root):
        export = root / 'model'; export.mkdir()
        layout = dict(num_stages=[5] * 100, dataset_to_model={str(i): i for i in range(100)})
        (export / 'task-layout.json').write_text(json.dumps(layout))
        (export / 'params').mkdir(); (export / 'params/values').write_bytes(b'original-ema')
        assets = export / 'assets/IliaLarchenko/behavior_224_rgb'; assets.mkdir(parents=True)
        (assets / 'norm_stats.json').write_text('{}')
        manifest = dict(schema='champion-ema-export-v2', usable_for_deployment=True,
            purpose='CHAMPION100_FORMAL_SFT', weights='EMA', restored_by_author_loader_exact=True,
            completed_updates=1000, task_layout_sha256=digest_json(layout),
            model=dict(num_tasks=100, task_num_stages=[5]*100),
            files=[dict(path=str(p.relative_to(export)), size=p.stat().st_size, sha256=sha256(p))
                   for p in export.rglob('*') if p.is_file()])
        config = dict(checkpoint_kind='formal_ema_v2', checkpoint='model', verified='model/export.json',
                      completed_updates=1000, model_task_count=100, trained_model_tasks=list(range(100)))
        self.publish(export, manifest, config)
        return export, manifest, config

    def publish(self, export, manifest, config):
        (export / 'export.json').write_text(json.dumps(manifest))
        config['verified_sha256'] = sha256(export / 'export.json')

    def test_valid_formal_export_uses_its_own_mapping_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _, cfg = self.make(root)
            result = verify_policy_assets(root, root, cfg)
            self.assertEqual(result['layout']['dataset_to_model']['99'], 99)
            self.assertEqual(result['expected_model']['num_tasks'], 100)

    def test_engineering_artifact_can_never_enter_physical_serving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export, manifest, cfg = self.make(root)
            manifest.update(purpose='ENGINEERING_ONLY_NOT_FOR_DEPLOYMENT', usable_for_deployment=False)
            self.publish(export, manifest, cfg)
            with self.assertRaises(ValueError): verify_policy_assets(root, root, cfg)

    def test_changed_parameter_file_and_wrong_step_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export, _, cfg = self.make(root)
            with self.assertRaises(ValueError): verify_policy_assets(root, root, dict(cfg, completed_updates=2000))
            (export / 'params/values').write_bytes(b'changed!-ema')
            with self.assertRaises(ValueError): verify_policy_assets(root, root, cfg)

    def test_task_mapping_not_taken_from_stale_public_50_task_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export, manifest, cfg = self.make(root)
            manifest['model']['task_num_stages'][99] = 9
            self.publish(export, manifest, cfg)
            with self.assertRaises(ValueError): verify_policy_assets(root, root, cfg)

    def test_existing_public_parent_receipt_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export, manifest, cfg = self.make(root)
            public = dict(status='all_selected_assets_verified', revision='public-revision', files=manifest['files'])
            (root / 'receipt.json').write_text(json.dumps(public))
            (root / 'layout.json').write_text(json.dumps(dict(layout=dict(dataset_to_model={'0': 2}))))
            config = dict(verified='receipt.json', verified_sha256=sha256(root / 'receipt.json'),
                          revision='public-revision', asset_root='model', checkpoint='model', layout='layout.json')
            result = verify_policy_assets(root, root, config)
            self.assertEqual(result['layout']['dataset_to_model']['0'], 2)
            self.assertIsNone(result['expected_model'])
            with self.assertRaises(ValueError):
                verify_policy_assets(root, root, dict(config, checkpoint='different-model'))

    def test_formal_unlisted_parameters_and_path_aliases_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export, manifest, cfg = self.make(root)
            (export / 'params/unregistered').write_bytes(b'not-in-verified-export')
            with self.assertRaises(ValueError): verify_policy_assets(root, root, cfg)
            manifest['files'][0]['path'] = 'params/../' + manifest['files'][0]['path']
            self.publish(export, manifest, cfg)
            with self.assertRaises(ValueError): verify_policy_assets(root, root, cfg)


if __name__ == '__main__': unittest.main()
