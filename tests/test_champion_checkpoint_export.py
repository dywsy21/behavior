import copy
import json
from pathlib import Path
import tempfile
import unittest

from flax import struct
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from behavior_champion.checkpoint import save_full
from behavior_champion.checkpoint_export import ema_key, read_checkpoint_ema, verify_parent_assets
from behavior_champion.export import export_ema
from behavior_champion.release import sha256
from behavior_champion.sampling import digest_json


@struct.dataclass
class SavedState:
    step: object
    params: object
    ema_params: object


class SavedExportTests(unittest.TestCase):
    def fixture(self, root):
        params = nnx.State(dict(table=nnx.VariableState(nnx.Param, jnp.zeros((4, 2), dtype=jnp.bfloat16)),
            correlation=nnx.VariableState(nnx.Intermediate, jnp.eye(2))))
        ema = nnx.State(dict(table=nnx.VariableState(nnx.Param, jnp.arange(8, dtype=jnp.bfloat16).reshape(4, 2)),
            correlation=nnx.VariableState(nnx.Intermediate, jnp.eye(2) * .9)))
        layout = dict(num_stages=[5]*100, dataset_to_model={str(i):i for i in range(100)})
        contract = dict(purpose='CHAMPION100_FORMAL_SFT', model=dict(num_tasks=100, task_num_stages=[5]*100),
            layout_sha256=digest_json(layout), sampler_plan_sha256='fixed')
        receipt = save_full(root / 'checkpoints', dict(state=SavedState(jnp.array(3), params, ema)),
            completed=3, contract=contract, cursor=dict(completed_updates=3, plan_sha256='fixed'), reserve_bytes=0)
        return receipt, layout, ema

    def test_saved_not_live_ema_exact_author_restore_including_intermediate_and_bfloat16(self):
        from openpi.models.model import restore_params
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); receipt, layout, ema = self.fixture(root)
            state, actual_receipt, contract = read_checkpoint_ema(receipt['path'],
                manifest_sha256=receipt['manifest_sha256'], layout=layout)
            self.assertEqual(receipt, actual_receipt)
            for name in ('table', 'correlation'):
                np.testing.assert_array_equal(state.ema_params[name].value, np.asarray(ema[name].value))
            assets = root / 'assets'; assets.mkdir(); (assets / 'norm_stats.json').write_text('{}')
            cache = assets / '__pycache__'; cache.mkdir(); (cache / 'generated.pyc').write_bytes(b'not-an-asset')
            exported = export_ema(root / 'export', state, receipt=actual_receipt,
                assets=assets, layout=layout, contract=contract, reserve_bytes=0,
                asset_files={'norm_stats.json':sha256(assets / 'norm_stats.json')})
            self.assertTrue((cache / 'generated.pyc').is_file())
            self.assertFalse((Path(exported['path']) / 'assets/IliaLarchenko/behavior_224_rgb/__pycache__').exists())
            values = restore_params(Path(exported['path']) / 'params', restore_type=np.ndarray)
            self.assertTrue(exported['usable_for_deployment'])
            for name in ('table', 'correlation'):
                self.assertEqual(values[name].dtype, ema[name].value.dtype)
                np.testing.assert_array_equal(values[name], np.asarray(ema[name].value))

    def test_wrong_layout_and_changed_array_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, layout, _ = self.fixture(Path(directory))
            wrong = copy.deepcopy(layout); wrong['num_stages'][0] = 6
            with self.assertRaises(ValueError):
                read_checkpoint_ema(receipt['path'], manifest_sha256=receipt['manifest_sha256'], layout=wrong)
            root = Path(receipt['path']); manifest = json.loads((root / 'manifest.json').read_text())
            row = next(r for r in manifest['leaves'] if '.ema_params' in r['path'])
            raw = root / row['file']; data = bytearray(raw.read_bytes()); data[0] ^= 1; raw.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'array bytes'):
                read_checkpoint_ema(root, manifest_sha256=receipt['manifest_sha256'], layout=layout)

    def test_incomplete_path_and_manifest_mutation_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, layout, _ = self.fixture(Path(directory)); root = Path(receipt['path'])
            incomplete = root.with_name('.incomplete-00000003'); root.rename(incomplete)
            with self.assertRaises(ValueError):
                read_checkpoint_ema(incomplete, manifest_sha256=receipt['manifest_sha256'], layout=layout)
            incomplete.rename(root)
            with (root / 'manifest.json').open('a') as stream: stream.write(' ')
            with self.assertRaises(ValueError):
                read_checkpoint_ema(root, manifest_sha256=receipt['manifest_sha256'], layout=layout)

    def test_path_decoder_refuses_calls_nonstring_keys_and_attribute_traversal(self):
        self.assertEqual(ema_key("['state'].ema_params['a']['b'].value"), ('a', 'b'))
        for suffix in ("['a'].danger.value", "[__import__('os')].value", '[0].value', "['a/b'].value", '.value'):
            with self.assertRaises((ValueError, SyntaxError)):
                ema_key("['state'].ema_params" + suffix)

    def test_parent_asset_subtree_exact_and_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = root / 'parent'
            assets = parent / 'assets/IliaLarchenko/behavior_224_rgb'; assets.mkdir(parents=True)
            norm = assets / 'norm_stats.json'; norm.write_text('{}')
            receipt = root / 'verified.json'
            receipt.write_text(json.dumps(dict(status='all_selected_assets_verified', files=[
                dict(path=norm.relative_to(parent).as_posix(), size=norm.stat().st_size, sha256=sha256(norm))])))
            self.assertEqual(verify_parent_assets(parent, receipt, expected_sha256=sha256(receipt)), assets)
            cache = assets / 'fast_tokenizer/__pycache__'; cache.mkdir(parents=True)
            (cache / 'processing_action_tokenizer.cpython-311.pyc').write_bytes(b'transient')
            selected_assets, selected = verify_parent_assets(parent, receipt,
                expected_sha256=sha256(receipt), with_ledger=True)
            self.assertEqual(selected_assets, assets)
            self.assertEqual(selected, {'norm_stats.json':sha256(norm)})
            (assets / 'unregistered.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_parent_assets(parent, receipt, expected_sha256=sha256(receipt))


if __name__ == '__main__': unittest.main()
