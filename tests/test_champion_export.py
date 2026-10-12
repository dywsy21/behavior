import json
from pathlib import Path
import tempfile
import types
import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from behavior_champion.export import export_ema
from behavior_champion.release import sha256
from behavior_champion.checkpoint import tree_fingerprints
from behavior_champion.sampling import digest_json


class ExportTests(unittest.TestCase):
    def test_author_loader_roundtrip_keeps_ema_and_bfloat16(self):
        from openpi.models.model import restore_params
        params = nnx.State(dict(table=nnx.VariableState(nnx.Param, jnp.arange(8, dtype=jnp.bfloat16).reshape(4, 2)),
                                correlation=nnx.VariableState(nnx.Intermediate, jnp.eye(2))))
        state = types.SimpleNamespace(step=3, ema_params=params)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'checkpoint'; source.mkdir()
            contract = dict(purpose='ENGINEERING_ONLY_NOT_FOR_DEPLOYMENT', model={})
            leaves = tree_fingerprints(params)['leaves']
            for row in leaves: row['path'] = "['state'].ema_params" + row['path']
            (source / 'manifest.json').write_text(json.dumps(dict(completed_updates=3,
                contract_sha256=digest_json(contract), leaves=leaves)))
            assets = root / 'assets'; assets.mkdir(); (assets / 'norm_stats.json').write_text('{"test":1}')
            receipt = dict(path=str(source), completed_updates=3, manifest_sha256=sha256(source / 'manifest.json'))
            exported = export_ema(root / 'exports', state, receipt=receipt, assets=assets,
                layout={'test': 1}, contract=contract, reserve_bytes=0)
            actual = restore_params(Path(exported['path']) / 'params', restore_type=np.ndarray)
            np.testing.assert_array_equal(actual['table'], np.asarray(params['table'].value))
            self.assertEqual(actual['table'].dtype, params['table'].value.dtype)
            self.assertFalse(exported['usable_for_deployment'])
            self.assertTrue(json.loads((Path(exported['path']) / 'export.json').read_text())['restored_by_author_loader_exact'])
            with self.assertRaises(FileExistsError):
                export_ema(root / 'exports', state, receipt=receipt, assets=assets,
                    layout={}, contract=contract, reserve_bytes=0)
            params['table'].value = params['table'].value + 1
            with self.assertRaises(ValueError):
                export_ema(root / 'other', state, receipt=receipt, assets=assets,
                    layout={}, contract=contract, reserve_bytes=0)


if __name__ == '__main__': unittest.main()
