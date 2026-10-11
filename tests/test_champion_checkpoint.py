import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import flax.nnx as nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from behavior_champion.checkpoint import save_full, restore_full, tree_fingerprints
from behavior_champion.optimizer import RowScaledAdamW
from behavior_champion.release import sha256
from behavior_champion.sampling import BalancedPlan
from test_champion_sampling import dataset


@struct.dataclass
class ToyState:
    step: object
    params: object
    opt_state: object
    ema: object
    intermediate: object


class CheckpointTests(unittest.TestCase):
    def setup_state(self):
        params = nnx.State({name: {'embedding': nnx.VariableState(nnx.Param, jnp.ones((5, 3)))}
                            for name in ('task_embeddings', 'task_stage_embeddings')})
        self.tx = RowScaledAdamW(task_old_rows=2, stage_old_rows=3).create(1e-3)
        self.plan = BalancedPlan(dataset(), updates=25)
        return dict(state=ToyState(jnp.array(0), params, self.tx.init(params), params,
                                  jnp.arange(12, dtype=jnp.bfloat16)),
                    train_rng=jax.random.PRNGKey(19))

    def step(self, payload):
        import optax
        state = payload['state']
        key = jax.random.fold_in(payload['train_rng'], state.step)
        grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape), state.params)
        updates, opt = self.tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        ema = jax.tree.map(lambda a, b: .99 * a + .01 * b, state.ema, params)
        return dict(state=dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt, ema=ema),
                    train_rng=payload['train_rng'])

    def test_restore_matches_uninterrupted_rng_optimizer_ema_and_bfloat16(self):
        payload = self.step(self.step(self.setup_state()))
        want = self.step(payload)
        replicate = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        shardings = jax.tree.map(lambda _: replicate, payload)
        template = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), payload)
        with tempfile.TemporaryDirectory() as root:
            contract = {'source': 'abc', 'stage_counts': (5, 6, 7)}
            receipt = save_full(root, payload, completed=2, contract=contract,
                                cursor=self.plan.cursor(2), reserve_bytes=0)
            restored = restore_full(receipt['path'], template, shardings, contract=contract,
                                    plan=self.plan, manifest_sha256=receipt['manifest_sha256'])
            self.assertEqual(tree_fingerprints(payload), tree_fingerprints(restored))
            self.assertEqual(tree_fingerprints(want), tree_fingerprints(self.step(restored)))
            with self.assertRaises(ValueError):
                save_full(root, payload, completed=2, contract={'source': 'abc'},
                          cursor=self.plan.cursor(2), reserve_bytes=0)
            with self.assertRaises(ValueError):
                restore_full(receipt['path'], template, shardings, contract={'source': 'changed'},
                             plan=self.plan, manifest_sha256=receipt['manifest_sha256'])

    def test_corrupt_file_and_cursor_rejected(self):
        payload = self.step(self.setup_state())
        placement = jax.tree.map(lambda _: jax.sharding.SingleDeviceSharding(jax.devices()[0]), payload)
        with tempfile.TemporaryDirectory() as root:
            receipt = save_full(root, payload, completed=1, contract={}, cursor=self.plan.cursor(1), reserve_bytes=0)
            path = Path(receipt['path']); manifest = json.loads((path / 'manifest.json').read_text())
            source = path / manifest['leaves'][0]['file']
            raw = bytearray(source.read_bytes()); raw[0] ^= 1; source.write_bytes(raw)
            with self.assertRaises(ValueError):
                restore_full(path, payload, placement, contract={}, plan=self.plan,
                             manifest_sha256=receipt['manifest_sha256'])
            manifest['sampler_cursor']['completed_updates'] = 0
            (path / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                restore_full(path, payload, placement, contract={}, plan=self.plan,
                             manifest_sha256=sha256(path / 'manifest.json'))


if __name__ == '__main__': unittest.main()
