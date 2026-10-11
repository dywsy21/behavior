import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx

from behavior_champion.optimizer import RowScaledAdamW


class OptimizerTests(unittest.TestCase):
    def test_actual_nnx_state_old_rows_and_moments_unchanged_new_updates_fourfold(self):
        params = nnx.State({'task_embeddings': {'embedding': nnx.VariableState(nnx.Param, jnp.ones((5, 3)))},
                            'task_stage_embeddings': {'embedding': nnx.VariableState(nnx.Param, jnp.ones((7, 3)))},
                            'other': nnx.VariableState(nnx.Param, jnp.ones((3, 3)))})
        cfg = RowScaledAdamW(task_old_rows=2, stage_old_rows=4)
        tx = cfg.create(1e-3)
        ref = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(1e-3, b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10))
        state, ref_state = tx.init(params), ref.init(params)
        for index in range(3):
            grads = jax.tree.map(lambda x: jnp.arange(x.size).reshape(x.shape) + .1 + index, params)
            updates, state = tx.update(grads, state, params)
            baseline, ref_state = ref.update(grads, ref_state, params)
            for a, b in zip(jax.tree.leaves(state[:2]), jax.tree.leaves(ref_state), strict=True):
                np.testing.assert_array_equal(a, b)
            for table, old in [('task_embeddings', 2), ('task_stage_embeddings', 4)]:
                got, wanted = updates[table]['embedding'].value, baseline[table]['embedding'].value
                np.testing.assert_array_equal(got[:old], wanted[:old])
                np.testing.assert_array_equal(got[old:], 4 * wanted[old:])
            np.testing.assert_array_equal(updates['other'].value, baseline['other'].value)

    def test_unexpanded_tables_rejected(self):
        tx = RowScaledAdamW().create(1e-3)
        with self.assertRaises(ValueError): tx.init({'unrelated': jnp.ones((3, 3))})


if __name__ == '__main__': unittest.main()
