import unittest
import jax
import jax.numpy as jnp
import numpy as np
import types
from unittest.mock import patch

from behavior_champion.losses import stage_metrics, compute_eval_losses


class StageMetricsTests(unittest.TestCase):
    def test_heldout_wrapper_stays_eval_and_reports_missing_stage_ce_explicitly(self):
        flags = []
        observation = types.SimpleNamespace(tokenized_prompt=jnp.array([[0, 1], [1, 2]]))
        prefix = jnp.zeros((2, 6, 3)).at[:, 2].set(jnp.array([[1., 2., 1000.], [1., 3., 2.]]))
        autoregressive = jnp.array([False, False, False, True, False, True])
        model = types.SimpleNamespace(task_num_stages=(2, 3),
            config=types.SimpleNamespace(use_fast_auxiliary=True, fast_loss_weight=.05, subtask_loss_weight=.1))
        model.eval = lambda: flags.append('model_eval')
        def detailed(*args, **kwargs):
            flags.append(('detailed_train', kwargs['train']))
            return dict(action_loss=jnp.array([1., 2.]), fast_loss=jnp.array([3., 4.]), total_loss=jnp.zeros(2))
        model.compute_detailed_loss = detailed
        model.embed_prefix = lambda obs: (prefix, jnp.ones((2, 6), bool), autoregressive)
        model.PaliGemma = types.SimpleNamespace(llm=lambda *a, **k: ((prefix, None), None))
        model.stage_pred_from_vlm = lambda x: x
        def preprocess(key, obs, *, train):
            flags.append(('preprocess_train', train)); return obs
        modules = {
            'b1k.models.observation': types.SimpleNamespace(preprocess_observation=preprocess),
            'openpi.models.pi0': types.SimpleNamespace(make_attn_mask=lambda *a: None),
        }
        with patch.dict('sys.modules', modules):
            losses = compute_eval_losses(model, jax.random.PRNGKey(31), observation, None)
        self.assertEqual(flags, ['model_eval', ('detailed_train', False), ('preprocess_train', False)])
        expected_ce, _ = stage_metrics(prefix[:, 2], jnp.array([0, 1]), jnp.array([1, 2]), (2, 3))
        np.testing.assert_array_equal(losses['subtask_loss'], expected_ce)
        np.testing.assert_allclose(losses['total_loss'], jnp.array([1., 2.]) + .05 * jnp.array([3., 4.]) + .1 * expected_ce)

    def test_invalid_stage_logits_cannot_win_and_all_examples_remain_separate(self):
        logits = jnp.asarray([[1., 2., 1000.], [1., 3., 2.]])
        ce, accuracy = jax.jit(stage_metrics, static_argnums=3)(logits, jnp.array([0, 1]),
                                                             jnp.array([1, 2]), (2, 3))
        expected = [np.log(np.exp(1.) + np.exp(2.)) - 2.,
                    np.log(np.exp(1.) + np.exp(3.) + np.exp(2.)) - 2.]
        np.testing.assert_allclose(ce, expected, rtol=1e-6)
        np.testing.assert_array_equal(accuracy, [1., 0.])
        self.assertEqual(ce.shape, (2,))


if __name__ == '__main__': unittest.main()
