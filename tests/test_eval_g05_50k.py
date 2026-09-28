import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1]/'scripts/experiments'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('bounded_50k_eval', DIRECTORY/'eval_g05_50k.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class FixedCheckpointTests(unittest.TestCase):
    def test_new_paths_and_same_budget(self):
        self.assertNotEqual(evaluation.ROOT, evaluation.previous.ROOT)
        self.assertEqual(evaluation.STEP, 50000)
        self.assertEqual(sum(evaluation.core.LIMITS[t] for t in evaluation.SIM_TASKS), 4248)
        self.assertEqual(sum((evaluation.core.LIMITS[t]+15)//16 for t in evaluation.SIM_TASKS), 266)
        self.assertEqual(evaluation.WALL_SECONDS, 3600)

    def test_previous_evidence_is_read_only_and_pinned(self):
        value = ({'mean': {'fm_loss': .1}}, [], {'checkpoint': 'old'}, evaluation.ROWS_SHA)
        with patch.object(evaluation.previous, 'read_candidate', return_value=value) as read:
            self.assertEqual(evaluation.evidence(), (value[0], value[2]))
            read.assert_called_once_with(50000)
        with patch.object(evaluation.previous, 'read_candidate', return_value=(*value[:3], 'changed')):
            with self.assertRaises(ValueError):
                evaluation.evidence()

    def test_deployment_must_match_validated_load(self):
        expected = {k: k for k in evaluation.LOAD_FIELDS}
        evaluation.check_load(dict(expected, source_commit='new'), expected)
        for key in evaluation.LOAD_FIELDS:
            with self.assertRaises(ValueError):
                evaluation.check_load(dict(expected, **{key: 'changed'}), expected)

    def test_identity_rejects_different_checkpoint_or_protocol(self):
        row = dict(checkpoint_step=50000, checkpoint=str(evaluation.core.checkpoint(50000)),
                   port=8933, action_source='fm', robot_action_dim=23, obs_steps=1,
                   predicted_steps=32, execute_steps=16, predict_cot=False, memlite=False)
        evaluation.check_identity(row)
        for key, value in [('checkpoint_step', 20000), ('port', 8932), ('action_source', 'ar'),
                           ('robot_action_dim', 27), ('obs_steps', 6), ('predict_cot', True)]:
            with self.assertRaises(ValueError):
                evaluation.check_identity(dict(row, **{key: value}))


if __name__ == '__main__':
    unittest.main()
