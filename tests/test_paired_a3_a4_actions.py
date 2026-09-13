"""Standard-library-only tests; real input/model checks also run on robo."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('paired_actions',
    Path(__file__).resolve().parents[1] / 'scripts/experiments/paired_a3_a4_actions.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PairedActionTests(unittest.TestCase):
    def test_raw_absolute_target_and_current_q(self):
        metric = probe.arm_metrics([[.1]*7]*16, [1.0]*7, [[1.1]*7]*16)
        self.assertLess(metric['error_rms_rad'], 1e-15)
        self.assertAlmostEqual(metric['hold_error_rms_rad'], .1)
        self.assertAlmostEqual(metric['projection_gain'], 1)
        self.assertTrue(metric['beats_current_q_hold'])

    def test_staying_still_is_not_progress(self):
        metric = probe.arm_metrics([[0]*7]*16, [1]*7, [[1.1]*7]*16)
        self.assertFalse(metric['beats_current_q_hold'])
        self.assertIsNone(metric['cosine'])

    def test_wrong_direction_has_negative_projection(self):
        metric = probe.arm_metrics([[-.1]*7]*16, [0]*7, [[.1]*7]*16)
        self.assertAlmostEqual(metric['projection_gain'], -1)
        self.assertFalse(metric['beats_current_q_hold'])

    def test_zero_motion_has_no_invented_cosine(self):
        metric = probe.arm_metrics([[0]*7]*16, [2]*7, [[2]*7]*16)
        self.assertEqual(metric['error_rms_rad'], 0)
        self.assertIsNone(metric['projection_gain'])

    def test_reject_shape_and_nonfinite(self):
        for prediction in ([[0]*7]*15, [[0]*6]*16, [[math.nan]*7]*16):
            with self.assertRaises(ValueError):
                probe.arm_metrics(prediction, [0]*7, [[1]*7]*16)

    def test_exact_pairs_and_repeats(self):
        rows = [dict(frame=f, seed=s, repeat=False) for f in probe.ANCHORS for s in probe.SEEDS]
        self.assertEqual(len(probe.index_rows(rows + [dict(rows[0], repeat=True)])), 84)
        for invalid in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                probe.index_rows(invalid)

    def test_summary_is_paired_and_not_success_rate(self):
        def make(magnitude):
            return [dict(frame=f, seed=s, repeat=False, right_gripper_command_rmse=.2,
                         right_arm=probe.arm_metrics([[magnitude]*7]*16, [0]*7, [[.01]*7]*16))
                    for f in probe.ANCHORS for s in probe.SEEDS]
        result = probe.paired_summary(make(0), make(.01))
        for group in result.values():
            self.assertEqual(group['pairs'], 42)
            self.assertEqual(group['A4_lower_error_count'], 42)
            self.assertLess(group['mean_A4_minus_A3_error_rad'], 0)
            self.assertNotIn('success_rate', group)

    def test_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            probe.write_new(path, {'ok': True})
            with self.assertRaises(FileExistsError):
                probe.write_new(path, {'ok': False})
            self.assertEqual(probe.read(path), {'ok': True})

    def test_diagnostic_budget_and_padding(self):
        self.assertEqual((len(probe.ANCHORS)*len(probe.SEEDS) + 1)*2, 170)
        self.assertEqual(probe.PAD_DIMS, [7, 8, 17, 18])
        self.assertEqual(probe.CHECKPOINTS['A4'][2], 2500)


if __name__ == '__main__':
    unittest.main()
