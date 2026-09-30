import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('io_probe', Path(__file__).resolve().parents[1] / 'scripts/infra/benchmark_behavior2026_io.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IOProbeBounds(unittest.TestCase):
    def test_anchors_keep_the_whole_future_horizon(self):
        for length in (64, 65, 1956, 100000):
            frames = module.anchors(length)
            self.assertEqual(len(set(frames)), 4)
            self.assertEqual(frames, sorted(frames))
            self.assertTrue(all(0 <= frame and frame + 32 <= length for frame in frames))

    def test_short_invalid_episodes_are_rejected(self):
        for value in (None, 0, 31, 63, '1956'):
            with self.assertRaises(ValueError):
                module.anchors(value)

    def test_stride16_changes_only_observation_starts(self):
        for length in (80, 81, 1956, 100000):
            frames = module.anchors(length, stride=16)
            self.assertEqual(len(set(frames)), 4)
            self.assertEqual(frames, sorted(frames))
            self.assertTrue(all(frame % 16 == 0 and frame + 32 <= length for frame in frames))
        for length in (64, 65, 79):
            with self.assertRaises(ValueError):
                module.anchors(length, stride=16)
        for stride in (0, 2, 32, True):
            with self.assertRaises(ValueError):
                module.anchors(1000, stride=stride)

    def test_scope_is_rgb_only(self):
        self.assertEqual(len(module.KEYS), 3)
        self.assertTrue(all(key.startswith('observation.rgb.') for key in module.KEYS))


if __name__ == '__main__':
    unittest.main()
