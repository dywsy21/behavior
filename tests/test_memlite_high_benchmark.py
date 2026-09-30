"""Pure CPU arithmetic/profile tests, without importing torch or opening SSH."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'scripts/infra'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load('benchmark_memlite_high')
counter = load('count_memlite_stride_candidates')


class BenchTests(unittest.TestCase):
    def test_global_256_for_all_capacity_options(self):
        for micro in (4, 8, 16, 32):
            row = bench.batch_plan(micro)
            self.assertEqual(row['world'] * row['micro'] * row['accumulation'], 256)
            self.assertEqual(row['warmup'] + row['measured'], 18)
            self.assertEqual(bench.batch_plan(micro, True)['measured'], 1)
        for micro in (True, 0, 2, 64):
            with self.assertRaises(ValueError):
                bench.batch_plan(micro)

    def test_global_sharding_includes_every_draw_once(self):
        for micro in (4, 8, 16, 32):
            positions = []
            for accum in range(32 // micro):
                for rank in range(8):
                    start = (accum * 8 + rank) * micro
                    positions.extend(range(start, start+micro))
            self.assertEqual(positions, list(range(256)))

    def test_token_weighted_accumulation_not_mean_of_means(self):
        weights, losses = [1., 4., 3., 2.], [8., 2., 4., 1.]
        global_weight = sum(weights)
        reduced = sum(l*w*8/global_weight for l,w in zip(losses, weights))/8
        self.assertEqual(reduced, sum(l*w for l,w in zip(losses, weights))/global_weight)
        self.assertNotEqual(reduced, sum(losses)/len(losses))

    def test_actual_longest_profile(self):
        rows = list(range(30))
        tokens = [dict(sequence_tokens=(i*7)%31) for i in rows]
        chosen = bench.profile_indices(rows, tokens, 'long')
        self.assertEqual(len(chosen), 8)
        self.assertEqual([tokens[i]['sequence_tokens'] for i in chosen], sorted(x['sequence_tokens'] for x in tokens)[-8:])
        self.assertEqual(bench.profile_indices(rows, tokens, 'mixed'), rows)

    def test_stride_phase_is_fixed_and_count_is_exact(self):
        for length in (0, 1, 15, 16, 17, 31, 32, 1956):
            for offset in range(16):
                self.assertEqual(counter.count_anchors(length, offset), len(range(offset, length, 16)))
        self.assertEqual(counter.phase(1, 230, 7), counter.phase(1, 230, 7))
        self.assertTrue(all(0 <= counter.phase(1, i, i) < 16 for i in range(200)))
        for offset in (-1, 16, True):
            with self.assertRaises(ValueError):
                counter.count_anchors(100, offset)


if __name__ == '__main__':
    unittest.main()
