"""CPU-only export contracts; no servers, torch model or dataset mutation."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('b_export', Path(__file__).resolve().parents[1] /
                                             'scripts/infra/export_memlite_b_benchmark_assets.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Array:
    def __init__(self, shape):
        self.shape = shape
    def __getitem__(self, item):
        assert item == slice(-1, None)
        return Array((1,) + self.shape[1:])
    def clone(self):
        return Array(self.shape)


class ExportTests(unittest.TestCase):
    def original(self):
        return {'samples': {'template': ''.join(f'<image{i}_image_!>' for i in range(18)) + '<EOC>',
                            **{f'image{i}': (256, 256, i) for i in range(18)},
                            'proprio': {'value': Array((6, 27))},
                            'memory': 'unchanged', 'outcome_supervision_mask': False},
                'pixel_values': {camera: Array((6, 3, 256, 256)) for camera in ('a', 'b', 'c')},
                'audit': 'not a model input'}

    def test_current_only_does_not_rewrite_labels(self):
        old = self.original()
        new = module.current_frame(old)
        self.assertEqual(new['samples']['memory'], 'unchanged')
        self.assertIs(new['samples']['outcome_supervision_mask'], False)
        self.assertNotIn('audit', new)
        self.assertEqual([new['samples'][f'image{i}'][2] for i in range(3)], [5, 11, 17])
        self.assertEqual(new['samples']['proprio']['value'].shape, (1, 27))
        self.assertEqual(old['samples']['proprio']['value'].shape, (6, 27))
        self.assertTrue(all(v.shape == (1, 3, 256, 256) for v in new['pixel_values'].values()))
        self.assertTrue(all(v.shape == (6, 3, 256, 256) for v in old['pixel_values'].values()))

    def test_rejects_wrong_camera_or_history(self):
        for field in ('camera', 'template', 'proprio'):
            item = self.original()
            if field == 'camera':
                item['pixel_values'].pop('a')
            elif field == 'template':
                item['samples']['template'] = '<image0_image_!><EOC>'
            else:
                item['samples']['proprio']['value'] = Array((1, 27))
            with self.assertRaises(ValueError):
                module.current_frame(item)

    def test_registered_final_parent(self):
        self.assertEqual(module.WEIGHT_BYTES, 26593013632)
        self.assertEqual(module.WEIGHT.name, 'step_1500.pt')
        self.assertEqual(len(module.WEIGHT_SHA), 64)


if __name__ == '__main__':
    unittest.main()
