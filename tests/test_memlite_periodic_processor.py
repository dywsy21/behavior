import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

spec = importlib.util.spec_from_file_location('periodic_processor_check',
    Path(__file__).resolve().parents[1] / 'scripts/infra/check_memlite_periodic_processor.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CameraIdentityTests(unittest.TestCase):
    def test_categorical_attention_counts_positions_not_modality_values(self):
        self.assertEqual(module.valid_token_count(torch.tensor([0, 1, 2, 3, 0, 4])), 4)
        self.assertEqual(module.valid_token_count(torch.tensor([0, 0])), 0)

    def fixture(self):
        pairs = (('head_rgb', 'exterior'), ('left_wrist_rgb', 'wrist_left'),
                 ('right_wrist_rgb', 'wrist_right'))
        original = {key: torch.full((1, 3, 4, 4), number + 10, dtype=torch.uint8)
                    for number, (key, _) in enumerate(pairs)}
        # Distinct RGB and distinct transforms catch key-only or shape-only checks.
        def transform(offset):
            return lambda value: value.float() / 255 + offset
        transforms = {key: [transform(number / 10)] for number, (key, _) in enumerate(pairs)}
        processor = SimpleNamespace(shape_meta={'images': [dict(key=key, camera_type=camera)
                                   for key, camera in pairs]}, val_transforms=transforms,
                                   samples_builder=SimpleNamespace(_image_keys=list(original)))
        sample = {'pixel_values': {camera: transforms[key][0](original[key]) for key, camera in pairs}}
        return processor, original, sample

    def test_raw_keys_are_intentionally_renamed_to_camera_types(self):
        processor, original, sample = self.fixture()
        mapping = module.validate_processed_cameras(processor, original, sample)
        self.assertEqual(mapping['head_rgb'], 'exterior')
        self.assertEqual(mapping['left_wrist_rgb'], 'wrist_left')

    def test_same_shape_wrist_swap_or_one_changed_pixel_rejected(self):
        processor, original, sample = self.fixture()
        swapped = copy.deepcopy(sample)
        swapped['pixel_values']['wrist_left'], swapped['pixel_values']['wrist_right'] = (
            swapped['pixel_values']['wrist_right'], swapped['pixel_values']['wrist_left'])
        with self.assertRaises(ValueError): module.validate_processed_cameras(processor, original, swapped)
        for value in (0., float('nan')):
            changed = copy.deepcopy(sample)
            changed['pixel_values']['exterior'][0, 0, 0, 0] = value
            with self.assertRaises(ValueError): module.validate_processed_cameras(processor, original, changed)

    def test_raw_template_or_output_reorder_rejected(self):
        processor, original, sample = self.fixture()
        with self.assertRaises(ValueError):
            module.validate_processed_cameras(processor, dict(reversed(list(original.items()))), sample)
        with self.assertRaises(ValueError):
            module.validate_processed_cameras(processor, original,
                {'pixel_values': dict(reversed(list(sample['pixel_values'].items())))})
        processor.samples_builder._image_keys.reverse()
        with self.assertRaises(ValueError): module.validate_processed_cameras(processor, original, sample)

    def test_metadata_camera_swap_rejected(self):
        processor, original, sample = self.fixture()
        processor.shape_meta['images'][1]['camera_type'] = 'wrist_right'
        with self.assertRaises(ValueError): module.validate_processed_cameras(processor, original, sample)


if __name__ == '__main__': unittest.main()
