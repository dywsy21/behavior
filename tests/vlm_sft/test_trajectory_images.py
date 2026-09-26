from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/vlm_sft'))
from trajectory_images import checked_current_pngs, VIEWS


class CurrentImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'images').mkdir()
        self.row = {'frame_index': 32, 'timestamp_s': 32 / 30,
                    'id': 't0_i1_e2_f000032', 'task_index': 0, 'task_instance_id': 1, 'episode_index': 2,
                    'image_references': [{'view': v, 'episode_start_timestamp_s': 2.,
                                          'requested_timestamp_s': 2 + 32 / 30} for v in VIEWS]}
        self.receipts = []
        for view, color in zip(VIEWS, ('red', 'green', 'blue')):
            image = Image.new('RGB', (720 if view == 'head' else 480,) * 2, color)
            path = self.root / 'images' / f'{self.row["id"]}_plus00_{view}.png'; image.save(path)
            data = path.read_bytes()
            self.receipts.append({'view': view, 'offset': 0, 'source_frame': 32,
                'requested_timestamp_s': 2 + 32 / 30, 'actual_timestamp_s': 2 + 32 / 30,
                'path': str(path.relative_to(self.root)), 'bytes': len(data), 'png_sha256': hashlib.sha256(data).hexdigest(),
                'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'resolution': list(image.size)})

    def test_exact_current_three_views_pass(self):
        images = checked_current_pngs(self.row, self.receipts, self.root)
        self.assertEqual(list(images), list(VIEWS))
        self.assertEqual(images['left_wrist'].getpixel((0, 0)), (0, 128, 0))

    def test_future_frame_under_legal_head_key_fails(self):
        for change in ({'offset': 8}, {'source_frame': 40}, {'actual_timestamp_s': 2 + 40 / 30},
                       {'requested_timestamp_s': 2 + 40 / 30}):
            receipts = deepcopy(self.receipts); receipts[0].update(change)
            with self.assertRaises(ValueError): checked_current_pngs(self.row, receipts, self.root)

    def test_swapped_wrist_pixels_fail_hash(self):
        (self.root / self.receipts[1]['path']).write_bytes((self.root / self.receipts[2]['path']).read_bytes())
        with self.assertRaises(ValueError): checked_current_pngs(self.row, self.receipts, self.root)

    def test_updated_hash_cannot_approve_placeholder_resolution(self):
        small = Image.new('RGB', (1, 1), 'red'); path = self.root / self.receipts[0]['path']; small.save(path)
        data = path.read_bytes(); receipts = deepcopy(self.receipts)
        receipts[0].update(bytes=len(data), png_sha256=hashlib.sha256(data).hexdigest(),
                           pixels_sha256=hashlib.sha256(small.tobytes()).hexdigest(), resolution=[1, 1])
        with self.assertRaises(ValueError): checked_current_pngs(self.row, receipts, self.root)

    def test_other_case_same_clock_receipts_cannot_join(self):
        other = deepcopy(self.row)
        other.update(id='t0_i3_e4_f000032', task_instance_id=3, episode_index=4)
        # Every timestamp, frame and original pixel hash is valid, but wrong case.
        with self.assertRaises(ValueError): checked_current_pngs(other, self.receipts, self.root)

    def test_duplicate_views_and_path_escape_fail(self):
        with self.assertRaises(ValueError): checked_current_pngs(self.row, [self.receipts[0]] * 3, self.root)
        receipts = deepcopy(self.receipts); receipts[0]['path'] = '../head.png'
        with self.assertRaises(ValueError): checked_current_pngs(self.row, receipts, self.root)


if __name__ == '__main__': unittest.main()
