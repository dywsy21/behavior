from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from behavior_champion.storage import check_storage, storage_plan


class StorageTests(unittest.TestCase):
    def test_full_run_reserves_all_eight_plus_emergency_and_ema(self):
        result = storage_plan(completed=0, updates=8000, checkpoint_every=1000)
        self.assertEqual(result['planned_new_snapshots'], 9)
        self.assertEqual(result['required_free_bytes'], 775 << 30)
        self.assertFalse(result['deletes_existing_artifacts'])

    def test_resume_counts_remaining_not_already_retained_snapshots(self):
        self.assertEqual(storage_plan(completed=3000, updates=8000, checkpoint_every=1000)['planned_new_snapshots'], 6)
        self.assertEqual(storage_plan(completed=3, updates=3, checkpoint_every=2)['planned_new_snapshots'], 1)
        self.assertEqual(storage_plan(completed=0, updates=3, checkpoint_every=2)['planned_new_snapshots'], 3)

    def test_insufficient_space_refuses_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'important'; original.write_bytes(b'keep')
            with patch('behavior_champion.storage.shutil.disk_usage', return_value=types.SimpleNamespace(free=774 << 30)):
                with self.assertRaises(OSError): check_storage(tmp, completed=0, updates=8000, checkpoint_every=1000)
            self.assertEqual(original.read_bytes(), b'keep')

    def test_valid_space_and_bad_schedule(self):
        with patch('behavior_champion.storage.shutil.disk_usage', return_value=types.SimpleNamespace(free=900 << 30)):
            self.assertEqual(check_storage('.', completed=0, updates=8000, checkpoint_every=1000)['observed_free_bytes'], 900 << 30)
        for values in ((-1, 8000, 1000), (0, 8000, 0), (True, 8000, 1000)):
            with self.assertRaises(ValueError): storage_plan(completed=values[0], updates=values[1], checkpoint_every=values[2])
