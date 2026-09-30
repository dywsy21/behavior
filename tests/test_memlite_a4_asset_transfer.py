"""CPU-only arithmetic/integrity tests; never connects to a server."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "asset_sync", Path(__file__).resolve().parents[1] / "scripts/infra/sync_memlite_a4_checkpoint.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class TransferTests(unittest.TestCase):
    def test_exact_nonoverlapping_ranges(self):
        for size, chunk in [(1, 8), (16, 8), (19, 8), (sync.SIZE, 16 * 1024**2)]:
            with self.subTest(size=size, chunk=chunk):
                rows = sync.chunks(size, chunk)
                self.assertEqual(sum(n for _, _, n in rows), size)
                end = 0
                for i, (index, start, length) in enumerate(rows):
                    self.assertEqual(index, i)
                    self.assertEqual(start, end)
                    self.assertTrue(0 < length <= chunk)
                    end += length
                self.assertEqual(end, size)


    def test_invalid_sizes(self):
        for size, chunk in [(0, 1), (1, 0), (-1, 8)]:
            with self.subTest(size=size, chunk=chunk), self.assertRaises(ValueError):
                sync.chunks(size, chunk)


    def test_digest_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "data.bin"
            p.write_bytes(b"abc")
            self.assertEqual(sync.digest(p), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            state = Path(directory) / "status.json"
            sync.publish_json(state, {"state": "running"})
            sync.publish_json(state, {"state": "complete"})
            self.assertIn('"complete"', state.read_text())
            self.assertFalse(state.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
