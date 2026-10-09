"""User-authorized unbounded preparation still preserves every attempt."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'tools/check_expert_prefix_sim.py'
spec = importlib.util.spec_from_file_location('prefix_check', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AttemptLedgerTest(unittest.TestCase):
    def test_no_quota_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'budget.json').write_text('{"historical": true}')
            for i in range(20):
                module.reserve(root, 'same_source', 'code', root/f'attempt-{i}')
            ledger = json.loads((root/'attempts.json').read_text())
            self.assertIsNone(ledger['quota'])
            self.assertEqual(len(ledger['attempts']), 20)
            self.assertEqual(json.loads((root/'budget.json').read_text()), {'historical': True})
            with self.assertRaises(ValueError):
                module.reserve(root, 'same_source', 'code', root/'attempt-0')


if __name__ == '__main__':
    unittest.main()
