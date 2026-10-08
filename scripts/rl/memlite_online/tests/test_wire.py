"""Standalone transport regression; GPU and simulator are not required."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
AVAILABLE = importlib.util.find_spec('msgpack') is not None


@unittest.skipUnless(AVAILABLE, 'Run in the server model/simulator Python with msgpack')
class WireContracts(unittest.TestCase):
    def test_array_scalar_nested_roundtrip(self):
        from a4_wire import packb, unpackb
        for value in [np.arange(27, dtype=np.float32),
                      np.arange(32*23, dtype=np.float32).reshape(32, 23)[:, ::2],
                      np.ones((3, 32, 32), dtype=np.uint8), np.float32(.5)]:
            decoded = unpackb(packb({'payload': [value], 'task': 'radio'}))
            np.testing.assert_array_equal(decoded['payload'][0], value)
            self.assertEqual(decoded['task'], 'radio')

    def test_reject_non_numeric_object_payload(self):
        from a4_wire import packb
        with self.assertRaises(ValueError):
            packb(np.array([object()], dtype=object))

    def test_g05_wire_byte_identity(self):
        from a4_wire import packb
        original = Path(__file__).resolve().parents[4]/'src/g05/utils/websocket/msgpack.py'
        spec = importlib.util.spec_from_file_location('reference_g05_wire', original)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        value = {'images': np.ones((3, 32, 32), dtype=np.uint8), 'value': np.float32(.5)}
        self.assertEqual(packb(value), reference.packb(value))


if __name__ == '__main__':
    unittest.main()
