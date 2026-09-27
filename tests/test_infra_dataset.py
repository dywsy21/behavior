import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "infra_download", Path(__file__).resolve().parents[1] / "scripts/infra/download_behavior2026.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_path, verify_file = _module.validate_path, _module.verify_file
DiskAdmission = _module.DiskAdmission
download_retry = _module.download_retry


class DatasetIntegrityTests(unittest.TestCase):
    def test_stream_retry_is_bounded(self):
        from requests.exceptions import ChunkedEncodingError
        attempts = []
        sleeps = []
        def broken():
            attempts.append(1)
            raise ChunkedEncodingError("broken stream")
        with self.assertRaises(ChunkedEncodingError):
            download_retry(broken, (ChunkedEncodingError,), sleep=sleeps.append)
        self.assertEqual(len(attempts), 5)
        self.assertEqual(sleeps, [2, 4, 8, 16])

    def test_http_error_classification(self):
        from requests import Response
        from requests.exceptions import HTTPError
        for status, expected in [(401, 1), (403, 1), (404, 1), (429, 5), (503, 5)]:
            calls = []
            response = Response()
            response.status_code = status
            def fail():
                calls.append(1)
                raise HTTPError(response=response)
            with self.assertRaises(HTTPError):
                download_retry(fail, (HTTPError,), sleep=lambda _: None)
            self.assertEqual(len(calls), expected)

    def test_stream_retry_does_not_hide_hash_errors(self):
        def invalid():
            raise ValueError("bad digest")
        with self.assertRaises(ValueError):
            download_retry(invalid, (ConnectionError,), sleep=lambda _: None)

    def test_inflight_reserve_and_release(self):
        admission = DiskAdmission(lambda: 100, 60)
        with admission.admit(30):
            with self.assertRaises(RuntimeError):
                with admission.admit(20):
                    self.fail("Overcommitted disk reserve")
            self.assertEqual(admission.inflight, 30)
        self.assertEqual(admission.inflight, 0)
        with self.assertRaises(ValueError):
            with admission.admit(40):
                raise ValueError("download failed")
        self.assertEqual(admission.inflight, 0)

    def test_paths(self):
        for name in ("/tmp/a", "a/../../etc/passwd", ""):
            with self.assertRaises(ValueError):
                validate_path(name)
        self.assertEqual(str(validate_path("meta/info.json")), "meta/info.json")

    def test_blob_and_lfs_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b"official content\n"
            (root / "item").write_bytes(data)
            item = dict(path="item", size=len(data),
                        blob_id=hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest())
            self.assertTrue(verify_file(root, item))
            item["sha256"] = hashlib.sha256(data).hexdigest()
            self.assertTrue(verify_file(root, item))
            (root / "item").write_bytes(b"x" * len(data))
            self.assertFalse(verify_file(root, item))

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target").write_bytes(b"ok")
            (root / "item").symlink_to(root / "target")
            self.assertFalse(verify_file(root, dict(path="item", size=2, blob_id="unknown")))


if __name__ == "__main__":
    unittest.main()
