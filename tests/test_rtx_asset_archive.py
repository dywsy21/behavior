import importlib.util
import io
from pathlib import Path
import stat
import sys
import unittest
import zipfile

FILE = Path(__file__).resolve().parents[1]/'scripts/rl/rtx_assets.py'
spec = importlib.util.spec_from_file_location('rtx_asset_preparation', FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.path.insert(0, str(FILE.parent))
from rtx_asset_resume import ranges


class ArchiveTests(unittest.TestCase):
    def test_range_resume_never_repeats_or_skips_a_byte(self):
        self.assertEqual(ranges(9, 20, 4), [(9, 12), (13, 16), (17, 19)])
        self.assertEqual(ranges(20, 20), [])
        for args in ((-1, 20), (21, 20), (1, 20, 0)):
            with self.assertRaises(ValueError): ranges(*args)

    def archive(self, name, symlink=False):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            info = zipfile.ZipInfo(name)
            if symlink: info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, 'x')
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_normal_member(self):
        with self.archive('models/radio/model.usd') as z:
            self.assertEqual(len(mod.safe_members(z)), 1)

    def test_escape_or_link_rejected(self):
        for name in ('../outside', '/absolute', 'models/../../outside', 'models\\outside'):
            with self.archive(name) as z, self.assertRaises(ValueError): mod.safe_members(z)
        with self.archive('link', True) as z, self.assertRaises(ValueError): mod.safe_members(z)


if __name__ == '__main__': unittest.main()
