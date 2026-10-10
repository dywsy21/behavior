import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('adapter_verify',Path(__file__).resolve().parents[1]/'scripts/verify_recovery_observer_adapter.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class DiagnosticGroups(unittest.TestCase):
    def test_disjoint_pinned_groups(self):
        rows=[dict(candidate=dict(source_group='a')),dict(candidate=dict(source_group='b'))]
        module.require_diagnostic_groups(rows,{'old'},['a','b'])
        for exposed,expected in [({'a'},['a','b']),({'old'},['a']),({'old'},['a','b','b'])]:
            with self.assertRaises(ValueError):module.require_diagnostic_groups(rows,exposed,expected)


if __name__=='__main__':unittest.main()
