from types import SimpleNamespace
import unittest

from semantic_robot.v2.run_budget import validate_run_budget


def args(**changes):
    values = dict(mode="agent", harness="grounded", prefix=0,
                  replay_prefix_spec=None, max_decisions=96,
                  max_controls=3072, max_seconds=2400)
    values.update(changes)
    return SimpleNamespace(**values)


class RunBudgetTests(unittest.TestCase):
    def test_default_preserves_pilot_ceiling(self):
        self.assertEqual(validate_run_budget(args()), (96, 3072, 2400))
        for field, value in [("max_decisions",97), ("max_controls",3073),
                             ("max_seconds",2401), ("prefix",449)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_run_budget(args(**{field:value}))

    def test_explicit_fullstart_within_registered_limits(self):
        self.assertEqual(validate_run_budget(args(budget_profile="fullstart192",
                         max_decisions=192,max_controls=6144,max_seconds=7200)),
                         (192,6144,7200))

    def test_fullstart_cannot_expand_gates_or_prefixed_diagnostics(self):
        for changes in [dict(prefix=1),dict(replay_prefix_spec="saved.json"),
                        dict(replay_prefix_spec=""),dict(mode="gate"),dict(harness="v2")]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_run_budget(args(budget_profile="fullstart192",**changes))

    def test_fullstart_never_bypasses_finite_caps(self):
        for field, value in [("max_decisions",193),("max_controls",6145),
                             ("max_seconds",7201),("max_seconds",float("inf")),
                             ("max_controls",0),("max_decisions",True)]:
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                validate_run_budget(args(budget_profile="fullstart192",**{field:value}))

    def test_unknown_profile_and_invalid_prefix_fail(self):
        for changes in [dict(budget_profile="unlimited"),dict(prefix=-1),dict(prefix=False)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_run_budget(args(**changes))


if __name__ == "__main__":
    unittest.main()
