import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_control_clock import persist_applied_control, require_nonterminal
from recovery_local_teacher import validate_terminal_observation


class ControlClockTests(unittest.TestCase):
    def test_native_end_is_written_before_stopping_and_final_observation_is_aligned(self):
        for terminated, truncated in ((True, False), (False, True), (True, True)):
            stream = io.StringIO()
            row = dict(control_step=0, simulator_apply_ack=True,
                       proprio_before=[0.] * 61, proprio_after=[1.] * 61,
                       physical_audit={'at': 's1'})
            count = persist_applied_control(stream, row, terminated=terminated, truncated=truncated)
            with self.assertRaisesRegex(RuntimeError, 'after recorded control'):
                require_nonterminal(terminated, truncated)
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(len(rows), count)
            self.assertEqual((rows[-1]['terminated'], rows[-1]['truncated']), (terminated, truncated))
            terminal = dict(control_step=count, proprio=[1.] * 61,
                            physical_audit={'at': 's1'}, outcome_candidate=None,
                            has_next_executed_action=False, training_approved=False)
            self.assertTrue(validate_terminal_observation(terminal, rows, 'left'))
            self.assertNotIn('outcome_candidate', rows[-1])

    def test_ordinary_step_and_rejected_unacknowledged_control(self):
        stream = io.StringIO()
        row = dict(control_step=7, simulator_apply_ack=True)
        self.assertEqual(persist_applied_control(stream, row, terminated=False, truncated=False), 8)
        require_nonterminal(False, False)
        for changes in ({'simulator_apply_ack': False}, {'control_step': -1}):
            rejected = io.StringIO()
            with self.assertRaises(ValueError):
                persist_applied_control(rejected, dict(row, **changes), terminated=False, truncated=False)
            self.assertEqual(rejected.getvalue(), '')

    def test_failed_write_cannot_advance_the_recorded_clock(self):
        class Broken(io.StringIO):
            def write(self, value):
                raise OSError('disk failure')
        with self.assertRaises(OSError):
            persist_applied_control(Broken(), dict(control_step=0, simulator_apply_ack=True),
                                    terminated=True, truncated=False)


if __name__ == '__main__':
    unittest.main()
