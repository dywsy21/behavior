"""Execute the REAL nested/outer failure handlers without importing a simulator."""
import ast
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class VirtualPath:
    def __truediv__(self, name):
        result = VirtualPath(); result.name = name
        return result
    def exists(self): return False


@contextmanager
def fake_environment(*args, **kwargs):
    yield object()


class RunnerFailureCleanupTests(unittest.TestCase):
    def run_handler(self, fault=None, terminal=False, primary=True, issued_grips=None):
        path = Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py"
        module = ast.parse(path.read_text())
        main = next(x for x in module.body if isinstance(x, ast.FunctionDef) and x.name == "main")
        inner = copy.deepcopy(next(x for x in main.body if isinstance(x, ast.FunctionDef) and x.name == "recorded_session"))
        # Same body, module test scope replaces main's closure only.
        inner.body = [ast.Global(names=x.names) if isinstance(x, ast.Nonlocal) else x for x in inner.body]
        outer = copy.deepcopy(next(x for x in main.body if isinstance(x, ast.Try) and x.finalbody))
        outer.body = ast.parse("with recorded_session():\n    raise primary_error" if primary else "with recorded_session():\n    pass").body
        fn = ast.FunctionDef(name="exercise", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=[outer], decorator_list=[])
        errors = []; records = []; error = RuntimeError("PRIMARY")
        trace, video, servo, io_trace = Mock(), Mock(), Mock(), Mock()
        servo.grips=[1.,1.]
        def writer(path, value):
            if fault in ("all_write", path.name): raise OSError("WRITE_SECONDARY")
            records.append((path.name, copy.deepcopy(value)))
        state = object(); servo.safe_hold.return_value = [0., 0., 0., -1.]
        state_now = Mock(return_value=state); step = Mock()
        if fault == "state": state_now.side_effect = RuntimeError("STATE_SECONDARY")
        if fault == "step": step.side_effect = RuntimeError("STEP_SECONDARY")
        if fault == "trace_write": trace.write.side_effect = OSError("TRACE_SECONDARY")
        if fault == "trace_close": trace.close.side_effect = OSError("CLOSE_SECONDARY")
        if fault == "video_close": video.close.side_effect = OSError("CLOSE_SECONDARY")
        if fault == "io_close": io_trace.close.side_effect = OSError("CLOSE_SECONDARY")
        ns = dict(contextmanager=contextmanager, session_factory=fake_environment, window=None,
                  args=SimpleNamespace(gpu=3, odometry_substep_controls=6, max_controls=12),
                  controls=6, out=VirtualPath(), write=writer, policy=None, phase="CONTROL", reset_completed=True,
                  terminal=terminal, prefix_count=0, replay_count=0, decisions=[], digest="test", servo=servo,
                  last_issued_grips=issued_grips,native_io=None,
                  step=step, state_now=state_now, trace=trace, video=video, io_trace=io_trace, json=json, sys=sys,
                  primary_error=error, secondary_error_report=errors.append)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[inner, fn], type_ignores=[])), str(path), "exec"), ns)
        caught = None
        try: ns["exercise"]()
        except BaseException as exc: caught = exc
        return ns, caught, error, records

    def test_primary_survives_each_record_stop_and_close_failure(self):
        for fault in (None, "failure.json", "all_write", "safety_hold_after_failure.json", "state", "step",
                      "trace_write", "trace_close", "video_close", "io_close"):
            with self.subTest(fault=fault):
                ns, caught, primary, records = self.run_handler(fault)
                self.assertIs(caught, primary)
                ns["trace"].close.assert_called_once(); ns["video"].close.assert_called_once()
                ns["io_trace"].close.assert_called_once()
                if fault not in ("state", "step"):
                    self.assertEqual(ns["controls"], 7)
                    ns["step"].assert_called_once_with([0., 0., 0., -1.])
                if fault == "trace_write":
                    stop = next(value for name, value in records if name == "safety_hold_after_failure.json")
                    self.assertTrue(stop["completed"])

    def test_terminal_does_not_take_extra_physics_step(self):
        ns, caught, primary, _ = self.run_handler(terminal=True)
        self.assertIs(caught, primary); ns["step"].assert_not_called()
        self.assertEqual(ns["controls"], 6)

    def test_cancelled_new_open_or_close_is_not_applied_by_cleanup(self):
        ns,caught,primary,_=self.run_handler(issued_grips=[-1.,1.])
        self.assertIs(caught,primary)
        self.assertEqual(ns["servo"].grips,[-1.,1.])
        ns["step"].assert_called_once()

    def test_close_error_without_primary_is_not_silently_discarded(self):
        ns, caught, _, _ = self.run_handler(fault="trace_close", primary=False)
        self.assertIsInstance(caught, OSError)
        ns["video"].close.assert_called_once()


if __name__ == "__main__": unittest.main()
