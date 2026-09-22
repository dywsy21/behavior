"""The reviewed boundary mode is explicit and cannot reuse an old gate."""
import argparse
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from semantic_robot.v2.protocol import Action
from semantic_robot.v2.servo import SafeServo, ServoLimits
from test_hand_body_collision import calibrated_fixture
from test_v2 import fixture


class JointBoundaryRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((Path(__file__).resolve().parents[2] / 'scripts/semantic_robot/run_v2.py').read_text())

    def test_cli_default_off_and_explicit_enable(self):
        node = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                    and n.args and isinstance(n.args[0], ast.Constant)
                    and n.args[0].value == '--joint-boundary-start-v1')
        parser = argparse.ArgumentParser()
        eval(compile(ast.Expression(node), 'boundary_cli', 'eval'), {'p': parser})
        self.assertIs(parser.parse_args([]).joint_boundary_start_v1, False)
        self.assertIs(parser.parse_args(['--joint-boundary-start-v1']).joint_boundary_start_v1, True)

    def test_dependencies_fail_before_output_or_simulator(self):
        node = next(n for n in ast.walk(self.tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test).startswith('args.joint_boundary_start_v1 and'))
        body = compile(ast.Module(body=[node], type_ignores=[]), 'boundary_dependency', 'exec')
        for enabled in (False, True):
            for geometry in (False, True):
                for gripper in (False, True):
                    args = SimpleNamespace(joint_boundary_start_v1=enabled,
                                           robot_geometry_guards=geometry, gripper_completion_v1=gripper)
                    if enabled and not (geometry and gripper):
                        with self.assertRaises(ValueError):
                            exec(body, {'args': args})
                    else:
                        exec(body, {'args': args})
        output = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'out.mkdir')
        self.assertLess(node.lineno, output.lineno)

    def test_gate_requires_the_same_mode_and_missing_means_legacy(self):
        node = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Compare)
                    and ast.unparse(n.left) == "g.get('joint_boundary_start_v1', False)")
        code = compile(ast.Expression(node), 'boundary_gate', 'eval')
        for gate in ({}, {'joint_boundary_start_v1': False}, {'joint_boundary_start_v1': True}):
            for requested in (False, True):
                self.assertEqual(eval(code, {'g': gate, 'args': SimpleNamespace(joint_boundary_start_v1=requested)}),
                                 gate.get('joint_boundary_start_v1', False) == requested)

    def test_result_records_mode_and_servo_receives_it(self):
        values = [v for n in ast.walk(self.tree) if isinstance(n, ast.Dict)
                  for k, v in zip(n.keys, n.values)
                  if isinstance(k, ast.Constant) and k.value == 'joint_boundary_start_v1']
        self.assertEqual([ast.unparse(v) for v in values], ['args.joint_boundary_start_v1'])
        constructor = next(n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name) and n.func.id == 'ServoLimits')
        code = compile(ast.Expression(constructor), 'boundary_actual_limits', 'eval')
        model, original, _ = calibrated_fixture()
        q = original.q.copy(); q[16] = model.upper[16] - 1.43e-6
        state = model.state(q, original.gripper, np.zeros(3))
        for enabled in (False, True):
            args = SimpleNamespace(robot_geometry_guards=True, gripper_completion_v1=True,
                                   carry_duration_v1=True, joint_boundary_start_v1=enabled)
            limits = eval(code, {'args': args, 'ServoLimits': ServoLimits})
            servo = SafeServo(model, state, limits=limits)
            self.assertEqual(servo.begin(Action('left', 'close'), state), enabled)

    def test_direct_runner_hold_has_no_boundary_check_bypass(self):
        model, original = fixture()
        q = original.q.copy(); q[16] = model.upper[16] - 1.43e-6
        state = model.state(q, original.gripper, np.zeros(3))
        servo = SafeServo(model, state, limits=ServoLimits(joint_boundary_start_v1=True))
        self.assertTrue(servo.begin(Action('left', 'hold'), state))
        command = servo.safe_hold(state)
        np.testing.assert_array_equal(command[:3], np.zeros(3))
        np.testing.assert_allclose(np.r_[command[3:14], command[15:22]], q, atol=1e-7, rtol=0)
        for value in (q[16] + 8e-7, model.upper[16] + 5e-6):
            bad = copy.deepcopy(state); bad.q[16] = value
            with self.assertRaises(ValueError):
                servo.safe_hold(bad)
            self.assertEqual(servo.status, 'JOINT_COMMAND_OUT_OF_BOUNDS')


if __name__ == '__main__':
    unittest.main()
