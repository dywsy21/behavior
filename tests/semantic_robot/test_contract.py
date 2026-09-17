"""CPU tests runnable with unittest; no torch, G05, simulator or model import."""
import copy
import unittest

import numpy as np

from semantic_robot.actions import parse_action, displacement, rotation, Unit, action_language
from semantic_robot.control import R1ProServo, RobotState, dls
from semantic_robot.prompts import user_text


def state():
    poses = {"left": (np.array([.4,.3,.8]), np.array([0.,0.,0.,1.])),
             "right": (np.array([.4,-.3,.8]), np.array([0.,0.,0.,1.])),
             "torso": (np.array([0.,0.,.8]), np.array([0.,0.,0.,1.]))}
    jac = {k: np.zeros((6,18)) for k in poses}
    jac["left"][:,4:10] = np.eye(6)
    jac["right"][:,11:17] = np.eye(6)
    jac["torso"][:3,:3] = np.eye(3)
    return RobotState(np.full(18,.1), np.full(18,-3.), np.full(18,3.), poses, jac, np.array([.05,.05]), np.zeros(3))


class ContractTests(unittest.TestCase):
    def test_entire_constrained_language_is_valid_unique(self):
        lines=action_language()
        self.assertEqual(len(lines),len(set(lines)))
        self.assertGreater(len(lines),800)
        for line in lines: parse_action(line)

    def test_parser_all_groups(self):
        for text in ("R FWD FINE", "L ROLL_NEG COARSE", "BASE YAW_POS COARSE", "TORSO UP FINE", "BOTH CLOSE", "L UP FINE ; R DOWN FINE", "HOLD", "DONE", "MODE CARRY"):
            with self.subTest(text=text):
                self.assertTrue(parse_action(text))

    def test_invalid_no_execution(self):
        for text in ("", "R FWD", "R FWD 100", "L OPEN COARSE", "BASE UP FINE", "TORSO LEFT FINE", "L CLOSE; L OPEN", "BOTH UP FINE; R UP FINE", "BASE FWD FINE; R UP FINE", "R UP FINE\nI think this is good", "GRASP radio", "R FWD FINE; L FWD FINE; R CLOSE"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_action(text)

    def test_coordinates_and_granularity(self):
        np.testing.assert_array_equal(displacement(Unit("R","LEFT","COARSE")), [0,.05,0])
        np.testing.assert_array_equal(displacement(Unit("L","FWD"),True), [.01,0,0])
        self.assertAlmostEqual(rotation(Unit("L","PITCH_NEG"))[1], -np.pi/36)

    def test_hold_is_not_zero_pose(self):
        s = state(); servo = R1ProServo(s)
        servo.begin(parse_action("HOLD"),s)
        action = servo.next_action(s)
        self.assertEqual(action.shape,(23,))
        np.testing.assert_allclose(action[3:14], .1)
        np.testing.assert_allclose(action[15:22], .1)
        np.testing.assert_array_equal(action[:3], 0)

    def test_other_arm_and_trunk_preserved(self):
        s=state(); servo=R1ProServo(s); servo.begin(parse_action("R UP FINE"),s)
        for _ in range(12): action=servo.next_action(s)
        np.testing.assert_allclose(action[3:14],.1)
        self.assertGreater(action[17], .1)
        self.assertEqual(action[14],1)

    def test_grip_latched_until_explicit_open(self):
        s=state(); servo=R1ProServo(s)
        for command in ("R CLOSE", "BASE FWD FINE", "TORSO UP FINE", "HOLD"):
            servo.begin(parse_action(command),s)
            self.assertEqual(servo.next_action(s)[22],-1)

    def test_single_delta_not_tick_accumulation(self):
        s=state(); servo=R1ProServo(s); servo.begin(parse_action("R UP FINE"),s)
        for _ in range(24): servo.next_action(s)
        self.assertAlmostEqual(servo.targets["right"][0][2],.82)
        with self.assertRaises(RuntimeError): servo.next_action(s)

    def test_base_integrated_pulse_and_stop(self):
        s=state(); servo=R1ProServo(s); servo.begin(parse_action("BASE FWD COARSE"),s)
        actions=np.array([servo.next_action(s) for _ in range(24)])
        self.assertAlmostEqual(actions[:,0].sum()*.75/30,.15,places=6)
        np.testing.assert_allclose(actions[-1,:3],0,atol=1e-12)
        np.testing.assert_allclose(actions[:,3:14],.1)

    def test_carry_reject_rotation_without_side_effect(self):
        s=state(); servo=R1ProServo(s); servo.begin(parse_action("MODE CARRY"),s)
        before=copy.deepcopy(servo.__dict__)
        with self.assertRaises(ValueError): servo.begin(parse_action("R OPEN; L ROLL_POS FINE"),s)
        np.testing.assert_array_equal(servo.grips,before["grips"])
        self.assertTrue(servo.carry)

    def test_singular_jacobian_finite_and_bounded(self):
        np.testing.assert_array_equal(dls(np.zeros((6,7)),np.ones(6)),0)
        self.assertLessEqual(abs(dls(np.ones((6,7)),np.ones(6))).max(),.025+1e-9)

    def test_nan_and_shape_rejected(self):
        s=state()
        s.q[0]=float("nan")
        with self.assertRaises(ValueError): s.__post_init__()

    def test_task_guard_and_observable_only(self):
        s=state()
        with self.assertRaises(ValueError): user_text(0,"other","test",s)
        result=user_text(3,"cleaning_up_plates_and_food","test",s)
        self.assertIn("BOTH pizzas",result)
        self.assertNotIn("jacobian",result)
        self.assertNotIn("global_pose",result)


if __name__ == "__main__":
    unittest.main()
