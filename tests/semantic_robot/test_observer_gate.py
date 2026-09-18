import unittest
from unittest.mock import patch
import numpy as np

from semantic_robot.v2.observer_gate import choose_gate_pair
from semantic_robot.v2.servo import SafeServo
from test_v2 import fixture


class ObserverGateTests(unittest.TestCase):
    def make(self):
        model,state=fixture()
        model.spec["metadata"].update(grasp_region_reference_gripper_m=[.05,.05],
                                      grasp_region_reference_fully_open={"left":True,"right":True})
        depths={v:np.full((100,100),1.2,np.float32) for v in ("head","left_wrist","right_wrist")}
        return model,state,SafeServo(model,state,[1,1]),depths

    def test_selects_only_a_checked_inverse_pair_with_other_joints_fixed(self):
        model,state,servo,depths=self.make()
        paths=[]
        class Guard:
            def __init__(self,*args):pass
            def check(self,path):
                paths.append(path.copy())
                return len(paths)>1,{"reason":"first_path_blocked" if len(paths)==1 else "test_clear"}
        with patch("semantic_robot.v2.observer_gate.ObservingArmGuard",Guard):
            forward,back,receipt=choose_gate_pair(model,state,servo,depths,{})
        self.assertIsNotNone(forward);self.assertEqual(forward.scale,"coarse")
        self.assertNotEqual(forward.move,back.move)
        self.assertEqual(forward.move.split("_")[0],back.move.split("_")[0])
        self.assertEqual(len(receipt["tested"]),2)
        for path in paths:
            self.assertLessEqual(len(path),64)
            np.testing.assert_allclose(path[:,list(range(4))+list(range(11,18))],0.)

    def test_no_safe_pair_does_not_lower_scale_or_bypass_guard(self):
        model,state,servo,depths=self.make()
        with patch("semantic_robot.v2.observer_gate.ObservingArmGuard") as factory:
            factory.return_value.check.return_value=(False,{"reason":"test_blocked"})
            forward,back,receipt=choose_gate_pair(model,state,servo,depths,{})
        self.assertIsNone(forward);self.assertIsNone(back)
        self.assertEqual(len(receipt["tested"]),6)
        self.assertTrue(all(row["forward"]["scale"]=="coarse" for row in receipt["tested"]))

    def test_unknown_opening_or_close_latch_refuses_gate(self):
        for change in ("closed_command","width","metadata"):
            model,state,servo,depths=self.make()
            if change=="closed_command":servo.grips[0]=-1
            elif change=="width":state.gripper[0]=.02
            else:model.spec["metadata"].pop("grasp_region_reference_fully_open")
            with patch("semantic_robot.v2.observer_gate.ObservingArmGuard") as factory:
                self.assertIsNone(choose_gate_pair(model,state,servo,depths,{})[0])
                factory.assert_not_called()
