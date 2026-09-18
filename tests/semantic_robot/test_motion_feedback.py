import copy
import unittest

import numpy as np

from semantic_robot.v2.motion_feedback import observed_motion_feedback
from semantic_robot.v2.protocol import Action, HOLD


class MotionFeedbackTests(unittest.TestCase):
    def raw(self, status="BASE_TRACKING_FAILED"):
        return {"status":status,"base_integral":[-.05743515,-.00604628,.03616708],"carry":False}

    def test_h10_false_alarm_not_rewritten_in_raw_evidence(self):
        raw=self.raw();original=copy.deepcopy(raw)
        # Synthetic dual-depth receipt using the independent video-PnP value;
        # the old physical run is NOT retrospectively certified by this test.
        receipt={"valid":True,"body_delta":[-.0566676,-.00102346,-.000250664]}
        result=observed_motion_feedback(Action("base","back"),raw,receipt)
        self.assertEqual(result["status"],"TARGET_REACHED")
        self.assertEqual(raw,original)
        self.assertEqual(result["base_velocity_integral_raw"],original["base_integral"])
        self.assertEqual(result["base_tracking_status_from_velocity_raw"],"BASE_TRACKING_FAILED")

    def test_inverse_false_pass_rejected_at_same_bounds(self):
        for delta in ([.0479,0.,0.],[.06,0.,np.deg2rad(2.01)]):
            result=observed_motion_feedback(Action("base","forward"),self.raw("TARGET_REACHED"),
                                            {"valid":True,"body_delta":delta})
            self.assertEqual(result["status"],"BASE_TRACKING_FAILED")

    def test_hard_controller_failures_are_never_erased(self):
        for status in ("TRACKING_DIVERGED","INTERRUPTED","ROBOT_COLLISION_RISK","JOINT_LIMIT_STALL"):
            result=observed_motion_feedback(Action("base","back"),self.raw(status),
                                            {"valid":True,"body_delta":[-.06,0.,0.]})
            self.assertEqual(result["status"],status)
            self.assertNotIn("base_tracking_status_from_velocity_raw",result)

    def test_missing_invalid_initial_nonfinite_receipts_cannot_pass(self):
        for receipt in ({},{"valid":False,"body_delta":[0,0,0]},
                        {"valid":True,"initial":True,"body_delta":[0,0,0]},
                        {"valid":True,"body_delta":[0,0]},
                        {"valid":True,"body_delta":[0,0,float("nan")]},
                        {"valid":True,"body_delta":[0,0,float("inf")]}):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                observed_motion_feedback(Action("base","back"),self.raw(),receipt)

    def test_yaw_direction_and_carry_scale_preserved(self):
        for carry in (False,True):
            for move in ("yaw_plus","yaw_minus"):
                action=Action("base",move,"coarse")
                yaw=action.amount(carry)*(1 if move=="yaw_plus" else -1)
                result=observed_motion_feedback(action,{**self.raw(),"carry":carry},
                                                {"valid":True,"body_delta":[0,0,yaw]})
                self.assertEqual(result["status"],"TARGET_REACHED")

    def test_nonbase_status_not_rejudged_and_no_alias_mutation(self):
        receipt={"valid":True,"body_delta":[0,0,0]};raw=self.raw("TRACKING_FAILED")
        result=observed_motion_feedback(HOLD,raw,receipt)
        self.assertEqual(result["status"],"TRACKING_FAILED")
        result["base_integral"][0]=1
        result["base_velocity_integral_raw"][0]=1
        self.assertEqual(receipt["body_delta"],[0,0,0])
        self.assertNotEqual(raw["base_integral"][0],1)


if __name__=="__main__":unittest.main()
