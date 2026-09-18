import unittest
from semantic_robot.v2.grounded_harness import GroundedHarness,GroundedController
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import Action,HOLD
from semantic_robot.v2.servo import SafeServo
from test_v2 import fixture,evidence


class VerificationProbeTests(unittest.TestCase):
    def make(self,hand="right"):
        m,s=fixture();s.gripper[:]=.035
        h=GroundedHarness([Goal("pick","object",hand,"moves",hand=="both")]);h.stage="VERIFY_GRASP"
        h.last_gripper=s.gripper.copy();h.observation=evidence(enclosed=True,co_moving=None)
        for arm in h.arms:h.pending_grasp[arm]=True
        c=GroundedController(m,SafeServo(m,s,[-1,-1]),h,visual_odometry=True,grasp_motion=True)
        c.target={"valid":True};return h,c

    def test_informative_prechecked_lift_is_selected_not_ineffective_micro(self):
        for hand in ("right","left","both"):
            h,c=self.make(hand);fine=Action(hand,"up","fine")
            action,r=c.verification_action((HOLD,Action(hand,"up","micro"),fine))
            self.assertEqual(action,fine);self.assertEqual(r["commanded_lift_m"],.01)
            self.assertTrue(r["measurement_gate_unchanged"]);self.assertFalse(r["success_claim"])
            self.assertFalse(h.completed);self.assertFalse(any(h.hold_verified.values()))

    def test_unsafe_fine_does_not_fallback_to_uninformative_motion(self):
        h,c=self.make();a,r=c.verification_action((HOLD,Action("right","up","micro")))
        self.assertEqual(a,HOLD);self.assertIn("PRESERVED",h.stop_reason)
        self.assertEqual(h.palette(),(HOLD,))

    def test_unknown_target_or_explicit_contradiction_preserves_load(self):
        for obs in (evidence(visible=False,view="none",target_uv=None),evidence(enclosed=False),evidence(co_moving=False),evidence(hazard="occluded")):
            h,c=self.make();h.observation=obs
            a,r=c.verification_action((HOLD,Action("right","up")))
            self.assertEqual(a,HOLD);self.assertIn("PRESERVED",h.stop_reason)

    def test_cannot_select_probe_for_approach_or_without_registration(self):
        h,c=self.make();h.stage="ALIGN"
        with self.assertRaises(ValueError):c.verification_action((Action("right","up"),))
        h.stage="VERIFY_GRASP";c.grasp_verifier=None
        with self.assertRaises(ValueError):c.verification_action((Action("right","up"),))


if __name__=="__main__":unittest.main()
