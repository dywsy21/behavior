import unittest
from types import SimpleNamespace
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.policy import grasp_tracking_instruction


class TrackingAnchorTests(unittest.TestCase):
    def test_only_grasp_verification_changes_localization_purpose(self):
        for kind,stage,active in (("pick","VERIFY_GRASP",True),("pick","ALIGN",False),("press","SEARCH",False)):
            h=SimpleNamespace(goal=Goal(kind,"object","right","visible effect"),stage=stage)
            prompt=grasp_tracking_instruction(h)
            self.assertEqual(bool(prompt),active)
            if active:self.assertIn("NOT holding evidence",prompt)

    def test_bimanual_instruction_does_not_collapse_two_contacts(self):
        h=SimpleNamespace(goal=Goal("pick","object","both","both move"),stage="VERIFY_GRASP")
        prompt=grasp_tracking_instruction(h)
        self.assertIn("each",prompt);self.assertIn("hand_contacts",prompt)
        self.assertIn("Do not click a robot finger",prompt)

