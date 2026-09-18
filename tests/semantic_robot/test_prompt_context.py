import copy
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest

from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action
from test_grounded import setup_controller, grounded_evidence
from test_v2 import feedback


class PromptContextTests(unittest.TestCase):
    def make_context(self):
        model,state,h,servo,c,depths,receipt=setup_controller()
        h.observation=grounded_evidence(enclosed=False,co_moving=None)
        h.pending_grasp["right"]=True
        h.stop_reason="VISUAL_ODOMETRY_UNCERTAIN"
        h.grounding={"valid":False,"reason":"OBSERVED_FREE_SPACE_CONTRADICTION",
                     "debug_mesh":"DO_NOT_FEED_THIS"*20000}
        h.motion_receipt={"valid":False,"reason":"INSUFFICIENT_STABLE_CORRESPONDENCES",
                          "current_rgb_sha256":"NOT_VISUAL_EVIDENCE"}
        for index in range(8):
            f=feedback();f.update(status="BASE_TRACKING_FAILED" if index==7 else "TARGET_REACHED",
                eef_delta_m={"left":[0.,0.,0.],"right":[0.,0.,.0123456]},debug_blob="HUGE"*20000)
            h.history.append({"action":asdict(Action("right","up")),"stage":"ALIGN","feedback":f})
        allowed=(Action("right","up"),Action("right","back","micro"))
        h.candidate_receipt={"state_q":["UNNEEDED_Q"*20000],"command_grip_latch":[-1,-1],
            "tested":[{"action":asdict(a),"accepted":True,"reason":"KINEMATIC_PATH_FEASIBLE",
                       "predicted_distance_gain_m":.0091234} for a in allowed]+
                      [{"action":asdict(Action("base","forward")),"accepted":False,"reason":"OBSERVED_BASE_OBSTACLE"}]}
        return h,state,SimpleNamespace(geometry={}),allowed

    def test_debug_volume_removed_but_all_recent_failures_and_unknowns_preserved(self):
        h,state,bundle,allowed=self.make_context()
        original=copy.deepcopy(h.context());raw=copy.deepcopy(h.candidate_receipt)
        text=actor_context(h,state,bundle,allowed);value=json.loads(text)
        self.assertLess(len(text),8000)
        for forbidden in ("DO_NOT_FEED_THIS","HUGE","NOT_VISUAL_EVIDENCE","UNNEEDED_Q"):
            self.assertNotIn(forbidden,text)
        self.assertEqual(h.context(),original);self.assertEqual(h.candidate_receipt,raw)
        scope=value["harness"]
        self.assertEqual(len(scope["recent_executed"]),len(original["recent_executed"]))
        self.assertEqual(scope["recent_executed"][-1]["feedback"]["status"],"BASE_TRACKING_FAILED")
        self.assertEqual(scope["stop_reason"],"VISUAL_ODOMETRY_UNCERTAIN")
        self.assertTrue(scope["unverified_close_latches"]["right"])
        self.assertFalse(value["current_visual_evidence"]["enclosed"])
        self.assertIsNone(value["current_visual_evidence"]["co_moving"])
        self.assertFalse(scope["target_surface_estimate"]["valid"])
        self.assertEqual(scope["target_surface_estimate"]["reason"],"OBSERVED_FREE_SPACE_CONTRADICTION")

    def test_scores_index_only_offered_commands_and_preserve_rejection_reasons(self):
        h,state,bundle,allowed=self.make_context()
        value=json.loads(actor_context(h,state,bundle,allowed[::-1]))["CURRENT preflight receipt"]
        self.assertEqual([row["command_index"] for row in value["scores_for_allowed_commands"]],[0,1])
        self.assertEqual(value["rejected_reason_counts"],{"OBSERVED_BASE_OBSTACLE":1})
        self.assertEqual(value["command_grip_latch"],[-1,-1])
        self.assertTrue(value["fresh_execution_recheck_required"])

    def test_bad_float_rejected_without_silent_repair(self):
        h,state,bundle,allowed=self.make_context();h.grounding["point_base_m"]=[float("nan"),0,0]
        with self.assertRaises(ValueError):actor_context(h,state,bundle,allowed)

    def test_independent_hand_contacts_and_unknowns_remain_separate(self):
        h,state,bundle,allowed=self.make_context()
        h.grounding={"valid":False,"hand_contacts":{"left":{"valid":True,"point_base_m":[.1,.2,.3]},
                                                     "right":{"valid":False,"reason":"NO_CONTACT_EVIDENCE"}}}
        value=json.loads(actor_context(h,state,bundle,allowed))["harness"]["target_surface_estimate"]["hand_contacts"]
        self.assertEqual(value["left"]["point_base_m"],[.1,.2,.3])
        self.assertFalse(value["right"]["valid"]);self.assertNotIn("point_base_m",value["right"])


if __name__=="__main__":unittest.main()
