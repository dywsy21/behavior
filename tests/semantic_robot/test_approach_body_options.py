import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantic_robot.v2.prompt_context import actor_context
from semantic_robot.v2.protocol import Action, ROTATIONS
from semantic_robot.v2.wall_budget import WallTimeBudgetReached
from test_approach_reorientation import setup
from test_v2 import evidence


class BodyOptionsTests(unittest.TestCase):
    def setup_body(self):
        result=setup()
        result[2].approach_reorientation=False
        result[2].approach_body_options=True
        return result

    def test_all_six_body_directions_survive_blocked_best_translation(self):
        _,s,h,servo,c=self.setup_body()
        c.depth_guard.check=lambda a,*_: (not (a.part=="base" and a.move=="forward"), "FIXTURE")
        before=s.q.copy();allowed=c.candidates(s)
        fine=[r for r in h.candidate_receipt["tested"] if r["action"]["part"]=="base" and r["action"]["scale"]=="fine"]
        self.assertEqual({r["action"]["move"] for r in fine},{"forward","back","left","right","yaw_plus","yaw_minus"})
        self.assertIn(Action("base","yaw_plus"),allowed)
        self.assertIn(Action("base","yaw_minus"),allowed)
        self.assertNotIn(Action("base","forward"),allowed)
        self.assertFalse(any(a.part=="right" and a.move in ROTATIONS for a in allowed))
        self.assertLessEqual(len(h.candidate_receipt["tested"])-1,32)
        np.testing.assert_array_equal(before,s.q);self.assertEqual(servo.status,"IDLE")
        context=json.loads(actor_context(h,s,SimpleNamespace(geometry={}),allowed))
        self.assertTrue(context["CURRENT preflight receipt"]["approach_body_options"]["eligible"])

    def test_micro_fallback_is_checked_after_all_fine_directions(self):
        _,s,h,_,c=self.setup_body()
        c.depth_guard.check=lambda a,*_: (not (a.part=="base" and a.scale=="fine"), "FIXTURE")
        allowed=c.candidates(s)
        rows=[r for r in h.candidate_receipt["tested"] if r["action"]["part"]=="base"]
        self.assertEqual([r["action"]["scale"] for r in rows], ["fine"]*6+["micro"]*6)
        self.assertEqual(len([a for a in allowed if a.part=="base"]),6)

    def test_progress_veto_still_applies_to_new_options(self):
        _,s,_,_,c=self.setup_body()
        c.approach_monitor=SimpleNamespace(allowed=lambda a: a!=Action("base","yaw_plus"))
        allowed=c.candidates(s)
        self.assertNotIn(Action("base","yaw_plus"),allowed)
        self.assertIn(Action("base","yaw_minus"),allowed)

    def test_unknown_load_close_latch_near_or_hazard_do_not_unlock(self):
        changes=[lambda h,s,v:h.pending_grasp.update(right=None),
                 lambda h,s,v:h.hold_verified.update(left=True),
                 lambda h,s,v:h.possible_contact_after_close.update(left=True),
                 lambda h,s,v:h.grounding.update(distance_to_active_closing_center_m=.10),
                 lambda h,s,v:setattr(h,"observation",evidence(hazard="collision")),
                 lambda h,s,v:v.grips.__setitem__(1,-1),
                 lambda h,s,v:v.grips.__setitem__(1,1.01),
                 lambda h,s,v:s.gripper.__setitem__(1,np.nan)]
        for change in changes:
            _,s,h,servo,c=self.setup_body();change(h,s,servo)
            try:allowed=c.candidates(s)
            except (ValueError,RuntimeError):continue  # Existing servo rejects nonfinite state.
            self.assertFalse(h.candidate_receipt["approach_body_options"]["eligible"])
            self.assertFalse(any(a.part=="base" and a.move in ROTATIONS for a in allowed))

    def test_disabled_preserves_single_legacy_body_candidate(self):
        _,s,h,_,c=self.setup_body();h.approach_body_options=False
        c.candidates(s)
        self.assertEqual(len([r for r in h.candidate_receipt["tested"] if r["action"]["part"]=="base"]),1)

    def test_deadline_blocks_candidates_before_any_preflight(self):
        _,s,_,_,c=self.setup_body()
        with patch.object(c.depth_guard,"check") as check:
            with self.assertRaises(WallTimeBudgetReached):c.candidates(s,deadline=0)
            check.assert_not_called()


if __name__=="__main__":unittest.main()
