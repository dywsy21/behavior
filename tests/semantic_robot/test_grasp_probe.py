import unittest
from types import SimpleNamespace
from semantic_robot.v2.diagnostics import grasp_audit
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import Action
from test_v2 import fixture, evidence, feedback


class GraspProbeTests(unittest.TestCase):
    def test_privileged_grasp_audit_is_explicit_and_physical_abstains(self):
        r=SimpleNamespace(grasping_mode="assisted",_ag_obj_in_hand={"right":SimpleNamespace(name="radio_1"),"left":None})
        self.assertEqual(grasp_audit(r)["assisted_objects"],{"right":"radio_1","left":None})
        self.assertTrue(grasp_audit(r)["not_available_to_actor"])
        r.grasping_mode="physical"
        self.assertIsNone(grasp_audit(r)["assisted_objects"])

    def make(self, **kwargs):
        _, state=fixture()
        h=GroundedHarness([Goal("pick","visible object","right","moves with hand")],active_grasp_probe=True)
        h.stage="ALIGN";h.observation=evidence(enclosed=None)
        h.grounding={"valid":True,"distance_to_active_closing_center_m":.026}
        for k,v in kwargs.items():setattr(h,k,v)
        h.update_grasp_probe(state)
        return h,state

    def test_opt_in_unknown_is_attempt_not_success(self):
        h,state=self.make()
        self.assertIn(Action("right","close"),h.palette())
        self.assertIsNone(h.observation.enclosed)
        h.executed(Action("right","close"),feedback())
        self.assertEqual(h.stage,"VERIFY_GRASP")
        self.assertEqual(h.completed,[])
        self.assertFalse(any(h.hold_verified.values()))
        self.assertTrue(h.pending_grasp["right"])
        self.assertEqual(h.grasp_probe_attempts,{0:1})
        self.assertFalse(h.grasp_probe["eligible"])

    def test_default_off(self):
        h,_=self.make(active_grasp_probe=False)
        self.assertNotIn(Action("right","close"),h.palette())

    def test_contrary_missing_far_hazard_and_wrong_stage_rejected(self):
        changes=[{"observation":evidence(enclosed=False)},
                 {"observation":evidence(enclosed=None,hazard="occluded")},
                 {"observation":evidence(enclosed=None,hazard="collision")},
                 {"observation":evidence(visible=False,view="none",target_uv=None,enclosed=None)},
                 {"grounding":{"valid":False,"distance_to_active_closing_center_m":.02}},
                 {"grounding":{"valid":True}},
                 {"grounding":{"valid":True,"distance_to_active_closing_center_m":.04001}},
                 {"grounding":{"valid":True,"distance_to_active_closing_center_m":float("nan")}},
                 {"stage":"APPROACH"},{"stop_reason":"SAFETY_STOP"}]
        for change in changes:
            with self.subTest(change=change):
                h,_=self.make(**change)
                self.assertFalse(h.grasp_probe["eligible"])
                self.assertNotIn(Action("right","close"),h.palette())

    def test_closed_or_loaded_hand_rejected(self):
        h,state=self.make();state.gripper[:]=.0;h.update_grasp_probe(state)
        self.assertFalse(h.grasp_probe["eligible"])
        h,state=self.make();h.pending_grasp["right"]=True;h.update_grasp_probe(state)
        self.assertFalse(h.grasp_probe["eligible"])
        h,state=self.make();h.hold_verified["left"]=True;h.update_grasp_probe(state)
        self.assertFalse(h.grasp_probe["eligible"])

    def test_budget_persists_across_open_recovery(self):
        h,state=self.make()
        for i in range(2):
            h.stage="ALIGN";h.update_grasp_probe(state)
            self.assertTrue(h.grasp_probe["eligible"])
            h.executed(Action("right","close"),feedback())
            h.executed(Action("right","open"),feedback())
        h.stage="ALIGN";h.update_grasp_probe(state)
        self.assertFalse(h.grasp_probe["eligible"])
        self.assertEqual(h.grasp_probe_attempts[0],2)

    def test_failed_motion_is_not_verified_or_free_retry(self):
        h,state=self.make();f={**feedback("TRACKING_DIVERGED"),"control_ticks":2}
        h.executed(Action("right","close"),f)
        self.assertEqual(h.grasp_probe_attempts[0],1)
        self.assertNotEqual(h.stage,"VERIFY_GRASP")
        self.assertEqual(h.completed,[])

    def test_attempt_alone_or_visual_claim_cannot_complete(self):
        h,state=self.make();h.executed(Action("right","close"),feedback())
        h.observe(evidence(enclosed=None,co_moving=True),state,measured_progress={"metric_co_motion":True})
        self.assertEqual(h.completed,[])
        h.observe(evidence(enclosed=True,co_moving=True),state,measured_progress={"metric_co_motion":False})
        self.assertEqual(h.completed,[])

    def test_two_hand_probe_requires_both_near_and_open(self):
        h,state=self.make();h.goals=[Goal("pick","visible plate","both","moves together",True)]
        h.update_grasp_probe(state)
        self.assertIn(Action("both","close"),h.palette())
        state.gripper[0]=.0;h.update_grasp_probe(state)
        self.assertNotIn(Action("both","close"),h.palette())
        self.assertNotIn(Action("right","close"),h.palette())


if __name__=="__main__":unittest.main()
