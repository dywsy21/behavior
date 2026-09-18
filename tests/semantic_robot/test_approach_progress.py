import unittest
import numpy as np
from semantic_robot.v2.approach_progress import ApproachProgress
from semantic_robot.v2.protocol import Action


class ApproachProgressTests(unittest.TestCase):
    def sample(self,p,i,distance=.04,**changes):
        body=np.eye(4);body[0,3]=.018
        args=dict(goal_index=0,kind="pick",stage="ALIGN",points={"right":np.array([distance,0,0])},
            poses={"right":np.eye(4)},execution=i,action=Action("base","forward","micro"),
            feedback={"status":"TARGET_REACHED"},motion={"valid":True,"body_transform_current_in_previous":body})
        args.update(changes);return p.observe(**args)

    def stalled(self):
        p=ApproachProgress()
        for i in range(4):self.sample(p,i)
        self.assertFalse(p.allowed(Action("base","forward")))
        return p

    def test_three_measured_ineffective_advances_veto_only_same_base_direction(self):
        p=self.stalled();r=p.context()
        self.assertEqual(r["blocked_base_directions"],["forward"])
        self.assertAlmostEqual(r["evidence"][0]["measured_base_travel_m"],.054)
        for action in (Action("right","forward"),Action("base","back"),Action("base","yaw_plus")):
            self.assertTrue(p.allowed(action))

    def test_legitimate_repeated_approach_is_not_penalized(self):
        p=ApproachProgress()
        for i,d in enumerate((.09,.072,.054,.036)):self.sample(p,i,d)
        self.assertTrue(p.allowed(Action("base","forward")))

    def test_duplicate_observations_do_not_count_as_three_actions(self):
        p=ApproachProgress()
        self.sample(p,0)
        for _ in range(6):self.sample(p,1)
        self.assertTrue(p.allowed(Action("base","forward")))
        self.assertEqual(len(p.window),1)

    def test_missing_measurement_far_target_navigation_and_loaded_hand_abstain(self):
        for changes in ({"motion":{"valid":False}}, {"kind":"navigate"}, {"stage":"VERIFY_GRASP"},
                        {"loaded":True}, {"points":{}}, {"feedback":{"status":"BASE_TRACKING_FAILED"}}):
            p=ApproachProgress()
            for i in range(5):self.sample(p,i,**changes)
            self.assertEqual(p.context()["blocked_base_directions"],[])
        p=ApproachProgress()
        for i in range(5):self.sample(p,i,.5)
        self.assertEqual(p.blocked,{})

    def test_unknown_target_open_or_new_pixel_cannot_forget_veto(self):
        for changes in ({"points":{}},{"action":Action("right","open")},{"distance":.025}):
            p=self.stalled();self.sample(p,4,**changes)
            self.assertFalse(p.allowed(Action("base","forward")))

    def test_actual_hand_pose_change_or_new_goal_clears_veto(self):
        p=self.stalled();pose=np.eye(4);pose[2,3]=.012
        self.sample(p,4,poses={"right":pose},action=Action("right","up"))
        self.assertTrue(p.allowed(Action("base","forward")))
        p=self.stalled();self.sample(p,4,goal_index=1)
        self.assertTrue(p.allowed(Action("base","forward")))

    def test_retreat_must_be_measured_not_commanded(self):
        p=self.stalled();self.sample(p,4,action=Action("base","back"),motion={"valid":False})
        self.assertFalse(p.allowed(Action("base","forward")))
        t=np.eye(4);t[0,3]=-.045
        self.sample(p,5,action=Action("base","back"),motion={"valid":True,"body_transform_current_in_previous":t})
        self.assertTrue(p.allowed(Action("base","forward")))

    def test_executor_candidates_and_reposition_both_obey_veto(self):
        from types import SimpleNamespace
        import test_grounded
        model,state,h,servo,c,_,_=test_grounded.setup_controller()
        h.stage="ALIGN";h.observation=test_grounded.grounded_evidence(enclosed=None)
        c.centers=model.grasp_centers(state.q)
        c.target={"valid":True,"point_base_m":c.centers["right"]+np.array([.04,0,0]),"distance_to_active_closing_center_m":.04}
        c.depth_guard=SimpleNamespace(check=lambda *a:(True,"FREE"),receipt=lambda:{})
        self.assertIn(Action("base","forward","micro"),c.candidates(state))
        c.approach_monitor=self.stalled()
        self.assertNotIn(Action("base","forward","micro"),c.candidates(state))
        self.assertTrue(any(r["reason"]=="REPEATED_BASE_APPROACH_WITHOUT_CONTACT_PROGRESS" for r in h.candidate_receipt["tested"]))
        c.reposition=Action("base","forward","micro");c.reposition_left=1
        action,receipt=c.search_action(state)
        self.assertEqual(action.move,"hold");self.assertEqual(receipt["source"],"observed_progress_veto")


if __name__=="__main__":unittest.main()
