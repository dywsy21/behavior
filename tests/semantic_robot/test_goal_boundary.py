import unittest
from types import SimpleNamespace
import numpy as np
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.protocol import Action,HOLD
from semantic_robot.v2.search import CoverageSearch
from test_grounded import setup_controller,grounded_evidence
from test_v2 import feedback


class GoalBoundaryTests(unittest.TestCase):
    def test_old_target_cannot_command_next_goal(self):
        m,s,h,servo,c,depths,receipt=setup_controller()
        h.goals=[Goal("pick","radio","right","moves"),Goal("press","button","left","on")]
        h.stage="VERIFY_GRASP";h.observation=grounded_evidence(enclosed=True)
        h.feedback=feedback();h.feedback["eef_delta_m"]["right"]=[0,0,.01]
        h.last_action=Action("right","up");h.pending_grasp["right"]=True
        c.grasp_verifier=SimpleNamespace(observe=lambda *args:{"source":"registered_onboard_RGBD_grasp_motion","verified":True})
        c.observe(grounded_evidence(enclosed=True,co_moving=None),s,depths,receipt,images={},self_geometry={})
        self.assertEqual(h.index,1);self.assertTrue(c.goal_changed)
        self.assertFalse(c.target["valid"]);self.assertEqual(c.hand_targets,{})
        self.assertIsNone(h.observation);self.assertIsNone(c.previous)
        self.assertEqual(c.candidates(s),(HOLD,));self.assertEqual(c.search_action(s)[0],HOLD)
        self.assertTrue(h.hold_verified["right"])

    def test_nontransition_keeps_fresh_target(self):
        m,s,h,servo,c,depths,receipt=setup_controller()
        c.observe(grounded_evidence(),s,depths,receipt)
        self.assertFalse(c.goal_changed);self.assertTrue(c.target["valid"])

    def test_approach_travel_does_not_exhaust_search(self):
        c=CoverageSearch();c.executed({"base_integral":[1.3,0,0]},exploratory=False)
        self.assertAlmostEqual(c.travel_m,1.3);self.assertIsNotNone(c.propose()[0])
        c.executed({"base_integral":[1.21,0,0]},exploratory=True)
        self.assertEqual(c.propose(),(None,"SEARCH_TRAVEL_BUDGET"))

    def test_new_fixed_goal_resets_only_search_allowance_not_episode_odometry(self):
        c=CoverageSearch();T=np.eye(4);K=np.eye(3)
        c.observe(0,T,K,100,1.,False)
        c.executed({"base_integral":[1.3,0,0]},exploratory=True)
        c.observe(0,T,K,100,1.,False);self.assertEqual(c.propose()[1],"SEARCH_TRAVEL_BUDGET")
        c.observe(1,T,K,100,1.,False)
        self.assertAlmostEqual(c.travel_m,1.3);self.assertEqual(c.search_travel_m,0.)
        self.assertIsNotNone(c.propose()[0])

    def test_controller_counts_execution_phase_not_next_observation_stage(self):
        m,s,h,servo,c,depths,receipt=setup_controller()
        h.stage="APPROACH"
        fb=feedback();fb["base_integral"]=[.06,0,0]
        c.executed(Action("base","forward"),fb)
        self.assertAlmostEqual(c.search.travel_m,.06);self.assertEqual(c.search.search_travel_m,0.)
        h.stage="SEARCH";c.executed(Action("base","forward"),fb)
        self.assertAlmostEqual(c.search.search_travel_m,.06)


if __name__=="__main__":unittest.main()
