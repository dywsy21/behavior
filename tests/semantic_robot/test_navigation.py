import copy
import unittest
import numpy as np
from semantic_robot.v2.navigation import navigation_workspace_check
from semantic_robot.v2.kinematics import RobotModel
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.grounded_harness import GroundedHarness
from semantic_robot.v2.protocol import Action
from test_v2 import fixture,evidence


class NavigationTests(unittest.TestCase):
    def model(self):
        m,s=fixture();spec=copy.deepcopy(m.spec);spec["metadata"]["arm_chains"]={}
        for arm in ("left","right"):
            spec["metadata"]["arm_chains"][arm]=[arm+"_shoulder",arm+"_elbow"]
            for suffix,offset in (("_shoulder",-.3),("_elbow",-.15)):
                row=copy.deepcopy(spec["links"][arm]);row["T_reference"][0][3]+=offset
                spec["links"][arm+suffix]=row
        return RobotModel(spec),s

    def test_far_point_cannot_be_arrived_even_when_facing_it(self):
        m,s=self.model();r=navigation_workspace_check(m,s.q,{"valid":True,"point_base_m":[2.243,.501,.677]},("left","right"))
        self.assertTrue(r["valid"]);self.assertFalse(r["within_optimistic_reach"])
        self.assertGreater(min(x["outside_reach_lower_bound_m"] for x in r["per_arm"].values()),1.)

    def test_near_only_removes_veto_not_certifies_ik_or_arrival(self):
        m,s=self.model();r=navigation_workspace_check(m,s.q,{"valid":True,"point_base_m":[.35,.3,.8]},("left","right"))
        self.assertTrue(r["within_optimistic_reach"])
        self.assertTrue(r["not_a_reachable_pose_or_arrival_certificate"])
        self.assertTrue(r["per_arm"]["left"]["within_optimistic_reach"])
        self.assertFalse(r["per_arm"]["right"]["within_optimistic_reach"])

    def test_unknown_geometry_or_target_fails_closed(self):
        m,s=fixture();self.assertFalse(navigation_workspace_check(m,s.q,{"valid":True,"point_base_m":[.2,.2,.2]},("left",))["valid"])
        m,s=self.model();self.assertFalse(navigation_workspace_check(m,s.q,{"valid":False},("left",))["valid"])

    def test_nonfinite_rejected(self):
        m,s=self.model()
        with self.assertRaises(ValueError):navigation_workspace_check(m,s.q,{"valid":True,"point_base_m":[np.nan,0,0]},("left",))

    def test_model_true_does_not_skip_distance_veto_on_either_confirmation(self):
        _,s=self.model();h=GroundedHarness([Goal("navigate","table","both","at table")]);h.stage="APPROACH";h.last_action=Action("base","forward")
        h.observe(evidence(effect=True),s,measured_progress={"navigation_aligned":True,"navigation_reach_possible":False})
        self.assertEqual(h.stage,"APPROACH");self.assertEqual(h.completed,[])
        h.stage="VERIFY_EFFECT";h.confirmations=1
        h.observe(evidence(effect=True),s,measured_progress={"navigation_aligned":True,"navigation_reach_possible":False})
        self.assertEqual(h.stage,"APPROACH");self.assertEqual(h.completed,[])


if __name__=="__main__":unittest.main()
