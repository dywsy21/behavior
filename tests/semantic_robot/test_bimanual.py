"""Two visible contacts / independent alignment / synchronized grasp regression."""
from dataclasses import asdict, replace
import json
import unittest

import numpy as np

from semantic_robot.v2.bimanual import BimanualEvidence, HandContact, all_claims, localize_hand_contacts
from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
from semantic_robot.v2.harness import Goal
from semantic_robot.v2.policy import BIMANUAL_OBSERVE_SYSTEM
from semantic_robot.v2.protocol import Action, HOLD
from semantic_robot.v2.servo import SafeServo
from semantic_robot.v2.vision import project
from test_v2 import fixture, evidence


def setup():
    model, state = fixture()
    manager = GroundedHarness([Goal("pick", "wide tray", "both", "both hands visibly support it", True)])
    controller = GroundedController(model, SafeServo(model,state), manager)
    depths = {view:np.full((100,100),1.2,np.float32) for view in ("head","left_wrist","right_wrist")}
    return model,state,manager,controller,depths


def observed_contacts(model,state,offsets=None,enclosed=(True,True),moving=(None,None)):
    contacts=[]
    for i,hand in enumerate(("left","right")):
        view=hand+"_wrist"
        point=model.grasp_centers(state.q)[hand]+np.asarray((offsets or {}).get(hand,[0,0,0]))
        camera=model.spec["metadata"]["cameras"][view]
        uv=project(point,model.forward(state.q,"camera_"+view),camera["K"])/np.array([camera["width"]-1,camera["height"]-1])
        contacts.append(HandContact(hand,view,tuple(uv),enclosed[i],moving[i]))
    return BimanualEvidence(**asdict(evidence(view="head",enclosed=True,co_moving=True)),
                             hand_contacts=tuple(contacts))


class BimanualTests(unittest.TestCase):
    def test_prompt_example_and_missing_contacts_abstain(self):
        example=BIMANUAL_OBSERVE_SYSTEM.split("Return only JSON with exactly these fields:\n")[1].split("\n")[0]
        parsed=BimanualEvidence.parse(example)
        self.assertEqual(len(parsed.hand_contacts),2)
        self.assertTrue(all(c.view=="none" for c in parsed.hand_contacts))
        missing=BimanualEvidence.parse(json.dumps(asdict(evidence())))
        self.assertEqual(missing.hand_contacts,())
        self.assertIsNone(all_claims([True]))
        self.assertIsNone(all_claims([True,None]))
        self.assertFalse(all_claims([True,False]))

    def test_contact_schema_rejects_duplicates_unknown_and_invented_hidden_points(self):
        model,state,*_=setup();good=asdict(observed_contacts(model,state))
        rows=list(good["hand_contacts"])
        for bad in ([rows[0],rows[0]], [dict(rows[0],hand="both")],
                    [dict(rows[0],enclosed=1)], [dict(rows[0],target_uv=[2,0])],
                    [dict(rows[0],world_pose=[1,2,3])], [dict(rows[0],view="none")]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                BimanualEvidence.parse(json.dumps({**good,"hand_contacts":bad}))
        with self.assertRaises(ValueError):
            BimanualEvidence.parse(json.dumps({**good,"visible":False,"view":"none","target_uv":None}))

    def test_two_contacts_are_separate_depth_rays_not_one_midpoint(self):
        model,state,manager,controller,depths=setup()
        obs=observed_contacts(model,state)
        targets=localize_hand_contacts(obs,depths,model,state.q)
        self.assertTrue(all(t["valid"] for t in targets.values()))
        self.assertGreater(np.linalg.norm(np.asarray(targets["left"]["point_base_m"])-targets["right"]["point_base_m"]),.5)
        controller.observe(obs,state,depths,{"head":{"valid_fraction":1.}})
        self.assertLess(controller.target["distance_to_active_closing_center_m"],.02)
        self.assertGreater(np.linalg.norm(np.asarray(controller.target["point_base_m"])-controller.centers["left"]),.25)
        controller.observe(obs,state,depths,{"head":{"valid_fraction":1.}})
        self.assertEqual(manager.stage,"ALIGN")
        controller.observe(obs,state,depths,{"head":{"valid_fraction":1.}})
        self.assertEqual(manager.stage,"GRASP")
        self.assertIn(Action("both","close"),manager.palette())
        self.assertNotIn(Action("left","close"),manager.palette())

    def test_one_hand_unknown_cannot_borrow_other_hand_or_global_boolean(self):
        model,state,manager,controller,depths=setup()
        obs=observed_contacts(model,state)
        for contacts in (obs.hand_contacts[:1],(obs.hand_contacts[0],HandContact("right","none",None,None,None))):
            manager.stage="ALIGN"
            controller.observe(replace(obs,hand_contacts=contacts),state,depths,{"head":{"valid_fraction":1.}})
            self.assertFalse(controller.target["valid"])
            self.assertNotEqual(manager.stage,"GRASP")
            self.assertNotIn(Action("both","close"),manager.palette())
        manager.stage="ALIGN"
        controller.observe(observed_contacts(model,state,enclosed=(True,False)),state,depths,{"head":{"valid_fraction":1.}})
        self.assertNotEqual(manager.stage,"GRASP")

    def test_independent_alignment_then_shared_verification_and_carry(self):
        model,state,manager,controller,depths=setup()
        manager.stage="ALIGN"
        for hand in ("left","right"):
            self.assertIn(Action(hand,"left","fine"),manager.palette())
            self.assertIn(Action(hand,"pitch_plus","fine","tool"),manager.palette())
        manager.stage="VERIFY_GRASP"
        self.assertEqual(set(manager.palette()),{HOLD,Action("both","up","micro"),Action("both","up")})
        manager.stage="ALIGN";manager.hold_verified={"left":True,"right":True}
        self.assertNotIn(Action("left","left"),manager.palette())

    def test_candidate_budget_retains_improving_independent_hands(self):
        model,state,manager,controller,depths=setup()
        obs=observed_contacts(model,state,offsets={"left":[0,.045,0],"right":[0,-.045,0]},enclosed=(False,False))
        controller.observe(obs,state,depths,{"head":{"valid_fraction":1.}})
        allowed=controller.candidates(state)
        rows=manager.candidate_receipt["tested"]
        self.assertLessEqual(len(rows),41)
        good=[r for r in rows if r["accepted"] and r.get("predicted_distance_gain_m",0)>0]
        for hand in ("left","right"):
            self.assertTrue(any(r["action"]["part"]==hand for r in good),good)
        # Moving one hand can reduce mean error while the other is unchanged.
        distances=controller._expected_distances(Action("left","left"),state)
        self.assertLess(distances["left"],controller.target["per_hand_distance_m"]["left"])
        self.assertAlmostEqual(distances["right"],controller.target["per_hand_distance_m"]["right"])
        self.assertTrue(allowed)

    def test_bimanual_abab_with_real_progress_is_not_zero_drift(self):
        _,_,manager,_,_=setup();manager.stage="APPROACH"
        for move in ("forward","up","forward","up"):
            manager.executed(Action("both",move),{"status":"TARGET_REACHED",
                "eef_delta_m":{"left":[.01,0,0],"right":[.01,0,0]}})
        self.assertEqual(manager.recoveries,0)

    def test_both_metric_contacts_must_follow_both_hands(self):
        for moving_both in (False,True):
            model,state,manager,controller,depths=setup()
            # A 100px fixture has ~17mm depth-ray spacing at 1.2m: it cannot
            # certify a 15mm lift. Use resolved contact rays, not relaxed gates.
            for camera in model.spec["metadata"]["cameras"].values():
                camera.update(width=500,height=500,K=[[350,0,250],[0,350,250],[0,0,1]])
            depths={v:np.full((500,500),1.2,np.float32) for v in depths}
            state=model.state(state.q,np.array([.02,.02]),np.zeros(3))
            manager.stage="VERIFY_GRASP"
            controller.observe(observed_contacts(model,state),state,depths,{"head":{"valid_fraction":1.}})
            q=state.q.copy();q[6]+=.015;q[13]+=.015
            moved=model.state(q,state.gripper,np.zeros(3))
            manager.feedback={"status":"TARGET_REACHED","base_integral":[0,0,0],
                              "eef_delta_m":{"left":[0,0,.015],"right":[0,0,.015]}}
            manager.last_action=Action("both","up")
            updated=observed_contacts(model,moved,moving=(True,True))
            depths["left_wrist"][:]=1.185
            if moving_both:depths["right_wrist"][:]=1.185
            else:
                updated=replace(updated,hand_contacts=(updated.hand_contacts[0],observed_contacts(model,state).hand_contacts[1]))
            controller.observe(updated,moved,depths,{"head":{"valid_fraction":1.}})
            self.assertEqual(bool(manager.completed),moving_both)
            self.assertEqual(controller.progress["metric_co_motion"],moving_both)
