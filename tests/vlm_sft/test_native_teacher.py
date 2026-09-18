import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import types
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts/vlm_sft")]
from native_teacher_contract import (select_sources, compile_candidates, actor_input, digest,
                                     validate_approval, release_reviewed_record)
from native_teacher_collect import rest_screen, require_release
from common import TOKENS, token_to_action


class NativeTeacherTests(unittest.TestCase):
    def fixture(self):
        proprio = {"eef_base_m": {a: [0., 0., 0.] for a in ("left", "right")},
                   "finger_opening_m": {a: .05 for a in ("left", "right")},
                   "base_velocity_local": [0., 0., 0.], "torso_joints_rad": [0.]*4}
        actor = actor_input("Move the object", "verb=GRASP; target=object", proprio,
                            {v: "a"*64 for v in ("head", "left_wrist", "right_wrist")}, [])
        request = {"actor": actor, "source_group": {"task": 1, "instance": 192, "episode": 310},
                   "allowed_tokens": [t for t in TOKENS if not t.startswith("BASE_")]}
        approval = {"request_sha256": digest(request), "reviewer": "independent reviewer",
                    "decision": "approve", "token": "RIGHT_UP", "intent_still_valid": True,
                    "judged_correct_next_action": True, "reviewed_current_and_source_views": True,
                    "reason": "Reviewed all three views and the source continuation; this is a fixture, not real evidence."}
        record = {"request": request, "approval": approval,
                  "execution": {"token": "RIGHT_UP", "action": asdict(token_to_action("RIGHT_UP")),
                                "status": "TARGET_REACHED", "interrupted": False, "native_controls": 18,
                                "native_trace_sha256": "b"*64}, "post_observation_sha256": "c"*64, "settle_passed": True}
        post = {"record_sha256": digest(record), "reviewer": "post reviewer", "decision": "approve",
                "correct_for_current_intent": True, "reviewed_full_before_after_and_native_trace": True,
                "not_based_only_on_safety_or_distance": True, "reason": "Fixture manual judgment of the action and preserved constraints, not a success certificate."}
        return request, approval, record, post

    def test_fixed_train_selection_excludes_all_protected_instances(self):
        counts = json.loads((ROOT/"configs/vlm_sft/h09r_train_feasibility_counts.json").read_text())
        self.assertEqual([(r["task"], r["episode"], r["instance"]) for r in select_sources(counts)],
                         [(0,66,70),(1,310,192),(3,629,30)])
        counts["exclusions"]["extra"] = [[0,70]]
        self.assertNotEqual(select_sources(counts)[0]["instance"], 70)

    def test_mixed_motion_is_only_a_proposal(self):
        s = np.zeros((17,61)); s[:,23] = 1; s[:,48] = 1
        s[:,17] = np.linspace(0,.02,17); s[:,43] = np.linspace(0,.03,17)
        out = compile_candidates(s, np.zeros((16,23)))
        self.assertIsNone(out["label"]); self.assertFalse(out["training_eligible"])
        self.assertIn("LEFT_FORWARD", out["ranked_proposals"])
        self.assertIn("RIGHT_LEFT", out["ranked_proposals"])
        s[:,0] = .021
        self.assertEqual(compile_candidates(s, np.zeros((16,23)))["ranked_proposals"], [])

    def test_nonfinite_or_misaligned_expert_rejected(self):
        for n in (16,18):
            with self.assertRaises(ValueError): compile_candidates(np.zeros((n,61)),np.zeros((16,23)))
        s=np.zeros((17,61));s[0,0]=np.nan
        with self.assertRaises(ValueError): compile_candidates(s,np.zeros((16,23)))

    def test_stale_approval_and_wrong_intent_fail(self):
        req,a,_,_=self.fixture()
        self.assertEqual(validate_approval(a,req),"RIGHT_UP")
        for key,val in (("request_sha256","d"*64),("intent_still_valid",False),
                        ("judged_correct_next_action",False),("token","BASE_FORWARD"),
                        ("reviewed_current_and_source_views",False),("reason","PASS")):
            b=dict(a);b[key]=val
            with self.subTest(key=key), self.assertRaises(ValueError):validate_approval(b,req)

    def test_arrival_alone_never_releases_label(self):
        _,_,r,p=self.fixture()
        with self.assertRaises(ValueError):release_reviewed_record(r,{})
        p["not_based_only_on_safety_or_distance"]=False
        with self.assertRaises(ValueError):release_reviewed_record(r,p)

    def test_wrong_execution_interruption_and_missing_motion_fail(self):
        _,_,r,p=self.fixture()
        for key,value in (("token","LEFT_UP"),("status","INTERRUPTED"),("interrupted",True),("native_controls",0)):
            bad=copy.deepcopy(r);bad["execution"][key]=value;p["record_sha256"]=digest(bad)
            with self.subTest(key=key), self.assertRaises(ValueError):release_reviewed_record(bad,p)

    def test_projection_and_split_provenance_are_separate(self):
        _,_,r,p=self.fixture();out=release_reviewed_record(r,p)
        self.assertEqual(set(out["actor"]),{"task","active_instruction","proprio","current_rgb_sha256","history"})
        self.assertNotIn("source_group",out["actor"]);self.assertEqual(out["source_group"]["instance"],192)
        self.assertFalse(out["official_success_claim"])

    def test_rest_and_release_fail_closed(self):
        def state(x=0.):return types.SimpleNamespace(q=np.full(18,x),gripper=np.ones(2)*.05,base_velocity=np.zeros(3),
                                                    poses={a:(np.zeros(3),[0.,0.,0.,1.]) for a in ("left","right")})
        self.assertFalse(rest_screen([state()]*3));self.assertTrue(rest_screen([state()]*4))
        self.assertFalse(rest_screen([state(),state(),state(),state(.02)]))
        self.assertFalse(rest_screen([state(),state(),state(),state(np.nan)]))
        with self.assertRaises(ValueError):require_release({},"code","executor")


if __name__ == "__main__": unittest.main()
