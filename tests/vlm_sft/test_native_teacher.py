import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts/vlm_sft")]
from native_teacher_contract import (select_sources, compile_candidates, actor_input, digest,
                                     validate_approval, release_reviewed_record, verify_prepared_source,
                                     verify_loaded_window, SCHEMA)
from native_teacher_collect import rest_screen, require_release, native_preflight, capture_snapshot
from common import TOKENS, token_to_action, sha, write_json

sys.path.insert(0, str(ROOT/"tests/semantic_robot"))
from test_hand_body_collision import calibrated_fixture


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

    def test_gate_flag_and_exact_executor_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f"gate_{task}.json" for task in (0,3)]
            rows = [{"task": task, "gate_ok": True, "implementation_digest": "executor",
                     "robot_geometry_guards": True} for task in (0,3)]
            release = {"schema": SCHEMA, "authorize_collection": True, "collector_commit": "code",
                       "executor_digest": "executor", "h14_body_and_finger_safety_reviewed": True,
                       "reviewer": "fixture", "max_resets": 3, "max_candidates_per_instance": 3,
                       "engineering_gate_paths": [str(p) for p in paths]}
            for p,row in zip(paths,rows):write_json(p,row)
            require_release(release,"code","executor")
            for key,value in (("robot_geometry_guards",False),("robot_geometry_guards",None),
                              ("implementation_digest","old"),("gate_ok",False)):
                row=dict(rows[0]);row[key]=value;write_json(paths[0],row)
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    require_release(release,"code","executor")

    def prepared_fixture(self, root):
        prepared=root/"task_0";prepared.mkdir()
        pilot={"task":0,"episode":66,"instance":70,"frame":2,"verb":"PRESS"}
        ref={"schema":SCHEMA,"pilot":pilot,"prefix_controls":2}
        np.save(prepared/"prefix.npy",np.zeros((2,23),np.float32),allow_pickle=False)
        window={"official_mode":"train","seed":0,"instance_id":70,"task_name":"turning_on_radio",
                "prefix_actions_path":str(prepared/"prefix.npy"),"prefix_actions_sha256":sha(prepared/"prefix.npy")}
        write_json(prepared/"teacher_reference.json",ref);write_json(prepared/"window.json",window)
        row={**pilot,"reference_sha256":sha(prepared/"teacher_reference.json"),
             "window_sha256":sha(prepared/"window.json"),"prefix_sha256":sha(prepared/"prefix.npy")}
        manifest={"schema":SCHEMA,"status":"PREPARED_NOT_COLLECTED_NOT_TRAINING_DATA",
                  "sources":[row,{"task":1},{"task":3}]}
        write_json(root/"preparation.json",manifest)
        return prepared,{"preparation_manifest_sha256":sha(root/"preparation.json")}

    def test_preparation_binds_all_bytes_and_factory_task_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared,release=self.prepared_fixture(Path(tmp))
            ref,prefix,binding=verify_prepared_source(prepared,release)
            window=types.SimpleNamespace(task_name=binding["task_name"],official_mode="train",instance_id=70,seed=0)
            verify_loaded_window(window,prefix,prefix.copy(),binding)
            wrong=prefix.copy();wrong[0,5]=.01
            with self.assertRaises(ValueError):verify_loaded_window(window,wrong,prefix,binding)
            window.task_name="different_task"
            with self.assertRaises(ValueError):verify_loaded_window(window,prefix,prefix,binding)
            for name in ("window.json","prefix.npy","teacher_reference.json"):
                path=prepared/name;original=path.read_bytes();path.write_bytes(original+b" ")
                with self.subTest(file=name),self.assertRaises(ValueError):verify_prepared_source(prepared,release)
                path.write_bytes(original)
            release["preparation_manifest_sha256"]="0"*64
            with self.assertRaises(ValueError):verify_prepared_source(prepared,release)

    def test_native_preflight_enables_real_guards_and_rejects_bad_current_geometry(self):
        model,state,geometry=calibrated_fixture()
        with patch("native_teacher_collect.observed_cloud",return_value=np.tile([1.5,0,.2],(50,1))):
            servo,receipt=native_preflight(model,state,[-1,1],token_to_action("RIGHT_UP"),{},geometry)
            self.assertTrue(servo.limits.robot_geometry_guards)
            self.assertIsNotNone(servo.collision.hand_body)
            self.assertTrue(receipt["depth_guard"]["actual_robot_box_self_exclusion"])
            self.assertTrue(servo.carry)
            bad=copy.deepcopy(geometry);bad["boxes"][0]["T_base_link"][0][0]=.1
            with self.assertRaises(ValueError):native_preflight(model,state,[-1,1],token_to_action("RIGHT_UP"),{},bad)

    def test_capture_binds_raw_depth_actual_boxes_q_and_clock(self):
        model,state,geometry=calibrated_fixture()
        views=("head","left_wrist","right_wrist")
        images={v+"_rgb":np.zeros((3,4,4),np.uint8) for v in views}
        depths={v:np.ones((4,4),np.float32) for v in views}
        onboard=types.SimpleNamespace(read=lambda *a,**kw:(images,depths,{v:{"fixture":True} for v in views}))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"before"
            _,_,_,_,receipt=capture_snapshot(folder,model,lambda:state,onboard,lambda:geometry,lambda:None,
                                             {"prefix_control":2,"native_control":12},Path(tmp))
            self.assertEqual(receipt["q"],state.q.tolist())
            self.assertEqual(receipt["kinematic_model_sha256"],model.sha)
            self.assertEqual(receipt["clock"]["native_control"],12)
            for file,expected in receipt["files_sha256"].items():self.assertEqual(sha(folder/file),expected)
            with np.load(folder/"depth.npz") as saved:
                for view in views:np.testing.assert_array_equal(saved[view],depths[view])
            bad=copy.deepcopy(geometry);bad["boxes"][0]["T_base_link"][0][0]=.1
            with self.assertRaises(ValueError):capture_snapshot(Path(tmp)/"bad",model,lambda:state,onboard,
                        lambda:bad,lambda:None,{},Path(tmp))
            changed=model.state(state.q+.01,state.gripper,np.zeros(3)); sequence=iter([state,changed])
            with self.assertRaises(RuntimeError):capture_snapshot(Path(tmp)/"stale",model,lambda:next(sequence),
                        onboard,lambda:geometry,lambda:None,{},Path(tmp))


if __name__ == "__main__": unittest.main()
