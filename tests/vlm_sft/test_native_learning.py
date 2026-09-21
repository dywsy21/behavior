import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts/vlm_sft"),str(ROOT/"src")]
import native_actor_protocol as protocol
from common import CAMERAS,sha,write_json
from native_dataset import checked_images,coverage,load_dataset,verified_run
from native_evaluation import features,NearestNeighbor,PublicExecution,choose
from native_eval_outcomes import score_grasp
from native_train import CONFIG,validate_config,require_training
from native_teacher_artifacts import json_bytes
import test_native_actor_protocol as protocol_fixtures
from test_native_teacher_automatic import frame


class NativeLearningTests(unittest.TestCase):
    def fixture(self):return protocol_fixtures.PoseProtocolTests().fixture()

    def test_training_images_are_raw_hashed_bytes_not_arbitrary_paths(self):
        _,_,_,actor,raw=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row=protocol.training_row(actor,"RIGHT_UP");row["images"]={}
            for view,data in raw.items():
                path=root/(view+".png");path.write_bytes(data);row["images"][view]=str(path)
            self.assertEqual(checked_images(row)["head"].size,(256,256))
            bad=copy.deepcopy(row);bad.pop("protocol")
            with self.assertRaises(ValueError):checked_images(bad)
            bad=copy.deepcopy(row);bad["actor"]["target_pose"]=[0,0,0]
            with self.assertRaises(ValueError):checked_images(bad)
            (root/"head.png").write_bytes(raw["head"]+b"extra")
            with self.assertRaises(ValueError):checked_images(row)

    def test_parent_review_digest_and_failure_rejected_before_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);review=root/"review.json";inventory=root/"inventory.json"
            write_json(inventory,{"files":{}});write_json(review,{"reviewer":"Codex-parent","inventory_sha256":sha(inventory),
                "candidate_single_trajectory_accepted":False,"local_success":True})
            entry={"run":str(root/"run"),"review":str(review),"inventory":str(inventory),"review_sha256":sha(review)}
            with self.assertRaises(ValueError):verified_run(entry,{})
            entry["review_sha256"]="0"*64
            with self.assertRaises(ValueError):verified_run(entry,{})

    def test_four_run_gate_is_not_single_trajectory_or_random_frame_split(self):
        _,_,_,actor,_=self.fixture();rows=[];runs=[]
        tokens=["RIGHT_LEFT","RIGHT_YAW_PLUS","RIGHT_LEFT","RIGHT_ROLL_MINUS","RIGHT_CLOSE","RIGHT_UP","RIGHT_UP","RIGHT_UP"]
        for i,group in enumerate(([1,192],[1,192],[1,114],[1,114])):
            a=copy.deepcopy(actor);a["proprio"]["eef_base_m"]["right"][0]+=.01*i
            run={"group":group,"prefix":390+i,"hand":"right","inventory_sha256":str(i)*64,"initial_proprio":a["proprio"],"measured_rotation_degrees":[3.]};runs.append(run)
            history=[]
            for j,t in enumerate(tokens):
                value=copy.deepcopy(a);value["history"]=history[-5:]
                rows.append({**protocol.training_row(value,t),"provenance":{"group":group,"macro":j,"run_inventory_sha256":str(i)*64}});history.append(t)
        self.assertTrue(coverage(rows,runs)["passed"])
        self.assertFalse(coverage(rows[:8],runs[:1])["passed"])
        bad=copy.deepcopy(runs);bad[1]["initial_proprio"]=bad[0]["initial_proprio"]
        self.assertFalse(coverage(rows,bad)["passed"])
        bad=copy.deepcopy(runs);bad[0]["group"]=[1,1]
        self.assertFalse(coverage(rows,bad)["passed"])
        bad=copy.deepcopy(runs)
        for run in bad:run["measured_rotation_degrees"]=[]
        self.assertFalse(coverage(rows,bad)["passed"])

    def test_training_identity_no_old_adapter_gpu_or_extra_updates(self):
        validate_config(copy.deepcopy(CONFIG))
        for key,value in (("protocol",None),("physical_gpu",1),("old_adapter","old"),("max_updates_including_gate",122),("seed",True)):
            cfg=copy.deepcopy(CONFIG);cfg[key]=value
            with self.assertRaises(ValueError):validate_config(cfg)
        release={"authorize_training":True,"code_commit":"a","config_sha256":"b","dataset_sha256":"c","reviewer":"parent"}
        require_training(release,"a","b","c")
        for key,value in (("authorize_training",1),("reviewer",""),("dataset_sha256","changed")):
            bad={**release,key:value}
            with self.assertRaises(ValueError):require_training(bad,"a","b","c")

    def test_nn_has_no_vision_instance_or_hidden_feature(self):
        _,_,_,actor,raw=self.fixture()
        rows=[{**protocol.training_row(actor,"RIGHT_UP"),"id":"one"}];nn=NearestNeighbor(rows)
        mutated=copy.deepcopy(actor);mutated["current_rgb_sha256"]={v:"f"*64 for v in CAMERAS}
        self.assertTrue(np.array_equal(features(actor),features(mutated)))
        self.assertEqual(choose("proprio_history_nn",mutated,{},nn=nn)["prediction"],"RIGHT_UP")
        for field in ("held","target_pose","seed","goal_status","teacher_stage"):
            with self.assertRaises(ValueError):features({**actor,field:True})
        for variant in ("base","finetuned"):
            seen=[]
            result=choose(variant,actor,raw,remote=lambda request:(seen.append(request) or {
                "prediction":"RIGHT_UP","variant":variant,"protocol":protocol.VERSION}))
            self.assertEqual(result["prediction"],"RIGHT_UP");self.assertEqual(seen[0]["actor"],actor)
        with self.assertRaises(ValueError):choose("base",actor,raw,remote=lambda _: {"prediction":"RIGHT_UP","variant":"base"})

    def test_rotation_qualification_is_robot_only_and_issued_not_proposed(self):
        model,state,_,_,_=self.fixture()
        model.spec["metadata"]={"grasp_region_reference_gripper_m":[.05,.05],"grasp_region_reference_fully_open":{"left":True,"right":True}}
        policy=PublicExecution([1,1]);self.assertTrue(policy.rotation_qualified("right",state,model))
        # A proposed CLOSE has not called issued(), so cannot change latch.
        self.assertFalse(policy.close_seen["right"])
        command=np.zeros(23);command[[14,22]]=[1,-1];policy.issued(command)
        self.assertFalse(policy.rotation_qualified("right",state,model));self.assertTrue(policy.rotation_qualified("left",state,model))
        command[22]=1;policy.issued(command)
        self.assertFalse(policy.rotation_qualified("right",state,model))
        unknown=PublicExecution([0,1]);self.assertFalse(unknown.rotation_qualified("left",state,model))
        model.spec["metadata"]["grasp_region_reference_fully_open"]["right"]=None
        self.assertFalse(PublicExecution([1,1]).rotation_qualified("right",state,model))

    def test_fresh_public_preflight_binding_and_real_call_signature(self):
        model,state,clock,_,_=self.fixture();model.spec["metadata"]={"grasp_region_reference_gripper_m":[.05,.05],
            "grasp_region_reference_fully_open":{"left":True,"right":True}}
        geometry={"scene_truth":False};depth={v:np.ones((2,2),np.float32) for v in CAMERAS}
        receipt={"clock":clock,"kinematic_model_sha256":model.sha,"q":state.q.tolist(),"gripper":state.gripper.tolist(),"scene_truth":False,
            "files_sha256":{"robot_self_geometry.json":hashlib.sha256(json_bytes(geometry)).hexdigest()},
            "depth_array_sha256":{v:hashlib.sha256(x.tobytes()).hexdigest() for v,x in depth.items()}}
        class Guard:
            def __init__(self,points,m,q,d,self_geometry):self.geometry=self_geometry
            def check(self,action,carry):return True,"visible only"
            def receipt(self):return {"scene_truth":False}
        class Servo:
            def __init__(self,m,s,gripper_command,limits):assert limits.robot_geometry_guards is True;self.total_ticks=18;self.status="RUNNING"
            def begin(self,action,state,carry):self.action=action;self.carry=carry;return True
        with patch("native_evaluation.LocalDepthGuard",Guard),patch("native_evaluation.SafeServo",Servo),patch("native_evaluation.observed_cloud",lambda *a:np.zeros((1,3))):
            policy=PublicExecution([1,1])
            servo,r=policy.preflight("RIGHT_YAW_PLUS",state,model,depth,geometry,capture_receipt=receipt,expected_clock=clock)
            self.assertFalse(servo.carry);self.assertAlmostEqual(r["amount"],np.deg2rad(3))
            _,r=policy.preflight("RIGHT_UP",state,model,depth,geometry,capture_receipt=receipt,expected_clock=clock)
            self.assertTrue(r["carry"]);self.assertEqual(r["amount"],.01)
            _,r=policy.preflight("BASE_YAW_PLUS",state,model,depth,geometry,capture_receipt=receipt,expected_clock=clock)
            self.assertFalse(r["public_full_open_rotation_qualification"])
            bad=copy.deepcopy(receipt);bad["q"][0]+=.001
            with self.assertRaises(ValueError):policy.preflight("HOLD",state,model,depth,geometry,capture_receipt=bad,expected_clock=clock)
            depth["head"][0,0]=2
            with self.assertRaises(ValueError):policy.preflight("HOLD",state,model,depth,geometry,capture_receipt=receipt,expected_clock=clock)

    def test_posthoc_any_hand_no_cross_hand_pooling_or_missed_close(self):
        spec={"verb":"GRASP","hand":"right","target":"target"};initial=frame();frames=[]
        for t in range(1,14):
            f=frame(t);f["held"]["left"]=f["finger_contact"]["left"]=True
            if t>=2:f["target_pose"][2][3]=f["hand_poses"]["left"][2][3]=.04
            frames.append(f)
        end=copy.deepcopy(frames[-1]);end["tick"]=14
        score=score_grasp(spec,initial,frames,{1:"LEFT_CLOSE"},end)
        self.assertTrue(score["any_hand_local_success"]);self.assertEqual(score["actual_successful_hands"],["left"])
        self.assertFalse(score_grasp(spec,initial,frames,{1:"LEFT_CLOSE"},end,specified_hand="right")["any_hand_local_success"])
        self.assertFalse(score_grasp(spec,initial,frames,{1:"RIGHT_CLOSE"},end)["any_hand_local_success"])
        split=copy.deepcopy(frames)
        for f in split:f["finger_contact"]={"left":False,"right":True}
        bad=copy.deepcopy(end);bad["finger_contact"]={"left":False,"right":True}
        self.assertFalse(score_grasp(spec,initial,split,{1:"LEFT_CLOSE",2:"RIGHT_CLOSE"},bad)["any_hand_local_success"])
        end["forbidden_contacts"]=[["finger","torso"]]
        self.assertFalse(score_grasp(spec,initial,frames,{1:"LEFT_CLOSE"},end)["any_hand_local_success"])

    def test_public_module_never_imports_oracle_or_reader(self):
        tree=ast.parse((ROOT/"scripts/vlm_sft/native_evaluation.py").read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(set(imports)&{"native_teacher_og","native_teacher_policy","native_teacher_outcomes","native_eval_outcomes"})


if __name__=="__main__":unittest.main()
