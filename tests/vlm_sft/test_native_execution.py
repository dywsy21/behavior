import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts/vlm_sft"),str(ROOT/"src"),str(ROOT/"tests/semantic_robot")]
from common import CAMERAS,token_to_action,write_json
from native_execution import PROFILE,authorization_profile,completed,metadata,servo_limits,require_dataset_profile
from native_teacher_capacity import H09Y_PROFILE
from native_teacher_near_grasp import SCHEMA
from native_teacher_pregrasp_seed import PROFILE as SEED_PROFILE
from native_teacher_collect import native_preflight,require_release
from native_teacher_contract import digest,release_reviewed_record
from native_dataset import check_execution
from native_evaluation import PublicExecution,VARIANTS
from native_eval_summary import summarize
from native_eval_prepare import SCHEMA as EVAL_SCHEMA
from native_eval_run import require_evaluation,BUDGET,ROOT as EXPERIMENT
from native_actor_protocol import VERSION as ACTOR_VERSION
from native_teacher_artifacts import json_bytes
from semantic_robot.v2.servo import SafeServo
from test_hand_body_collision import calibrated_fixture
import test_native_teacher as teacher_fixtures


def receipt(token="RIGHT_CLOSE",profile=PROFILE,drift=.0029833103417):
    model,state,_=calibrated_fixture()
    servo=SafeServo(model,state,limits=servo_limits(profile))
    assert servo.begin(token_to_action(token),state)
    for _ in range(18):servo.next_action(state)
    end=copy.deepcopy(state);end.poses[token.split("_")[0].lower()][0][0]+=drift
    return servo.finish(end)


class NativeExecutionTests(unittest.TestCase):
    def test_explicit_profile_unknown_null_and_mixed_collection_fail_closed(self):
        self.assertIsNone(authorization_profile({}))
        self.assertFalse(servo_limits(None).gripper_completion_v1)
        auth={"execution_profile":PROFILE,"schema":SCHEMA,"capacity_profile":H09Y_PROFILE,"seed_profile":SEED_PROFILE}
        self.assertEqual(authorization_profile(auth,collection=True),PROFILE)
        self.assertTrue(servo_limits(PROFILE).gripper_completion_v1)
        for key,value in (("execution_profile",None),("execution_profile",True),("execution_profile","future"),
                          ("schema","manual"),("capacity_profile","near100"),("seed_profile",None)):
            bad={**auth,key:value}
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                authorization_profile(bad,collection=True)
            with self.assertRaises(ValueError):require_release(bad,"unused","unused")
        cfg=json.loads((ROOT/"configs/vlm_sft/h09y_gripper_completion_profile.json").read_text())
        self.assertEqual(cfg["execution_profile"],PROFILE)
        self.assertEqual(cfg["collection_requires_capacity_profile"],H09Y_PROFILE)
        self.assertFalse(cfg["authorize_collection"]);self.assertFalse(cfg["authorize_evaluation"])

    def test_real_receipts_are_action_bound_and_do_not_claim_grasp(self):
        for token in ("LEFT_OPEN","LEFT_CLOSE","RIGHT_OPEN","RIGHT_CLOSE"):
            f=receipt(token)
            self.assertTrue(completed(token,f,profile=PROFILE))
            self.assertEqual(f["pose_tracking_status"],"TRACKING_FAILED")
            self.assertEqual(f["holding"],"UNKNOWN");self.assertFalse(f["gripper_execution"]["success_claim"])
            self.assertFalse(completed(token,f))
            self.assertFalse(completed("RIGHT_UP",f,profile=PROFILE))
            for bad_token in ("LEFT_CLOSE","RIGHT_OPEN"):
                if bad_token!=token:self.assertFalse(completed(bad_token,f,profile=PROFILE))
        self.assertFalse(completed("RIGHT_CLOSE",{"status":"GRIPPER_COMMAND_COMPLETED"},profile=PROFILE))
        legacy=receipt(profile=None,drift=0)
        self.assertTrue(completed("RIGHT_CLOSE",legacy))
        self.assertFalse(completed("RIGHT_CLOSE",legacy,profile=PROFILE))
        self.assertFalse(completed("RIGHT_CLOSE",receipt(profile=None)))
        self.assertTrue(completed("RIGHT_UP",{"status":"TARGET_REACHED"},profile=PROFILE))
        self.assertFalse(completed("RIGHT_UP",{"status":"TRACKING_FAILED"},profile=PROFILE))

    def test_malformed_or_unsafe_receipt_is_not_a_status_whitelist(self):
        f=receipt()
        mutations=[lambda r:r.update(visual_gate_failure={}),lambda r:r.update(control_ticks=17),
            lambda r:r.update(pose_tracking_status="TARGET_REACHED"),lambda r:r.update(holding=True),
            lambda r:r["target_error_m"].update(left=.003),
            lambda r:r["gripper_execution"].update(success_claim=True),
            lambda r:r["gripper_execution"]["final_state_checks"].update(joint_bounds=False),
            lambda r:r["gripper_execution"]["limits"].update(active_position_m=.02)]
        for mutate in mutations:
            bad=copy.deepcopy(f);mutate(bad)
            self.assertFalse(completed("RIGHT_CLOSE",bad,profile=PROFILE))

    def test_collector_and_public_preflight_enable_same_real_servo_limits(self):
        model,state,geometry=calibrated_fixture();cloud=np.tile([1.5,0,.2],(50,1))
        depths={v:np.ones((2,2),np.float32) for v in CAMERAS};clock={"prefix_control":832,"native_control":12}
        capture={"clock":clock,"kinematic_model_sha256":model.sha,"q":state.q.tolist(),
            "gripper":state.gripper.tolist(),"scene_truth":False,
            "files_sha256":{"robot_self_geometry.json":hashlib.sha256(json_bytes(geometry)).hexdigest()},
            "depth_array_sha256":{v:hashlib.sha256(d.tobytes()).hexdigest() for v,d in depths.items()}}
        for profile in (None,PROFILE):
            with patch("native_teacher_collect.observed_cloud",return_value=cloud):
                collector,c= native_preflight(model,state,[1,1],token_to_action("RIGHT_CLOSE"),{},geometry,
                                              execution_profile=profile)
            for variant in VARIANTS:
                policy=PublicExecution([1,1],execution_profile=profile)
                with patch("native_evaluation.observed_cloud",return_value=cloud):
                    public,p=policy.preflight("RIGHT_CLOSE",state,model,depths,geometry,
                        capture_receipt=capture,expected_clock=clock)
                self.assertEqual(public.limits,collector.limits,variant)
                self.assertEqual(public.limits.gripper_completion_v1,profile==PROFILE)
                self.assertEqual(authorization_profile(c),profile);self.assertEqual(authorization_profile(p),profile)
                feedback=receipt(profile=profile,drift=0)
                self.assertTrue(policy.completed("RIGHT_CLOSE",feedback))
                self.assertFalse(policy.completed("RIGHT_CLOSE",{"status":"GRIPPER_COMMAND_COMPLETED"}))

    def test_dataset_receipt_profile_and_actual_controls_are_bound(self):
        for profile in (None,PROFILE):
            fields=metadata(profile);f=receipt(profile=profile,drift=0)
            record={"token":"RIGHT_CLOSE",**fields}
            request={"token":"RIGHT_CLOSE","preflight":fields.copy()}
            execution={"token":"RIGHT_CLOSE","control_start":12,"control_end":30,"feedback":f,**fields}
            check_execution(record,request,execution,profile)
            bad=copy.deepcopy(execution);bad["feedback"]={"status":"GRIPPER_COMMAND_COMPLETED"}
            with self.assertRaises(ValueError):check_execution(record,request,bad,profile)
            if profile:
                for field,value in (("control_end",29),("execution_profile",None)):
                    with self.assertRaises(ValueError):check_execution(record,request,{**execution,field:value},profile)
                with self.assertRaises(ValueError):check_execution({"token":"RIGHT_CLOSE"},request,execution,profile)
                with self.assertRaises(ValueError):check_execution(record,{**request,"preflight":{}},execution,profile)

    def test_manual_postreview_still_requires_real_full_receipt_and_judgment(self):
        request,approval,record,post=teacher_fixtures.NativeTeacherTests().fixture()
        request["execution_profile"]=PROFILE
        approval.update(token="RIGHT_CLOSE",request_sha256=digest(request))
        from dataclasses import asdict
        record["execution"].update(token="RIGHT_CLOSE",action=asdict(token_to_action("RIGHT_CLOSE")),
                                   status="GRIPPER_COMMAND_COMPLETED")
        record["feedback"]=receipt();post["record_sha256"]=digest(record)
        result=release_reviewed_record(record,post)
        self.assertFalse(result["official_success_claim"])
        record.pop("feedback");post["record_sha256"]=digest(record)
        with self.assertRaises(ValueError):release_reviewed_record(record,post)

    def test_actual_entrypoint_call_sites_use_explicit_receipt_boundary(self):
        for filename,callname,count in (("native_teacher_collect.py","execution_completed",4),
                                       ("native_dataset.py","check_execution",1)):
            tree=ast.parse((ROOT/"scripts/vlm_sft"/filename).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==callname]
            self.assertEqual(len(calls),count)
            if callname=="execution_completed":
                self.assertTrue(all({k.arg:ast.unparse(k.value) for k in c.keywords}=={"profile":"execution_profile"} for c in calls))
        tree=ast.parse((ROOT/"scripts/vlm_sft/native_eval_run.py").read_text())
        checks=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="not policy.completed(token, feedback)"]
        self.assertEqual(len(checks),1)
        for good in (True,False):
            policy=PublicExecution([1,1],execution_profile=PROFILE)
            feedback=receipt() if good else {"status":"GRIPPER_COMMAND_COMPLETED"}
            code=compile(ast.fix_missing_locations(ast.Module(body=checks,type_ignores=[])),"actual_eval_receipt_gate","exec")
            if good:exec(code,{"policy":policy,"token":"RIGHT_CLOSE","feedback":feedback})
            else:
                with self.assertRaises(RuntimeError):exec(code,{"policy":policy,"token":"RIGHT_CLOSE","feedback":feedback})

    def test_pair_summary_rejects_mixed_execution_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for variant in ("base","finetuned"):
                folder=root/f"eval_t1_i1_{variant}_v1";folder.mkdir()
                auth={"schema":EVAL_SCHEMA,"source":[1,200,1],"variant":variant}
                if variant=="finetuned":auth["execution_profile"]=PROFILE
                write_json(folder/"manifest.json",{"authorization":auth,"preparation":{"prefix_controls":832},
                    "oracle_actor_feedback":False,"specified_hand":None,"code_commit":"same","protocol":"same","dataset_sha256":"same"})
            with self.assertRaisesRegex(ValueError,"execution profile"):summarize(root)

    def test_new_dataset_needs_same_paired_executor_but_preserves_historical_w(self):
        dataset={"runs":[{},metadata(PROFILE)]}
        require_dataset_profile(dataset,PROFILE)
        with self.assertRaises(ValueError):require_dataset_profile(dataset,None)
        require_dataset_profile({"runs":[{}]},None)
        with self.assertRaises(ValueError):require_dataset_profile({"runs":[{}]},PROFILE)

    def test_evaluation_requires_new_matching_engineering_gates_for_all_three_policies(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f"gate{task}.json" for task in (0,3)]
            for variant in VARIANTS:
                auth={"schema":EVAL_SCHEMA,"authorize_evaluation":True,"execution_profile":PROFILE,
                    "code_commit":"code","executor_digest":"executor","protocol":ACTOR_VERSION,
                    "dataset_sha256":"data","source":[1,200,1],"variant":variant,"physical_gpu":3,
                    "specified_hand":None,"oracle_actor_feedback":False,"reviewer":"parent",
                    "experiment_root":str(EXPERIMENT),"budget":BUDGET,"engineering_gate_paths":list(map(str,paths))}
                args=("code","executor",{"source":[1,200,1]},variant,EXPERIMENT/f"eval_t1_i1_{variant}_v1","data")
                for flag in (False,True):
                    for task,path in zip((0,3),paths):write_json(path,{"task":task,"gate_ok":True,
                        "robot_geometry_guards":True,"gripper_completion_v1":flag,"implementation_digest":"executor"})
                    if flag:require_evaluation(auth,*args)
                    else:
                        with self.assertRaises(ValueError):require_evaluation(auth,*args)
                with self.assertRaises(ValueError):require_evaluation({**auth,"execution_profile":None},*args)
                with self.assertRaises(ValueError):require_evaluation({**auth,"seed_profile":SEED_PROFILE},*args)

    def test_collector_requires_new_matching_engineering_gates_before_any_reset(self):
        from native_teacher_capacity import H09Y_ROOT
        auth=json.loads((ROOT/"configs/vlm_sft/h09w_capacity_authorization_template.json").read_text())
        auth.update(authorize_collection=True,collector_commit="code",reviewer="parent",execution_profile=PROFILE,
            seed_profile=SEED_PROFILE,capacity_profile=H09Y_PROFILE,experiment_root=H09Y_ROOT,
            source=[1,264,114],paid_prefix_controls=989,held_out_instance_groups=[[1,1],[1,71]],
            registered_gpu=3,initialization_seconds=900,seconds_after_reset=1200,total_MiB=6144)
        for field in ("prior_experiment_roots","combined_total_MiB"):auth.pop(field)
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f"gate{task}.json" for task in (0,3)]
            auth["engineering_gate_paths"]=list(map(str,paths))
            for flag in (None,False,True,1):
                for task,path in zip((0,3),paths):
                    gate={"task":task,"gate_ok":True,"robot_geometry_guards":True,"implementation_digest":auth["executor_digest"]}
                    if flag is not None:gate["gripper_completion_v1"]=flag
                    write_json(path,gate)
                if flag is True:require_release(auth,"code",auth["executor_digest"])
                else:
                    with self.assertRaises(ValueError):require_release(auth,"code",auth["executor_digest"])


if __name__=="__main__":unittest.main()
