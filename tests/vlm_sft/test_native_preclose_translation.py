"""Public timing qualification is not a privileged empty-hand certificate."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts/vlm_sft"),str(ROOT/"tests/semantic_robot")]
from common import CAMERAS,token_to_action,sha
from native_execution import (PROFILE,PRECLOSE_PROFILE,PublicGripperHistory,preclose_translation,
    authorization_profile,require_pipeline_profile,require_same_pipeline,servo_limits,timing_metadata)
from native_teacher_collect import native_preflight,require_release
from native_evaluation import PublicExecution,VARIANTS
from native_dataset import check_public_timing
from native_teacher_artifacts import json_bytes
import native_teacher_capacity as capacity
import native_storage as storage
from native_teacher_near_grasp import SCHEMA
from native_teacher_pregrasp_seed import PROFILE as SEED_PROFILE
from semantic_robot.v2.protocol import Action
from test_hand_body_collision import calibrated_fixture


def fixture():
    model,state,geometry=calibrated_fixture()
    state.gripper[:]=.05
    model.spec["metadata"]["grasp_region_reference_gripper_m"]=[.05,.05]
    return model,state,geometry


def authorization():
    return {"schema":SCHEMA,"execution_profile":PRECLOSE_PROFILE,"capacity_profile":capacity.DIVERSE_PROFILE,
        "seed_profile":SEED_PROFILE,"storage":copy.deepcopy(storage.DIVERSE_SPEC),
        "experiment_root":capacity.H09Y_ROOT,"source":[1,310,192],"paid_prefix_controls":380,
        "held_out_instance_groups":[[1,1],[1,71]],"registered_gpu":3,"initialization_seconds":900,
        "run_MiB":384,"total_MiB":6144}


class PrecloseTranslationTests(unittest.TestCase):
    def test_exact_public_qualification_per_arm_and_unchanged_legacy(self):
        model,state,_=fixture();history=PublicGripperHistory([1,1])
        for arm in ("left","right"):
            for move in ("forward","back","left","right","up","down"):
                action=Action(arm,move)
                self.assertTrue(preclose_translation(action,state,model,history,PRECLOSE_PROFILE))
                self.assertEqual(action.amount(False),action.amount(True));self.assertEqual(action.amount(False),.01)
                for profile in (None,PROFILE):
                    self.assertFalse(preclose_translation(action,state,model,history,profile))
        for action in (Action("base","forward"),Action("both","up"),Action("torso","up"),Action("all","hold"),
                       Action("right","close"),Action("right","open"),Action("right","yaw_plus"),
                       Action("right","up","micro"),Action("right","up","coarse"),Action("right","up",frame="tool")):
            self.assertFalse(preclose_translation(action,state,model,history,PRECLOSE_PROFILE))
        self.assertEqual(servo_limits(PRECLOSE_PROFILE),servo_limits(PROFILE))

    def test_unknown_nonfinite_partial_open_or_wrong_hand_never_qualifies(self):
        model,state,_=fixture();action=Action("right","down")
        for history in (None,{},PublicGripperHistory([1,.998]),PublicGripperHistory([1,-1])):
            self.assertFalse(preclose_translation(action,state,model,history,PRECLOSE_PROFILE))
        for opening in (.049,.02,float("nan"),float("inf")):
            state.gripper[1]=opening
            self.assertFalse(preclose_translation(action,state,model,PublicGripperHistory([1,1]),PRECLOSE_PROFILE))
        state.gripper[:]=.05
        for value in (False,None,1):
            model.spec["metadata"]["grasp_region_reference_fully_open"]["right"]=value
            self.assertFalse(preclose_translation(action,state,model,PublicGripperHistory([1,1]),PRECLOSE_PROFILE))
        model.spec["metadata"].pop("grasp_region_reference_gripper_m")
        self.assertFalse(preclose_translation(action,state,model,PublicGripperHistory([1,1]),PRECLOSE_PROFILE))

    def test_actual_close_latch_never_cleared_by_open_and_is_arm_specific(self):
        model,state,_=fixture();history=PublicGripperHistory([1,1]);command=np.zeros(23);command[[14,22]]=[1,-1]
        history.issued(command)
        self.assertFalse(preclose_translation(Action("right","up"),state,model,history,PRECLOSE_PROFILE))
        self.assertTrue(preclose_translation(Action("left","up"),state,model,history,PRECLOSE_PROFILE))
        command[22]=1;history.issued(command)
        self.assertTrue(history.close_seen["right"])
        self.assertFalse(preclose_translation(Action("right","down"),state,model,history,PRECLOSE_PROFILE))
        for bad in (np.zeros(22),np.full(23,np.nan),np.full(23,2)):
            with self.assertRaises(ValueError):history.issued(bad)
        history.close_seen["left"]=None
        self.assertFalse(preclose_translation(Action("left","up"),state,model,history,PRECLOSE_PROFILE))

    def test_actual_collector_and_each_public_variant_share_timing_and_safety(self):
        model,state,geometry=fixture();cloud=np.tile([1.5,0,.2],(50,1))
        depths={v:np.ones((2,2),np.float32) for v in CAMERAS};clock={"prefix_control":832,"native_control":12}
        capture={"clock":clock,"kinematic_model_sha256":model.sha,"q":state.q.tolist(),
            "gripper":state.gripper.tolist(),"scene_truth":False,
            "files_sha256":{"robot_self_geometry.json":hashlib.sha256(json_bytes(geometry)).hexdigest()},
            "depth_array_sha256":{v:hashlib.sha256(d.tobytes()).hexdigest() for v,d in depths.items()}}
        for profile in (None,PROFILE,PRECLOSE_PROFILE):
            for closed in (False,True):
                history=PublicGripperHistory([1,1]);command=np.zeros(23);command[[14,22]]=[1,-1]
                if closed:history.issued(command)
                with patch("native_teacher_collect.observed_cloud",return_value=cloud):
                    collector,c=native_preflight(model,state,history.grips,Action("right","down"),depths,geometry,
                        execution_profile=profile,public_history=history)
                for variant in VARIANTS:
                    policy=PublicExecution([1,1],execution_profile=profile)
                    if closed:policy.issued(command)
                    with patch("native_evaluation.observed_cloud",return_value=cloud):
                        public,p=policy.preflight("RIGHT_DOWN",state,model,depths,geometry,capture_receipt=capture,expected_clock=clock)
                    expected=not (profile==PRECLOSE_PROFILE and not closed)
                    self.assertIs(c["carry"],expected,variant);self.assertEqual(c["carry"],p["carry"])
                    self.assertEqual(collector.limits,public.limits);self.assertEqual(collector.total_ticks,public.total_ticks)
                    self.assertEqual(c["amount"],.01)
                    for key,value in timing_metadata(profile,not expected).items():self.assertEqual(c[key],p[key])

    def test_dataset_recomputes_public_qualification_from_actual_issued_history(self):
        model,state,_=fixture();command=np.zeros(23);command[[14,22]]=1
        rows=[{"action23":command.tolist()} for _ in range(13)]
        before={"q":state.q.tolist(),"gripper":state.gripper.tolist()}
        request={"preflight":{**timing_metadata(PRECLOSE_PROFILE,True),"carry":False,"amount":.01}}
        check_public_timing(request,"RIGHT_DOWN",before,model,rows,1,12)
        for mutation in ({"carry":True},{"amount":.03},{"public_preclose_translation_qualification":1},
                         {"empty_hand_or_environment_contact_not_certified":False}):
            bad={"preflight":{**request["preflight"],**mutation}}
            with self.assertRaises(ValueError):check_public_timing(bad,"RIGHT_DOWN",before,model,rows,1,12)
        rows[3]=copy.deepcopy(rows[3]);rows[3]["action23"][22]=-1
        with self.assertRaises(ValueError):check_public_timing(request,"RIGHT_DOWN",before,model,rows,1,12)
        request={"preflight":{**timing_metadata(PRECLOSE_PROFILE,False),"carry":True,"amount":.01}}
        check_public_timing(request,"RIGHT_DOWN",before,model,rows,1,12)

    def test_new_capacity_exactly_two_not_old_five_and_mixed_profile_refused(self):
        auth=authorization()
        for task,episode,instance,prefix in capacity.DIVERSE_STARTS:
            a={**auth,"source":[task,episode,instance],"paid_prefix_controls":prefix}
            self.assertEqual(authorization_profile(a,collection=True),PRECLOSE_PROFILE)
            self.assertEqual(capacity.capacity_limits(a),(384*1024**2,6144*1024**2))
            capacity.validate_collection_location(a,Path(capacity.H09Y_ROOT)/f"native_t1_i{instance}_p{prefix:04d}",3)
            with self.assertRaises(ValueError):capacity.capacity_limits({**a,"capacity_profile":capacity.H09Y_PROFILE})
        for task,episode,instance,prefix in capacity.H09Y_STARTS:
            with self.assertRaises(ValueError):capacity.capacity_limits({**auth,"source":[task,episode,instance],"paid_prefix_controls":prefix})
        for mutation in ({"execution_profile":PROFILE},{"storage":storage.SHARED_SPEC},{"capacity_profile":capacity.H09Y_PROFILE},
                         {"paid_prefix_controls":True},{"source":[1,200,1]},{"registered_gpu":2},{"total_MiB":6145}):
            with self.assertRaises(ValueError):authorization_profile({**auth,**mutation},collection=True)
        with self.assertRaises(ValueError):require_release(auth,"unused","unused")

    def test_storage_adds_exact_two_aliases_without_changing_old_specs_or_limits(self):
        self.assertEqual(set(storage.DIVERSE_RUNS)-set(storage.SHARED_RUNS),{"native_t1_i192_p0380","native_t1_i114_p0969"})
        for name in storage.DIVERSE_RUNS[-2:]:
            out=storage.EXPERIMENT_ROOT/name
            new=storage.RuntimeStorage({"storage":storage.DIVERSE_SPEC},out)
            self.assertEqual(len(new.aliases),len(storage.SHARED_RUNS)+2)
            with self.assertRaises(ValueError):storage.RuntimeStorage({"storage":storage.SHARED_SPEC},out)
        for key in ("runtime_max_MiB","nvme_min_free_GiB","sda_min_free_GiB"):
            self.assertEqual(storage.DIVERSE_SPEC[key],storage.SHARED_SPEC[key])
        for mutation in ({"profile":storage.SHARED_PROFILE},{"runtime_max_MiB":16385},{"sda_min_free_GiB":31}):
            with self.assertRaises(ValueError):storage.validate_spec({"storage":{**storage.DIVERSE_SPEC,**mutation}})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/"target";target.mkdir();(target/"file").write_bytes(b"a")
            alias=root/"new/cache";alias.parent.mkdir();alias.symlink_to(target)
            self.assertEqual(storage.tree_bytes(root,root.stat().st_dev,{alias:target}),1)
            with self.assertRaises(ValueError):storage.tree_bytes(root,root.stat().st_dev,{})

    def test_actual_collection_entry_requires_separate_active_authority_and_original_gates(self):
        auth=authorization();auth.update(authorize_collection=True,collector_commit="fixed",executor_digest="executor",
            h14_body_and_finger_safety_reviewed=True,reviewer="parent",max_resets=1,max_candidates_per_instance=12,
            authorize_offline_teacher=True,allow_known_empty_rotation=True,allow_grasp_cell_attempt=True,
            native_controls_max=420,seconds_after_reset=1200,max_teacher_primitives=12,model_calls=0)
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for task in (0,3):
                path=Path(directory)/f"gate{task}.json"
                path.write_text(json.dumps({"task":task,"gate_ok":True,"robot_geometry_guards":True,
                    "gripper_completion_v1":True,"implementation_digest":"executor"}));paths.append(str(path))
            auth["engineering_gate_paths"]=paths
            require_release(auth,"fixed","executor")
            for mutation in ({"authorize_collection":False},{"authorize_collection":1},{"collector_commit":"other"},
                    {"native_controls_max":421},{"max_resets":2},{"seconds_after_reset":1201},{"max_teacher_primitives":13},
                    {"execution_profile":PROFILE},{"storage":storage.SHARED_SPEC}):
                with self.assertRaises(ValueError):require_release({**auth,**mutation},"fixed","executor")
            gate=json.loads(Path(paths[0]).read_text());gate["gripper_completion_v1"]=False
            Path(paths[0]).write_text(json.dumps(gate))
            with self.assertRaises(ValueError):require_release(auth,"fixed","executor")

    def test_dataset_train_service_and_all_three_evaluations_require_same_new_profile(self):
        dataset={"runs":[{}, {"execution_profile":PROFILE},{"execution_profile":PRECLOSE_PROFILE}]}
        release={"execution_profile":PRECLOSE_PROFILE,"storage":storage.DIVERSE_SPEC}
        self.assertEqual(require_pipeline_profile(release,dataset),PRECLOSE_PROFILE)
        for bad in ({},{"execution_profile":PROFILE},{"execution_profile":PRECLOSE_PROFILE},
                    {"execution_profile":PRECLOSE_PROFILE,"storage":storage.SHARED_SPEC}):
            with self.assertRaises(ValueError):require_pipeline_profile(bad,dataset)
        identity={"execution_profile":PRECLOSE_PROFILE,"storage":{"profile":storage.DIVERSE_STORAGE_PROFILE}}
        require_same_pipeline(release,identity)
        for bad in ({},{"execution_profile":PROFILE,"storage":identity["storage"]},
                    {"execution_profile":PRECLOSE_PROFILE,"storage":{"profile":storage.SHARED_PROFILE}}):
            with self.assertRaises(ValueError):require_same_pipeline(release,bad)
        # Old reviewed W + gripper-v1 remain admitted, without retroactive rewriting.
        self.assertIsNone(require_pipeline_profile({}, {"runs":[{}, {"execution_profile":PROFILE}]}))

    def test_registration_does_not_grant_run_training_or_evaluation(self):
        cfg=json.loads((ROOT/"configs/vlm_sft/h09y_public_preclose_translation_profile.json").read_text())
        self.assertEqual(cfg["execution_profile"],PRECLOSE_PROFILE)
        self.assertEqual({tuple(s) for s in cfg["starts"]},capacity.DIVERSE_STARTS)
        for key in ("authorize_collection","authorize_training","authorize_service","authorize_evaluation"):
            self.assertIs(cfg[key],False)

    def test_actual_model_service_loader_checks_identity_and_each_response_profile(self):
        import native_eval_run as runner
        from native_actor_protocol import VERSION
        identity={"protocol":VERSION,"code_commit":"code","dataset_sha256":"data","physical_gpu":3,
            "port":8919,"max_calls":48,"execution_profile":PRECLOSE_PROFILE,
            "storage":{"profile":storage.DIVERSE_STORAGE_PROFILE},"adapter_sha256":"adapter","training_result_sha256":"training"}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"service_v1").mkdir();path=root/"service_v1/identity.json"
            path.write_text(json.dumps(identity))
            auth={"execution_profile":PRECLOSE_PROFILE,"storage":storage.DIVERSE_SPEC,"service_identity_sha256":sha(path)}
            answer={k:identity[k] for k in ("code_commit","dataset_sha256","adapter_sha256","training_result_sha256","execution_profile")}
            answer["storage_profile"]=storage.DIVERSE_STORAGE_PROFILE
            for mutation in ({},{"execution_profile":PROFILE},{"storage_profile":storage.SHARED_PROFILE}):
                streams=[io.BytesIO(json.dumps({**identity,"calls":0}).encode()),io.BytesIO(json.dumps({**answer,**mutation}).encode())]
                with patch.object(runner,"ROOT",root),patch.object(runner,"urlopen",side_effect=streams):
                    _,remote=runner.load_service(auth,"code","data")
                    if mutation:
                        with self.assertRaisesRegex(ValueError,"profile drift"):remote({})
                    else:self.assertEqual(remote({}),answer)
            bad={**identity,"execution_profile":PROFILE};path.write_text(json.dumps(bad));auth["service_identity_sha256"]=sha(path)
            with patch.object(runner,"ROOT",root),patch.object(runner,"urlopen") as network:
                with self.assertRaises(ValueError):runner.load_service(auth,"code","data")
                network.assert_not_called()


if __name__=="__main__":unittest.main()
