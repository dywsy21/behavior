import ast
import copy
import io
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts/vlm_sft"),str(ROOT/"src"),str(ROOT/"scripts/semantic_robot")]
from common import sha,write_json
from native_actor_protocol import VERSION
from native_eval_prepare import select,STARTS,INSTRUCTION,SCHEMA
from native_eval_run import require_evaluation,BUDGET,ROOT as EXPERIMENT
from native_grasp_sources import reserve_groups
from native_teacher_group_prepare import COUNTS_SHA
from native_evaluation import PublicExecution
from native_serve import validate_checkpoint
from native_train import BASE
from native_eval_summary import summarize


class EvalEntrypointTests(unittest.TestCase):
    def test_actual_runner_rechecks_live_state_after_policy_latency(self):
        from unittest.mock import Mock
        tree=ast.parse((ROOT/"scripts/vlm_sft/native_eval_run.py").read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="preflight_at_execution")
        before=NS(q=np.zeros(18),gripper=np.ones(2)*.05)
        policy=Mock();model=object();receipt={};depths={};geometry={}
        for field in (None,"q","gripper"):
            current=copy.deepcopy(before)
            if field:getattr(current,field)[0]+=.001
            state=Mock(return_value=current);policy.reset_mock()
            ns={"np":np,"state":state,"model":model,"policy":policy,"prefix_count":832,"controls":12}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),"actual_live_preflight","exec"),ns)
            if field:
                with self.assertRaisesRegex(RuntimeError,"stale token"):
                    ns["preflight_at_execution"]("RIGHT_UP",before,depths,geometry,receipt)
                policy.preflight.assert_not_called()
            else:
                ns["preflight_at_execution"]("RIGHT_UP",before,depths,geometry,receipt)
                self.assertIs(policy.preflight.call_args.args[1],current)
            state.assert_called_once_with()
        # The actual loop calls this fresh-state boundary, not only a tested
        # helper disconnected from the runner.
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
            n.func.id=="preflight_at_execution" for n in ast.walk(tree)),1)

    def test_missing_comparison_slots_are_not_completed_or_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            result=summarize(d)
            self.assertEqual(len(result["rows"]),6)
            self.assertIn("INCOMPLETE",result["status"])
            self.assertTrue(all(v==[0,2] for v in result["per_variant_registered_numerator_denominator"].values()))
            self.assertFalse(result["full_task_SR_claim"])

    def test_fixed_heldout_source_and_no_expert_hand_in_instruction(self):
        counts=json.loads((ROOT/"configs/vlm_sft/h09r_train_feasibility_counts.json").read_text())
        audit={"counts_sha256":COUNTS_SHA,"groups":[]}
        for split,source in reserve_groups(counts):
            audit["groups"].append({"split":split,"source":source,"selected_earliest_eligible":{
                "near_prefix_controls":STARTS.get(source["instance"],{}).get("prefix",396),
                "skill":{"verb":"GRASP","target":"trash_can_116","source":"floors","arm":"UNSPECIFIED"}}})
        for instance in (1,71):self.assertEqual(select(counts,audit,instance)["source"]["instance"],instance)
        for instance in (192,114,True):
            with self.assertRaises(ValueError):select(counts,audit,instance)
        for key,value in (("near_prefix_controls",831),("skill",{"verb":"GRASP","target":"trash_can_116","source":"floors","arm":"LEFT"})):
            bad=copy.deepcopy(audit);row=next(r for r in bad["groups"] if r["source"]["instance"]==1)
            row["selected_earliest_eligible"][key]=value
            with self.assertRaises(ValueError):select(counts,bad,1)

    def test_single_evaluation_authority_not_six_implicit_retries(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for task in (0,3):
                p=Path(d)/f"gate{task}.json";write_json(p,{"task":task,"gate_ok":True,"robot_geometry_guards":True,"implementation_digest":"executor"});paths.append(str(p))
            prepared={"source":[1,200,1]};output=EXPERIMENT/"eval_t1_i1_base_v1"
            auth={"schema":SCHEMA,"authorize_evaluation":True,"code_commit":"code","executor_digest":"executor",
                "protocol":VERSION,"dataset_sha256":"data","source":[1,200,1],"variant":"base","physical_gpu":3,
                "specified_hand":None,"oracle_actor_feedback":False,"reviewer":"parent","experiment_root":str(EXPERIMENT),
                "budget":BUDGET,"engineering_gate_paths":paths}
            require_evaluation(auth,"code","executor",prepared,"base",output,"data")
            changes=[{"authorize_evaluation":1},{"code_commit":"old"},{"source":[1,264,114]},
                {"physical_gpu":True},{"specified_hand":"left"},{"oracle_actor_feedback":True},
                {"variant":"finetuned"},{"dataset_sha256":"old"},{"budget":{**BUDGET,"resets":6}},
                {"budget":{**BUDGET,"new_controls_including_final_hold":421}},{"budget":{**BUDGET,"max_decisions":True}}]
            for mutation in changes:
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    require_evaluation({**auth,**mutation},"code","executor",prepared,"base",output,"data")
            with self.assertRaises(ValueError):require_evaluation(auth,"code","executor",prepared,"base",EXPERIMENT/"retry","data")

    def test_only_fixed_terminal_fresh_checkpoint_with_bound_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/"adapter_0120";folder.mkdir()
            (folder/"adapter_model.safetensors").write_bytes(b"synthetic_no_model");write_json(folder/"adapter_config.json",{})
            identity={"dataset_sha256":"data","protocol":VERSION,"old_adapter_loaded":False,"base_model":BASE}
            write_json(root/"identity.json",identity)
            checkpoint={"step":120,"adapter_sha256":sha(folder/"adapter_model.safetensors"),"adapter_config_sha256":sha(folder/"adapter_config.json")}
            result={"status":"COMPLETE","optimizer_updates":120,"protocol":VERSION,"dataset_sha256":"data",
                "identity_sha256":sha(root/"identity.json"),"checkpoints":[checkpoint]}
            write_json(root/"result.json",result);write_json(root/"restore_gate.json",{"passed":True})
            self.assertEqual(validate_checkpoint(root,"data")[0],folder)
            for change in ({"optimizer_updates":2},{"status":"FAILED"},{"identity_sha256":"stale"}):
                write_json(root/"result.json",{**result,**change})
                with self.assertRaises(ValueError):validate_checkpoint(root,"data")
            write_json(root/"result.json",result);write_json(root/"restore_gate.json",{"passed":1})
            with self.assertRaises(ValueError):validate_checkpoint(root,"data")

    def test_real_evaluation_step_latches_before_partial_failure_and_never_uses_measured_truth(self):
        tree=ast.parse((ROOT/"scripts/vlm_sft/native_eval_run.py").read_text())
        fn=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="step"))
        fn.body[0]=ast.Global(names=fn.body[0].names)
        command=np.zeros(23,np.float32);command[[14,22]]=[1,-1]
        state=NS(q=np.zeros(18),gripper=np.ones(2)*.05);issued=io.StringIO();completed=io.StringIO();measurements=[]
        def fail(*args,**kwargs):raise RuntimeError("partially executed CLOSE")
        evaluator=NS(_preprocess_obs=lambda x:x,_sync_lights_and_get_obs=lambda x:x)
        policy=PublicExecution([1,1]);writer=NS(check=lambda *a:None,append_text=lambda s,line:s.write(line))
        ns={"np":np,"json":json,"controls":12,"prefix_count":832,"issued_native":12,"issued_prefix":832,
            "grips":np.ones(2),"terminal":False,"last_info":{},"started":0,"time":NS(monotonic=lambda:1),
            "check_storage":lambda *a:None,"storage":None,"check_actual_joint_bounds":lambda *a:None,
            "state":lambda:state,"model":object(),"current_writer":writer,"trace":completed,"issue_trace":issued,
            "reader":object(),"measure":lambda:measurements.append({"held":True,"forbidden_contacts":True}),
            "policy":policy,"og":NS(sim=NS(render_on_step=lambda _:nullcontext())),
            "env":NS(step=fail),"session":NS(evaluator=evaluator)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),"real_evaluation_step","exec"),ns)
        with self.assertRaisesRegex(RuntimeError,"partially"):ns["step"](command,"candidate")
        self.assertEqual((ns["issued_native"],ns["controls"]),(13,12));self.assertTrue(policy.close_seen["right"])
        self.assertEqual(json.loads(issued.getvalue())["action23"][22],-1);self.assertEqual(completed.getvalue(),"")
        ns.update(issued_native=12,env=NS(step=lambda *a,**k:({},0,False,False,{})))
        ns["step"](command,"candidate")
        self.assertEqual(ns["controls"],13);self.assertEqual(len(measurements),1)
        # Measuring hidden failure does not stop this public control function;
        # the separately imported posthoc scorer alone interprets it.


if __name__=="__main__":unittest.main()
