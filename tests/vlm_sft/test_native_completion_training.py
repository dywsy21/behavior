import copy
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts/vlm_sft"), str(ROOT/"src")]
from common import CAMERAS, sha
import native_actor_protocol as old
import modeling as old_modeling
import native_completion_protocol as protocol
from native_completion_modeling import encode, check_mask, collate
from native_completion_runtime import (CONFIG, SERVICE_CONFIG, validate_config, INITIAL_ADAPTER_SHA,
    INITIAL_ADAPTER_CONFIG_SHA, INITIAL_RESULT_SHA, require_authorization, CompletionStorage,
    EXECUTOR_DIGEST, CORE_COMMIT, ROOT as EXPERIMENT_ROOT, WALL2100_SPEC, initial_checkpoint,
    completed_checkpoint)
from native_completion_training_data import (project_examples, BalancedSampler, load_reviewed,
    DATA_SHA, STATES_SHA, REVIEW_SHA)
from native_completion_serve import DecisionEngine, validate_decision
from native_completion import loss_mask
from native_motion_codec import ACTOR_VERSION
import test_native_actor_protocol as fixtures


def fixture():
    _, _, _, actor, raw = fixtures.PoseProtocolTests().fixture()
    actor["protocol"] = ACTOR_VERSION
    return actor, raw


def prediction(token):
    return {"prediction": token, "latency_s": .01, "input_tokens": 42, "output_tokens": 3, "input_ids_sha256": "a"*64}


class CompletionProtocolTests(unittest.TestCase):
    def test_status_formatter_separate_and_motion_messages_exact_old(self):
        actor, raw = fixture(); images = {k: Image.open(BytesIO(v)).convert("RGB").resize((256,256)) for k,v in raw.items()}
        motion = protocol.query_row(actor, "motion", "RIGHT_UP")
        self.assertEqual(protocol.messages(motion, images), old_modeling.messages(old.training_row(actor, "RIGHT_UP"), images))
        status = protocol.query_row(actor, "status", "REQUEST_VERIFY")
        self.assertNotEqual(protocol.messages(status, images)[0], protocol.messages(motion, images)[0])
        for forbidden in ("Next motion:", "Available motions:", "SUCCEEDED", "prefix_control", "run_inventory", "same_state_tick", "target_pose"):
            self.assertNotIn(forbidden, status["text"])
        self.assertEqual(set(status["actor"]), old.ACTOR_KEYS)
        for field in ("outcome", "teacher", "instance", "capture_clock", "held", "future_q"):
            with self.assertRaises(ValueError): protocol.query_row({**actor,field:True}, "status")
        with self.assertRaises(ValueError): protocol.query_row(actor, "status", "RIGHT_UP")
        with self.assertRaises(ValueError): protocol.query_row(actor, "motion", "REQUEST_VERIFY")
        with self.assertRaises(ValueError): protocol.validate_query({**status,"private":{}}, supervised=True)
        with self.assertRaises(ValueError): protocol.validate_query({**status,"text":old.prompt(actor)}, supervised=True)

    def test_real_tensor_masks_inference_prefix_eos_and_padding(self):
        import torch
        actor, raw = fixture(); images = {k: Image.open(BytesIO(v)).convert("RGB") for k,v in raw.items()}
        class Tokenizer:
            unk_token_id=0; eos_token_id=300; pad_token_id=0
            def convert_tokens_to_ids(self,x): return 300 if x=="<|im_end|>" else 0
            def convert_ids_to_tokens(self,x): return "<|im_end|>" if x==300 else "?"
            def encode(self,text,**_): return {"RIGHT_UP":[20,21,22],"CONTINUE":[30],"REQUEST_VERIFY":[40,41]}[text]
            def decode(self,ids,**_): return {(20,21,22,300):"RIGHT_UP",(30,300):"CONTINUE",(40,41,300):"REQUEST_VERIFY"}[tuple(ids)]
        class Processor:
            tokenizer=Tokenizer()
            def apply_chat_template(self,msg,**kwargs):
                self.last=kwargs
                ids=torch.tensor([[1]+[c+1 for c in (msg[0]["content"]+msg[1]["content"][-1]["text"]).encode()[::8]]])
                return {"input_ids":ids,"attention_mask":torch.ones_like(ids),"mm_token_type_ids":torch.zeros_like(ids)}
        proc=Processor(); all_encoded=[]
        for kind,target in (("motion","RIGHT_UP"),("status","CONTINUE"),("status","REQUEST_VERIFY")):
            row=protocol.query_row(actor,kind,target); inference=protocol.query_row(actor,kind)
            train=encode(proc,row,images,supervised=True); infer=encode(proc,inference,images,supervised=False)
            check_mask(proc,row,train); n=infer["input_ids"].shape[1]
            self.assertTrue(torch.equal(train["input_ids"][:,:n],infer["input_ids"]))
            self.assertTrue((train["labels"][:,:n]==-100).all()); self.assertFalse(proc.last["enable_thinking"])
            bad={k:v.clone() for k,v in train.items()};bad["labels"][0,0]=7
            with self.assertRaises((RuntimeError,KeyError)):check_mask(proc,row,bad)
            all_encoded.append(train)
        batch=collate(all_encoded,0)
        self.assertTrue((batch["labels"][batch["attention_mask"]==0]==-100).all())
        with self.assertRaises(ValueError): encode(proc,{**row,"target":"HOLD"},images,supervised=True)

    def test_two_adapter_same_snapshot_service_and_verify_no_motion(self):
        actor,raw=fixture(); identity={"max_calls":64,"variants":list(protocol.VARIANTS),
            "instructions":[actor["active_instruction"]],"adapter_sha256":{protocol.VARIANTS[0]:"a"*64,protocol.VARIANTS[1]:"b"*64}}
        seen=[];ledger=[]
        def predict(variant,row,images):
            seen.append((variant,copy.deepcopy(row),{k:v.tobytes() for k,v in images.items()}))
            return prediction("CONTINUE" if row["query_kind"]=="status" else "RIGHT_UP")
        engine=DecisionEngine(identity,"c"*64,predict,check=lambda:None,record=ledger.append)
        for i,variant in enumerate(protocol.VARIANTS):
            request=protocol.request_payload(actor,raw,variant,f"r{i}","c"*64)
            result=engine.decide(request); validate_decision(result,request,identity,"c"*64)
            self.assertEqual(result["motion"],"RIGHT_UP");self.assertFalse(result["success_claim"])
            self.assertEqual(seen[-2][1]["actor"],seen[-1][1]["actor"]);self.assertEqual(seen[-2][2],seen[-1][2])
            with self.assertRaises(ValueError):engine.decide(request)
        self.assertEqual(engine.calls,4);self.assertEqual(len(ledger),8)
        engine.predict=lambda *args:prediction("REQUEST_VERIFY")
        result=engine.decide(protocol.request_payload(actor,raw,protocol.VARIANTS[1],"verify","c"*64))
        self.assertIsNone(result["motion"]);self.assertEqual(len(result["queries"]),1);self.assertEqual(engine.calls,5)

    def test_service_strict_identity_images_history_and_prediction_receipts(self):
        actor,raw=fixture();variant=protocol.VARIANTS[1]
        identity={"max_calls":64,"variants":list(protocol.VARIANTS),"instructions":[actor["active_instruction"]],
            "adapter_sha256":{v:"a"*64 for v in protocol.VARIANTS}}
        request=protocol.request_payload(actor,raw,variant,"request","c"*64)
        engine=DecisionEngine(identity,"c"*64,lambda *args:prediction("REQUEST_VERIFY"),check=lambda:None,record=lambda _:None)
        for field,value in (("identity_sha256","d"*64),("snapshot_sha256","d"*64),("variant","base"),("protocol","old"),("request_id","bad id")):
            with self.assertRaises(ValueError):engine.decide({**request,field:value})
        for field,value in (("proprio",{**actor["proprio"],"held":True}),("history",["STOP"]),
                            ("current_rgb_sha256",{v:"d"*64 for v in CAMERAS})):
            wrong=copy.deepcopy(request);wrong["actor"][field]=value
            with self.assertRaises(ValueError):engine.decide(wrong)
        wrong=copy.deepcopy(request);wrong["images"]["head"]=wrong["images"]["left_wrist"]
        with self.assertRaises(ValueError):engine.decide(wrong)
        self.assertEqual(engine.calls,0)
        result=engine.decide(request)
        for field,value in (("success_claim",True),("motion","HOLD"),("adapter_sha256","d"*64),("variant",protocol.VARIANTS[0])):
            with self.assertRaises(ValueError):validate_decision({**result,field:value},request,identity,"c"*64)
        for field,value in (("call",True),("latency_s",float("nan")),("prediction","SUCCEEDED"),("query_kind","motion")):
            wrong=copy.deepcopy(result);wrong["queries"][0][field]=value
            with self.assertRaises(ValueError):validate_decision(wrong,request,identity,"c"*64)

    def test_budget_failures_consumed_and_no_hidden_retry_or_extra_motion(self):
        actor,raw=fixture();identity={"max_calls":64,"variants":list(protocol.VARIANTS),
            "instructions":[actor["active_instruction"]],"adapter_sha256":{v:"a"*64 for v in protocol.VARIANTS}}
        events=[];engine=DecisionEngine(identity,"c"*64,lambda *args:prediction("CONTINUE"),check=lambda:None,record=events.append)
        engine.calls=63;request=protocol.request_payload(actor,raw,protocol.VARIANTS[1],"cap","c"*64)
        with self.assertRaises(RuntimeError):engine.decide(request)
        self.assertEqual(engine.calls,64);self.assertEqual(len(events),2)
        with self.assertRaises(ValueError):engine.decide(request)
        engine=DecisionEngine(identity,"c"*64,lambda *args:(_ for _ in ()).throw(RuntimeError("backend")),check=lambda:None,record=events.append)
        with self.assertRaises(RuntimeError):engine.decide(request)
        self.assertEqual(engine.calls,1)
        with self.assertRaises(ValueError):engine.decide(request)


class CompletionTrainingAdmissionTests(unittest.TestCase):
    def examples(self):
        actor,_=fixture(); original=[];states=[]
        for group,count in enumerate((10,4,6,6,8)):
            history=[];last=None
            for index in range(count):
                a=copy.deepcopy(actor);a["history"]=history[-5:]
                label={"skill_status":"CONTINUE","motion":"RIGHT_UP"}
                row={**old.training_row(a,"RIGHT_UP"),"id":f"g{group}m{index}","images":{},
                    "provenance":{"run_inventory_sha256":str(group)*64,"capture_sha256":f"{index:064x}"}}
                state={"schema":"h09aa-reviewed-completion-state-v1","completion_protocol":"h09aa-completion-request-v1",
                    **{k:copy.deepcopy(row[k]) for k in ("protocol","actor","text","images","provenance")},
                    "id":row["id"]+"_continue","label":label,"loss_mask":loss_mask(label),
                    "source_motion_row_id":row["id"],"offline_label_evidence":{}}
                original.append(row);states.append(state);history.append("RIGHT_UP");last=state
            terminal=copy.deepcopy(last);terminal["id"]+= "terminal";terminal["source_motion_row_id"]=None
            terminal["actor"]["history"]=history[-5:];terminal["text"]=old.prompt(terminal["actor"])
            terminal["label"]={"skill_status":"REQUEST_VERIFY","motion":None};terminal["loss_mask"]=loss_mask(terminal["label"])
            terminal["provenance"]["capture_sha256"]="f"*64;states.append(terminal)
        return original,states
    def test_exact73_queries_preserved34_and_balanced120_reproducible_draws(self):
        original,states=self.examples();examples=project_examples(original,states)
        self.assertEqual(len(examples),73);self.assertEqual(sum(e["category"]=="motion" for e in examples),34)
        a,b=BalancedSampler(examples),BalancedSampler(examples);draws=0;groups=set()
        for _ in range(120):
            one,two=a.update(),b.update();self.assertEqual(one,two)
            self.assertEqual([[x["category"] for x in batch] for batch in one],[["motion"]*2,["motion"]*2,["CONTINUE"]*2,["REQUEST_VERIFY"]*2])
            draws+=sum(map(len,one));groups.update(e["group"] for batch in one for e in batch)
        self.assertEqual(draws,960);self.assertEqual(len(groups),5)
        for e in examples:self.assertEqual(set(e["query"]["actor"]),old.ACTOR_KEYS)
    def test_duplicates_history_target_or_terminal_motion_do_not_admit(self):
        original,states=self.examples()
        for mutate in (lambda s:s.append(copy.deepcopy(s[0])),lambda s:s[1].update(id=s[0]["id"]),
            lambda s:s[1]["provenance"].update(capture_sha256=s[0]["provenance"]["capture_sha256"]),
            lambda s:s[0]["label"].update(motion="HOLD"),lambda s:s[-1]["label"].update(motion="HOLD"),
            lambda s:s[0]["actor"].update(history=["RIGHT_CLOSE"])):
            wrong=copy.deepcopy(states);mutate(wrong)
            with self.assertRaises(ValueError):project_examples(original,wrong)
        with patch("native_completion_training_data.sha",return_value="0"*64):
            with self.assertRaises(ValueError):load_reviewed("a","b","c")
    def test_exact_config_no_extra_updates_or_freshbase_fallback(self):
        validate_config(CONFIG);validate_config(SERVICE_CONFIG,service=True)
        for key,value in (("max_updates_including_gate",122),("initial_adapter_sha256",None),("seed",True),
            ("per_update",[4,4,0]),("max_artifact_MiB",512),("learning_rate",.001)):
            with self.assertRaises(ValueError):validate_config({**CONFIG,key:value})
        for key,value in (("max_calls",56),("port",8919),("variants",["base","finetuned"]),("physical_gpu",True)):
            with self.assertRaises(ValueError):validate_config({**SERVICE_CONFIG,key:value},service=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):initial_checkpoint(directory)
    def test_new_storage_only_two_model_paths_old_spec_unchanged(self):
        before=copy.deepcopy(WALL2100_SPEC)
        for name in ("training_completion_v1","service_completion_v1"):
            obj=CompletionStorage({"storage":WALL2100_SPEC},EXPERIMENT_ROOT/name)
            self.assertEqual(obj.root.name,name);self.assertFalse(any(name in str(k) for k in obj.aliases))
            self.assertTrue(all(str(obj.root) in value for key,value in obj.expected.items() if key!="PYTHONDONTWRITEBYTECODE"))
        for name in ("training_v1","other","native_t1_i1"):
            with self.assertRaises(ValueError):CompletionStorage({"storage":WALL2100_SPEC},EXPERIMENT_ROOT/name)
        self.assertEqual(before,WALL2100_SPEC)

    def test_parent_exact_authorization_and_unique_output_are_required(self):
        from native_completion_runtime import execution_metadata,CARRY_PROFILE
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);release=root/"parent.json";output=root/"training_completion_v1"
            auth={"protocol":protocol.VERSION,"code_commit":"c"*40,"config_sha256":"d"*64,
                "dataset_sha256":DATA_SHA,"states_sha256":STATES_SHA,"data_review_sha256":REVIEW_SHA,
                "initial_adapter_sha256":INITIAL_ADAPTER_SHA,"initial_result_sha256":INITIAL_RESULT_SHA,
                "physical_gpu":3,"reviewer":"Codex-parent","output":str(output),
                "executor_digest":EXECUTOR_DIGEST,"carry_duration_core_commit":CORE_COMMIT,
                **execution_metadata(CARRY_PROFILE),"authorize_training":True,"authorize_service":False,
                "authorize_physical":False,"storage":WALL2100_SPEC}
            release.write_text(json.dumps(auth));auth.update(parent_release=str(release),parent_release_sha256=sha(release))
            def check(value): require_authorization(value,stage="training",code="c"*40,config_sha="d"*64,output=output)
            with patch("native_completion_runtime.ROOT",root),patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":"3"}):
                check(auth)
                for key,value in (("authorize_training",1),("authorize_physical",True),("authorize_service",True),
                    ("dataset_sha256","a"*64),("states_sha256","a"*64),("initial_adapter_sha256","a"*64),
                    ("code_commit","e"*40),("reviewer","self"),("parent_release_sha256","a"*64),
                    ("physical_gpu",True),("storage",{}),("max_workspace_macros",3)):
                    with self.assertRaises(ValueError):check({**auth,key:value})
                altered=copy.deepcopy(auth);altered["reviewer"]="self";release.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):check({**auth,"parent_release_sha256":sha(release)})
                release.write_text(json.dumps({**altered,"reviewer":"Codex-parent"}));auth["parent_release_sha256"]=sha(release)
                output.mkdir()
                with self.assertRaises(ValueError):check(auth)
            self.assertEqual(list(output.iterdir()),[])

    def test_original_adapter_file_and_config_are_pinned_not_suffix_guessed(self):
        from native_completion_runtime import initial_checkpoint
        old_local=ROOT/"artifacts/h09y-resume-20260921/training_complete/training_v1"
        if not old_local.exists():self.skipTest("Optional local original adapter archive absent")
        self.assertEqual(sha(old_local/"adapter_0120/adapter_config.json"),INITIAL_ADAPTER_CONFIG_SHA)
        cfg=json.loads((old_local/"adapter_0120/adapter_config.json").read_text())
        self.assertTrue(all("." not in name for name in cfg["target_modules"]))
        with patch("native_completion_runtime.ROOT",old_local.parent),patch("native_completion_runtime.base_identity",return_value={}):
            folder,identity=initial_checkpoint(old_local)
            self.assertEqual(folder,old_local/"adapter_0120");self.assertEqual(identity["trainable_parameters"],16819200)
        with patch("native_completion_runtime.sha",return_value="0"*64):
            with self.assertRaises(ValueError):initial_checkpoint(EXPERIMENT_ROOT/"training_v1")

    def test_verification_bridge_does_not_convert_public_holding_to_success(self):
        from native_completion_protocol import verification_bridge
        calls=[]
        def verify(*args,**kwargs):calls.append((args,kwargs));return {"public_holding_verified":True,"not_skill_success":True,"not_official":True}
        good={"protocol":protocol.VERSION,"skill_status":"REQUEST_VERIFY","motion":None,"success_claim":False}
        kwargs=dict(model="model",goal="goal",records=[],request_capture="capture",localize="callback",deadline=10.)
        result=verification_bridge(good,verify,**kwargs)
        self.assertTrue(result["not_skill_success"]);self.assertEqual(calls[0][1]["requested_status"],"REQUEST_VERIFY")
        for wrong in ({**good,"skill_status":"CONTINUE"},{**good,"motion":"HOLD"},{**good,"success_claim":True}):
            with self.assertRaises(ValueError):verification_bridge(wrong,verify,**kwargs)
        self.assertEqual(len(calls),1)


if __name__=="__main__":unittest.main()
