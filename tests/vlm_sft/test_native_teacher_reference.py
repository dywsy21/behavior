"""Reference/attribution contract tests; synthetic success is NOT collected data."""
import copy
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts/vlm_sft")]
from native_teacher_toggle import ToggleObserver,PressCausality
from native_teacher_reference_prepare import validate_segment
from native_teacher_reference_contract import compare_source_state
from native_teacher_seed import seed_identity,validate_seed_release,extract_seed
from native_teacher_contract import digest
from native_teacher_outcomes import LocalOutcome
from native_teacher_og import resolve_bound_native_objects
from common import sha
from test_native_teacher_automatic import frame,toggle_events


class ReferenceTests(unittest.TestCase):
    def identity_fixture(self):
        def obj(name):return types.SimpleNamespace(name=name,prim_path='/World/'+name,links={'root':object()},states={})
        target,destination=obj('trash_can_116'),obj('table_12')
        scope={'ashcan.n.01_1':target,'table.n.02_1':destination,'future.n.01_1':None}
        metadata={'ashcan.n.01_1':'trash_can_116','table.n.02_1':'table_12','future.n.01_1':'unborn_1'}
        registry=lambda field,name:next((o for o in (target,destination) if o.name==name),None)
        spec={'target':'trash_can_116','destination':'table_12','payloads':[]}
        return scope,metadata,registry,spec

    def test_exact_native_name_resolves_bound_bddl_identity_and_distinct_roles(self):
        scope,meta,reg,spec=self.identity_fixture()
        objects,receipt=resolve_bound_native_objects(scope,meta,reg,spec)
        self.assertIs(objects['trash_can_116'],scope['ashcan.n.01_1'])
        self.assertIs(objects['table_12'],scope['table.n.02_1'])
        self.assertEqual(receipt['trash_can_116']['bddl_instance'],'ashcan.n.01_1')
        swapped={**spec,'target':'table_12','destination':'trash_can_116'}
        swapped_objects,_=resolve_bound_native_objects(scope,meta,reg,swapped)
        self.assertIs(swapped_objects[swapped['target']],scope['table.n.02_1'])

    def test_native_identity_unbound_unknown_fuzzy_and_role_alias_fail_closed(self):
        for replacement in ('trash_can','trash_can_116 ','ashcan.n.01_1','unborn_1'):
            scope,meta,reg,spec=self.identity_fixture();spec['target']=replacement
            with self.assertRaises(ValueError):resolve_bound_native_objects(scope,meta,reg,spec)
        scope,meta,reg,spec=self.identity_fixture();scope['ashcan.n.01_1']=None
        with self.assertRaises(ValueError):resolve_bound_native_objects(scope,meta,reg,spec)
        scope,meta,reg,spec=self.identity_fixture();spec['destination']=spec['target']
        with self.assertRaises(ValueError):resolve_bound_native_objects(scope,meta,reg,spec)

    def test_duplicate_bddl_alias_wrong_metadata_or_registry_proxy_rejected(self):
        scope,meta,reg,spec=self.identity_fixture()
        scope['ashcan.n.01_2']=scope['ashcan.n.01_1'];meta['ashcan.n.01_2']='trash_can_116'
        with self.assertRaises(ValueError):resolve_bound_native_objects(scope,meta,reg,spec)
        scope,meta,reg,spec=self.identity_fixture();meta['ashcan.n.01_1']='another_1'
        with self.assertRaises(ValueError):resolve_bound_native_objects(scope,meta,reg,spec)
        scope,meta,reg,spec=self.identity_fixture()
        with self.assertRaises(ValueError):resolve_bound_native_objects(scope,meta,lambda field,name:copy.copy(reg(field,name)),spec)

    def test_identity_receipt_remains_private_not_actor_proprio(self):
        import test_native_teacher
        from native_teacher_contract import actor_input
        args=self.identity_fixture();_,receipt=resolve_bound_native_objects(*args)
        request,_,_,_=test_native_teacher.NativeTeacherTests().fixture();actor=request['actor']
        bad={**actor['proprio'],'identity_bindings':receipt}
        with self.assertRaises(ValueError):actor_input(actor['task'],actor['active_instruction'],bad,actor['current_rgb_sha256'],[])

    def test_instance_hook_exact_original_call_result_and_restore(self):
        class State:
            value=False;robot_can_toggle_steps=0
            def __init__(self):self.calls=[]
            def _update(self,*args,**kwargs):
                self.calls.append((args,kwargs));self.robot_can_toggle_steps+=1;return token
        token=object();s=State();other=State();original=State._update
        o=ToggleObserver(s,lambda:toggle_events()[0]["before"]["measurement"])
        with o.installed():
            self.assertIs(s._update(3,a=4),token)
            self.assertEqual(s.calls,[((3,),{"a":4})]);self.assertNotIn("_update",other.__dict__)
            self.assertEqual(o.drain()[0]["update_index"],1)
        self.assertNotIn("_update",s.__dict__);self.assertIs(State._update,original)

    def test_hook_original_exception_identity_and_existing_override_restored(self):
        primary=RuntimeError("physics original")
        s=types.SimpleNamespace(value=False,robot_can_toggle_steps=0)
        def update():raise primary
        s._update=update;o=ToggleObserver(s,lambda:{})
        with self.assertRaises(RuntimeError) as cm:
            with o.installed():s._update()
        self.assertIs(cm.exception,primary);self.assertIs(s._update,update);self.assertFalse(o.active)

    def test_measurement_failure_does_not_skip_original_update(self):
        calls=[];s=types.SimpleNamespace(value=False,robot_can_toggle_steps=0,_update=lambda:calls.append(1))
        def bad():raise ValueError("missing marker")
        o=ToggleObserver(s,bad)
        with o.installed():
            s._update();self.assertEqual(calls,[1])
            with self.assertRaises(RuntimeError):o.drain()

    def test_marker_attribution_rejects_other_hand_missing_and_repeated_updates(self):
        for events in (toggle_events("left"),toggle_events(other=True),toggle_events()[1:]):
            with self.assertRaises(ValueError):PressCausality("right").consume(events)
        p=PressCausality("right");self.assertTrue(p.consume(toggle_events()))
        with self.assertRaises(ValueError):p.consume(toggle_events())

    def test_marker_counter_cannot_skip_or_be_assumed_per_control(self):
        events=toggle_events();events[0]["before"]["counter"]=1
        with self.assertRaises(ValueError):PressCausality("right").consume(events)
        o=LocalOutcome({"verb":"PRESS","hand":"right","target":"target"});o.update(frame())
        f=frame(1);f["toggled"]=True;f["finger_contact"]["right"]=True;f["toggle_events"]=toggle_events("left")
        self.assertEqual(o.update(f)["outcome"],"UNKNOWN")

    def fixture(self,extra=None):
        states=np.zeros((4,61));actions=np.zeros((4,23));semantic='[{"verb":"PRESS"}]'
        labels=[{"frame_index":i,"source_kind":"original_demo","memlite_branch":"low",
                 "active_skills_semantic_json":semantic,"low_action_supervision_mask":True,
                 "segment_start":1,"segment_end":3,"action_horizon_end":3} for i in range(4)]
        if extra:labels.append({**labels[1],**extra})
        h=hashlib.sha256(states.tobytes()+actions.tobytes());h.update(json.dumps(labels,sort_keys=True,separators=(",",":")).encode())
        source={"frames":4,"episode":66,"extracted_arrays_and_labels_sha256":h.hexdigest()}
        return states,actions,np.arange(4),labels,source,1,3,semantic,[]

    def test_complete_segment_branch_selection_retains_duplicate_audit(self):
        args=self.fixture({"memlite_branch":"high"});prefix,seg,states,audit=validate_segment(*args)
        self.assertEqual((prefix.shape,seg.shape,states.shape),((1,23),(2,23),(3,61)))
        self.assertEqual(audit["duplicate_frame_rows"],1);self.assertEqual(audit["selected_segment_row_indexes"],[1,2])

    def test_selected_branch_conflict_sha_and_quarantine_reject(self):
        with self.assertRaises(ValueError):validate_segment(*self.fixture({"memlite_branch":"low"}))
        args=list(self.fixture());args[1][1,0]=1
        with self.assertRaises(ValueError):validate_segment(*args)
        args=list(self.fixture());args[-1]=[{"episode_index":66,"frame_start":2,"frame_end":2}]
        with self.assertRaises(ValueError):validate_segment(*args)

    def test_real_state_clock_mismatch_stops_not_calibrates(self):
        s=types.SimpleNamespace(q=np.zeros(18),gripper=np.zeros(2));compare_source_state(s,np.zeros(61))
        s.q[0]=.021
        with self.assertRaises(RuntimeError):compare_source_state(s,np.zeros(61))

    def test_random_hash_or_relabelled_seed_is_not_a_release(self):
        spec={"verb":"PRESS","hand":"right","target":"target","destination":"","support_hand":None,
              "payloads":[],"goal_frame":"toggle_link","goal_pose_local":np.eye(4).tolist(),"pose_reviewer":"parent"}
        ref={"source":{"task":0,"episode":66,"instance":70,"extracted_arrays_and_labels_sha256":"a"*64},
             "source_label":{"segment_start":0,"segment_end":2}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"seed.json";review=Path(d)/"review.json";review.write_text('{}')
            for seed in ({},{"schema":"h09u-measured-reference-seed-v1","identity":{**seed_identity(ref,spec),"hand":"left"},
                             "goal_pose_local":spec["goal_pose_local"],"is_native_bc_label":False,"training_eligible":False,
                             "reference_preparation_sha256":"b"*64}):
                p.write_text(json.dumps(seed));spec["pose_evidence_sha256"]=sha(p)
                with self.assertRaises(ValueError):validate_seed_release(p,review,spec,ref,"b"*64)

    def test_full_seed_positive_review_and_pose_evidence_corruption(self):
        spec={"verb":"PRESS","hand":"right","target":"target","destination":"","support_hand":None,
              "payloads":[],"goal_frame":"toggle_link","goal_pose_local":np.eye(4).tolist(),"pose_reviewer":"parent"}
        ref={"source":{"task":0,"episode":66,"instance":70,"extracted_arrays_and_labels_sha256":"a"*64},
             "source_label":{"segment_start":0,"segment_end":2}}
        rows=[{"frame":frame(),"actual_action23":None}]
        for t in range(1,16):
            f=frame(t);f["toggled"]=True
            if t==1:f["toggle_events"]=toggle_events()
            rows.append({"frame":f,"actual_action23":[0.]*23})
        identity=seed_identity(ref,spec)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"PRIVATE_reference_trace.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result={"status":"REFERENCE_LOCAL_SUCCEEDED","identity":identity,"final_hold_completed":True,
                    "failure":None,"completed_controls":15,"model_calls":0,"reference_preparation_sha256":"b"*64}
            (root/"result.json").write_text(json.dumps(result));captures=[]
            for folder,tick in (("before",0),("segment_end",2),("after_stability",14),("after_final_hold",15)):
                (root/folder).mkdir();hashes={}
                for name in ("head.png","left_wrist.png","right_wrist.png","depth.npz","robot_self_geometry.json","sensors.json","proprio.json","capture.json"):
                    p=root/folder/name;p.write_bytes(b'synthetic fixture');hashes[folder+'/'+name]=sha(p)
                captures.append({"control":tick,"files_sha256":hashes})
            (root/"captures.json").write_text(json.dumps(captures))
            seed={"schema":"h09u-measured-reference-seed-v1","identity":identity,**extract_seed(spec,rows),
                  "training_eligible":False,"reference_preparation_sha256":"b"*64,
                  "evidence_sha256":{n:sha(root/n) for n in ("PRIVATE_reference_trace.jsonl","result.json","captures.json")}}
            p=root/"seed.json";p.write_text(json.dumps(seed));spec["pose_evidence_sha256"]=sha(p)
            review={"seed_sha256":sha(p),"identity_sha256":digest(identity),"goal_pose_sha256":digest(seed["goal_pose_local"]),
                    "reviewer":"parent","decision":"approve","reviewed_source_current_and_terminal_views":True,
                    "reviewed_contact_update_and_control_ledger":True,"local_outcome_and_pose_accepted":True,
                    "reason":"Synthetic test fixture only; not a real review or acquired sample."}
            rp=root/"review.json";rp.write_text(json.dumps(review))
            self.assertEqual(validate_seed_release(p,rp,spec,ref,"b"*64)["identity"],identity)
            with self.assertRaises(ValueError):validate_seed_release(p,rp,spec,ref,"c"*64)
            spec["goal_pose_local"][0][3]=.01
            with self.assertRaises(ValueError):validate_seed_release(p,rp,spec,ref,"b"*64)
            spec["goal_pose_local"][0][3]=0
            (root/"after_final_hold/head.png").write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_seed_release(p,rp,spec,ref,"b"*64)

    def test_actual_runner_cleanup_attempts_hold_despite_all_writes_failing(self):
        tree=ast.parse((ROOT/"scripts/vlm_sft/native_teacher_reference_replay.py").read_text())
        final=next(n.finalbody for n in ast.walk(tree) if isinstance(n,ast.Try) and n.finalbody and
                   isinstance(n.finalbody[0],ast.If) and 'max_controls' in ast.unparse(n.finalbody[0].test))
        program=compile(ast.fix_missing_locations(ast.Module(body=final,type_ignores=[])),"runner_cleanup","exec")
        for terminal,issued,expected in ((False,2,1),(True,2,0),(False,6,0)):
            steps=[];primary=RuntimeError('original sample failure');errors=[]
            def broken(*a,**k):raise OSError('disk failed')
            ns={"kin":types.SimpleNamespace(state=lambda:types.SimpleNamespace(q=np.zeros(18))),
                "grips":np.ones(2),"terminal":terminal,"issued":issued,"completed":issued,"budget":{"max_controls":6},
                "np":np,"native_action":lambda q,g:np.zeros(23),"writer":types.SimpleNamespace(write_json=broken),
                "a":types.SimpleNamespace(output=Path('/unused')),"error":lambda e:errors.append(e),
                "env":types.SimpleNamespace(step=lambda *a,**k:(steps.append(1) or {},0,False,False,{})),
                "og":types.SimpleNamespace(sim=types.SimpleNamespace(render_on_step=lambda _:nullcontext())),
                "reader":None,"trace":None,"private":None,"primary":primary,"final_hold":False}
            exec(program,ns);self.assertEqual(len(steps),expected);self.assertIs(ns['primary'],primary)
            self.assertEqual(ns['completed'],issued+expected)


if __name__=="__main__":unittest.main()
