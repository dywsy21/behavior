"""Later-prefix/known-empty teacher counterexamples; no real collection here."""
import ast
import copy
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch,MagicMock
import numpy as np
from scipy.spatial.transform import Rotation
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/vlm_sft')]
from common import sha,write_json,token_to_action
from native_teacher_near_grasp import SCHEMA,prepare,verify_prepared,derive
from native_teacher_policy import empty_hand_rotation_allowed,PoseTeacher
from native_teacher_collect import native_preflight,require_release
from native_teacher_reference_contract import check_actual_joint_bounds
from native_teacher_contract import actor_input
from test_native_teacher_automatic import frame
import test_native_teacher


class NearTeacherTests(unittest.TestCase):
    def empty(self):
        f=frame();f.update(held_any={'left':False,'right':False},finger_external_contact={'left':False,'right':False})
        s=NS(gripper=np.full(2,.05),q=np.zeros(18))
        m=NS(spec={'metadata':{'grasp_region_reference_gripper_m':[.05,.05],
              'grasp_region_reference_fully_open':{'left':True,'right':True}}})
        return f,s,m,np.ones(2)

    def test_empty_requires_object_agnostic_false_and_real_open_not_target_false_only(self):
        f,s,m,g=self.empty();self.assertTrue(empty_hand_rotation_allowed('right',s,f,g,m,False))
        for field in ('held_any','held','finger_external_contact','finger_contact'):
            for value in (None,True,0):
                bad=copy.deepcopy(f);bad[field]['right']=value
                self.assertFalse(empty_hand_rotation_allowed('right',s,bad,g,m,False))
        for command in (-1,.998,1.001,float('nan')):
            self.assertFalse(empty_hand_rotation_allowed('right',s,f,[1,command],m,False))
        self.assertFalse(empty_hand_rotation_allowed('right',s,f,g,m,True))
        s.gripper[1]=.049
        self.assertFalse(empty_hand_rotation_allowed('right',s,f,g,m,False))

    def test_default_teacher_does_not_gain_rotations_but_optin_uses_same_base_3deg(self):
        f,s,m,g=self.empty();goal=np.eye(4);goal[:3,:3]=Rotation.from_euler('z',9,degrees=True).as_matrix()
        spec={'verb':'GRASP','hand':'right'}
        self.assertEqual(PoseTeacher(spec).ranked(s,f,goal,np.eye(4),g,m),[])
        t=PoseTeacher(spec,True);ranked=t.ranked(s,f,goal,np.eye(4),g,m)
        self.assertEqual(ranked[0],'RIGHT_YAW_PLUS');a=token_to_action(ranked[0])
        self.assertEqual(a.frame,'base');self.assertAlmostEqual(a.amount(False),np.deg2rad(3))
        t.close_issued=True;f['held']['right']=True;f['finger_contact']['right']=True
        self.assertFalse(any('YAW' in x for x in t.ranked(s,f,goal,np.eye(4),g,m)))

    def test_unreachable_centimetre_grid_stops_without_close_or_cycle(self):
        f,s,m,g=self.empty();goal=np.eye(4);goal[:3,3]=[.0048,.0048,.0048]
        teacher=PoseTeacher({'verb':'GRASP','hand':'right'},True)
        self.assertGreater(np.linalg.norm(goal[:3,3]),.004)
        self.assertEqual(teacher.ranked(s,f,goal,np.eye(4),g,m),[])
        for _ in range(4):teacher.executed('HOLD',f)
        with self.assertRaises(RuntimeError):teacher.executed('HOLD',f)

    def test_grid_cell_is_optin_one_proposal_not_success_or_retry(self):
        f,s,m,g=self.empty();goal=np.eye(4);goal[:3,3]=[.0048]*3
        t=PoseTeacher({'verb':'GRASP','hand':'right'},True,True)
        self.assertEqual(t.ranked(s,f,goal,np.eye(4),g,m),['RIGHT_CLOSE'])
        r=t.proposal_receipt
        self.assertEqual(r['proposal_basis'],'GRASP_GRID_CELL_SINGLE_CLOSE_ATTEMPT')
        self.assertEqual(len(r['unfiltered_neighbor_costs']),12)
        self.assertEqual(r['strictly_improving_geometry'],[])
        self.assertAlmostEqual(r['grid_cell_half_diagonal_m'],np.sqrt(3)*.005)
        self.assertGreater(r['position_error_m'],.004)
        self.assertTrue(r['not_success_evidence']);self.assertTrue(r['not_actor_input'])
        with self.assertRaisesRegex(RuntimeError,'already proposed'):t.ranked(s,f,goal,np.eye(4),g,m)
        t.executed('RIGHT_CLOSE',f)
        with self.assertRaisesRegex(RuntimeError,'no retry'):t.ranked(s,f,goal,np.eye(4),g,m)

    def test_grid_cell_requires_known_empty_real_open_and_never_applies_to_other_verbs(self):
        f,s,m,g=self.empty();goal=np.eye(4);goal[:3,3]=[.0048]*3
        for field in ('held_any','held','finger_contact','finger_external_contact'):
            for value in (True,None,0):
                bad=copy.deepcopy(f);bad[field]['right']=value
                t=PoseTeacher({'verb':'GRASP','hand':'right'},True,True)
                with self.subTest(field=field,value=value):
                    if field=='held' and value is None:
                        with self.assertRaises(RuntimeError):t.ranked(s,bad,goal,np.eye(4),g,m)
                    else:self.assertNotIn('RIGHT_CLOSE',t.ranked(s,bad,goal,np.eye(4),g,m))
        for commands,opening in (([1,-1],.05),([1,.998],.05),([1,1.001],.05),([1,1],.049)):
            s.gripper[1]=opening;t=PoseTeacher({'verb':'GRASP','hand':'right'},True,True)
            self.assertNotIn('RIGHT_CLOSE',t.ranked(s,f,goal,np.eye(4),commands,m))
        s.gripper[1]=.05
        t=PoseTeacher({'verb':'PRESS','hand':'right'},True,True)
        self.assertEqual(t.ranked(s,f,goal,np.eye(4),g,m),[])
        for enable in (False,None,1):
            self.assertEqual(PoseTeacher({'verb':'GRASP','hand':'right'},True,enable).ranked(s,f,goal,np.eye(4),g,m),[])

    def test_grid_cell_checks_all_geometry_including_ineligible_rotations_and_tiny_gains(self):
        f,s,m,g=self.empty();goal=np.eye(4);goal[:3,3]=[.0048]*3
        goal[:3,:3]=Rotation.from_euler('z',2,degrees=True).as_matrix()
        t=PoseTeacher({'verb':'GRASP','hand':'right'},False,True)
        self.assertEqual(t.ranked(s,f,goal,np.eye(4),g,m),[])
        self.assertIn('RIGHT_YAW_PLUS',t.proposal_receipt['strictly_improving_geometry'])
        # Less than the old 1e-6 ranking margin is STILL a real improvement;
        # returning an empty filtered ranking must not admit a CLOSE.
        goal=np.eye(4);goal[:3,3]=[.005000001,0,0]
        t=PoseTeacher({'verb':'GRASP','hand':'right'},True,True)
        self.assertEqual(t.ranked(s,f,goal,np.eye(4),g,m),[])
        self.assertIn('RIGHT_FORWARD',t.proposal_receipt['strictly_improving_geometry'])
        for displacement,angle in (([.009,0,0],0),([.0048]*3,4.001)):
            goal=np.eye(4);goal[:3,3]=displacement;goal[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()
            self.assertNotIn('RIGHT_CLOSE',PoseTeacher({'verb':'GRASP','hand':'right'},True,True).ranked(s,f,goal,np.eye(4),g,m))

    def test_original_exact_pose_gate_is_unchanged(self):
        f,s,m,g=self.empty();goal=np.eye(4);goal[0,3]=.004
        goal[:3,:3]=Rotation.from_euler('z',3.9,degrees=True).as_matrix()
        for cell in (False,True):
            t=PoseTeacher({'verb':'GRASP','hand':'right'},True,cell)
            self.assertEqual(t.ranked(s,f,goal,np.eye(4),g,m),['RIGHT_CLOSE'])
            self.assertEqual(t.proposal_receipt['proposal_basis'],'EXACT_POSE_GATE')
            self.assertFalse(t.cell_close_proposed)

    def test_actual_collector_safety_veto_does_not_fallback_to_cell_close(self):
        f,s,m,g=self.empty();goal=np.eye(4);goal[0,3]=.006
        teacher=PoseTeacher({'verb':'GRASP','hand':'right'},True,True)
        tree=ast.parse((ROOT/'scripts/vlm_sft/native_teacher_collect.py').read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and
                  isinstance(n.target,ast.Name) and n.target.id=='index' and
                  isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and
                  'max_teacher_primitives' in ast.unparse(n.iter))
        start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and
                   any(isinstance(v,ast.Name) and v.id=='ranked' for v in n.targets))
        end=next(i for i,n in enumerate(loop.body[start:],start) if isinstance(n,ast.If) and ast.unparse(n.test)=='choice is None')
        reject=MagicMock(side_effect=RuntimeError('fixture fresh depth veto'));store=MagicMock()
        ns={'teacher':teacher,'teacher_frame':f,'teacher_reader':NS(goal=lambda:goal,base=lambda:np.eye(4)),
            'before':s,'grips':g,'model':m,'artifacts':store,'folder':Path('/not-written'),
            'depths':{},'geometry':None,'near':True,'execution_profile':None,'public_history':None,
            'native_preflight':reject,'token_to_action':token_to_action}
        fragment=ast.Module(body=copy.deepcopy(loop.body[start:end+1]),type_ignores=[])
        with self.assertRaisesRegex(RuntimeError,'No safe'):exec(compile(ast.fix_missing_locations(fragment),'actual_selection','exec'),ns)
        self.assertEqual(reject.call_count,1);self.assertEqual(ns['ranked'],['RIGHT_FORWARD'])
        self.assertFalse(teacher.cell_close_proposed)
        self.assertEqual(store.write_json.call_args.args[0].name,'PRIVATE_proposal.json')
        self.assertEqual(teacher.proposal_receipt['strictly_improving_geometry'],['RIGHT_FORWARD'])

    def test_cell_receipt_cannot_enter_actor_and_no_lift_never_becomes_bc(self):
        from native_teacher_outcomes import LocalOutcome
        f,s,m,g=self.empty();goal=np.eye(4);goal[:3,3]=[.0048]*3
        t=PoseTeacher({'verb':'GRASP','hand':'right'},True,True);t.ranked(s,f,goal,np.eye(4),g,m)
        req,*_=test_native_teacher.NativeTeacherTests().fixture();actor=req['actor']
        bad={**actor['proprio'],'proposal_basis':t.proposal_receipt}
        with self.assertRaises(ValueError):actor_input(actor['task'],actor['active_instruction'],bad,actor['current_rgb_sha256'],[])
        o=LocalOutcome({'verb':'GRASP','hand':'right','target':'target'});o.update(frame())
        for tick in range(1,15):
            evidence=frame(tick);evidence['held']['right']=True;evidence['finger_contact']['right']=True
            verdict=o.update(evidence,'RIGHT_CLOSE' if tick==1 else None)
        self.assertNotEqual(verdict['outcome'],'SUCCEEDED')

    def test_preflight_itself_rechecks_empty_and_never_changes_translation_carry(self):
        f,s,m,g=self.empty();e={'frame':f,'close_issued':False}
        with patch('native_teacher_collect.observed_cloud',return_value=np.empty((0,3))), \
             patch('native_teacher_collect.LocalDepthGuard') as guard,patch('native_teacher_collect.SafeServo') as factory:
            guard.return_value.check.return_value=(True,'CPU_FIXTURE');guard.return_value.receipt.return_value={}
            factory.return_value.begin.return_value=True;factory.return_value.total_ticks=18;factory.return_value.status='RUNNING'
            _,receipt=native_preflight(m,s,g,token_to_action('RIGHT_YAW_PLUS'),{},None,e)
            self.assertFalse(factory.return_value.begin.call_args.kwargs['carry']);self.assertAlmostEqual(receipt['amount'],np.deg2rad(3))
            _,receipt=native_preflight(m,s,g,token_to_action('RIGHT_UP'),{},None,e)
            self.assertTrue(factory.return_value.begin.call_args.kwargs['carry']);self.assertEqual(receipt['amount'],.01)
            f['held_any']['right']=None
            with self.assertRaises(RuntimeError):native_preflight(m,s,g,token_to_action('RIGHT_YAW_PLUS'),{},None,e)
        req,*_=test_native_teacher.NativeTeacherTests().fixture();bad={**req['actor']['proprio'],'held_any':{'right':False}}
        with self.assertRaises(ValueError):actor_input('task','GRASP',bad,req['actor']['current_rgb_sha256'],[])

    def origin(self,root):
        original=root/'original';folder=original/'task_1';folder.mkdir(parents=True)
        source={'task':1,'episode':2,'instance':42,'cohort':'additional_train','extracted_arrays_and_labels_sha256':'a'*64}
        semantic=json.dumps([{'verb':'GRASP','arm':'RIGHT','target':'object_1','destination':'','unbound_relation':''}])
        labels=[{'frame_index':i,'active_skills_semantic_json':semantic,'source_kind':'original_demo','memlite_branch':'low',
                 'low_action_supervision_mask':True,'segment_start':2,'segment_end':40,'action_horizon_end':40} for i in range(2,40)]
        ref={'source':source,'source_label':labels[0],'private_original_semantic_json':semantic,
             'pilot':{'task':1,'episode':2,'instance':42,'frame':2,'verb':'GRASP'},'active_instruction':'verb=GRASP; target=object',
             'schema':'old','training_eligible':False}
        a=np.zeros((40,23),np.float32);a[:,[14,22]]=1;a[:,3]=np.arange(40)/100
        for name,arr in [('prefix',a[:2]),('segment',a[2:]),('source_states',np.zeros((39,61),np.float32))]:np.save(folder/(name+'.npy'),arr)
        write_json(folder/'teacher_reference.json',ref);write_json(folder/'segment_labels.json',labels)
        write_json(folder/'private_spec.json',{});write_json(folder/'label_selection_audit.json',{})
        write_json(folder/'window.json',{'task_name':'synthetic','official_mode':'train','seed':0,'instance_id':42,
            'prefix_actions_path':str(folder/'prefix.npy'),'prefix_actions_sha256':sha(folder/'prefix.npy'),'max_steps':54})
        row={'task':1,'episode':2,'instance':42,'start':2,'end':40,'files_sha256':{x.name:sha(x) for x in folder.iterdir()}}
        write_json(original/'preparation.json',{'schema':'h09u-full-expert-reference-v1','status':'PREPARED_NO_RESET_NO_OUTCOME_NO_SEED','sources':[row]})
        counts=root/'counts.json';write_json(counts,{'exclusions':{},'sources':[source]})
        return folder,counts,sha(original/'preparation.json'),a

    def test_paid_prefix_is_exact_full_source_not_shifted_skill_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder,counts,h,a=self.origin(root);out=root/'near'
            manifest=prepare(folder,out,20,h,counts,[])
            release={'schema':SCHEMA,'preparation_manifest_sha256':sha(out/'preparation.json'),'source':[1,2,42],
                'paid_prefix_controls':20,'train_counts_sha256':sha(counts),'train_counts_path':str(counts),
                'held_out_instance_groups':[],'seed_reference_preparation_sha256':h}
            ref,prefix,binding=verify_prepared(out/'task_1',release)
            np.testing.assert_array_equal(prefix,a[:20]);self.assertEqual(ref['source_label']['segment_start'],2)
            self.assertEqual(ref['source_label']['frame_index'],20);self.assertFalse(ref['is_original_skill_phase_start'])
            for bad in ({**release,'paid_prefix_controls':21},{**release,'source':[1,2,43]},
                        {**release,'held_out_instance_groups':[[1,42]]}):
                with self.assertRaises(ValueError):verify_prepared(out/'task_1',bad)
            prefix[0,0]=.1;np.save(out/'task_1/prefix.npy',prefix)
            manifest['files_sha256']['prefix.npy']=sha(out/'task_1/prefix.npy');write_json(out/'preparation.json',manifest)
            release['preparation_manifest_sha256']=sha(out/'preparation.json')
            with self.assertRaisesRegex(ValueError,'not exactly'):verify_prepared(out/'task_1',release)

    def test_source_exclusion_and_close_latch_stop_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder,counts,h,_=self.origin(root);c=json.loads(counts.read_text())
            with self.assertRaises(ValueError):derive(folder,h,c,[[1,42]],20)
            with self.assertRaises(ValueError):derive(folder,h,c,[],2)
            m=json.loads((folder.parent/'preparation.json').read_text());a=np.load(folder/'segment.npy');a[17,22]=-1
            np.save(folder/'segment.npy',a);m['sources'][0]['files_sha256']['segment.npy']=sha(folder/'segment.npy');write_json(folder.parent/'preparation.json',m)
            with self.assertRaisesRegex(ValueError,'OPEN'):derive(folder,sha(folder.parent/'preparation.json'),c,[],20)

    def test_near_authority_keeps_exact_new_budget_and_real_geometry_gates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);gates=[]
            for task in (0,3):
                path=root/f'gate{task}.json';write_json(path,{'task':task,'gate_ok':True,'robot_geometry_guards':True,'implementation_digest':'executor'});gates.append(str(path))
            release={'schema':SCHEMA,'authorize_collection':True,'collector_commit':'code','executor_digest':'executor',
                     'h14_body_and_finger_safety_reviewed':True,'reviewer':'independent','max_resets':1,'max_candidates_per_instance':12,
                     'authorize_offline_teacher':True,'allow_known_empty_rotation':True,'allow_grasp_cell_attempt':True,'native_controls_max':420,
                     'seconds_after_reset':900,'run_MiB':100,'total_MiB':384,'max_teacher_primitives':12,
                     'model_calls':0,'experiment_root':d,'engineering_gate_paths':gates}
            require_release(release,'code','executor')
            for key,bad in [('max_resets',2),('native_controls_max',421),('seconds_after_reset',901),
                            ('run_MiB',101),('allow_known_empty_rotation',False),('allow_grasp_cell_attempt',False),
                            ('allow_grasp_cell_attempt',None),('collector_commit','old')]:
                with self.subTest(key=key),self.assertRaises(ValueError):require_release({**release,key:bad},'code','executor')
            gate=json.loads(Path(gates[0]).read_text());gate['robot_geometry_guards']=False;write_json(gates[0],gate)
            with self.assertRaises(ValueError):require_release(release,'code','executor')

    def test_partial_step_close_keeps_latch_and_issued_ledger_not_fake_completion(self):
        from native_execution import PublicGripperHistory
        tree=ast.parse((ROOT/'scripts/vlm_sft/native_teacher_collect.py').read_text())
        fn=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step'))
        fn.body[0]=ast.Global(names=fn.body[0].names)
        primary=RuntimeError('partial step fault');stream=io.StringIO();command=np.zeros(23);command[[14,22]]=[1,-1]
        def fail(*a,**k):raise primary
        state=NS(q=np.zeros(18),gripper=np.ones(2)*.05)
        ns={'np':np,'json':json,'near':True,'native_limit':420,'issued_native':12,'issued_prefix':396,'controls':12,'prefix_count':396,
            'terminal':False,'grips':np.ones(2),'public_history':PublicGripperHistory([1,1]),
            'teacher_frame':None,'teacher_reader':None,'started':0,'wall_limit':900,'storage':None,
            'time':NS(monotonic=lambda:1),'shutil':NS(disk_usage=lambda p:NS(free=90*1024**3)),
            'x':NS(output=Path('/unused')),'state':lambda:state,'model':NS(lower=-np.ones(18),upper=np.ones(18)),
            'check_actual_joint_bounds':check_actual_joint_bounds,'issue_trace':stream,
            'current_writer':NS(check=lambda *a:None,append_text=lambda s,v:s.write(v)),
            'og':NS(sim=NS(render_on_step=lambda v:nullcontext())),'env':NS(step=fail)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_step','exec'),ns)
        try:ns['step'](command,'candidate')
        except RuntimeError as exc:self.assertIs(exc,primary)
        else:self.fail('partial failure swallowed')
        self.assertEqual(ns['issued_native'],13);self.assertEqual(ns['controls'],12)
        self.assertEqual(ns['grips'][1],-1);self.assertEqual(json.loads(stream.getvalue())['action23'][22],-1)
        self.assertIs(ns['public_history'].close_seen['right'],True)
        command[22]=1;ns['public_history'].issued(command)
        self.assertIs(ns['public_history'].close_seen['right'],True)


if __name__=='__main__':unittest.main()
