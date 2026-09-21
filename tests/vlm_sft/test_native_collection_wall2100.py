"""One explicit953 collection wall, not extra actions or longer six-slot evaluation."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/vlm_sft')]
import native_teacher_capacity as capacity
import native_storage as storage
from native_teacher_collect import require_release
from native_execution import CARRY_PROFILE,metadata,authorization_profile,require_pipeline_profile,require_same_pipeline,actor_protocol
from native_dataset import collection_time_metadata,checked_collection_time_provenance
from native_eval_run import evaluation_budget,require_evaluation
from native_eval_prepare import SCHEMA,STARTS,ROOT as EXPERIMENT
from native_evaluation import VARIANTS
from native_eval_summary import summarize
from test_native_train953_registration import new_auth
from test_native_carry_duration import receipts
from test_native_summary_storage import suite

NAME='native_t1_i114_p0953_ws45_cd1_b2_wall2100'


def release():
    return {**new_auth(),'capacity_profile':capacity.WALL2100_PROFILE,
        'storage':copy.deepcopy(storage.WALL2100_SPEC),'seconds_after_reset':2100,
        'authorize_collection':True,'collector_commit':'d'*40,'h14_body_and_finger_safety_reviewed':True,
        'reviewer':'Codex-parent','max_resets':1,'max_candidates_per_instance':14,
        'max_teacher_primitives':14,'native_controls_max':640,'model_calls':0,
        'allow_known_empty_rotation':True,'allow_grasp_cell_attempt':True}


class CollectionWallTests(unittest.TestCase):
    def test_exact_new_source_name_time_and_original_scopes(self):
        a=release();self.assertEqual(authorization_profile(a,collection=True),CARRY_PROFILE)
        capacity.validate_collection_location(a,Path(capacity.H09Y_ROOT)/NAME,3)
        self.assertEqual(capacity.collection_wall_seconds(a),2100)
        self.assertEqual(capacity.collection_wall_seconds(new_auth()),1200)
        self.assertEqual(capacity.collection_wall_seconds({}),900)
        for change in ({'seconds_after_reset':1200},{'seconds_after_reset':2100.0},
                       {'seconds_after_reset':True},{'seconds_after_reset':2101},
                       {'paid_prefix_controls':961},{'paid_prefix_controls':969},
                       {'source':[1,310,192]},{'source':[1,200,1]},{'initialization_seconds':901}):
            with self.subTest(change=change),self.assertRaises(ValueError):capacity.capacity_limits({**a,**change})
        for wrong in (NAME+'_retry',NAME.replace('_wall2100',''),NAME.replace('2100','1800')):
            with self.assertRaises(ValueError):capacity.validate_collection_location(a,Path(capacity.H09Y_ROOT)/wrong,3)
        with self.assertRaises(ValueError):capacity.validate_collection_location(new_auth(),Path(capacity.H09Y_ROOT)/NAME,3)

    def test_exact_collector_gate_and_old1200_stay_separate(self):
        a=release()
        with patch('native_carry_admission.require_engineering_gates') as gates:
            require_release(a,'d'*40,a['executor_digest']);gates.assert_called_once()
            for k,v in (('seconds_after_reset',1200),('native_controls_max',641),('max_resets',2),
                        ('max_teacher_primitives',15),('max_candidates_per_instance',15),('model_calls',1)):
                with self.subTest(k=k),self.assertRaises(ValueError):require_release({**a,k:v},'d'*40,a['executor_digest'])
            old={**a,'capacity_profile':capacity.CARRY_PROFILE,'storage':storage.CARRY953_SPEC,'seconds_after_reset':1200}
            require_release(old,'d'*40,a['executor_digest'])
            with self.assertRaises(ValueError):require_release({**old,'seconds_after_reset':2100},'d'*40,a['executor_digest'])

    def test_only_new_alias_old_specs_unchanged_and_mixing_denied(self):
        self.assertEqual(storage.WALL2100_RUNS,storage.CARRY953_RUNS+(NAME,))
        self.assertNotIn(NAME,storage.CARRY953_RUNS)
        old=copy.deepcopy(storage.WALL2100_SPEC);old['profile']=storage.CARRY953_STORAGE_PROFILE
        old['shared_og_cache']['allowed_aliases'].pop();self.assertEqual(old,storage.CARRY953_SPEC)
        for bad in (storage.CARRY953_SPEC,storage.CARRY_SPEC):
            with self.assertRaises(ValueError):authorization_profile({**release(),'storage':bad},collection=True)
        with self.assertRaises(ValueError):authorization_profile({**new_auth(),'storage':storage.WALL2100_SPEC},collection=True)
        storage.RuntimeStorage(release(),Path(capacity.H09Y_ROOT)/NAME)
        with self.assertRaises(ValueError):storage.RuntimeStorage(new_auth(),Path(capacity.H09Y_ROOT)/NAME)

    def test_complete_wall_receipt_and_parent_binding_strict(self):
        a=release();path=Path(capacity.H09Y_ROOT)/NAME
        result={'collection_time_budget':{'capacity_profile':capacity.WALL2100_PROFILE,
            'seconds_after_reset':2100,'wall_seconds_after_reset':1500.}}
        review={'capacity_profile':capacity.WALL2100_PROFILE,'seconds_after_reset':2100}
        expected={'collection_capacity_profile':capacity.WALL2100_PROFILE,'collection_seconds_after_reset':2100}
        self.assertEqual(collection_time_metadata(path,a,result,review),expected)
        self.assertEqual(collection_time_metadata(Path('/local/archive')/NAME,a,result,review),expected)
        for bad in (None,[],{}, {**result['collection_time_budget'],'wall_seconds_after_reset':2100.001},
                    {**result['collection_time_budget'],'wall_seconds_after_reset':float('nan')},
                    {**result['collection_time_budget'],'wall_seconds_after_reset':True},
                    {**result['collection_time_budget'],'seconds_after_reset':1200},
                    {**result['collection_time_budget'],'extra':1}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):collection_time_metadata(path,a,{'collection_time_budget':bad},review)
        for changed in ({},{'capacity_profile':capacity.CARRY_PROFILE,'seconds_after_reset':2100},
                        {'capacity_profile':capacity.WALL2100_PROFILE,'seconds_after_reset':1200}):
            with self.assertRaises(ValueError):collection_time_metadata(path,a,result,changed)
        with self.assertRaises(ValueError):collection_time_metadata(path,new_auth(),result,review)
        self.assertEqual(collection_time_metadata(path,new_auth(),{},{}),{})

    def test_dataset_wall_provenance_cannot_be_partial_or_from_another_group(self):
        row={**metadata(CARRY_PROFILE),'group':[1,114],'prefix':953,
            'collection_capacity_profile':capacity.WALL2100_PROFILE,'collection_seconds_after_reset':2100}
        self.assertEqual(checked_collection_time_provenance(row)['collection_seconds_after_reset'],2100)
        for change in ({'collection_seconds_after_reset':1200},{'collection_seconds_after_reset':2100.0},
                       {'collection_capacity_profile':capacity.CARRY_PROFILE},{'prefix':961},{'group':[1,192]}):
            with self.assertRaises(ValueError):checked_collection_time_provenance({**row,**change})
        value=dict(row);value.pop('collection_seconds_after_reset')
        with self.assertRaises(ValueError):checked_collection_time_provenance(value)
        self.assertEqual(checked_collection_time_provenance({}),{})

    def test_pipeline_storage_extended_but_six_eval_time_stays1200(self):
        with tempfile.TemporaryDirectory() as d:
            a=receipts(Path(d),release(),integrated=True)
            data={'protocol':actor_protocol(CARRY_PROFILE),'runs':[{**metadata(CARRY_PROFILE),'group':[1,114],
                'prefix':953,'collection_capacity_profile':capacity.WALL2100_PROFILE,'collection_seconds_after_reset':2100}]}
            self.assertEqual(require_pipeline_profile(a,data),CARRY_PROFILE)
            for bad in (storage.CARRY953_SPEC,storage.CARRY_SPEC):
                with self.assertRaises(ValueError):require_pipeline_profile({**a,'storage':bad},data)
            for variant in VARIANTS:require_same_pipeline(a,{**a,'variant':variant})
            for k in ('capacity_profile','seed_profile','authorize_offline_teacher'):a.pop(k)
            a.update(schema=SCHEMA,authorize_evaluation=True,code_commit='code',dataset_sha256='data',
                physical_gpu=3,specified_hand=None,oracle_actor_feedback=False,budget=evaluation_budget(CARRY_PROFILE))
            self.assertEqual(a['budget']['seconds_after_reset'],1200)
            with patch('native_carry_admission.require_engineering_gates'):
                for instance,item in STARTS.items():
                    for variant in VARIANTS:
                        a.update(source=[1,item['episode'],instance],variant=variant)
                        args=('code',a['executor_digest'],{'source':a['source']},variant,EXPERIMENT/f'eval_t1_i{instance}_{variant}_v1','data')
                        require_evaluation(a,*args)
                        with self.assertRaises(ValueError):require_evaluation({**a,'budget':{**a['budget'],'seconds_after_reset':2100}},*args)
                        with self.assertRaises(ValueError):require_evaluation({**a,'capacity_profile':capacity.WALL2100_PROFILE},*args)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);suite(root,storage.WALL2100_SPEC)
            self.assertEqual(summarize(root)['status'],'ALL_SIX_ATTEMPTS_FINISHED')

    def test_real_collector_selects_wall_and_new_receipt_only(self):
        source=(ROOT/'scripts/vlm_sft/native_teacher_collect.py').read_text();tree=ast.parse(source)
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id=='wall_limit' for t in n.targets))
        for value,expected in ((release(),2100),(new_auth(),1200),({},900)):
            scope={'release':value,'collection_wall_seconds':capacity.collection_wall_seconds}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'actual_wall','exec'),scope)
            self.assertEqual(scope['wall_limit'],expected)
        self.assertIn("if time.monotonic()-started >= wall_limit:",source)
        self.assertIn("'collection_time_budget'",source)
        receipt=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
            "release.get('capacity_profile') == WALL2100_PROFILE"==ast.unparse(n.test) and
            any(isinstance(c,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='collection_time'
                for t in c.targets) for c in n.body))
        actual=compile(ast.fix_missing_locations(ast.Module(body=[receipt],type_ignores=[])),'actual_receipt','exec')
        scope={'release':release(),'WALL2100_PROFILE':capacity.WALL2100_PROFILE,'collection_time':{},
            'time':SimpleNamespace(monotonic=lambda:2100.),'started':0.,'wall_limit':2100}
        exec(actual,scope)
        self.assertEqual(scope['collection_time']['collection_time_budget']['wall_seconds_after_reset'],2100.)
        scope['time']=SimpleNamespace(monotonic=lambda:2100.001)
        with self.assertRaises(TimeoutError):exec(actual,scope)
        scope.update(release=new_auth(),collection_time={})
        exec(actual,scope);self.assertEqual(scope['collection_time'],{})
        data_source=(ROOT/'scripts/vlm_sft/native_dataset.py').read_text()
        self.assertIn("collection_time_metadata(run,manifest['authorization'],result,review)",data_source)
        self.assertIn("Row collection wall provenance differs",data_source)

    def test_new_registration_never_grants_a_run_or_training(self):
        r=json.loads((ROOT/'configs/vlm_sft/h09z_train953_wall2100_registration_v1.json').read_text())
        self.assertEqual(r['registered_run_name'],NAME);self.assertEqual(r['source'],[1,264,114])
        self.assertEqual(r['seconds_after_reset'],2100)
        for key in ('authorize_collection','authorize_training','authorize_service','authorize_evaluation'):
            self.assertIs(r[key],False)


if __name__=='__main__':unittest.main()
