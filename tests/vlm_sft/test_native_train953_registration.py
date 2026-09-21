"""One later TRAIN start, immutable old storage manifests, same paired physics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/vlm_sft')]
import native_storage as storage
import native_teacher_capacity as capacity
from native_execution import (CARRY_PROFILE,metadata,authorization_profile,require_pipeline_profile,
    require_same_pipeline,actor_protocol,episode_limits,body_limit,service_call_limit)
from native_carry_admission import require_carry_reviews
from native_eval_run import require_evaluation,evaluation_budget
from native_evaluation import VARIANTS
from native_motion_codec import tokens,VERSION
from test_native_carry_duration import auth,receipts


def new_auth():
    return {**auth(),'source':[1,264,114],'paid_prefix_controls':953,
        'storage':copy.deepcopy(storage.CARRY953_SPEC),'carry_duration_bootstrap':False}


class Train953RegistrationTests(unittest.TestCase):
    def test_only_exact_one_additional_source_and_name(self):
        a=new_auth();name='native_t1_i114_p0953_ws45_cd1_b2'
        self.assertEqual(authorization_profile(a,collection=True),CARRY_PROFILE)
        capacity.validate_collection_location(a,Path(capacity.H09Y_ROOT)/name,3)
        for change in ({'paid_prefix_controls':961},{'paid_prefix_controls':952},
                       {'paid_prefix_controls':954},{'source':[1,310,192]},
                       {'source':[1,200,1]},{'paid_prefix_controls':953.0}):
            with self.subTest(change=change),self.assertRaises(ValueError):capacity.capacity_limits({**a,**change})
        for wrong in (name+'_retry',name.replace('_cd1_b2',''),name.replace('0953','0961')):
            with self.assertRaises(ValueError):capacity.validate_collection_location(a,Path(capacity.H09Y_ROOT)/wrong,3)
        for old in (capacity.H09Y_PROFILE,capacity.DIVERSE_PROFILE,capacity.WORKSPACE_PROFILE):
            with self.assertRaises(ValueError):capacity.capacity_limits({**a,'capacity_profile':old})

    def test_storage_adds_one_explicit_alias_without_mutating_old_spec(self):
        name='native_t1_i114_p0953_ws45_cd1_b2'
        self.assertEqual(storage.CARRY953_RUNS,storage.CARRY_RUNS+(name,))
        self.assertNotIn(name,storage.CARRY_RUNS)
        for spec in (storage.SHARED_SPEC,storage.DIVERSE_SPEC,storage.WORKSPACE_SPEC,storage.CARRY_SPEC,storage.CARRY953_SPEC):
            self.assertTrue(storage.validate_spec({'storage':spec}))
        old=copy.deepcopy(storage.CARRY953_SPEC);old['profile']=storage.CARRY_STORAGE_PROFILE
        old['shared_og_cache']['allowed_aliases'].pop()
        self.assertEqual(old,storage.CARRY_SPEC)
        for change in ('profile','alias','target'):
            value=copy.deepcopy(storage.CARRY953_SPEC)
            if change=='profile':value['profile']=storage.CARRY_STORAGE_PROFILE
            elif change=='alias':value['shared_og_cache']['allowed_aliases'].append('/tmp/extra')
            else:value['shared_og_cache']['canonical_target']='/tmp/cache'
            with self.assertRaises(ValueError):storage.validate_spec({'storage':value})
        for name in ('native_t1_i114_p0953_ws45_cd1_b2','training_v1','service_v1','eval_prepare_i1_v1'):
            storage.RuntimeStorage(new_auth(),storage.EXPERIMENT_ROOT/name)
        with self.assertRaises(ValueError):storage.RuntimeStorage(auth(),storage.EXPERIMENT_ROOT/'native_t1_i114_p0953_ws45_cd1_b2')

    def test_collection_does_not_mix_old_and_new_storage_contracts(self):
        self.assertEqual(authorization_profile(auth(),collection=True),CARRY_PROFILE)
        with self.assertRaises(ValueError):authorization_profile({**new_auth(),'storage':storage.CARRY_SPEC},collection=True)
        with self.assertRaises(ValueError):authorization_profile({**auth(),'storage':storage.CARRY953_SPEC},collection=True)
        self.assertEqual(episode_limits(CARRY_PROFILE),(14,640))
        self.assertEqual(body_limit(CARRY_PROFILE),2);self.assertEqual(service_call_limit(CARRY_PROFILE),56)
        self.assertEqual(len(tokens(VERSION)),45)

    def test_new_candidate_requires_existing_real_integration_not_bootstrap(self):
        with tempfile.TemporaryDirectory() as d:
            a=receipts(Path(d),new_auth())
            with self.assertRaises(ValueError):require_carry_reviews(a)
            with self.assertRaises(ValueError):require_carry_reviews({**a,'carry_duration_bootstrap':True},bootstrap=True)
            receipts(Path(d),a,integrated=True);require_carry_reviews(a)

    def test_training_service_and_three_eval_variants_keep_same_extended_storage(self):
        data={'protocol':actor_protocol(CARRY_PROFILE),'runs':[
            {},metadata(CARRY_PROFILE),{**metadata(CARRY_PROFILE),'group':[1,114],'prefix':953}]}
        with tempfile.TemporaryDirectory() as d:
            a=receipts(Path(d),new_auth(),integrated=True)
            self.assertEqual(require_pipeline_profile(a,data),CARRY_PROFILE)
            with self.assertRaises(ValueError):require_pipeline_profile({**a,'storage':storage.CARRY_SPEC},data)
            old_data={**data,'runs':data['runs'][:-1]}
            self.assertEqual(require_pipeline_profile({**a,'storage':storage.CARRY_SPEC},old_data),CARRY_PROFILE)
            for role in ('training','service',*VARIANTS):
                identity={**a,'role':role};require_same_pipeline(a,identity)
                with self.assertRaises(ValueError):require_same_pipeline(a,{**identity,'storage':storage.CARRY_SPEC})

    def test_all_six_evaluation_admissions_accept_same_explicit_storage_only(self):
        from native_eval_prepare import SCHEMA,ROOT as EXPERIMENT
        with tempfile.TemporaryDirectory() as d:
            a=receipts(Path(d),new_auth(),integrated=True)
            for k in ('capacity_profile','seed_profile','authorize_offline_teacher'):a.pop(k)
            a.update(schema=SCHEMA,authorize_evaluation=True,code_commit='code',dataset_sha256='data',physical_gpu=3,
                specified_hand=None,oracle_actor_feedback=False,reviewer='Codex-parent',budget=evaluation_budget(CARRY_PROFILE))
            with patch('native_carry_admission.require_engineering_gates'):
                for instance,episode in ((1,200),(71,247)):
                    for variant in VARIANTS:
                        a.update(source=[1,episode,instance],variant=variant)
                        require_evaluation(a,'code',a['executor_digest'],{'source':a['source']},variant,
                            EXPERIMENT/f'eval_t1_i{instance}_{variant}_v1','data')
                        with self.assertRaises(ValueError):require_evaluation({**a,'budget':{**a['budget'],'max_decisions':15}},
                            'code',a['executor_digest'],{'source':a['source']},variant,EXPERIMENT/f'eval_t1_i{instance}_{variant}_v1','data')

    def test_registration_grants_nothing_and_names_only953(self):
        value=json.loads((ROOT/'configs/vlm_sft/h09z_train953_registration_v1.json').read_text())
        self.assertEqual(value['source'],[1,264,114]);self.assertEqual(value['paid_prefix_controls'],953)
        self.assertEqual(value['registered_run_name'],'native_t1_i114_p0953_ws45_cd1_b2')
        self.assertEqual(value['executor_digest'],'8fdfcd3e1b8bdfd95530eec09710d1c44efc215885b49c64c1a82f55bbc501a8')
        for key in ('authorize_collection','authorize_training','authorize_service','authorize_evaluation'):
            self.assertIs(value[key],False)


if __name__=='__main__':unittest.main()
