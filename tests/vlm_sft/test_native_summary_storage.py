"""Whole six-slot reports accept either exact carry storage, never mixed identities."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/vlm_sft')]
from common import write_json,sha
from native_actor_protocol import VERSION
from native_carry_admission import require_carry_reviews
from native_eval_prepare import SCHEMA,STARTS
from native_eval_run import evaluation_budget
from native_eval_summary import summarize
from native_evaluation import VARIANTS
from native_execution import CARRY_PROFILE,actor_protocol
from native_storage import CARRY_SPEC,CARRY953_SPEC
from test_native_carry_duration import auth,receipts


def suite(root,storage=None):
    profile=CARRY_PROFILE if storage is not None else None
    if profile is not None:
        common=receipts(root,auth(),integrated=True)
        for key in ('capacity_profile','seed_profile','authorize_offline_teacher'):
            common.pop(key)
        common['storage']=copy.deepcopy(storage)
        require_carry_reviews(common)
    else:
        common={'code_commit':'d'*40}
    slots=[]
    for instance,item in STARTS.items():
        for variant in VARIANTS:
            folder=root/f'eval_t1_i{instance}_{variant}_v1';folder.mkdir()
            a={**copy.deepcopy(common),'schema':SCHEMA,'source':[1,item['episode'],instance],
                'variant':variant,'budget':evaluation_budget(profile)}
            m={'code_commit':a['code_commit'],'protocol':actor_protocol(profile) if profile else VERSION,
                'dataset_sha256':'a'*64,'authorization':a,'preparation':{'prefix_controls':item['prefix']},
                'specified_hand':None,'oracle_actor_feedback':False,'budget':evaluation_budget(profile),
                'storage':None if storage is None else {'profile':storage['profile']}}
            write_json(folder/'manifest.json',m)
            r={'manifest_sha256':sha(folder/'manifest.json'),'variant':variant,'instance':instance,
                'status':'COMPLETE','final_hold':True,'issued_native':1,'native_controls':1,
                'issued_prefix':item['prefix'],'prefix_controls':item['prefix'],'decisions':[],
                'wall_seconds_after_reset':1.,'score':{'any_hand_local_success':False}}
            write_json(folder/'result.json',r)
            slots.append((folder,m,r))
    return slots


def changed(slot):
    folder,m,r=slot;write_json(folder/'manifest.json',m)
    r['manifest_sha256']=sha(folder/'manifest.json');write_json(folder/'result.json',r)


class SummaryStorageTests(unittest.TestCase):
    def test_six_new953_same_profile_complete_and_original_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);suite(root,CARRY953_SPEC);r=summarize(root)
            self.assertEqual(r['status'],'ALL_SIX_ATTEMPTS_FINISHED')
            self.assertEqual(len(r['rows']),6)
            self.assertTrue(all(v['valid_bounded_result'] for v in r['rows']))
            self.assertEqual(r['per_variant_registered_numerator_denominator'],{v:[0,2] for v in VARIANTS})
            self.assertFalse(r['full_task_SR_claim'])

    def test_original_carry_and_legacy_still_pass(self):
        for storage in (CARRY_SPEC,None):
            with self.subTest(storage=storage is not None),tempfile.TemporaryDirectory() as d:
                root=Path(d);suite(root,storage)
                self.assertEqual(summarize(root)['status'],'ALL_SIX_ATTEMPTS_FINISHED')

    def test_two_valid_but_mixed_carry_storage_profiles_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);slots=suite(root,CARRY953_SPEC)
            _,m,_=slots[-1];m['authorization']['storage']=copy.deepcopy(CARRY_SPEC)
            m['storage']['profile']=CARRY_SPEC['profile'];changed(slots[-1])
            with self.assertRaisesRegex(ValueError,'same|share exact'):summarize(root)

    def test_invalid_storage_or_reported_profile_mismatch_rejected(self):
        for mutation in ('unknown','extra_alias','manifest_profile'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                root=Path(d);slots=suite(root,CARRY953_SPEC);_,m,_=slots[0]
                if mutation=='unknown':m['authorization']['storage']['profile']='unknown'
                elif mutation=='extra_alias':m['authorization']['storage']['shared_og_cache']['allowed_aliases'].append('/tmp/extra')
                else:m['storage']['profile']=CARRY_SPEC['profile']
                changed(slots[0])
                with self.assertRaises(ValueError):summarize(root)

    def test_same_new_storage_cannot_mix_source_dataset_core_or_executor(self):
        for mutation in ('code_commit','dataset_sha256','carry_duration_core_commit','executor_digest'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                root=Path(d);slots=suite(root,CARRY953_SPEC);_,m,_=slots[-1]
                if mutation=='code_commit':
                    m[mutation]='e'*40;m['authorization'][mutation]='e'*40
                elif mutation=='dataset_sha256':m[mutation]='e'*64
                else:m['authorization'][mutation]='e'*(40 if mutation.endswith('commit') else 64)
                changed(slots[-1])
                with self.assertRaises(ValueError):summarize(root)

    def test_old_or_expanded_carry_budget_rejected(self):
        for key,value in (('max_decisions',12),('max_decisions',15),
                          ('new_controls_including_final_hold',420),('new_controls_including_final_hold',641)):
            with self.subTest(key=key,value=value),tempfile.TemporaryDirectory() as d:
                root=Path(d);slots=suite(root,CARRY953_SPEC);_,m,_=slots[0]
                m['budget'][key]=value;m['authorization']['budget'][key]=value;changed(slots[0])
                with self.assertRaisesRegex(ValueError,'budgets differ'):summarize(root)


if __name__=='__main__':unittest.main()
