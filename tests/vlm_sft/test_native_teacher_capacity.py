"""Capacity-only counterexamples; no simulator/model or large test files."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/vlm_sft')]
from common import write_json
import native_teacher_capacity as capacity
from native_teacher_artifacts import ArtifactBudget,MIB,capture_upper_bound,action_evidence_bound
from native_teacher_collect import require_release


class CapacityTests(unittest.TestCase):
    def release(self):
        return json.loads((ROOT/'configs/vlm_sft/h09w_capacity_authorization_template.json').read_text())

    def test_legacy_default_and_explicit_complete_are_not_mixed(self):
        old={'run_MiB':100,'total_MiB':384}
        self.assertEqual(capacity.capacity_limits(old),(100*MIB,384*MIB))
        self.assertEqual(capacity.capacity_limits({**old,'capacity_profile':'near100'}),(100*MIB,384*MIB))
        new=self.release();self.assertEqual(capacity.capacity_limits(new),(384*MIB,512*MIB))
        for mutation in ({'capacity_profile':None},{'capacity_profile':True},{'capacity_profile':'unknown'},
                         {'run_MiB':100},{'total_MiB':384},{'combined_total_MiB':512},
                         {'run_MiB':384.0},{'combined_total_MiB':True},{'experiment_root':'/different'},
                         {'prior_experiment_roots':[]},{'prior_experiment_roots':[capacity.H09W_ROOT]},
                         {'prior_experiment_roots':[capacity.LEGACY_ROOT,capacity.LEGACY_ROOT]}):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):capacity.capacity_limits({**new,**mutation})
        with self.assertRaises(ValueError):capacity.capacity_limits({**old,'combined_total_MiB':768})

    def test_new_profile_retains_exact_authority_source_and_hard_noncapacity_limits(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.release();r.update(authorize_collection=True,collector_commit='fixed-code',reviewer='reviewer')
            gates=[]
            for task in (0,3):
                p=Path(d)/f'gate{task}.json';write_json(p,{'task':task,'gate_ok':True,'robot_geometry_guards':True,
                    'implementation_digest':r['executor_digest']});gates.append(str(p))
            r['engineering_gate_paths']=gates
            require_release(r,'fixed-code',r['executor_digest'])
            for key,value in [('authorize_collection',1),('authorize_collection','true'),('authorize_collection',False),
                    ('authorize_offline_teacher',1),('allow_grasp_cell_attempt',1),('collector_commit','old'),
                    ('native_controls_max',421),('seconds_after_reset',901),('max_resets',2),('max_resets',True),
                    ('max_resets',1.0),('model_calls',False),
                    ('max_teacher_primitives',13),('max_candidates_per_instance',13),('model_calls',1)]:
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):require_release({**r,key:value},'fixed-code',r['executor_digest'])
            bad=json.loads(Path(gates[0]).read_text());bad['robot_geometry_guards']=False;write_json(gates[0],bad)
            with self.assertRaises(ValueError):require_release(r,'fixed-code',r['executor_digest'])

    def test_new_writer_counts_old_root_before_reserve_and_allows_reserved_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);old=p/'old';new=p/'new';run=new/'run';old.mkdir();run.mkdir(parents=True)
            b=capacity.CombinedArtifactBudget(run,new,old)
            used={old:300*MIB,new:380*MIB,run:0}
            with patch.object(b,'used',side_effect=lambda root:used[root]):
                with b.transaction(5*MIB) as tx:
                    used[old]=383*MIB
                    with self.assertRaisesRegex(RuntimeError,'Combined'):tx.check(1)
                self.assertEqual(b._held,0)
                b.check(1024,cleanup=True)  # 763 MiB leaves the separate cleanup space.
                used[new]=386*MIB
                with self.assertRaisesRegex(RuntimeError,'Combined'):b.check(0,cleanup=True)
                used[new]=0;used[old]=385*MIB
                with self.assertRaisesRegex(RuntimeError,'Combined'):b.check(0)

    def test_missing_or_overlapping_prior_root_never_discounts_old_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);old=p/'old';new=p/'new';run=new/'run';old.mkdir();run.mkdir(parents=True)
            for prior in (p/'missing',new,run,p):
                with self.assertRaises(ValueError):capacity.CombinedArtifactBudget(run,new,prior)
            b=capacity.CombinedArtifactBudget(run,new,old);old.rename(p/'renamed-preserved')
            with self.assertRaisesRegex(RuntimeError,'disappeared'):b.check(0)

    def test_factory_selects_new_writer_only_for_explicit_profile(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);old=p/'old';new=p/'new';run=new/'run';old.mkdir();run.mkdir(parents=True)
            r=self.release();r.update(experiment_root=str(new),prior_experiment_roots=[str(old)])
            with patch.object(capacity,'LEGACY_ROOT',str(old)),patch.object(capacity,'H09W_ROOT',str(new)):
                b=capacity.near_artifact_budget(run,r)
            self.assertIsInstance(b,capacity.CombinedArtifactBudget)
            self.assertEqual((b.run_limit,b.total_limit),(384*MIB,512*MIB))
            legacy=capacity.near_artifact_budget(run,{'experiment_root':str(new),'run_MiB':100,'total_MiB':384})
            self.assertIs(type(legacy),ArtifactBudget)
            self.assertEqual((legacy.run_limit,legacy.total_limit),(100*MIB,384*MIB))

    def test_real_h09v_raw_layout_bound_fits_384_not_192_mib(self):
        layout={name:{'rgb_shape':[3,n,n],'rgb_bytes':3*n*n,'depth_bytes':4*n*n}
                for name,n in [('head',720),('left_wrist',480),('right_wrist',480)]}
        self.assertEqual(capture_upper_bound(layout),9276026)
        self.assertEqual(action_evidence_bound(layout),19879156)
        bound=12*(capture_upper_bound(layout)+action_evidence_bound(layout))+396*4096+4*MIB+3126989
        self.assertEqual(bound,358805493)
        self.assertLess(bound,384*MIB);self.assertGreater(bound,192*MIB)

    def test_h09y_is_explicit_five_starts_not_extension_of_old_profile(self):
        release=self.release();release.update(capacity_profile=capacity.H09Y_PROFILE,experiment_root=capacity.H09Y_ROOT,
            source=[1,264,114],paid_prefix_controls=993,held_out_instance_groups=[[1,1],[1,71]],registered_gpu=3,
            initialization_seconds=900,seconds_after_reset=1200,total_MiB=6144)
        for key in ('prior_experiment_roots','combined_total_MiB'):release.pop(key)
        self.assertEqual(capacity.capacity_limits(release),(384*MIB,6144*MIB))
        capacity.validate_collection_location(release,Path(capacity.H09Y_ROOT)/'native_t1_i114_p0993',3)
        for mutation in ({'source':[1,200,1]},{'paid_prefix_controls':994},{'total_MiB':512},
                         {'registered_gpu':True},{'held_out_instance_groups':[]},{'initialization_seconds':900.0},
                         {'prior_experiment_roots':[]},{'experiment_root':capacity.H09W_ROOT}):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):capacity.capacity_limits({**release,**mutation})
        for output,gpu in ((Path(capacity.H09Y_ROOT)/'retry',3),(Path(capacity.H09Y_ROOT)/'native_t1_i114_p0993',1)):
            with self.assertRaises(ValueError):capacity.validate_collection_location(release,output,gpu)


if __name__=='__main__':unittest.main()
