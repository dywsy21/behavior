"""Explicit six-slot wall identity, not a motion or success threshold change."""
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
from native_execution import CARRY_PROFILE,WORKSPACE_PROFILE,actor_protocol
from native_eval_run import evaluation_budget,evaluation_time_profile,require_evaluation,EVALUATION_TIME_PROFILE
from native_eval_prepare import SCHEMA,STARTS,ROOT as EXPERIMENT
from native_eval_summary import summarize
from native_evaluation import VARIANTS
from native_storage import WALL2100_SPEC
from test_native_summary_storage import suite,changed
from test_native_carry_duration import auth,receipts


def new_suite(root):
    slots=suite(root,WALL2100_SPEC)
    for slot in slots:
        _,m,r=slot
        m['authorization']['evaluation_time_profile']=EVALUATION_TIME_PROFILE
        m['authorization']['budget']=evaluation_budget(CARRY_PROFILE,EVALUATION_TIME_PROFILE)
        m['budget']=copy.deepcopy(m['authorization']['budget'])
        m['evaluation_time_profile']=r['evaluation_time_profile']=EVALUATION_TIME_PROFILE
        r['wall_seconds_after_reset']=1800.
        changed(slot)
    return slots


class EvaluationTimeTests(unittest.TestCase):
    def test_old_defaults_and_only_explicit_carry2100(self):
        for profile in (None,WORKSPACE_PROFILE,CARRY_PROFILE):
            self.assertEqual(evaluation_budget(profile)['seconds_after_reset'],1200)
        old=evaluation_budget(CARRY_PROFILE);new=evaluation_budget(CARRY_PROFILE,EVALUATION_TIME_PROFILE)
        self.assertEqual(new,{**old,'seconds_after_reset':2100})
        for tag in ('unknown',True,2100,{},[], ''):
            with self.subTest(tag=tag),self.assertRaises(ValueError):evaluation_budget(CARRY_PROFILE,tag)
        for profile in (None,WORKSPACE_PROFILE):
            with self.assertRaises(ValueError):evaluation_budget(profile,EVALUATION_TIME_PROFILE)
        with self.assertRaises(ValueError):evaluation_time_profile({'evaluation_time_profile':None})
        self.assertIsNone(evaluation_time_profile({}))

    def test_all_six_actual_admission_paths_and_budget_mixing(self):
        with tempfile.TemporaryDirectory() as d:
            a=receipts(Path(d),auth(),integrated=True)
            for k in ('capacity_profile','seed_profile','authorize_offline_teacher'):a.pop(k)
            a.update(schema=SCHEMA,authorize_evaluation=True,code_commit='d'*40,dataset_sha256='e'*64,
                protocol=actor_protocol(CARRY_PROFILE),physical_gpu=3,specified_hand=None,oracle_actor_feedback=False,
                reviewer='Codex-parent',experiment_root=str(EXPERIMENT),storage=copy.deepcopy(WALL2100_SPEC),
                evaluation_time_profile=EVALUATION_TIME_PROFILE,budget=evaluation_budget(CARRY_PROFILE,EVALUATION_TIME_PROFILE))
            with patch('native_carry_admission.require_engineering_gates'):
                for instance,item in STARTS.items():
                    for variant in VARIANTS:
                        a.update(source=[1,item['episode'],instance],variant=variant)
                        args=('d'*40,a['executor_digest'],{'source':a['source']},variant,EXPERIMENT/f'eval_t1_i{instance}_{variant}_v1','e'*64)
                        require_evaluation(a,*args)
                        for value in (1200,2101,2100.,True):
                            with self.assertRaises(ValueError):require_evaluation({**a,'budget':{**a['budget'],'seconds_after_reset':value}},*args)
                        missing=dict(a);missing.pop('evaluation_time_profile')
                        with self.assertRaises(ValueError):require_evaluation(missing,*args)
                        with self.assertRaises(ValueError):require_evaluation({**a,'evaluation_time_profile':None},*args)
                        old={**missing,'budget':evaluation_budget(CARRY_PROFILE)};require_evaluation(old,*args)

    def test_complete_six_new_and_old_results_have_exact_time_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);slots=new_suite(root)
            report=summarize(root)
            self.assertEqual(report['status'],'ALL_SIX_ATTEMPTS_FINISHED')
            self.assertTrue(all(r['valid_bounded_result'] for r in report['rows']))
            for r in report['rows']:self.assertEqual(r['evaluation_time_profile'],EVALUATION_TIME_PROFILE)
            for elapsed,valid in ((2100.,True),(2100.001,False),(float('inf'),False),(float('nan'),False),(-1.,False),(True,False),(None,False)):
                # Deliberately malformed JSON numerics test the report reader;
                # the production writer itself rejects NaN/Infinity.
                slots[0][2]['wall_seconds_after_reset']=elapsed
                (slots[0][0]/'result.json').write_text(json.dumps(slots[0][2],allow_nan=True))
                self.assertIs(summarize(root)['rows'][0]['valid_bounded_result'],valid)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);slots=suite(root,WALL2100_SPEC)
            slots[0][2]['wall_seconds_after_reset']=1200.001;changed(slots[0])
            self.assertFalse(summarize(root)['rows'][0]['valid_bounded_result'])

    def test_even_one_old_slot_cannot_mix_with_five_new_slots(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);slots=new_suite(root);_,m,r=slots[-1]
            m['authorization'].pop('evaluation_time_profile');m.pop('evaluation_time_profile');r.pop('evaluation_time_profile')
            m['budget']=m['authorization']['budget']=evaluation_budget(CARRY_PROFILE);r['wall_seconds_after_reset']=1.
            changed(slots[-1])
            with self.assertRaisesRegex(ValueError,'share exact'):summarize(root)

    def test_manifest_result_authorization_and_missing_attempt_binding(self):
        for where in ('manifest','result','authorization','manifest_budget','auth_budget','float_budget'):
            with self.subTest(where=where),tempfile.TemporaryDirectory() as d:
                root=Path(d);slots=new_suite(root);_,m,r=slots[0]
                if where=='manifest':m.pop('evaluation_time_profile')
                elif where=='result':r.pop('evaluation_time_profile')
                elif where=='authorization':m['authorization']['evaluation_time_profile']='unknown'
                elif where=='manifest_budget':m['budget']['seconds_after_reset']=1200
                elif where=='auth_budget':m['authorization']['budget']['seconds_after_reset']=1200
                else:m['budget']['seconds_after_reset']=2100.
                changed(slots[0])
                with self.assertRaises(ValueError):summarize(root)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);slots=new_suite(root);(slots[0][0]/'result.json').unlink()
            self.assertNotEqual(summarize(root)['status'],'ALL_SIX_ATTEMPTS_FINISHED')
            slots[0][1]['budget']['seconds_after_reset']=1200;changed(slots[0]);(slots[0][0]/'result.json').unlink()
            with self.assertRaises(ValueError):summarize(root)

    def test_real_runner_step_loop_and_last_instant_query_checks(self):
        tree=ast.parse((ROOT/'scripts/vlm_sft/native_eval_run.py').read_text())
        nodes=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.If):continue
            if any(isinstance(c,ast.Raise) and isinstance(c.exc,ast.Call) and isinstance(c.exc.func,ast.Name)
                    and c.exc.func.id=='TimeoutError' and c.exc.args and isinstance(c.exc.args[0],ast.Constant)
                    and c.exc.args[0].value in ('Reset-to-end budget exhausted','No new call after deadline','No new call after snapshot deadline') for c in node.body):
                nodes.append(node)
        self.assertEqual(len(nodes),3)
        for node in nodes:
            compiled=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_deadline','exec')
            for limit in (1200,2100):
                scope={'time':SimpleNamespace(monotonic=lambda:limit-.001),'started':0.,
                    'budget':{'seconds_after_reset':limit},'time_profile':EVALUATION_TIME_PROFILE}
                exec(compiled,scope)
                scope['time']=SimpleNamespace(monotonic=lambda:limit)
                with self.assertRaises(TimeoutError):exec(compiled,scope)
        source=ast.unparse(tree);guard=source.index("raise TimeoutError('No new call after snapshot deadline')")
        self.assertLess(guard,source.index('neural_requests += 1',guard))
        self.assertLess(guard,source.index('answer = choose(',guard))
        self.assertEqual(source.count('**time_identity'),2)


if __name__=='__main__':unittest.main()
