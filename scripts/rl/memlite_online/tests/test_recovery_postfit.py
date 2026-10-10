from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from recovery_postfit import calibration_gate, exposure_manifest, select_postfit_sources


class PostfitTests(unittest.TestCase):
    def calibration(self):
        return dict(schema='recovery_observer_calibration_v1', ready=True, blockers=[], mechanism='GRASP',
            certifies_other_mechanisms=False, minimum_confidence=.85, temperature=1.77,
            observer_sha256='b'*64, high_sha256='a'*64, events=90, false_success_count=1,
            source_groups=['cal:'+str(i) for i in range(90)], classes={
                name:dict(actual_events=30, confident_predictions=n, correct=c)
                for name,n,c in [('IN_PROGRESS',25,25),('SUCCEEDED',29,28),('FAILED',27,26)]})

    def exposure(self):
        cal=self.calibration()
        sel=dict(schema='independent_calibration_pool_selection_v4',selected_groups=cal['source_groups'],
            declared_groups=cal['source_groups']+['reserved:1'],selected_observer_sha256='b'*64,high_sha256='a'*64)
        rows=[dict(candidate=dict(source_group=g,split=s)) for g,s in [('fit:1','train'),('dev:1','dev')]]
        return exposure_manifest(rows,sel,cal,['historical:1'],['test:1'],{})

    def row(self,group,split='train',branch='open_gripper_joint_jitter',positive=True):
        return dict(source_group=group,case=group.replace(':','_'),split=split,branch=branch,
            physical_recovery_candidate=positive,failure=None,proposed_outcomes={})

    def test_original_actual_gates_pass_but_two_false_successes_fail(self):
        calibration_gate(self.calibration())
        bad=self.calibration();bad['false_success_count']=2
        with self.assertRaises(ValueError):calibration_gate(bad)

    def test_changed_scope_threshold_temperature_or_class_support_fails(self):
        for field,value in [('mechanism','OPEN_DOOR'),('minimum_confidence',.8),
                            ('temperature',float('nan')),('ready',False)]:
            c=self.calibration();c[field]=value
            with self.assertRaises(ValueError):calibration_gate(c)
        c=self.calibration();c['classes']['FAILED']['confident_predictions']=19
        with self.assertRaises(ValueError):calibration_gate(c)

    def test_all_fit_selection_calibration_and_reserves_excluded(self):
        e=self.exposure()
        queue=[self.row(g) for g in e['excluded_groups']]+[self.row('new:1'),self.row('new:2','dev')]
        selected,ledger=select_postfit_sources(queue,e)
        self.assertEqual({r['source_group'] for r in selected},{'new:1','new:2'})
        self.assertEqual({r['source_group']:r['split'] for r in selected},{'new:1':'train','new:2':'dev'})
        self.assertEqual(len(ledger),len(queue))

    def test_failed_sources_stay_and_branch_order_is_deterministic(self):
        e=self.exposure();q=[self.row('new:1',positive=False),self.row('new:1',branch='open_gripper'),
            self.row('new:2',positive=False),self.row('new:3'),self.row('new:3',branch='open_gripper')]
        a,ledger=select_postfit_sources(q,e);b,_=select_postfit_sources(list(reversed(q)),e)
        self.assertEqual(a,b)
        self.assertEqual({r['source_group']:r['branch'] for r in a},
            {'new:1':'open_gripper','new:3':'open_gripper_joint_jitter'})
        self.assertEqual(next(r for r in ledger if r['source_group']=='new:2')['status'],
            'no_physically_verified_positive_continuation')

    def test_missing_exclusion_and_cross_split_duplicate_fail(self):
        e=self.exposure();bad=deepcopy(e);bad['excluded_groups'].remove('dev:1')
        with self.assertRaises(ValueError):select_postfit_sources([self.row('new:1')],bad)
        for q in ([self.row('new:1'),self.row('new:1')],
                  [self.row('new:1'),self.row('new:1','dev','open_gripper')]):
            with self.assertRaises(ValueError):select_postfit_sources(q,e)


if __name__ == '__main__':
    unittest.main()
