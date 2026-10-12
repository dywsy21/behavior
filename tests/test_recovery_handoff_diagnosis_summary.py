import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl/memlite_online/code'))
from recovery_handoff_diagnosis_summary import paired_diagnostics, summarize_diagnostic_rows, SCORES


class DiagnosisSummaryTests(unittest.TestCase):
    def row(self,index=0,count=1,mean=2.):
        return dict(global_schedule_index=index,scheduled_sample_index=index,split='train_handoff',
            sample_id=str(index),identity=dict(task=index),expected=dict(bundle=str(index)),error=None,
            input_audit=dict(optimizer_exposures=1),score={key:False for key in SCORES},
            teacher_forced_fields=dict(ce_field_bundle_tokens=count,ce_field_bundle_mean=mean),
            first_choice=dict(bundle=dict(available=True,teacher_forced=True,
                preceding_fields_are_teacher_forced=True,target_log_probability=-1.,
                repeated_log_probability=-2.,target_minus_repeated_logit=1.,target_is_greedy=True)))

    def test_token_weighting_not_mean_of_sample_means(self):
        actual=summarize_diagnostic_rows([self.row(count=1,mean=2),self.row(1,count=3,mean=4)])
        self.assertEqual(actual['field_ce']['bundle'],dict(tokens=4,token_weighted_ce=3.5))
        self.assertEqual(actual['conditional_first_choice']['bundle']['target_beats_repeat'],2)

    def test_missing_choice_and_masked_field_are_not_zero_probability(self):
        row=self.row(count=0,mean=0);row['first_choice']['bundle']=dict(available=False)
        actual=summarize_diagnostic_rows([row])
        self.assertIsNone(actual['field_ce']['bundle']['token_weighted_ce'])
        self.assertEqual(actual['conditional_first_choice'],{})

    def test_pair_counts_include_invalid_and_exposure_strata(self):
        left=[self.row(),self.row(1)];left[1]['input_audit']['optimizer_exposures']=0
        right=copy.deepcopy(left);right[0]['score'].update(valid=True,exact=True)
        left[1]['error']='strict parser rejection'
        report=paired_diagnostics(left,right)['cohorts']
        self.assertEqual(report['handoff']['paired_counts']['exact'],dict(gains=1,losses=0))
        self.assertEqual(report['exposed_handoff']['control']['rows'],1)
        self.assertEqual(report['unexposed_handoff']['control']['rows'],1)
        self.assertEqual(report['all']['control']['strict_parse_errors'],1)

    def test_foreign_target_input_order_split_exposure_and_duplicates_rejected(self):
        for key in ('sample_id','expected','input_audit','identity','split','global_schedule_index'):
            left=[self.row()];right=copy.deepcopy(left);right[0][key]='changed'
            with self.assertRaises(ValueError):paired_diagnostics(left,right)
        with self.assertRaises(ValueError):paired_diagnostics([self.row()],[self.row(),self.row(1)])
        with self.assertRaises(ValueError):paired_diagnostics([self.row(),self.row()],[self.row(),self.row()])
        row=self.row();row['split']='eval'
        with self.assertRaises(ValueError):paired_diagnostics([row],[row])

    def test_nonfinite_or_invalid_ce_and_probability_refused(self):
        for count,mean in ((-1,1),(1.2,1),(1,float('nan')),(1,-.1)):
            with self.assertRaises(ValueError):summarize_diagnostic_rows([self.row(count=count,mean=mean)])
        row=self.row();row['first_choice']['bundle']['target_log_probability']=.1
        with self.assertRaises(ValueError):summarize_diagnostic_rows([row])


if __name__=='__main__':unittest.main()
