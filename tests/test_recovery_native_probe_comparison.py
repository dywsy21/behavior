from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_native_probe_comparison import compare_native_probes, ARMS
from recovery_corpus import digest


class NativeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.configs=[json.loads((REPO/f'configs/recovery_sft/a800_native_learner_{a}_probes_heartbeat_v2.json').read_text()) for a in ARMS]
        self.shas=['a'*64,'b'*64,'c'*64]; self.services=[];self.audits=[];self.logs=[]
        for arm,cfg,sha in zip(ARMS,self.configs,self.shas):
            jobs=[dict(id=digest([arm,c,s]),case=c,seed=s,status='done',phase='evaluation',round=0)
                  for c in cfg['cases'] for s in cfg['evaluation_seeds']]
            self.services.append(dict(status='completed_read_only_skill_probes',error=None,evaluation_only=True,
                optimizer_updates=0,actor_updates=0,active_episodes=0,whole_task_sr_evaluated=False,
                config_sha256=sha,parent_sha256=cfg['model']['sha256'],policy_identity_sha256=cfg['model']['sha256'],
                actor_before_sha256='d'*64,actor_after_sha256='d'*64,frozen_before_sha256='e'*64,
                frozen_after_sha256='e'*64,finish_acknowledged=list(cfg['cases']),jobs=jobs))
            rows=[dict(case=j['case'],seed=j['seed'],job_id=j['id'],success=True,outcome='SUCCEEDED',actual_controls=20,
                reset_proprio_error=0.,reset_target_error=0.,observation_archive=dict(status='exact_boundaries_and_lossless_roundtrip_passed')) for j in jobs]
            self.audits.append(dict(status='machine_control_reward_reset_audit_passed',baseline_protocol_checked=True,
                optimizer_authorized=False,whole_task_sr=False,config_sha256=sha,policy_sha256=cfg['model']['sha256'],episodes=rows))
            self.logs.append([dict(job=dict(j,status='active'),success=True,outcome='SUCCEEDED',actual_controls=20,
                whole_task_sr=False,policy_version=0,policy_sha256=cfg['model']['sha256'],
                reference_end_control=cfg['cases'][j['case']]['end_control'],policy_end_control=cfg['cases'][j['case']]['end_control']) for j in jobs])

    def compare(self):
        return compare_native_probes(self.configs,self.services,self.audits,self.logs,self.shas,repo=REPO)

    def mark_unsuccessful(self,arm,index):
        for r in (self.logs[arm][index],self.audits[arm]['episodes'][index]): r.update(success=False,outcome='UNKNOWN')

    def test_complete_pairing_preserves_both_new_wins_and_new_losses(self):
        self.mark_unsuccessful(1,0);self.mark_unsuccessful(2,1)
        r=self.compare();c=r['paired_contrasts']['turning_on_radio_155']['control_vs_candidate']
        self.assertEqual(c['gained_seeds'],[433]);self.assertEqual(c['lost_seeds'],[467])
        self.assertEqual(len(r['pairs']),24);self.assertEqual(r['total_scored_controls'],1440)
        self.assertFalse(r['whole_task_sr']);self.assertFalse(r['policy_promotion_authorized'])

    def test_service_and_physics_must_agree_per_job_not_only_total(self):
        original=deepcopy(self.logs)
        for field,value in [('success',False),('actual_controls',21),('policy_sha256','wrong'),
                             ('whole_task_sr',True),('policy_end_control',1000)]:
            self.logs=deepcopy(original);self.logs[2][0][field]=value
            with self.assertRaises(ValueError):self.compare()
        self.logs=original;self.logs[2][0]['job']['id']='foreign'
        with self.assertRaises(ValueError):self.compare()

    def test_missing_repeated_or_incomplete_arm_never_averaged(self):
        self.logs[2].pop()
        with self.assertRaises(ValueError):self.compare()
        self.setUp();self.logs[2][-1]=deepcopy(self.logs[2][0])
        with self.assertRaises(ValueError):self.compare()
        self.setUp();self.services[2]['finish_acknowledged']=[]
        with self.assertRaises(ValueError):self.compare()

    def test_archive_source_and_readonly_guards(self):
        self.audits[2]['episodes'][0]['observation_archive']['status']='incomplete'
        with self.assertRaises(ValueError):self.compare()
        self.setUp();self.audits[2]['episodes'][0]['job_id']='other'
        with self.assertRaises(ValueError):self.compare()
        self.setUp();self.services[2]['optimizer_updates']=1
        with self.assertRaises(ValueError):self.compare()
        self.setUp();self.configs[2]['evaluation_seeds'][0]=999
        with self.assertRaises(ValueError):self.compare()


if __name__=='__main__':unittest.main()
