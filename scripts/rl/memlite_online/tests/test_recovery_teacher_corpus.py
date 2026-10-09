from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest
from recovery_teacher_corpus import validate_branch,physical_proposal,episode_identity,PhysicalProposalIndex


def fixture():
    plan=dict(control_step=0,decision='EXECUTE',active_skills_semantic_json='[{"verb":"GRASP","target":"cup"}]',parent_goal='goal')
    plan['event_sha256']=digest(plan)
    state=dict(target_name='cup',entity='cup.n.01_1',position=[0.,0.,0.],orientation=[0.,0.,0.,1.],
        grasp=dict(left='TRUE',right='FALSE'),gripper_aperture=dict(left=.01,right=.05),all_held={})
    rows=[]
    for i in range(40):
        after=deepcopy(state)
        if i>=1:after['grasp']['left']='FALSE';after['gripper_aperture']['left']=.05
        a=np.zeros(23,dtype=np.float32);a[14]=-1
        n=np.zeros(23,dtype=np.float32);n[14]=2
        e=a+n
        rows.append(dict(control_step=i,proprio_before=[float(i)]*61,proprio_after=[float(i+1)]*61,
            a_intended=a.tolist(),a_sampled_noise=n.tolist(),a_executed_raw23=e.tolist(),action_executed_raw23=e.tolist(),
            simulator_apply_ack=True,label_kind='injected_fault_not_BC',physical_before=deepcopy(state),physical_audit=after,
            context=dict(context_id=plan['event_sha256'],active_skills_semantic_json=plan['active_skills_semantic_json'],parent_goal='goal')))
        state=after
    return rows,dict(controls=40),[plan]


class TeacherCorpusTests(unittest.TestCase):
    def test_index_exactly_matches_causal_scan_including_future_changes(self):
        import random
        rng=random.Random(17)
        for _ in range(12):
            rows,_,_=fixture()
            for row in rows:
                row['physical_audit']['grasp']['left']=rng.choice(['TRUE','FALSE','FALSE'])
                row['label_kind']=rng.choice(['same_state_local_teacher_candidate','injected_fault_not_BC'])
            index=PhysicalProposalIndex(rows,'left')
            for start in (0,4,16):
                for t in range(start,40):
                    self.assertEqual(physical_proposal(rows,t,'left',attempt_start=start),
                        physical_proposal(rows,t,'left',attempt_start=start,index=index))
            before=physical_proposal(rows,16,'left',attempt_start=0,index=index)
            for row in rows[16:]:row['physical_audit']['grasp']['left']='TRUE'
            self.assertEqual(before,physical_proposal(rows,16,'left',attempt_start=0,index=PhysicalProposalIndex(rows,'left')))
            with self.assertRaises(ValueError):physical_proposal(deepcopy(rows),16,'left',attempt_start=0,index=index)

    def test_copied_evidence_keeps_identity_but_different_physical_attempt_does_not(self):
        source=dict(task='task',instance_id=1,source_group='task:1',episode_index=12)
        result=dict(source_commit='a'*40,proposal_sha256='b'*64,full_snapshot_sha256='c'*64)
        manifest=dict(transitions_sha256='d'*64)
        first=episode_identity(source,result,manifest,'open_gripper')
        copied=dict(result,directory='/different/machine/local-copy')
        self.assertEqual(first,episode_identity(source,copied,manifest,'open_gripper'))
        self.assertNotEqual(first,episode_identity(source,result,dict(transitions_sha256='e'*64),'open_gripper'))
        self.assertNotEqual(first,episode_identity(source,result,manifest,'open_gripper_joint_jitter'))

    def test_true_applied_noise_and_causal_chain(self):
        self.assertTrue(validate_branch(*fixture()))
        for field,value in [('simulator_apply_ack',False),('label_kind','same_state_local_teacher_candidate'),
                             ('proprio_before',[999.]*61),('a_executed_raw23',[0.]*23)]:
            rows,m,p=fixture();rows[4][field]=value
            with self.assertRaises(ValueError):validate_branch(rows,m,p)

    def test_future_success_cannot_relabel_failure(self):
        rows,_,_=fixture()
        self.assertEqual(physical_proposal(rows,16,'left',attempt_start=0),'FAILED')
        for row in rows[16:]:row['physical_audit']['grasp']['left']='TRUE'
        self.assertEqual(physical_proposal(rows,16,'left',attempt_start=0),'FAILED')
        self.assertIsNone(physical_proposal(rows,4,'left',attempt_start=0))

    def test_empty_hand_never_implies_failure_without_prior_hold(self):
        rows,_,_=fixture()
        for row in rows:
            row['physical_before']['grasp']['left']='FALSE';row['physical_audit']['grasp']['left']='FALSE'
        self.assertIsNone(physical_proposal(rows,16,'left',attempt_start=0))


if __name__=='__main__':unittest.main()
