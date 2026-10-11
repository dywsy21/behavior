import unittest
from pathlib import Path
import numpy as np

from behavior_champion.session import ChampionSession


def observation():
    state=np.zeros(61,dtype=np.float32)
    state[24:26]=state[49:51]=.025
    return dict(proprio=state,images={k:np.zeros((3,4,4),dtype=np.uint8)
        for k in ('head_rgb','left_wrist_rgb','right_wrist_rgb')})


def predictor(obs,task,stage,previous,rng):
    random=np.random.default_rng(int(rng[1]))
    actions=random.normal(size=(30,23)).astype(np.float32)*.1
    actions[:,[14,22]]=.5
    logits=np.full(15,-np.inf,np.float32);logits[:5]=-2.;logits[min(stage+1,4)]=2.
    return dict(actions=actions,subtask_logits=logits),np.array([0,int(rng[1])+1],np.uint32)


def variation(actions,n):
    a,b=np.ptp(actions[:n,14]),np.ptp(actions[:n,22])
    return bool(a>.2 or b>.2),float(a),float(b)


def session(name='a',task=0,**kwargs):
    return ChampionSession(episode=name,task=task,model_task=task,num_stages=5,seed=17,
        predictor=kwargs.pop('predictor',predictor),correct_actions=lambda t,s,p,a:(a,s),
        gripper_variation=variation,**kwargs)


def acknowledge(s,result,n=20):
    return s.ack(token=result['token'],controls=[dict(control_step=s.control_step+i,
        simulator_apply_ack=True,action_executed_raw23=a.tolist())
        for i,a in enumerate(result['action_chunk'][:n])])


class SessionTests(unittest.TestCase):
    @unittest.skipUnless(Path('artifacts/champion-reference-20261011/src/b1k/shared/eval_b1k_wrapper.py').is_file(),
                         'Pinned read-only upstream reference unavailable')
    def test_stage_voting_against_actual_pinned_upstream_method(self):
        import ast
        import logging
        from collections import deque
        from types import SimpleNamespace
        from behavior_champion.contracts import LEGACY_STAGES
        text=Path('artifacts/champion-reference-20261011/src/b1k/shared/eval_b1k_wrapper.py').read_text()
        cls=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=='B1KPolicyWrapper')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='update_current_stage')
        namespace=dict(np=np,TASK_NUM_STAGES=LEGACY_STAGES,logger=logging.getLogger('reference'))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pinned-upstream-voting>','exec'),namespace)
        for task in (0,1,30,49):
            ours=ChampionSession(episode='oracle',task=task,model_task=task,num_stages=LEGACY_STAGES[task],
                seed=17,predictor=predictor,correct_actions=None,gripper_variation=variation)
            reference=SimpleNamespace(task_id=task,current_stage=0,prediction_history=deque(maxlen=3),
                config=SimpleNamespace(history_len=3,votes_to_promote=2),step_count=0)
            votes=list(range(LEGACY_STAGES[task]))*4
            votes+=np.random.default_rng(17).integers(0,LEGACY_STAGES[task],1000).tolist()
            for vote in votes:
                logits=np.full(15,-np.inf);logits[vote]=0
                namespace['update_current_stage'](reference,logits)
                ours._vote(vote)
                self.assertEqual(ours.stage,reference.current_stage)
                self.assertEqual(list(ours.history),list(reference.prediction_history))

    def test_interleaving_cannot_change_actions_stage_votes_rng_or_inpainting(self):
        alone=session();alone.begin()
        expected=[]
        for _ in range(6):
            result=alone.infer(observation());expected.append(result);acknowledge(alone,result)
        a,b=session(),session('b',1);a.begin();b.begin()
        for target in expected:
            foreign=b.infer(observation());acknowledge(b,foreign)
            result=a.infer(observation())
            np.testing.assert_array_equal(result['action_chunk'],target['action_chunk'])
            self.assertEqual(result['display'],target['display'])
            self.assertEqual(result['token'],target['token'])
            acknowledge(a,result)
        np.testing.assert_array_equal(a.initial_actions,alone.initial_actions)
        self.assertEqual(a.stage,2)
        self.assertEqual(a.control_step,120)

    def test_ack_rejects_foreign_stale_wrong_action_and_incomplete_before_infer(self):
        a,b=session(),session('b');a.begin();b.begin()
        x,y=a.infer(observation()),b.infer(observation())
        row=dict(control_step=0,simulator_apply_ack=True,action_executed_raw23=x['action_chunk'][0].tolist())
        with self.assertRaises(ValueError):a.ack(token=y['token'],controls=[row])
        row['action_executed_raw23'][0]+=1
        with self.assertRaises(ValueError):a.ack(token=x['token'],controls=[row])
        self.assertEqual(a.control_step,0)
        acknowledge(a,x,5)
        with self.assertRaises(ValueError):a.infer(observation())
        with self.assertRaises(ValueError):a.ack(token=x['token'],controls=[row])
        ended=a.finish();self.assertEqual(ended['unused_offered_controls'],15)
        self.assertEqual(ended['control_step'],5)
        with self.assertRaises(ValueError):a.begin()
        with self.assertRaises(ValueError):a.infer(observation())

    def test_inpainting_slices_match_both_original_execution_branches(self):
        for high_variation,start in ((False,26),(True,20)):
            calls=[]
            def fake(obs,task,stage,previous,rng):
                actions=np.tile(np.arange(30,dtype=np.float32)[:,None],(1,23))/100
                actions[:,[14,22]]=np.arange(30)[:,None]/10 if high_variation else .5
                calls.append((previous,actions.copy()))
                logits=np.full(15,-np.inf,np.float32);logits[:5]=0
                return dict(actions=actions,subtask_logits=logits),rng
            s=session(predictor=fake);s.begin()
            first=s.infer(observation());acknowledge(s,first)
            s.infer(observation())
            np.testing.assert_array_equal(calls[1][0],calls[0][1][start:start+4])
            self.assertEqual(first['display']['compressed'],not high_variation)
            if high_variation:
                np.testing.assert_array_equal(first['action_chunk'],calls[0][1][:20])
            else:
                self.assertAlmostEqual(first['action_chunk'][-1,0],.25*1.3,places=6)
                self.assertAlmostEqual(first['action_chunk'][-1,3],.25,places=6)

    def test_illegal_stage_privileged_input_and_old_rules_on_new_tasks_rejected(self):
        with self.assertRaises(ValueError):session(task=50)
        new=session(task=50,apply_legacy_rules=False);new.begin()
        obs=observation();obs['physical_success']=True
        with self.assertRaises(ValueError):new.infer(obs)
        def invalid(*args):
            output,rng=predictor(*args);output['subtask_logits'][8]=99;return output,rng
        s=session(predictor=invalid);s.begin()
        with self.assertRaises(ValueError):s.infer(observation())

    def test_stage_voting_skip_and_final_boundary_match_original(self):
        s=session()
        for vote in (2,2,2):s._vote(vote)
        self.assertEqual(s.stage,1)
        for vote in (0,0,0):s._vote(vote)
        self.assertEqual(s.stage,0)
        s.stage=4
        for vote in (3,3,3):s._vote(vote)
        self.assertEqual(s.stage,4)


if __name__=='__main__':unittest.main()
