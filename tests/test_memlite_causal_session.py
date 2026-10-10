from dataclasses import replace
import json
import unittest

import torch

from g05.utils.memlite_causal_session import CausalModelIdentity, CausalSessionIdentity, CausalPlannerSession
from g05.utils.memlite_skill_protocol import append_b_memory_idempotent, canonical_json, semantic_active_skills_text


def identity(task='task', episode='episode', session='slot0'):
    return CausalSessionIdentity(session, task, 3, episode,
        CausalModelIdentity('a'*64, 'b'*64, 'c'*64, 'd'*64, 'e'*64))


def bundle(parallel=False, target='cup'):
    row = dict(verb='GRASP', target=target, source='', destination='', target_part='',
        arm='LEFT', unbound_relation='')
    return [row, dict(row, target='plate', arm='RIGHT')] if parallel else [row]


def event(session, *, decision='EXECUTE', parallel=False, target='cup'):
    members = bundle(parallel, target)
    return dict(validated=True, previous_outcome='UNKNOWN', ar_previous_outcome='UNKNOWN',
        task_complete=False, task_complete_claimed=False, decision=decision,
        parent_goal='Task goal: '+session.identity.task, active_skills_semantic_json=canonical_json(members),
        active_skills_text=semantic_active_skills_text(members), memory_update=append_b_memory_idempotent(
            session.memory, session.previous_intent, task_name=session.identity.task))


def issue(s, **kwargs):
    token, causal = s.begin_planning(s.identity, s.feedback.control_step)
    s.stage(s.identity, token, event(s, **kwargs))
    s.commit(s.identity, token)
    return causal


class SessionTests(unittest.TestCase):
    def test_real_counters_same_refresh_retry_and_separate_low_input(self):
        i=identity();s=CausalPlannerSession(i)
        causal=issue(s)
        self.assertEqual(causal['execution_feedback'],'none')
        self.assertEqual(json.loads(s.memory)['issued_command_history'],[])
        s.observe(i,128);causal=issue(s)
        self.assertEqual(len(json.loads(s.memory)['issued_command_history']),1)
        self.assertEqual(json.loads(causal['execution_feedback'])['same_intent_controls'],128)
        s.observe(i,256);causal=issue(s,decision='RETRY')
        feedback=json.loads(causal['execution_feedback'])
        self.assertEqual(feedback['same_intent_controls'],256)
        self.assertEqual(feedback['same_intent_planner_refreshes'],1)
        self.assertEqual(s.feedback.started,256);self.assertEqual(s.feedback.attempt,2)
        self.assertEqual(set(s.low_goal(i)),{'task','parent_goal','semantic_bundle'})

    def test_unready_shadow_does_not_leak_confident_prediction(self):
        i=identity();s=CausalPlannerSession(i);issue(s,parallel=True)
        predictions=[dict(outcome='FAILED',confidence=.99),dict(outcome='SUCCEEDED',confidence=.999)]
        for step in (16,32):
            s.observe(i,step);s.shadow_outcomes(i,step,predictions)
        _,causal=s.begin_planning(i,32)
        feedback=json.loads(causal['execution_feedback'])
        self.assertEqual(feedback['estimated_bundle_outcome'],'UNKNOWN')
        self.assertEqual(causal['known_previous_outcome'],'UNKNOWN')
        self.assertEqual([r['confidence'] for r in feedback['estimated_member_outcomes']],[0.,0.])
        self.assertTrue(feedback['stalled_is_not_failed'])

    def test_slot_task_episode_and_every_model_identity_are_isolated(self):
        i=identity();s=CausalPlannerSession(i);issue(s)
        alternatives=[identity(task='other'),identity(episode='new'),identity(session='slot1'),replace(i,instance=4)]
        alternatives += [replace(i,models=replace(i.models,**{name:'f'*64})) for name in
            ('planner','low','observer_backbone','observer_adapter','normalization')]
        for bad in alternatives:
            with self.assertRaises(ValueError):s.begin_planning(bad,0)
            with self.assertRaises(ValueError):s.observe(bad,16)
            with self.assertRaises(ValueError):s.observer_request(bad,0)
        other=CausalPlannerSession(identity(session='slot1'))
        token,_=s.begin_planning(i,0)
        other_token,_=other.begin_planning(other.identity,0)
        self.assertNotEqual(token,other_token)
        with self.assertRaises(ValueError):other.stage(other.identity,token,event(other))

    def test_staging_rejection_discard_and_delayed_response_are_transactional(self):
        i=identity();s=CausalPlannerSession(i)
        token,causal=s.begin_planning(i,0)
        bad=event(s);bad['memory_update']='not canonical'
        with self.assertRaises(ValueError):s.stage(i,token,bad)
        self.assertEqual(s.revision,0);self.assertIsNone(s.installed)
        self.assertEqual(s.memory,causal['memory'])
        with self.assertRaises(ValueError):s.observe(i,16)
        s.stage(i,token,event(s));s.discard(i,token)
        self.assertIsNone(s.installed);self.assertIsNone(s.feedback.bundle)
        new,_=s.begin_planning(i,0)
        self.assertNotEqual(new,token)
        with self.assertRaises(ValueError):s.stage(i,token,event(s))
        s.stage(i,new,event(s));s.commit(i,new)
        with self.assertRaises(ValueError):s.commit(i,new)

    def test_observer_uses_separate_parent_and_bound_ordered_member_cache(self):
        i=identity();s=CausalPlannerSession(i);issue(s,parallel=True)
        self.assertNotEqual(i.feedback_identity().high_sha256,i.observer_identity().high_sha256)
        for step in (0,16,32,48,64):
            s.observe(i,step)
            for member in range(2):
                token,check=s.observer_request(i,member)
                self.assertEqual(check['served_controls'],step)
                values=s.append_observer_feature(i,member,token,torch.full((8,),float(member+step)),torch.ones(27))
        self.assertEqual(values['steps'].tolist(),[[16,32,48,64]])
        self.assertEqual(float(values['context'][0,-1,0]),65.)
        self.assertEqual(float(s.windows[0].rows[-1][1][0]),64.)
        token,_=s.observer_request(i,0)
        with self.assertRaises(ValueError):s.append_observer_feature(i,1,token,torch.ones(8),torch.ones(27))
        with self.assertRaises(ValueError):s.append_observer_feature(i,0,token,torch.ones(8),torch.ones(27))

    def test_same_command_keeps_history_but_retry_clears_it(self):
        i=identity();s=CausalPlannerSession(i);issue(s)
        token,_=s.observer_request(i,0)
        s.append_observer_feature(i,0,token,torch.ones(8),torch.ones(27))
        s.observe(i,128);old,_=s.observer_request(i,0);issue(s)
        self.assertEqual(len(s.windows[0].rows),1)
        with self.assertRaises(ValueError):s.append_observer_feature(i,0,old,torch.ones(8),torch.ones(27))
        s.observe(i,256);before,_=s.observer_request(i,0);issue(s,decision='RETRY')
        self.assertEqual(len(s.windows[0].rows),0)
        with self.assertRaises(ValueError):s.append_observer_feature(i,0,before,torch.ones(8),torch.ones(27))
        current,check=s.observer_request(i,0)
        self.assertEqual(check['served_controls'],0)
        s.append_observer_feature(i,0,current,torch.ones(8),torch.ones(27))

    def test_same_rgb_clock_cannot_accumulate_fake_history_and_close_drops_cache(self):
        i=identity();s=CausalPlannerSession(i,initial_control_step=270);issue(s)
        token,_=s.observer_request(i,0);s.append_observer_feature(i,0,token,torch.ones(8),torch.ones(27))
        s.observe(i,280);token,_=s.observer_request(i,0)
        with self.assertRaises(ValueError):s.append_observer_feature(i,0,token,torch.ones(8),torch.ones(27))
        with self.assertRaises(ValueError):s.observe(i,269)
        s.close(i,280)
        self.assertEqual(s.windows,[])
        with self.assertRaises(ValueError):s.low_goal(i)
        with self.assertRaises(ValueError):s.observe(i,296)

    def test_explicit_nonphysical_planner_grammar_and_canonical_history(self):
        for field,value in [('previous_outcome','SUCCEEDED'),('ar_previous_outcome','FAILED'),
                ('task_complete',True),('task_complete_claimed',True),('validated',False),('decision','STOP')]:
            s=CausalPlannerSession(identity());token,_=s.begin_planning(s.identity,0)
            bad=event(s);bad[field]=value
            with self.assertRaises(ValueError):s.stage(s.identity,token,bad)
        s=CausalPlannerSession(identity())
        for k,target in enumerate(['cup','plate','radio','bottle','box']):
            s.observe(s.identity,k*128);issue(s,target=target)
        history=json.loads(s.memory)
        self.assertEqual(history['verified_world_facts'],[])
        self.assertEqual(len(history['issued_command_history']),3)
        self.assertNotIn('box',s.memory)


if __name__=='__main__':unittest.main()
