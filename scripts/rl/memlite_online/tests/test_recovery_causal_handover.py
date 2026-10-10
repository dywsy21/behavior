from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest
from recovery_causal_handover import teacher_prefix_projection,restore_teacher_prefix,AppliedActionClock
from g05.utils.memlite_causal_session import CausalModelIdentity,CausalSessionIdentity
from g05.utils.memlite_skill_protocol import canonical_json,append_b_memory_idempotent,semantic_active_skills_text


def fixture():
    identity=CausalSessionIdentity('slot','some task',3,'new-joint-episode',
        CausalModelIdentity('a'*64,'b'*64,'c'*64,'d'*64,'e'*64,'f'*64,'e'*64))
    members=[dict(verb='GRASP',target='cup',source='',destination='',target_part='',arm='LEFT',unbound_relation='')]
    memory=canonical_json(dict(task_name=identity.task,issued_command_history=[],verified_world_facts=[]))
    previous='None';plans=[]
    for tick,decision in [(0,'EXECUTE'),(32,'RETRY'),(160,'EXECUTE')]:
        updated=append_b_memory_idempotent(memory,previous,task_name=identity.task)
        event=dict(control_step=tick,decision=decision,parent_goal='Task goal: '+identity.task,
            active_skills_semantic_json=canonical_json(members),active_skills_text=semantic_active_skills_text(members),
            memory_before=memory,memory_update=updated,previous_intent=previous,
            source='offline_verified_local_teacher_proposal',not_success_label=True)
        event['event_sha256']=digest(event);plans.append(event)
        memory,previous=updated,event['active_skills_text']
    rows=[]
    for step in range(180):
        event=[p for p in plans if p['control_step']<=step][-1]
        rows.append(dict(control_step=step,simulator_apply_ack=True,action_executed_raw23=[0.]*23,
            terminated=False,truncated=False,context=dict(context_id=event['event_sha256'],
                parent_goal=event['parent_goal'],active_skills_semantic_json=event['active_skills_semantic_json']),
            physical_audit=dict(grasp=True),outcome_target='SUCCEEDED',teacher={'future':'must not leak'}))
    return identity,plans,rows


def restore(identity,plans,rows,tick):
    projection=teacher_prefix_projection(plans,rows,at_control_step=tick)
    session,receipt=restore_teacher_prefix(identity,projection,source_task='some_task',source_instance=3)
    return session,receipt,projection


class HandoverTests(unittest.TestCase):
    def test_explicit_calibrated_factory_is_fresh_and_bound(self):
        from g05.utils.memlite_causal_session import CausalPlannerSession
        from recovery_calibrated_observer import GraspCalibrationBinding,CalibratedGraspPilotSession
        i,plans,rows=fixture();projection=teacher_prefix_projection(plans,rows,at_control_step=32)
        binding=GraspCalibrationBinding(i.models,'0'*64,1.77)
        session,_=restore_teacher_prefix(i,projection,source_task='some_task',source_instance=3,
            session_factory=lambda identity:CalibratedGraspPilotSession(identity,calibration=binding))
        self.assertEqual(session.observer_feedback_mode,'calibrated_grasp_estimate_pilot_v1')
        self.assertEqual(session.feedback.control_step,32)
        self.assertEqual(len(session.feedback._proposals),0)
        for factory in (lambda _:session,lambda _:CausalPlannerSession(replace(i,episode='foreign')),
                        lambda _:object()):
            with self.assertRaises(ValueError):restore_teacher_prefix(i,projection,
                source_task='some_task',source_instance=3,session_factory=factory)

    def test_predecision_never_restores_current_target_or_future_memory(self):
        i,plans,rows=fixture();s,receipt,projection=restore(i,plans,rows,32)
        self.assertEqual(s.feedback.attempt,1);self.assertEqual(s.feedback.started,0)
        self.assertEqual(s.feedback.control_step,32);self.assertEqual(s.revision,1)
        self.assertTrue(s.takeover_pending);self.assertTrue(s.planning_due(i))
        self.assertEqual(json.loads(s.memory)['issued_command_history'],[])
        self.assertFalse(receipt['historical_commands_generated_by_destination_model'])
        self.assertEqual(receipt['future_commands_restored'],0)
        self.assertEqual([len(w.rows) for w in s.windows],[0])
        for forbidden in ('physical_audit','outcome_target','teacher','action_executed_raw23','RETRY'):
            self.assertNotIn('"'+forbidden+'"',json.dumps(projection))
        # Future corruption cannot supply a recovery target at the handover.
        plans[1]['decision']='future invalid';rows[32]['context']={}
        self.assertEqual(teacher_prefix_projection(plans,rows,at_control_step=32),projection)

    def test_only_previously_executed_retry_changes_attempt_clock(self):
        i,plans,rows=fixture();s,_,_=restore(i,plans,rows,176)
        self.assertEqual(s.feedback.attempt,2);self.assertEqual(s.feedback.started,32)
        self.assertEqual(s.feedback.refreshes,1)
        self.assertEqual(json.loads(s.feedback.projection(i.feedback_identity(),176))['same_intent_controls'],144)
        self.assertEqual(len(json.loads(s.memory)['issued_command_history']),1)

    def test_wrong_task_instance_holes_unacked_or_terminal_prefix_rejected(self):
        i,plans,rows=fixture();_,_,projection=restore(i,plans,rows,32)
        for kwargs in [dict(source_task='other',source_instance=3),dict(source_task='some_task',source_instance=4)]:
            with self.assertRaises(ValueError):restore_teacher_prefix(i,projection,**kwargs)
        for field,value in [('control_step',4),('simulator_apply_ack',False),('terminated',True)]:
            bad=deepcopy(rows);bad[3][field]=value
            with self.assertRaises(ValueError):teacher_prefix_projection(plans,bad,at_control_step=32)
        bad=deepcopy(rows);bad[3]['context']['context_id']='f'*64
        with self.assertRaises(ValueError):teacher_prefix_projection(plans,bad,at_control_step=32)
        with self.assertRaises(ValueError):teacher_prefix_projection(plans[1:],rows,at_control_step=32)

    def test_projection_tampering_and_extra_physics_fail_closed(self):
        i,plans,rows=fixture();_,_,projection=restore(i,plans,rows,32)
        for mutate in [lambda p:p.update(physical_audit={}),
                       lambda p:p['events'][0]['event'].update(physical_audit={}),
                       lambda p:p['events'][0]['event'].update(parent_goal='Task goal: another goal'),
                       lambda p:p['events'][0].update(control_step=32),
                       lambda p:p['consumed_controls'][2].update(control_step=0),
                       lambda p:p['events'][0].update(memory_before='fake')]:
            bad=deepcopy(projection);mutate(bad)
            with self.assertRaises(ValueError):restore_teacher_prefix(i,bad,source_task='some_task',source_instance=3)

    def test_partial_ack_advances_real_count_and_blocks_inflight_planning(self):
        i,plans,rows=fixture();s,_,_=restore(i,plans,rows,32)
        clock=AppliedActionClock(s);actions=np.zeros((16,23),dtype=np.float32)
        token=clock.offer(i,actions)
        with self.assertRaises(ValueError):s.begin_planning(i,32)
        with self.assertRaises(ValueError):s.observe(i,48)
        with self.assertRaises(ValueError):s.close(i,32)
        with self.assertRaises(ValueError):AppliedActionClock(s).offer(i,actions)
        ack=[dict(control_step=32+k,simulator_apply_ack=True,action_executed_raw23=actions[k].tolist()) for k in range(3)]
        result=clock.acknowledge(i,token,ack)
        self.assertEqual(result['control_step'],35);self.assertEqual(result['unused_controls'],13)
        self.assertEqual(s.feedback.control_step,35)
        with self.assertRaises(ValueError):clock.acknowledge(i,token,ack)
        with self.assertRaises(ValueError):clock.offer(i,np.zeros((32,27)))

    def test_foreign_or_wrong_applied_action_ack_does_not_mutate(self):
        i,plans,rows=fixture();s,_,_=restore(i,plans,rows,32)
        clock=AppliedActionClock(s);token=clock.offer(i,np.zeros((16,23)))
        ack=[dict(control_step=32,simulator_apply_ack=True,action_executed_raw23=[0.]*23)]
        with self.assertRaises(ValueError):clock.acknowledge(replace(i,episode='other'),token,ack)
        for mutate in [lambda x:x[0].update(control_step=33),lambda x:x[0].update(simulator_apply_ack=False),
                       lambda x:x[0]['action_executed_raw23'].__setitem__(0,1.),lambda x:x[0].update(reward=1)]:
            bad=deepcopy(ack);mutate(bad)
            with self.assertRaises(ValueError):clock.acknowledge(i,token,bad)
            self.assertEqual(s.feedback.control_step,32);self.assertEqual(s.action_in_flight,token)
        clock.acknowledge(i,token,ack)


if __name__=='__main__':unittest.main()
