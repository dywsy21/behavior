"""Per-control reward -> same-skill semi-Markov PPO transitions.

Never splice different tasks, attempts, or policy versions into one GAE chain.
An early-completed action chunk contains only the controls actually ACKed by
the simulator. Time limits bootstrap but do not recurse across a reset.
"""
from dataclasses import asdict
import math

from skill_aligned_reward import SkillIdentity


def scalar(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid finite '+name)
    return float(value)


class SkillRollout:
    def __init__(self,identity,*,policy_version,policy_sha256,control_step,lambda_per_control=.998):
        if (not isinstance(identity,SkillIdentity) or type(policy_version) is not int or policy_version<0
                or not isinstance(policy_sha256,str) or len(policy_sha256)!=64
                or any(c not in '0123456789abcdef' for c in policy_sha256)
                or type(control_step) is not int or control_step<0):
            raise ValueError('Unbound on-policy short-skill episode')
        self.identity=identity;self.version=policy_version;self.sha=policy_sha256;self.step=control_step
        self.trace_lambda=scalar(lambda_per_control,'per-control trace lambda')
        if not 0<self.trace_lambda<=1:raise ValueError('Invalid trace decay')
        self.pending=None;self.records=[];self.used=set();self.ended=False

    def begin_chunk(self,*,experience_id,old_value,policy_version,policy_sha256,control_step,max_controls=16):
        if self.ended or self.pending is not None or policy_version!=self.version or policy_sha256!=self.sha:
            raise ValueError('Ended episode, pending chunk or changed behavior policy')
        if (type(experience_id) is not int or experience_id<0 or experience_id in self.used
                or control_step!=self.step or type(control_step) is not int
                or type(max_controls) is not int or not 1<=max_controls<=16):
            raise ValueError('Reused experience, wrong control clock or action chunk')
        self.used.add(experience_id)
        self.pending=dict(experience_id=experience_id,old_value=scalar(old_value,'old value'),
            start_control=self.step,max_controls=max_controls,controls=0,reward=0.,discount=1.,
            terminated=False,truncated=False)

    def acknowledge(self,row):
        p=self.pending
        if p is None or p['terminated'] or p['truncated'] or p['controls']>=p['max_controls']:
            raise ValueError('No unconsumed action in this attempt')
        identity=row['identity']
        if isinstance(identity,SkillIdentity):identity=asdict(identity)
        if identity!=asdict(self.identity) or type(row['control_step']) is not int or row['control_step']!=self.step+1:
            raise ValueError('Cross-task/attempt reward or missing control ACK')
        for key in ('terminated','truncated','bootstrap'):
            if type(row[key]) is not bool:raise ValueError('Unknown transition boundary')
        if row['terminated'] and row['truncated']:raise ValueError('Ambiguous terminal/truncated boundary')
        gamma=scalar(row['discount'],'physical control discount')
        if ((row['terminated'] and (gamma!=0 or row['bootstrap']))
                or (not row['terminated'] and (not 0<gamma<=1 or not row['bootstrap']))):
            raise ValueError('Terminal/truncation bootstrap mismatch')
        p['reward']+=p['discount']*scalar(row['reward'],'physical control reward')
        p['discount']*=gamma;p['controls']+=1;self.step+=1
        p.update(terminated=row['terminated'],truncated=row['truncated'])

    def finish_chunk(self,*,next_value,observation_control_step):
        p=self.pending
        if p is None or not p['controls'] or observation_control_step!=self.step:
            raise ValueError('Bootstrap must observe the actual post-ACK state')
        if not (p['terminated'] or p['truncated']) and p['controls']!=p['max_controls']:
            raise ValueError('Unacknowledged part of an active action chunk')
        value=scalar(next_value,'next value')
        if p['terminated'] and value!=0:raise ValueError('True terminal must bootstrap zero')
        row=dict(p,end_control=self.step,next_value=value,identity=asdict(self.identity),
                 policy_version=self.version,policy_sha256=self.sha,
                 trace_discount=p['discount']*self.trace_lambda**p['controls'])
        self.records.append(row);self.pending=None;self.ended=p['terminated'] or p['truncated']
        return row

    def advantages(self):
        if self.pending is not None or not self.records:raise ValueError('No completed on-policy chunks')
        advantage=0.;result=[]
        for i in reversed(range(len(self.records))):
            row=self.records[i]
            delta=row['reward']+row['discount']*row['next_value']-row['old_value']
            contiguous=(i+1<len(self.records) and not row['terminated'] and not row['truncated'])
            if contiguous:
                nxt=self.records[i+1]
                if (row['end_control']!=nxt['start_control'] or row['identity']!=nxt['identity']
                        or row['policy_sha256']!=nxt['policy_sha256'] or row['policy_version']!=nxt['policy_version']
                        or not math.isclose(row['next_value'],nxt['old_value'],abs_tol=1e-5,rel_tol=1e-5)):
                    raise ValueError('GAE does not describe contiguous same-policy, same-skill states')
            advantage=delta+(row['trace_discount']*advantage if contiguous else 0.)
            result.append(dict(experience_id=row['experience_id'],advantage=advantage,
                               returns=advantage+row['old_value'],actual_controls=row['controls']))
        return list(reversed(result))
