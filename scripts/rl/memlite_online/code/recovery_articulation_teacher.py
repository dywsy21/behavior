"""Offline measured-state reference-path servo, and causal articulation evidence.

No simulator state setter and no deployed actor dependency. This is a local
corrective candidate generator, never an automatic demonstration approval.
"""
from collections import deque
import math
import numpy as np


POSITIONS=list(range(3,14))+list(range(15,22))


def yaw_of(quat):
    x,y,z,w=map(float,quat)
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi


def servo(current,base,waypoint):
    """R1Pro absolute joints + body-frame normalized base velocity (0.75 m/s)."""
    current=np.asarray(current,float);base=np.asarray(base,float)
    goal=np.asarray(waypoint['q'],float);target=np.asarray(waypoint['base'],float)
    if current.shape!=(23,) or goal.shape!=(23,) or base.shape!=(3,) or target.shape!=(3,):
        raise ValueError('Invalid R1Pro waypoint dimensions')
    if not np.isfinite(np.r_[current,goal,base,target]).all():raise ValueError('Nonfinite servo state')
    world=target[:2]-base[:2];heading=wrap(target[2]-base[2]);c,s=math.cos(base[2]),math.sin(base[2])
    local=np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1]])
    action=goal.copy();action[POSITIONS]=current[POSITIONS]+np.clip(goal[POSITIONS]-current[POSITIONS],-.015,.015)
    action[:2]=np.clip(2*local/.75,-.15,.15);action[2]=np.clip(2*heading,-.2,.2)
    action[[14,22]]=np.clip(goal[[14,22]],-1,1)
    error=float(np.max(np.abs(goal[POSITIONS]-current[POSITIONS])))
    return action.astype(np.float32),dict(joint_error=error,base_distance=float(np.linalg.norm(world)),
        heading_error=abs(heading),reached=bool(error<.025 and np.linalg.norm(world)<.01 and abs(heading)<.02))


class CausalArticulation:
    """Known goal loss after verified attainment; not timeout/segment-end labels."""
    def __init__(self,stable=12):
        self.stable=stable;self.last=-1;self.true_streak=0;self.false_streak=0
        self.seen_false=False;self.achieved=False;self.retry=False

    def update(self,tick,predicate,*,retry=False,moving=False):
        if type(tick) is not int or tick!=self.last+1:raise ValueError('Nonconsecutive physical clock')
        self.last=tick
        if retry:self.retry=True;self.true_streak=0
        if predicate is None:
            self.true_streak=self.false_streak=0
            return 'UNKNOWN'
        if type(predicate) is not bool:raise ValueError('Predicate must be measured bool or None')
        self.true_streak=self.true_streak+1 if predicate else 0
        self.false_streak=self.false_streak+1 if not predicate else 0
        self.seen_false=self.seen_false or not predicate
        if self.seen_false and self.true_streak>=self.stable:
            self.achieved=True;return 'SUCCEEDED'
        if self.achieved and self.false_streak>=self.stable and not self.retry:return 'FAILED'
        return 'IN_PROGRESS' if moving else 'UNKNOWN'


class StationaryServo:
    def __init__(self):self.states=deque(maxlen=64);self.commands=deque(maxlen=64)
    def observe(self,q,base,object_joints,action):
        self.states.append(np.r_[q,base,object_joints]);self.commands.append(np.asarray(action))
        return (len(self.states)==64 and np.ptp(np.stack(self.states),axis=0).max()<1e-3
                and np.ptp(np.stack(self.commands),axis=0).max()<1e-3)
