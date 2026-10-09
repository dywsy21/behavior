"""Same-state joint feedback near an actually demonstrated grasp pose.

This is a restricted OFFLINE DART-inspired teacher, not the expert demonstrator
and not a deployment actor. It refuses moved targets instead of claiming that
old actions remain correct on arbitrary disturbed states. Physical execution
and human review, never its self-report, decide dataset admission.
"""
import numpy as np
from collections import deque


class NonGraspingFixedPoint:
    """Stationary command AND measured robot/object, not an elapsed-time quota."""
    def __init__(self,window=64):
        self.window=window;self.commands=deque(maxlen=window);self.states=deque(maxlen=window)

    def observe(self,command,q,position,aperture,grasp):
        if any(value!='FALSE' for value in grasp.values()):
            self.commands.clear();self.states.clear();return False
        self.commands.append(np.asarray(command,dtype=np.float32))
        self.states.append(np.r_[q,position,aperture].astype(np.float32))
        if len(self.states)<self.window:return False
        spread=np.ptp(np.stack(self.states),axis=0)
        return bool(np.ptp(np.stack(self.commands),axis=0).max()<1e-6
                    and spread[:23].max()<1e-3 and spread[23:26].max()<1e-3 and spread[26]<5e-4)


class LocalGraspTeacher:
    def __init__(self, target_q, target_position, *, arm, arm_indices, gripper_index):
        self.target_q = np.asarray(target_q,dtype=np.float32).copy()
        self.target_position = np.asarray(target_position,dtype=np.float32).copy()
        self.arm = arm; self.indices = np.asarray(arm_indices,dtype=int)
        self.gripper = int(gripper_index)
        if (self.target_q.shape != (23,) or self.target_position.shape != (3,)
                or arm not in ('left','right') or len(self.indices) != 7
                or self.gripper not in (14,22) or not np.isfinite(self.target_q).all()):
            raise ValueError('Wrong local teacher embodiment')
        self.stage = 'APPROACH_OPEN'

    def action(self, current_q, target_position, aperture, held):
        current = np.asarray(current_q,dtype=np.float32)
        position = np.asarray(target_position,dtype=np.float32)
        if (current.shape != (23,) or position.shape != (3,)
                or not np.isfinite(np.r_[current,position,aperture]).all()
                or held not in ('TRUE','FALSE','UNKNOWN')):
            raise ValueError('Invalid measured teacher state')
        if held == 'UNKNOWN': raise ValueError('Unknown grasp state: no corrective proposal')
        displacement = float(np.linalg.norm(position-self.target_position))
        if displacement > .025:
            raise ValueError('Target left verified local grasp neighborhood; needs another teacher')
        error = float(np.max(np.abs(current[self.indices]-self.target_q[self.indices])))
        if held == 'TRUE': self.stage = 'HOLD'
        elif self.stage == 'HOLD': self.stage = 'APPROACH_OPEN'
        if self.stage == 'APPROACH_OPEN' and error < .012 and aperture >= .048:
            self.stage = 'CLOSE'
        action = self.target_q.copy()
        # Rate-limited ABSOLUTE joint positions, recalculated at this state.
        positions = list(range(3,14))+list(range(15,22))
        action[positions] = current[positions]+np.clip(self.target_q[positions]-current[positions],-.015,.015)
        action[:3] = 0
        action[self.gripper] = 1. if self.stage == 'APPROACH_OPEN' else -1.
        return action, dict(teacher='local_measured_joint_servo_v1',stage=self.stage,
                            joint_error_max_rad=error,target_displacement_m=displacement,
                            uses_privileged_target_guard=True,not_actor_input=True,
                            action_quality_approved=False)


def perturb(action, *, arm_indices, gripper_index, rng, kind):
    """Return intended / sampled / executed separately, including clipping."""
    intended = np.asarray(action,dtype=np.float32)
    sampled = np.zeros(23,dtype=np.float32)
    if kind not in ('open_gripper','open_gripper_joint_jitter'): raise ValueError('Unknown intervention')
    sampled[gripper_index] = 2.
    if kind == 'open_gripper_joint_jitter': sampled[np.asarray(arm_indices)] = rng.normal(0,.012,7)
    executed = intended+sampled
    executed[[14,22]] = np.clip(executed[[14,22]],-1.,1.)
    return sampled,executed
