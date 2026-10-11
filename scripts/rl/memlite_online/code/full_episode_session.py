"""Read-only fresh-start session; actual simulator ACKs own the control clock."""
from copy import deepcopy
from dataclasses import asdict
import json

from recovery_causal_handover import AppliedActionClock
from recovery_calibrated_observer import CalibratedGraspPilotSession
from skill_training_protocol import actor_observation, observation_hash


class FullEpisodeSession:
    def __init__(self, identity, *, calibration, planner, low, observer_factory, emit):
        self.session = CalibratedGraspPilotSession(identity, calibration=calibration)
        self.clock = AppliedActionClock(self.session)
        self.planner, self.low, self.emit = planner, low, emit
        self.observer = observer_factory(self.session)
        self.last_observer = None
        self.last_plan = None
        self.emit('begin', dict(identity=asdict(identity), initial_control_step=0,
            teacher_prefix_controls=0, memory=self.session.memory))

    def observe(self, observation):
        s = self.session
        if s.installed is None:
            return
        result = self.observer.check(s.identity, observation)
        if result.get('members'):
            self.last_observer = deepcopy(result)
        self.emit('observer', dict(result=result, control_step=s.feedback.control_step,
            revision=s.revision, attempt_generation=s.attempt_generation))

    def infer(self, identity, observation, sample):
        s = self.session
        s._check(identity, s.feedback.control_step)
        if self.clock.pending is not None:
            raise ValueError('Do not infer before actual actions are acknowledged')
        actor_observation(observation)
        self.observe(observation)
        plan = self.planner.ensure_context(s, identity, observation,
            validate_low_goal=lambda goal: self.low.prepare(observation, goal), interval_controls=128)
        if not plan['reused']:
            self.last_plan = deepcopy(plan)
        self.emit('plan', dict(result=plan, observation_sha256=observation_hash(observation)))
        if not plan['reused']:
            self.observe(observation)
        if self.last_observer is None or self.last_observer['attempt_generation'] != s.attempt_generation:
            raise ValueError('Current intent needs its own actual observer output')
        raw = sample(observation, plan['goal'], s.feedback.control_step)
        token = self.clock.offer(identity, raw)
        display = dict(control_step=s.feedback.control_step, revision=s.revision,
            attempt_generation=s.attempt_generation, goal=plan['goal'],
            decision=self.last_plan['event']['decision'], observer=deepcopy(self.last_observer))
        self.emit('action', dict(token=token, display=display,
            raw_actions=self.clock.pending['actions'], observation_sha256=observation_hash(observation)))
        return dict(token=token, action_chunk=raw, display=display)

    def acknowledge(self, identity, token, controls):
        result = self.clock.acknowledge(identity, token, controls)
        self.emit('ack', dict(token=token, **result))
        return result

    def close(self, identity, observation):
        s = self.session
        s._check(identity, s.feedback.control_step)
        if self.clock.pending is not None:
            raise ValueError('Final partial chunk must be acknowledged before close')
        actor_observation(observation)
        self.observe(observation)
        s.close(identity, s.feedback.control_step)
        result = dict(control_step=s.feedback.control_step, revisions=s.revision,
            episode=identity.episode, optimizer_updates=0)
        self.emit('close', result)
        return result
