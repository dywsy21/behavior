"""Transactional, episode-local planner state for the causal-feedback pilot.

This first serving contract deliberately keeps learned results in SHADOW mode:
the high planner receives UNKNOWN/0 plus real command counters, exactly as in
the accepted H1 training recipe. An offline calibration certificate is not a
streaming/deployment certificate. The separately loaded observer uses its OWN
backbone identity; it must never borrow the updated planner's feature cache.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import re

from g05.utils.memlite_causal_feedback import CausalFeedbackLedger, CausalFeatureWindow, FeedbackIdentity
from g05.utils.memlite_skill_protocol import (
    append_b_memory_idempotent, canonical_json, parse_active_skills_semantic_json,
    semantic_active_skills_text, validate_semantic_parent_goal,
)


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("A complete immutable SHA256 is required")
    return value


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class CausalModelIdentity:
    planner: str
    low: str
    observer_backbone: str
    observer_adapter: str
    normalization: str

    def __post_init__(self):
        for value in asdict(self).values():
            _sha(value)


@dataclass(frozen=True)
class CausalSessionIdentity:
    session: str
    task: str
    instance: int
    episode: str
    models: CausalModelIdentity

    def __post_init__(self):
        if not isinstance(self.models, CausalModelIdentity):
            raise ValueError("Explicit separate planner/observer/low model identities required")
        # Reuse the checked episode/model identity grammar without placing
        # any identity, source instance, or hash into model input text.
        self.feedback_identity()

    def feedback_identity(self):
        return FeedbackIdentity(self.session, self.task, self.instance, self.episode,
            self.models.planner, self.models.observer_adapter)

    def observer_identity(self):
        return FeedbackIdentity(self.session, self.task, self.instance, self.episode,
            self.models.observer_backbone, self.models.observer_adapter)


class CausalPlannerSession:
    """One instance per environment/episode; never reset/reuse it in place.

    ``observe`` is called with the actual consumed-control clock, not chunk
    count or wall time. Proposals are bound to one exact input revision and
    clock. A discarded or rejected proposal cannot change memory or feedback.
    The transport must verify actual action ACKs before advancing this clock.
    """
    def __init__(self, identity, *, initial_control_step=0):
        if not isinstance(identity, CausalSessionIdentity):
            raise ValueError("Explicit session identity required")
        if type(initial_control_step) is not int or initial_control_step < 0:
            raise ValueError("Actual initial control clock required")
        self.identity = identity
        self.feedback = CausalFeedbackLedger(identity.feedback_identity(),
            uncertainty_protocol="unready_zero_confidence_v1")
        self.feedback.observe_clock(identity.feedback_identity(), initial_control_step)
        self.memory = canonical_json(dict(task_name=identity.task,
            issued_command_history=[], verified_world_facts=[]))
        self.previous_intent, self.previous_parent_goal = "None", "none"
        self.revision = self.attempt_generation = 0
        self.request_generation = 0
        self.request = self.pending = self.installed = None
        self.windows = []
        self.closed = False

    def _check(self, identity, control_step):
        if identity != self.identity or self.closed:
            raise ValueError("Foreign/closed task, slot, episode or model session")
        if type(control_step) is not int or control_step != self.feedback.control_step:
            raise ValueError("Request does not match the latest real control clock")

    def observe(self, identity, control_step):
        if identity != self.identity or self.closed or self.request is not None:
            raise ValueError("Foreign/closed observation or unresolved planner transaction")
        self.feedback.observe_clock(identity.feedback_identity(), control_step)

    def begin_planning(self, identity, control_step):
        self._check(identity, control_step)
        if self.request is not None:
            raise ValueError("Commit or discard the pending planner request first")
        causal = dict(task_name=identity.task, memory=self.memory,
            previous_parent_goal=self.previous_parent_goal, previous_intent=self.previous_intent,
            known_previous_outcome="UNKNOWN",
            execution_feedback=self.feedback.projection(identity.feedback_identity(), control_step))
        self.request_generation += 1
        token = _digest(dict(identity=asdict(identity), revision=self.revision,
            request_generation=self.request_generation, control_step=control_step, causal=causal))
        self.request = dict(token=token, control_step=control_step, causal=causal)
        return token, deepcopy(causal)

    def stage(self, identity, token, event):
        self._check(identity, self.feedback.control_step)
        if self.request is None or token != self.request['token'] or self.pending is not None:
            raise ValueError("Stale/foreign/already-staged planner response")
        if (event.get('validated') is not True or event.get('previous_outcome') != 'UNKNOWN'
                or event.get('ar_previous_outcome', 'UNKNOWN') != 'UNKNOWN'
                or event.get('task_complete') is not False or event.get('task_complete_claimed') is not False
                or event.get('decision') not in {'EXECUTE', 'RETRY', 'REPLAN'}):
            raise ValueError("Only parsed planner-only nonphysical proposals can be issued")
        parent = validate_semantic_parent_goal(event['parent_goal'], field='parent_goal')
        members = parse_active_skills_semantic_json(event['active_skills_semantic_json'], allow_empty=False)
        if event['active_skills_text'] != semantic_active_skills_text(members):
            raise ValueError("Bundle text/semantic identity differs")
        expected = append_b_memory_idempotent(self.memory, self.previous_intent, task_name=identity.task)
        if event['memory_update'] != expected:
            raise ValueError("Memory must record only previously issued commands")
        self.pending = dict(parent_goal=parent, semantic_bundle=canonical_json(members),
            text=event['active_skills_text'], memory=expected, decision=event['decision'])
        return self.low_goal(identity, staged=True)

    def commit(self, identity, token):
        self._check(identity, self.feedback.control_step)
        if self.request is None or self.request['token'] != token or self.pending is None:
            raise ValueError("No matching staged command to commit")
        next_goal = self.pending
        changed = (self.installed is None or next_goal['decision'] == 'RETRY'
            or (next_goal['parent_goal'], next_goal['semantic_bundle']) !=
               (self.installed['parent_goal'], self.installed['semantic_bundle']))
        self.feedback.issued(identity.feedback_identity(), self.feedback.control_step,
            next_goal['semantic_bundle'], next_goal['parent_goal'], next_goal['decision'])
        self.memory = next_goal['memory']
        self.previous_intent, self.previous_parent_goal = next_goal['text'], next_goal['parent_goal']
        self.installed = next_goal
        self.revision += 1
        self.pending = self.request = None
        if changed:
            self.attempt_generation += 1
            self.windows = [CausalFeatureWindow(identity.observer_identity(), maximum=4)
                for _ in json.loads(next_goal['semantic_bundle'])]
        return self.low_goal(identity)

    def discard(self, identity, token):
        self._check(identity, self.feedback.control_step)
        if self.request is None or token != self.request['token']:
            raise ValueError("No matching planner request to discard")
        self.request = self.pending = None

    def low_goal(self, identity, *, staged=False):
        self._check(identity, self.feedback.control_step)
        goal = self.pending if staged else self.installed
        if goal is None:
            raise ValueError("No actually installed skill for low-level execution")
        return dict(task=identity.task, parent_goal=goal['parent_goal'], semantic_bundle=goal['semantic_bundle'])

    def observer_request(self, identity, member):
        self._check(identity, self.feedback.control_step)
        if self.installed is None or type(member) is not int or not 0 <= member < len(self.windows):
            raise ValueError("Wrong or missing actually issued skill member")
        check = dict(task_name=identity.task, parent_goal=self.installed['parent_goal'],
            issued_bundle=self.installed['semantic_bundle'], member_index=member,
            memory=self.memory, served_controls=self.feedback.control_step-self.feedback.started)
        token = _digest(dict(identity=asdict(identity), generation=self.attempt_generation,
            control_step=self.feedback.control_step, check=check))
        return token, check

    def append_observer_feature(self, identity, member, token, context, proprio):
        expected, _ = self.observer_request(identity, member)
        if token != expected or self.request is not None:
            raise ValueError("Cross-member/model/attempt or stale observer feature")
        window = self.windows[member]
        # Match the trained cadence16 protocol. A final partial chunk can be
        # read by the planner but must not silently change observer history.
        if window.rows and self.feedback.control_step-window.rows[-1][0] < 16:
            raise ValueError("Observer features must be at least 16 real controls apart")
        window.append(identity.observer_identity(), self.feedback.control_step, context, proprio)
        return window.tensors(identity.observer_identity(), self.feedback.control_step)

    def shadow_outcomes(self, identity, control_step, member_predictions):
        self._check(identity, control_step)
        if self.request is not None:
            raise ValueError("Cannot change an in-flight planner input")
        # An offline phase-balanced certificate is NOT permission to use a
        # prediction as a real-time transition, failure or physical success.
        self.feedback.estimated(identity.feedback_identity(), control_step,
            member_predictions, calibrated=False)

    def close(self, identity, control_step):
        self._check(identity, control_step)
        if self.request is not None:
            raise ValueError("Close only after resolving the planner transaction")
        self.closed = True
        self.windows = []
