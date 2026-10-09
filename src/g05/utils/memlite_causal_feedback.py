"""Deployment-observable feedback bookkeeping, isolated per episode/model.

This module never reads reward, simulator contacts, annotations, or future
frames. Its text is a NEW opt-in feedback interface, not a silent alteration
of the frozen stage-1 planner prompt. Predictions remain explicitly estimated.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
import re

from g05.utils.memlite_skill_protocol import canonical_json, parse_active_skills_semantic_json

OUTCOMES = {"IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN"}


@dataclass(frozen=True)
class FeedbackIdentity:
    session: str
    task: str
    instance: int
    episode: str
    high_sha256: str
    observer_sha256: str

    def __post_init__(self):
        if (not all(isinstance(s, str) and s for s in (self.session, self.task, self.episode))
                or type(self.instance) is not int or self.instance < 0
                or any(not isinstance(s, str) or not re.fullmatch(r"[a-f0-9]{64}", s) for s in
                       (self.high_sha256, self.observer_sha256))):
            raise ValueError("Unbound feedback identity")


class CausalFeedbackLedger:
    def __init__(self, identity: FeedbackIdentity, *, minimum_confidence=.85, confirmations=2):
        if not isinstance(identity, FeedbackIdentity) or not 0 < minimum_confidence <= 1:
            raise ValueError("Invalid feedback contract")
        if type(confirmations) is not int or confirmations < 2:
            raise ValueError("Require at least two distinct observer checks")
        self.identity = identity
        self.minimum_confidence, self.confirmations = minimum_confidence, confirmations
        self.control_step = 0
        self.bundle = self.parent = None
        self.started = 0
        self.refreshes = 0
        self.attempt = 0
        self._attempts = OrderedDict()
        self._proposals = deque(maxlen=confirmations)

    def _check(self, identity, control_step):
        if identity != self.identity:
            raise ValueError("Cross-session/task/episode/high-model feedback rejected")
        if type(control_step) is not int or control_step < self.control_step:
            raise ValueError("Causal control clock cannot go backward")

    def observe_clock(self, identity, control_step):
        self._check(identity, control_step)
        self.control_step = control_step

    def issued(self, identity, control_step, bundle, parent_goal, decision="EXECUTE"):
        self._check(identity, control_step)
        parsed = parse_active_skills_semantic_json(bundle)
        semantic = canonical_json(parsed)
        if not isinstance(parent_goal, str) or not parent_goal or decision not in {"EXECUTE", "RETRY", "REPLAN"}:
            raise ValueError("Feedback must describe an actually issued valid bundle")
        same = (semantic, parent_goal) == (self.bundle, self.parent)
        # A repeated EXECUTE is not magically a new retry. It must remain
        # visible as an increasing elapsed/refresh count instead of vanishing.
        if same and decision != "RETRY":
            self.refreshes += 1
        else:
            key = (semantic, parent_goal)
            self._attempts[key] = self._attempts.get(key, 0) + 1
            self._attempts.move_to_end(key)
            if len(self._attempts) > 64:
                self._attempts.popitem(last=False)
            self.attempt = self._attempts[key]
            self.started, self.refreshes = control_step, 0
            self._proposals.clear()
        self.bundle, self.parent, self.control_step = semantic, parent_goal, control_step

    def estimated(self, identity, control_step, member_predictions, *, calibrated):
        """Consume only the observer's calibrated probabilities at THIS check.

        Calibration receipt validation is the serving loader's responsibility;
        this local gate cannot itself sign a readiness certificate.
        """
        self._check(identity, control_step)
        if self.bundle is None or control_step != self.control_step:
            raise ValueError("Observer must match the latest real observation clock")
        if self._proposals and control_step <= self._proposals[-1][0]:
            raise ValueError("Duplicate/future observer check cannot satisfy persistence")
        if len(member_predictions) != len(parse_active_skills_semantic_json(self.bundle)):
            raise ValueError("Outcome rows do not match ordered parallel skill members")
        safe = []
        for row in member_predictions:
            if set(row) != {"outcome", "confidence"} or row["outcome"] not in OUTCOMES:
                raise ValueError("Only whitelisted predicted outcome fields are accepted")
            confidence = row["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid observer confidence")
            safe.append(dict(outcome=row["outcome"] if calibrated and confidence >= self.minimum_confidence else "UNKNOWN",
                             confidence=float(confidence), calibrated=bool(calibrated)))
        self._proposals.append((control_step, safe))

    def projection(self, identity, control_step):
        self.observe_clock(identity, control_step)
        if self.bundle is None:
            return "none"
        count = len(parse_active_skills_semantic_json(self.bundle))
        outcomes = []
        for index in range(count):
            rows = [items[index] for _, items in self._proposals]
            fresh = bool(self._proposals) and self._proposals[-1][0] == control_step
            agreed = fresh and len(rows) == self.confirmations and len({r['outcome'] for r in rows}) == 1
            value = rows[-1]['outcome'] if agreed else "UNKNOWN"
            outcomes.append(dict(member=index, estimated_outcome=value,
                confidence=min((r['confidence'] for r in rows), default=0.) if agreed else 0.))
        values = [r['estimated_outcome'] for r in outcomes]
        aggregate = ("SUCCEEDED" if all(v == "SUCCEEDED" for v in values) else
                     "FAILED" if "FAILED" in values else
                     "UNKNOWN" if "UNKNOWN" in values else "IN_PROGRESS")
        return canonical_json(dict(schema="causal_execution_feedback_v1",
            same_intent_controls=control_step-self.started, same_intent_planner_refreshes=self.refreshes,
            attempt_index=self.attempt, estimated_member_outcomes=outcomes,
            estimated_bundle_outcome=aggregate, source="observable_counter_and_learned_observer",
            stalled_is_not_failed=True))


class CausalFeatureWindow:
    """Bounded detached context cache. Identity never becomes an actor feature."""
    def __init__(self, identity, maximum=4):
        if not isinstance(identity, FeedbackIdentity) or type(maximum) is not int or not 1 <= maximum <= 4:
            raise ValueError("Invalid bounded feature window")
        self.identity, self.rows = identity, deque(maxlen=maximum)

    def append(self, identity, control_step, context_hidden, normalized_proprio):
        import torch
        if identity != self.identity or type(control_step) is not int or control_step < 0:
            raise ValueError("Wrong feature-cache identity/clock")
        if self.rows and control_step <= self.rows[-1][0]:
            raise ValueError("Duplicate/backward feature observation")
        if (context_hidden.ndim != 1 or normalized_proprio.shape != (27,)
                or not torch.isfinite(context_hidden).all() or not torch.isfinite(normalized_proprio).all()):
            raise ValueError("Expected one causal context feature and normalized 27D proprio")
        self.rows.append((control_step, context_hidden.detach().clone(), normalized_proprio.detach().clone()))

    def tensors(self, identity, at_control_step):
        import torch
        if identity != self.identity or not self.rows or at_control_step != self.rows[-1][0]:
            raise ValueError("Cannot retrieve another episode's or future context window")
        return dict(context=torch.stack([r[1] for r in self.rows])[None],
                    proprio=torch.stack([r[2] for r in self.rows])[None],
                    steps=torch.tensor([[r[0] for r in self.rows]], device=self.rows[-1][1].device),
                    valid=torch.ones((1, len(self.rows)), dtype=torch.bool, device=self.rows[-1][1].device))
