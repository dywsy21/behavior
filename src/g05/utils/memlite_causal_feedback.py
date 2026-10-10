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

from g05.utils.memlite_skill_protocol import (canonical_json, parse_active_skills_semantic_json,
    semantic_active_skills_text, validate_b_memory_text, validate_semantic_parent_goal)

OUTCOMES = {"IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN"}


def single_frame_planner_prefix(builder, prepared, causal):
    """Build a serving prefix without ever calling the teacher-forcing builder.

    ``causal`` must already be the six-field deployment input projection.
    Refuse extra fields rather than silently dropping a caller's answer/audit.
    Images and normalized proprio are copied individually from the processor.
    """
    from g05.utils.memlite_skill_protocol import V6_PLANNER_INPUT_FIELDS
    if set(causal) != set(V6_PLANNER_INPUT_FIELDS):
        raise ValueError('Planner inference requires the exact causal input whitelist')
    if (builder.num_input_images != 3 or tuple(builder._image_sizes) !=
            ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')
            or prepared.get('_instructions') != causal['task_name']):
        raise ValueError('Planner task/camera mismatch')
    if causal['known_previous_outcome'] not in OUTCOMES:
        raise ValueError('Invalid known previous outcome')
    validate_b_memory_text(causal['memory'], task_name=causal['task_name'])
    validate_semantic_parent_goal(causal['previous_parent_goal'], field='previous_parent_goal', allow_none=True)
    if builder.template.count('<EOC>') != 1:
        raise ValueError('Ambiguous planner prefix boundary')
    result = dict(causal, template=builder.template.split('<EOC>', 1)[0] + '<EOC>',
        command=causal['task_name'], known_previous_outcome='Known previous outcome: ' + causal['known_previous_outcome'],
        planner_prompt='First report the previous outcome only when it is evidenced; then output '
                       'the decision, complete active-skills JSON bundle, memory update, and task completion.',
        schema_version=6, memlite_schema_version=6, memlite_branch='high', memlite_causal_prompt=True,
        embodiment=builder.embodiment_type,
        proprio=dict(value=prepared['proprio'], proprio_dim_is_pad=prepared['proprio_dim_is_pad']))
    for index, camera in enumerate(builder._image_sizes):
        result[f'image{index}'] = builder._image_sizes[camera]
    return result


def last_context_hidden(hidden, modality_mask):
    """Gather the last non-padding position, NOT the sum of modality IDs.

    G0.5 masks encode IMAGE=1/PROPRIO=2/TEXT=4/etc., not binary attention.
    Taking their numeric sum causes an out-of-bounds CUDA gather. Explicit
    positions also support left/right padding and an internal masked history.
    """
    import torch
    if hidden.ndim != 3 or modality_mask.shape != hidden.shape[:2] or hidden.shape[1] == 0:
        raise ValueError('Unaligned/empty causal hidden states')
    positions = torch.arange(hidden.shape[1],device=hidden.device).expand(hidden.shape[0],-1)
    positions = positions.masked_fill(modality_mask == 0,-1).amax(dim=1)
    if (positions < 0).any():
        raise ValueError('Outcome prefix has no non-padding context token')
    return hidden[torch.arange(hidden.shape[0],device=hidden.device),positions]


def single_frame_member_prefix(builder, prepared, *, task_name, parent_goal, issued_bundle,
                               member_index, memory, served_controls):
    """Target-free 3-camera observer prefix; does NOT call a training builder.

    The legacy outcome helper is strictly 18-image/six-frame. This separate
    version keeps its contract unchanged and handles the real single-frame
    Stage-1 parent. All caller inputs are deployment-observable, no answer
    fields or physical audit dictionaries are copied from ``prepared``.
    """
    if (builder.num_input_images != 3 or tuple(builder._image_sizes) !=
            ('head_rgb','left_wrist_rgb','right_wrist_rgb')
            or prepared.get('_instructions') != task_name):
        raise ValueError('Single-frame member prefix requires matching task and exactly three cameras')
    members = parse_active_skills_semantic_json(issued_bundle)
    if type(member_index) is not int or not 0 <= member_index < len(members):
        raise ValueError('Invalid parallel member index')
    if type(served_controls) is not int or served_controls < 0:
        raise ValueError('Real served-control count required')
    validate_semantic_parent_goal(parent_goal, field='previous_parent_goal')
    validate_b_memory_text(memory, task_name=task_name)
    template = builder.template
    if template.count('<EOC>') != 1:
        raise ValueError('Ambiguous causal template boundary')
    result = dict(template=template.split('<EOC>',1)[0]+'<EOC>',
        command=task_name,task_name=task_name,previous_parent_goal=parent_goal,
        previous_intent=semantic_active_skills_text([members[member_index]]),memory=memory,
        known_previous_outcome='Known previous outcome: UNKNOWN',
        execution_feedback=f'served_action_count={served_controls}; observer_scope=one_previous_skill_member',
        planner_prompt='Assess only this previous skill member from observable past and current evidence.',
        schema_version=6,memlite_schema_version=6,memlite_causal_prompt=True,
        embodiment=builder.embodiment_type,
        proprio=dict(value=prepared['proprio'],proprio_dim_is_pad=prepared['proprio_dim_is_pad']))
    for index,camera in enumerate(builder._image_sizes):
        result[f'image{index}'] = builder._image_sizes[camera]
    return result


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
    def __init__(self, identity: FeedbackIdentity, *, minimum_confidence=.85, confirmations=2,
                 uncertainty_protocol='raw_observer_confidence_v1'):
        if not isinstance(identity, FeedbackIdentity) or not 0 < minimum_confidence <= 1:
            raise ValueError("Invalid feedback contract")
        if type(confirmations) is not int or not 2 <= confirmations <= 8:
            raise ValueError("Require two to eight distinct observer checks")
        if uncertainty_protocol not in ('raw_observer_confidence_v1', 'unready_zero_confidence_v1'):
            raise ValueError('Unknown observer uncertainty input contract')
        self.identity = identity
        self.uncertainty_protocol = uncertainty_protocol
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
        if type(calibrated) is not bool:
            raise ValueError("Calibration readiness must be a verified boolean")
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
            usable_confidence = (0. if not calibrated and self.uncertainty_protocol == 'unready_zero_confidence_v1'
                                 else float(confidence))
            safe.append(dict(outcome=row["outcome"] if calibrated and confidence >= self.minimum_confidence else "UNKNOWN",
                             confidence=usable_confidence, calibrated=bool(calibrated)))
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
            attempt_count_scope="last_64_distinct_intents_this_episode",
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
