"""Frozen, episode-bound streaming observer diagnostics; never deployment.

The caller owns loading/SHA verification and the actual applied-action clock.
One lightweight wrapper belongs to ONE CausalPlannerSession; only the frozen
model/processor/head may be shared. No label, reward or privileged state is
accepted. All predictions stay outside the planner's UNKNOWN/0 interface.
"""
from contextlib import nullcontext
from copy import deepcopy

from recovery_causal_inference import configuration_stats_sha256, observer_prefix
from skill_training_protocol import observation_hash


OUTCOMES = ('IN_PROGRESS', 'SUCCEEDED', 'FAILED', 'UNKNOWN')


class BoundObserverShadowInference:
    feedback_mode = 'shadow_unknown_v1'

    def __init__(self, policy, processor, config, head, session, *,
                 loaded_backbone_sha256, loaded_adapter_sha256,
                 history_protocol='cadence16_v1', device='cuda'):
        self.session = session
        self.identity = session.identity
        models = self.identity.models
        if (loaded_backbone_sha256 != models.observer_backbone
                or loaded_adapter_sha256 != models.observer_adapter
                or configuration_stats_sha256(config) != models.observer_normalization
                or history_protocol != 'cadence16_v1'
                or policy.training or head.training
                or any(p.requires_grad for model in (policy, head) for p in model.parameters())
                or not head.include_absolute_proprio or head.include_served_controls):
            raise ValueError('Require exact separately loaded frozen cadence16 observer, not planner features')
        self.policy, self.processor, self.config, self.head = policy, processor, config, head
        self.device = device
        self.last = None

    def prediction(self, logits, member):
        confidence, predicted = logits.softmax(-1).max(-1)
        return dict(outcome=OUTCOMES[int(predicted)], confidence=float(confidence))

    def commit_predictions(self, identity, step, predictions):
        self.session.shadow_outcomes(identity, step, predictions)

    def feedback_metadata(self):
        return dict(observer_feedback_mode=self.feedback_mode, entered_planner=False,
                    calibration_applied=False, status='shadow_only_observer_check')

    def check(self, identity, observation):
        """All members commit together; failed prefill/head leaves no history.

        Duplicate value/action RPCs cannot manufacture two observations or
        two confirmations. A shorter final partial chunk does not silently
        switch from the trained cadence16 to the rejected short-history arm.
        """
        import torch
        session = self.session
        step = session.feedback.control_step
        session._check(identity, step)
        if (identity != self.identity or session.request is not None
                or session.action_in_flight is not None or session.installed is None):
            raise ValueError('Observer needs its own committed intent and an ACKed observation')
        digest = observation_hash(observation)  # Strict observable-only projection.
        generation = session.attempt_generation
        if self.last is not None and self.last['generation'] == generation and self.last['control_step'] == step:
            if self.last['observation_sha256'] != digest:
                raise ValueError('Different RGB/proprio at the same actual observation clock')
            return dict(status='duplicate_check_not_recomputed', control_step=step,
                observer_feedback_mode=self.feedback_mode, entered_planner=False)
        endings = [w.rows[-1][0] if w.rows else None for w in session.windows]
        if not endings or len(set(endings)) != 1:
            raise ValueError('All members require aligned, separately conditioned feature windows')
        if endings[0] is not None and step - endings[0] < 16:
            return dict(status='partial_chunk_skipped_by_original_cadence', control_step=step,
                observer_feedback_mode=self.feedback_mode, entered_planner=False)
        tokens, proposed, predictions, evidence = [], deepcopy(session.windows), [], []
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device == 'cuda' else nullcontext()
        with torch.inference_mode(), amp:
            for member, window in enumerate(proposed):
                token, prefix, pixels = observer_prefix(self.processor, self.config, session,
                    identity, member, observation, loaded_backbone_sha256=identity.models.observer_backbone)
                pixels = {key: value.unsqueeze(0).to(self.device) for key, value in pixels.items()}
                context = self.policy.outcome_context_from_prefix([prefix], pixels)
                if context.ndim != 2 or context.shape[0] != 1:
                    raise ValueError('Exactly one conditioned context per issued member')
                window.append(identity.observer_identity(), step, context[0],
                    prefix['proprio']['value'].reshape(27).to(self.device))
                values = window.tensors(identity.observer_identity(), step)
                logits = self.head(**values).float()
                if logits.shape != (1, 4) or not torch.isfinite(logits).all():
                    raise ValueError('Finite four-class shadow logits required')
                confidence, predicted = logits[0].softmax(-1).max(-1)
                uncalibrated = dict(outcome=OUTCOMES[int(predicted)], confidence=float(confidence))
                predictions.append(self.prediction(logits[0], member))
                evidence.append(dict(member=member, raw_logits=logits[0].cpu().tolist(),
                    uncalibrated_prediction=uncalibrated, context_steps=values['steps'][0].cpu().tolist(),
                    request_token=token))
                tokens.append(token)
        # Revalidate after every expensive call, before either history or
        # feedback changes. No cross-attempt or partially completed bundle.
        session._check(identity, step)
        if session.request is not None or session.action_in_flight is not None:
            raise ValueError('Planner/action transaction changed during observer computation')
        for member, token in enumerate(tokens):
            if session.observer_request(identity, member)[0] != token:
                raise ValueError('Issued attempt changed while observer was running')
        self.commit_predictions(identity, step, predictions)
        session.windows = proposed
        self.last = dict(generation=generation, control_step=step, observation_sha256=digest)
        return dict(control_step=step,
            attempt_generation=generation, observation_sha256=digest, members=evidence,
            optimizer_updates=0, **self.feedback_metadata())
