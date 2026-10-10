"""Explicit GRASP-only calibrated-feedback pilot; not a deployment default.

Only the owning, preflighted pilot may opt in after its planner fit/generation
checks. The immutable receipt is checked again here. Predictions are estimates,
never a known physical outcome, success reward, world fact, or terminal signal.
The ordinary CausalPlannerSession / shadow wrapper remain unchanged defaults.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from g05.utils.memlite_causal_session import CausalModelIdentity, CausalPlannerSession
from recovery_corpus import file_sha
from recovery_observer_shadow import BoundObserverShadowInference, OUTCOMES
from recovery_postfit import calibration_gate


@dataclass(frozen=True)
class GraspCalibrationBinding:
    """Value object for an already verified, separately loaded model triple."""
    models: CausalModelIdentity
    calibration_sha256: str
    temperature: float
    minimum_confidence: float = .85
    confirmations: int = 2

    def __post_init__(self):
        sha = self.calibration_sha256
        if (not isinstance(self.models, CausalModelIdentity)
                or not isinstance(sha, str) or len(sha) != 64
                or any(c not in '0123456789abcdef' for c in sha)
                or isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or not .5 <= self.temperature <= 5
                or self.minimum_confidence != .85 or type(self.confirmations) is not int
                or self.confirmations != 2):
            raise ValueError('Exact original GRASP calibration contract required')


def load_grasp_calibration(path, expected_sha256, models):
    """Never retune a temperature/threshold or substitute planner features."""
    path = Path(path)
    if file_sha(path) != expected_sha256:
        raise ValueError('Changed/unpinned calibration receipt')
    calibration = json.loads(path.read_text())
    calibration_gate(calibration)
    if (calibration['high_sha256'] != models.observer_backbone
            or calibration['observer_sha256'] != models.observer_adapter):
        raise ValueError('Calibration belongs to a different observer backbone/adapter')
    return GraspCalibrationBinding(models, expected_sha256, calibration['temperature'])


def calibrated_prediction(logits, *, temperature, verb):
    """Use the same Python stable softmax as fixed-feedback data generation."""
    values = list(logits)
    if (len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) for v in values)
            or isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not .5 <= temperature <= 5):
        raise ValueError('Four finite logits and the original temperature required')
    if verb != 'GRASP':
        return dict(outcome='UNKNOWN', confidence=0.)
    z = [v / temperature for v in values]
    top = max(z)
    p = [math.exp(v - top) for v in z]
    index = max(range(4), key=p.__getitem__)
    return dict(outcome=OUTCOMES[index], confidence=p[index] / sum(p))


class CalibratedGraspPilotSession(CausalPlannerSession):
    observer_feedback_mode = 'calibrated_grasp_estimate_pilot_v1'

    def __init__(self, identity, *, calibration, initial_control_step=0):
        if not isinstance(calibration, GraspCalibrationBinding) or calibration.models != identity.models:
            raise ValueError('Each pilot session must bind its exact planner/low/observer identities')
        super().__init__(identity, initial_control_step=initial_control_step)
        self.calibration = calibration
        if (self.feedback.minimum_confidence != calibration.minimum_confidence
                or self.feedback.confirmations != calibration.confirmations):
            raise ValueError('Serving and calibrated data persistence gates differ')

    def shadow_outcomes(self, identity, control_step, member_predictions):
        raise ValueError('Do not attach an uncalibrated wrapper to the calibrated pilot')

    def calibrated_outcomes(self, identity, control_step, predictions, *, calibration):
        self._check(identity, control_step)
        if (self.request is not None or self.action_in_flight is not None
                or calibration != self.calibration or self.installed is None):
            raise ValueError('Stale/in-flight/foreign calibration transaction')
        members = json.loads(self.installed['semantic_bundle'])
        if len(predictions) != len(members):
            raise ValueError('Ordered issued members and feedback differ')
        for member, prediction in zip(members, predictions):
            if member['verb'] != 'GRASP' and prediction != dict(outcome='UNKNOWN', confidence=0.):
                raise ValueError('GRASP calibration never licenses OPEN/PLACE/other outcomes')
        self.feedback.estimated(identity.feedback_identity(), control_step, predictions, calibrated=True)


class BoundCalibratedGraspInference(BoundObserverShadowInference):
    feedback_mode = CalibratedGraspPilotSession.observer_feedback_mode

    def __init__(self, policy, processor, config, head, session, **kwargs):
        if not isinstance(session, CalibratedGraspPilotSession):
            raise ValueError('Explicit calibrated pilot session required; shadow is the default')
        super().__init__(policy, processor, config, head, session, **kwargs)
        self.calibration = session.calibration

    def prediction(self, logits, member):
        verb = json.loads(self.session.installed['semantic_bundle'])[member]['verb']
        return calibrated_prediction(logits.cpu().tolist(), temperature=self.calibration.temperature, verb=verb)

    def commit_predictions(self, identity, step, predictions):
        self.session.calibrated_outcomes(identity, step, predictions, calibration=self.calibration)

    def feedback_metadata(self):
        return dict(status='calibrated_grasp_pilot_observer_check', observer_feedback_mode=self.feedback_mode,
            calibration_applied=True, calibration_sha256=self.calibration.calibration_sha256,
            entered_planner=False, feedback_available_to_planner=True,
            physical_success_asserted=False, known_previous_outcome='UNKNOWN')
