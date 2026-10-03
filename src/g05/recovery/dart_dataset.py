"""Private, non-training DART episode bundles.

The candidate collector in :mod:`dart_collection` deliberately stops at an
in-memory candidate receipt.  This module is the small persistence seam for a
runtime owner that has actually captured one fresh transition.  It accepts
captured PNG bytes (or an array converted by :meth:`RGBAsset.from_array`), the
already validated DART receipts, and writes an immutable private bundle.

The bundle is intentionally *not* a training dataset or an authority record:
all release/training flags are false, actor input is a projection containing
only the current RGB/proprioception, and delayed outcome evidence is stored in
a separate private file.  Raw ``state61`` values and arbitrary simulator
objects never enter the on-disk format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .common import RecoveryContractError, canonical_json, canonical_sha256, require_sha256, validate_raw23_action
from .dart_collection import (
    AppliedActionReceipt,
    CandidateSourceReceipt,
    DartObservation,
    OutcomeEvidenceReceipt,
    RuntimeSessionReceipt,
    TeacherCommand,
    TeacherReceipt,
    VerifiedDartSourceMembership,
)


BUNDLE_SCHEMA = "p107_dart_episode_bundle_v1"
TRANSITION_SCHEMA = "p107_dart_transition_v1"
ACTOR_PROJECTION_SCHEMA = "p107_dart_actor_projection_v1"
PRIVATE_STATUS = "PRIVATE_DART_DIAGNOSTIC_ONLY"
REQUIRED_VIEWS = ("head", "left_wrist", "right_wrist")
COLLECTION_MODE_ORIGINAL_GAUSSIAN = "original_gaussian_clean_intended_feedback"
COLLECTION_MODE_BOUNDED_ACTUAL = "dart_inspired_bounded_actual_clean_recovery"
_ALLOWED_COLLECTION_MODES = frozenset({COLLECTION_MODE_ORIGINAL_GAUSSIAN, COLLECTION_MODE_BOUNDED_ACTUAL})
_FORBIDDEN_ACTOR_KEYS = frozenset(
    {
        "state61",
        "privileged",
        "privileged_state",
        "scene_state",
        "target_pose",
        "target_velocity",
        "contacts",
        "is_grasping",
        "official_task_success",
        "official_terminal",
        "outcome_evidence",
        "post_observation",
        "next_observation",
        "future",
        "applied23",
        "requested_noisy23",
        "sampled_noise23",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RecoveryContractError(f"{field} must be a finite number")
    return float(value)


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecoveryContractError(f"{field} must be nonempty text")
    return value


def _png_info(payload: bytes) -> tuple[int, int]:
    """Validate a captured PNG without trusting caller-supplied dimensions."""

    if not isinstance(payload, bytes) or len(payload) < 8:
        raise RecoveryContractError("RGB asset must contain nonempty PNG bytes")
    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            if image.format != "PNG":
                raise RecoveryContractError("RGB asset must be encoded as PNG")
            if image.mode not in {"RGB", "RGBA"}:
                raise RecoveryContractError("RGB asset PNG must have RGB or RGBA channels")
            width, height = image.size
            image.verify()
        if width < 1 or height < 1:
            raise RecoveryContractError("RGB asset dimensions must be positive")
        return int(width), int(height)
    except RecoveryContractError:
        raise
    except Exception as exc:  # Pillow uses several exception types for bad bytes.
        raise RecoveryContractError("RGB asset is not a readable PNG") from exc


@dataclass(frozen=True)
class RGBAsset:
    """One captured RGB view, bound to a canonical view and policy clock."""

    view: str
    camera_key: str
    payload: bytes
    width: int
    height: int
    policy_clock: int
    source_locator: str
    capture_time_s: float | None = None
    pts: float | None = None

    def __post_init__(self) -> None:
        if self.view not in REQUIRED_VIEWS:
            raise RecoveryContractError(f"RGB asset view must be one of {REQUIRED_VIEWS}")
        _nonempty_text(self.camera_key, "camera_key")
        _nonempty_text(self.source_locator, "source_locator")
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise RecoveryContractError("RGB asset payload must be captured bytes")
        payload = bytes(self.payload)
        actual_width, actual_height = _png_info(payload)
        if type(self.width) is not int or self.width != actual_width:
            raise RecoveryContractError("RGB asset width does not match decoded PNG")
        if type(self.height) is not int or self.height != actual_height:
            raise RecoveryContractError("RGB asset height does not match decoded PNG")
        if type(self.policy_clock) is not int or self.policy_clock < 0:
            raise RecoveryContractError("RGB asset policy_clock must be a nonnegative integer")
        if self.capture_time_s is not None:
            _finite_number(self.capture_time_s, "capture_time_s")
            if float(self.capture_time_s) < 0:
                raise RecoveryContractError("capture_time_s must be nonnegative")
        if self.pts is not None:
            _finite_number(self.pts, "pts")
        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_array(
        cls,
        *,
        view: str,
        camera_key: str,
        array: Any,
        policy_clock: int,
        source_locator: str,
        capture_time_s: float | None = None,
        pts: float | None = None,
    ) -> "RGBAsset":
        """Encode a caller-provided captured image array as PNG.

        This method does not manufacture an observation: it only serializes an
        array already returned by the runtime.  The array is deliberately not
        retained after encoding.
        """

        try:
            from PIL import Image

            if isinstance(array, Image.Image):
                image = array
            else:
                image = Image.fromarray(array)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
        except Exception as exc:
            raise RecoveryContractError("captured RGB array could not be encoded as PNG") from exc
        return cls(
            view=view,
            camera_key=camera_key,
            payload=payload,
            width=int(image.width),
            height=int(image.height),
            policy_clock=policy_clock,
            source_locator=source_locator,
            capture_time_s=capture_time_s,
            pts=pts,
        )

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.payload)

    def descriptor(self, *, relative_path: str) -> dict[str, Any]:
        return {
            "relative_path": relative_path,
            "sha256": self.sha256,
            "bytes": len(self.payload),
            "view": self.view,
            "camera_key": self.camera_key,
            "width": self.width,
            "height": self.height,
            "policy_clock": self.policy_clock,
            "source_locator": self.source_locator,
            "capture_time_s": self.capture_time_s,
            "pts": self.pts,
        }


@dataclass(frozen=True)
class ActorObservation:
    """The only structured observation allowed in the actor projection.

    ``proprioception`` is a caller-provided actor-safe vector (for example the
    OGKinematics joint/EEF/finger/base state).  It is not derived from or
    serialized as the full simulator ``state61``.  The explicit field shape
    avoids accepting arbitrary dicts that could hide privileged scene truth.
    """

    policy_clock: int
    assets: Mapping[str, RGBAsset]
    proprioception: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if type(self.policy_clock) is not int or self.policy_clock < 0:
            raise RecoveryContractError("actor observation policy_clock must be a nonnegative integer")
        if not isinstance(self.assets, Mapping) or set(self.assets) != set(REQUIRED_VIEWS):
            raise RecoveryContractError(f"actor observation must contain exactly {REQUIRED_VIEWS}")
        seen_camera_keys: set[str] = set()
        normalized: dict[str, RGBAsset] = {}
        for view in REQUIRED_VIEWS:
            asset = self.assets.get(view)
            if not isinstance(asset, RGBAsset) or asset.view != view:
                raise RecoveryContractError("actor observation camera map is inconsistent with RGB asset view")
            if asset.policy_clock != self.policy_clock:
                raise RecoveryContractError("RGB asset policy_clock does not match actor observation")
            if asset.camera_key in seen_camera_keys:
                raise RecoveryContractError("actor observation reuses one camera_key for multiple views")
            seen_camera_keys.add(asset.camera_key)
            normalized[view] = asset
        if self.proprioception is not None:
            if isinstance(self.proprioception, (str, bytes, bytearray, Mapping)):
                raise RecoveryContractError("actor proprioception must be a finite numeric sequence")
            try:
                values = tuple(_finite_number(value, "actor proprioception") for value in self.proprioception)
            except TypeError as exc:
                raise RecoveryContractError("actor proprioception must be a finite numeric sequence") from exc
            if not values:
                raise RecoveryContractError("actor proprioception cannot be empty")
            object.__setattr__(self, "proprioception", values)
        object.__setattr__(self, "assets", normalized)

    def public(self, *, paths: Mapping[str, str]) -> dict[str, Any]:
        return {
            "policy_clock": self.policy_clock,
            "rgb": {view: self.assets[view].descriptor(relative_path=paths[view]) for view in REQUIRED_VIEWS},
            "proprioception": None if self.proprioception is None else list(self.proprioception),
            "actor_visible": True,
            "privileged_state_included": False,
        }


@dataclass(frozen=True)
class StepClock:
    """Actual runtime tick/time binding for one applied action."""

    simulation_tick_start: int
    simulation_tick_end: int
    action_start_time_s: float | None = None
    action_end_time_s: float | None = None

    def __post_init__(self) -> None:
        if type(self.simulation_tick_start) is not int or self.simulation_tick_start < 0:
            raise RecoveryContractError("simulation_tick_start must be a nonnegative integer")
        if type(self.simulation_tick_end) is not int or self.simulation_tick_end <= self.simulation_tick_start:
            raise RecoveryContractError("simulation_tick_end must be after simulation_tick_start")
        if self.action_start_time_s is not None:
            _finite_number(self.action_start_time_s, "action_start_time_s")
            if float(self.action_start_time_s) < 0:
                raise RecoveryContractError("action_start_time_s must be nonnegative")
        if self.action_end_time_s is not None:
            _finite_number(self.action_end_time_s, "action_end_time_s")
            if float(self.action_end_time_s) < 0:
                raise RecoveryContractError("action_end_time_s must be nonnegative")
        if self.action_start_time_s is not None and self.action_end_time_s is not None:
            if float(self.action_end_time_s) < float(self.action_start_time_s):
                raise RecoveryContractError("action_end_time_s must not precede action_start_time_s")

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunProvenance:
    """Explicit origin/provenance for a capture invocation."""

    run_id: str
    source_episode_id: str
    run_origin: str
    collector_code_sha256: str
    capture_adapter_sha256: str
    actor_observation_schema_sha256: str

    def __post_init__(self) -> None:
        _nonempty_text(self.run_id, "run_id")
        _nonempty_text(self.source_episode_id, "source_episode_id")
        if self.run_origin not in {"live_runtime", "cpu_fixture"}:
            raise RecoveryContractError("run_origin must be live_runtime or cpu_fixture")
        for field in ("collector_code_sha256", "capture_adapter_sha256", "actor_observation_schema_sha256"):
            require_sha256(getattr(self, field), field=field)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DartEpisodeStep:
    """One causal transition: observe -> teacher -> noisy apply -> observe."""

    step_index: int
    pre_observation: DartObservation
    pre_actor_observation: ActorObservation
    teacher_command: TeacherCommand
    requested_noisy23: Sequence[float]
    sampled_noise23: Sequence[float]
    applied: AppliedActionReceipt
    post_observation: DartObservation
    post_actor_observation: ActorObservation
    clock: StepClock
    label_kind: str = "dart_inspired_actual_clean_recovery"

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise RecoveryContractError("DART step_index must be a nonnegative integer")
        for name, value in (
            ("pre_observation", self.pre_observation),
            ("post_observation", self.post_observation),
        ):
            if not isinstance(value, DartObservation):
                raise RecoveryContractError(f"{name} must be DartObservation")
        for name, value in (
            ("pre_actor_observation", self.pre_actor_observation),
            ("post_actor_observation", self.post_actor_observation),
        ):
            if not isinstance(value, ActorObservation):
                raise RecoveryContractError(f"{name} must be ActorObservation")
        if not isinstance(self.teacher_command, TeacherCommand):
            raise RecoveryContractError("teacher_command must be TeacherCommand")
        if not isinstance(self.applied, AppliedActionReceipt):
            raise RecoveryContractError("applied must be AppliedActionReceipt")
        if not isinstance(self.clock, StepClock):
            raise RecoveryContractError("clock must be StepClock")
        if self.label_kind not in {
            "dart_clean_supervisor_feedback",
            "dart_inspired_noisy_injection",
            "dart_inspired_actual_clean_recovery",
        }:
            raise RecoveryContractError("step label_kind must preserve the existing DART candidate kind")
        requested = tuple(validate_raw23_action(self.requested_noisy23))
        sampled = tuple(validate_raw23_action(self.sampled_noise23))
        object.__setattr__(self, "requested_noisy23", requested)
        object.__setattr__(self, "sampled_noise23", sampled)
        if self.applied.status != "APPLIED":
            raise RecoveryContractError("episode bundle requires an actually APPLIED noisy action")
        if self.pre_actor_observation.policy_clock != self.pre_observation.policy_clock:
            raise RecoveryContractError("pre actor observation clock does not match DartObservation")
        if self.post_actor_observation.policy_clock != self.post_observation.policy_clock:
            raise RecoveryContractError("post actor observation clock does not match DartObservation")
        if self.post_observation.policy_clock <= self.pre_observation.policy_clock:
            raise RecoveryContractError("post observation must be later than pre observation")
        if self.teacher_command.observed_policy_clock != self.pre_observation.policy_clock:
            raise RecoveryContractError("teacher command is not bound to pre-action policy clock")
        if self.teacher_command.observed_state61_sha256 != self.pre_observation.state61_sha256:
            raise RecoveryContractError("teacher command state digest is not bound to pre-action state")
        if self.teacher_command.observed_observation_sha256 != self.pre_observation.observation_sha256:
            raise RecoveryContractError("teacher command observation digest is not bound to pre-action observation")
        if self.applied.pre_action_policy_clock != self.pre_observation.policy_clock:
            raise RecoveryContractError("applied receipt clock is not bound to pre-action observation")
        if self.applied.pre_action_state61_sha256 != self.pre_observation.state61_sha256:
            raise RecoveryContractError("applied receipt state digest is not bound to pre-action state")
        if self.applied.pre_action_observation_sha256 != self.pre_observation.observation_sha256:
            raise RecoveryContractError("applied receipt observation digest is not bound to pre-action observation")


@dataclass(frozen=True)
class DartEpisodeBundle:
    source: CandidateSourceReceipt
    verified_membership: VerifiedDartSourceMembership
    runtime_session: RuntimeSessionReceipt
    teacher: TeacherReceipt
    intent_bundle_id: str
    collection_mode: str
    label_kind: str
    run_provenance: RunProvenance
    steps: tuple[DartEpisodeStep, ...]
    outcome_evidence: OutcomeEvidenceReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, CandidateSourceReceipt):
            raise RecoveryContractError("bundle source must be CandidateSourceReceipt")
        if not isinstance(self.verified_membership, VerifiedDartSourceMembership):
            raise RecoveryContractError("bundle requires DATA-verified source membership")
        if not isinstance(self.runtime_session, RuntimeSessionReceipt):
            raise RecoveryContractError("bundle runtime_session must be RuntimeSessionReceipt")
        if not isinstance(self.teacher, TeacherReceipt):
            raise RecoveryContractError("bundle teacher must be TeacherReceipt")
        if not isinstance(self.run_provenance, RunProvenance):
            raise RecoveryContractError("bundle run_provenance must be RunProvenance")
        _nonempty_text(self.intent_bundle_id, "intent_bundle_id")
        if self.collection_mode not in _ALLOWED_COLLECTION_MODES:
            raise RecoveryContractError("bundle collection_mode must preserve an existing DART mode")
        if self.label_kind not in {
            "dart_clean_supervisor_feedback",
            "dart_inspired_noisy_injection",
            "dart_inspired_actual_clean_recovery",
            "mixed",
        }:
            raise RecoveryContractError("bundle label_kind must preserve an existing DART candidate kind")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise RecoveryContractError("DART episode bundle needs at least one captured transition")
        if not all(isinstance(step, DartEpisodeStep) for step in self.steps):
            raise RecoveryContractError("bundle steps must be DartEpisodeStep records")
        if self.runtime_session.task_instance_id != self.source.parent_task_instance_id:
            raise RecoveryContractError("runtime session task instance does not match source lineage")
        if self.runtime_session.runtime_session_id != self.source.collection_run_id:
            raise RecoveryContractError("runtime session ID does not match source collection run")
        candidate = self.verified_membership.candidate
        if (
            candidate.source_group_id != self.source.source_group_id
            or candidate.source_release_sha256 != self.source.source_release_sha256
            or candidate.task_index != self.source.parent_task_index
            or candidate.task_instance_id != self.source.parent_task_instance_id
            or candidate.original_split != "train"
            or candidate.usage_role != "student_candidate"
        ):
            raise RecoveryContractError("bundle source does not match sealed TRAIN student-candidate membership")
        if self.verified_membership.calibration_groups and any(
            group.source_group_id == self.source.source_group_id
            for group in self.verified_membership.calibration_groups
        ):
            raise RecoveryContractError("candidate source cannot also be a calibration group")
        for index, step in enumerate(self.steps):
            if step.step_index != index:
                raise RecoveryContractError("bundle step indices must be contiguous from zero")
            if step.teacher_command.intent_bundle_id != self.intent_bundle_id:
                raise RecoveryContractError("teacher intent bundle does not match episode intent")
            if self.collection_mode == COLLECTION_MODE_ORIGINAL_GAUSSIAN and tuple(step.applied.applied23) != tuple(step.requested_noisy23):
                raise RecoveryContractError(
                    "original Gaussian bundle cannot claim exact requested execution after runtime clipping"
                )
            if self.collection_mode == COLLECTION_MODE_ORIGINAL_GAUSSIAN and step.label_kind != "dart_clean_supervisor_feedback":
                raise RecoveryContractError("original Gaussian bundle cannot contain a bounded-DART step kind")
            if index and (
                step.pre_observation.policy_clock != self.steps[index - 1].post_observation.policy_clock
                or step.pre_observation.observation_sha256 != self.steps[index - 1].post_observation.observation_sha256
                or step.clock.simulation_tick_start != self.steps[index - 1].clock.simulation_tick_end
            ):
                raise RecoveryContractError("episode transitions are not causally linked at the observed boundary")
        step_kinds = {step.label_kind for step in self.steps}
        if self.label_kind == "mixed":
            if self.collection_mode == COLLECTION_MODE_ORIGINAL_GAUSSIAN or len(step_kinds) < 2:
                raise RecoveryContractError("mixed bundle label_kind must describe multiple bounded-DART step kinds")
        elif step_kinds != {self.label_kind}:
            raise RecoveryContractError("bundle label_kind does not match every transition")
        if self.outcome_evidence is not None:
            if not isinstance(self.outcome_evidence, OutcomeEvidenceReceipt):
                raise RecoveryContractError("outcome_evidence must be OutcomeEvidenceReceipt")
            if self.run_provenance.run_origin == "cpu_fixture" and self.outcome_evidence.label != "OUTCOME_UNKNOWN":
                raise RecoveryContractError("CPU fixtures may not serialize a claimed fault or survival outcome")
            if self.outcome_evidence.observed_policy_clock < self.steps[0].pre_observation.policy_clock:
                raise RecoveryContractError("outcome evidence precedes the captured episode")

    def _provenance(self) -> dict[str, Any]:
        return {
            "source": self.source.public(),
            "verified_source_membership": self.verified_membership.public(),
            "runtime_session": self.runtime_session.public(),
            "teacher": self.teacher.public(),
            "intent_bundle_id": self.intent_bundle_id,
            "collection_mode": self.collection_mode,
            "label_kind": self.label_kind,
            "run_provenance": self.run_provenance.public(),
        }


@dataclass(frozen=True)
class BundleReceipt:
    path: Path
    manifest_sha256: str
    transition_count: int
    image_count: int
    training_eligible: bool = False
    release_eligible: bool = False


@dataclass(frozen=True)
class ActorProjection:
    """Loaded current-state actor view; no post/future/private evidence."""

    step_index: int
    policy_clock: int
    rgb_by_view: Mapping[str, bytes]
    proprioception: tuple[float, ...] | None
    clean_intended23: tuple[float, ...]
    intent_bundle_id: str


class LoadedDartEpisodeBundle:
    """Validated bundle reader with explicit asset and actor-view accessors."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], rows: tuple[Mapping[str, Any], ...]) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.rows = rows

    def read_asset(self, step_index: int, phase: str, view: str) -> bytes:
        row = _row_for_step(self.rows, step_index)
        if phase not in {"pre", "post"} or view not in REQUIRED_VIEWS:
            raise RecoveryContractError("read_asset requires phase pre/post and a canonical actor view")
        actor_key = f"{phase}_actor_observation"
        descriptor = row[actor_key]["rgb"][view]
        path = _safe_relative_path(self.root, descriptor["relative_path"])
        payload = path.read_bytes()
        if _sha256_bytes(payload) != descriptor["sha256"]:
            raise RecoveryContractError("RGB asset SHA-256 does not match transition descriptor")
        return payload

    def actor_projection(self, step_index: int) -> ActorProjection:
        row = _row_for_step(self.rows, step_index)
        projection = row["actor_projection"]
        _reject_forbidden_actor_content(projection)
        return ActorProjection(
            step_index=step_index,
            policy_clock=projection["policy_clock"],
            rgb_by_view={view: self.read_asset(step_index, "pre", view) for view in REQUIRED_VIEWS},
            proprioception=(
                None
                if projection["proprioception"] is None
                else tuple(float(value) for value in projection["proprioception"])
            ),
            clean_intended23=tuple(float(value) for value in projection["bc_target_clean_intended23"]),
            intent_bundle_id=projection["intent_bundle_id"],
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "schema": self.manifest["schema"],
            "status": self.manifest["status"],
            "transition_count": len(self.rows),
            "image_count": len(self.manifest["files"]) - 1 - (1 if "private/outcome_evidence.json" in self.manifest["files"] else 0),
            "source_group_id": self.manifest["source"]["source_group_id"],
            "source_original_split": self.manifest["source"]["original_split"],
            "source_usage_role": self.manifest["verified_source_membership"]["candidate"]["usage_role"],
            "run_origin": self.manifest["run_provenance"]["run_origin"],
            "training_eligible": self.manifest["training_eligible"],
            "release_eligible": self.manifest["release_eligible"],
            "outcome_evidence_separate": "private/outcome_evidence.json" in self.manifest["files"],
        }


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _safe_relative_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RecoveryContractError("bundle file path must be relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RecoveryContractError("bundle file path escapes bundle root") from exc
    return path


def _asset_paths(step_index: int, phase: str) -> dict[str, str]:
    if phase not in {"pre", "post"}:
        raise RecoveryContractError("asset phase must be pre or post")
    return {
        view: f"assets/step-{step_index:06d}/{phase}-{view}.png" for view in REQUIRED_VIEWS
    }


def _teacher_public(command: TeacherCommand) -> dict[str, Any]:
    return {
        "clean_intended23": list(command.clean_intended23),
        "observed_policy_clock": command.observed_policy_clock,
        "observed_state61_sha256": command.observed_state61_sha256,
        "observed_observation_sha256": command.observed_observation_sha256,
        "intent_bundle_id": command.intent_bundle_id,
        "fresh_query_receipt_sha256": command.fresh_query_receipt_sha256,
    }


def _applied_public(applied: AppliedActionReceipt) -> dict[str, Any]:
    return {
        "status": applied.status,
        "applied23": list(applied.applied23),
        "applied_action_bytes_sha256": applied.applied_action_bytes_sha256,
        "runtime_step_receipt_sha256": applied.runtime_step_receipt_sha256,
        "pre_action_policy_clock": applied.pre_action_policy_clock,
        "pre_action_state61_sha256": applied.pre_action_state61_sha256,
        "pre_action_observation_sha256": applied.pre_action_observation_sha256,
    }


def _action_binding_sha256(step: DartEpisodeStep) -> str:
    """Fingerprint controls together with the pre/post clocks they affect."""

    return canonical_sha256(
        {
            "requested_noisy23": list(step.requested_noisy23),
            "sampled_noise23": list(step.sampled_noise23),
            "applied23": list(step.applied.applied23),
            "pre_policy_clock": step.pre_observation.policy_clock,
            "post_policy_clock": step.post_observation.policy_clock,
            "simulation_tick_start": step.clock.simulation_tick_start,
            "simulation_tick_end": step.clock.simulation_tick_end,
        }
    )


def _transition_row(step: DartEpisodeStep, *, intent_bundle_id: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    pre_paths = _asset_paths(step.step_index, "pre")
    post_paths = _asset_paths(step.step_index, "post")
    assets: dict[str, bytes] = {}
    for view in REQUIRED_VIEWS:
        assets[pre_paths[view]] = step.pre_actor_observation.assets[view].payload
        assets[post_paths[view]] = step.post_actor_observation.assets[view].payload
    row = {
        "schema": TRANSITION_SCHEMA,
        "step_index": step.step_index,
        "label_kind": step.label_kind,
        "pre_observation": step.pre_observation.public(),
        "post_observation": step.post_observation.public(),
        "pre_actor_observation": step.pre_actor_observation.public(paths=pre_paths),
        "post_actor_observation": step.post_actor_observation.public(paths=post_paths),
        "teacher": _teacher_public(step.teacher_command),
        "requested_noisy23": list(step.requested_noisy23),
        "sampled_noise23": list(step.sampled_noise23),
        "applied": _applied_public(step.applied),
        "action_binding_sha256": _action_binding_sha256(step),
        "clock": step.clock.public(),
        "actor_projection": {
            "schema": ACTOR_PROJECTION_SCHEMA,
            "phase": "pre_action",
            "step_index": step.step_index,
            "policy_clock": step.pre_observation.policy_clock,
            "rgb": {view: pre_paths[view] for view in REQUIRED_VIEWS},
            "proprioception": (
                None
                if step.pre_actor_observation.proprioception is None
                else list(step.pre_actor_observation.proprioception)
            ),
            "bc_target_clean_intended23": list(step.teacher_command.clean_intended23),
            "intent_bundle_id": intent_bundle_id,
            "future_excluded": True,
            "privileged_state_excluded": True,
            "outcome_excluded": True,
        },
    }
    return row, assets


def _reject_forbidden_actor_content(value: Any, *, path: str = "actor_projection") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecoveryContractError(f"{path} contains a non-text key")
            if key in _FORBIDDEN_ACTOR_KEYS:
                raise RecoveryContractError(f"{path} contains forbidden actor field {key}")
            _reject_forbidden_actor_content(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_actor_content(item, path=f"{path}[{index}]")


def _row_for_step(rows: Sequence[Mapping[str, Any]], step_index: int) -> Mapping[str, Any]:
    if type(step_index) is not int or step_index < 0 or step_index >= len(rows):
        raise RecoveryContractError("requested bundle step is out of range")
    row = rows[step_index]
    if row.get("step_index") != step_index:
        raise RecoveryContractError("bundle step indices are not canonical")
    return row


def _manifest_files(staging: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        payload = path.read_bytes()
        files[relative] = {"sha256": _sha256_bytes(payload), "bytes": len(payload)}
    return files


def write_episode_bundle(output: Path, bundle: DartEpisodeBundle) -> BundleReceipt:
    """Validate and atomically publish one fresh private DART episode bundle."""

    output = Path(output)
    bundle.__post_init__()
    if os.path.lexists(output):
        raise RecoveryContractError("refusing to overwrite an existing DART bundle path")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        records: list[dict[str, Any]] = []
        for step in bundle.steps:
            row, assets = _transition_row(step, intent_bundle_id=bundle.intent_bundle_id)
            records.append(row)
            for relative, payload in assets.items():
                path = _safe_relative_path(staging, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
        records_bytes = b"".join(_json_bytes(row) for row in records)
        (staging / "records.jsonl").write_bytes(records_bytes)
        if bundle.outcome_evidence is not None:
            private = staging / "private"
            private.mkdir(parents=True, exist_ok=True)
            (private / "outcome_evidence.json").write_bytes(_json_bytes(bundle.outcome_evidence.public()))
        files = _manifest_files(staging)
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "status": PRIVATE_STATUS,
            "role": "private_dart_episode_capture",
            "candidate_only": True,
            "authority_minted": False,
            "authority_status": "NO_AUTHORITY",
            "reviewed": False,
            "ready_for_training": False,
            "training_eligible": False,
            "release_eligible": False,
            "actor_projection_schema": ACTOR_PROJECTION_SCHEMA,
            "records_path": "records.jsonl",
            "files": files,
            **bundle._provenance(),
            "transition_count": len(records),
            "image_count": 2 * len(records) * len(REQUIRED_VIEWS),
            "outcome_evidence_separate": bundle.outcome_evidence is not None,
        }
        manifest_bytes = _json_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        if os.path.lexists(output):
            raise RecoveryContractError("refusing to overwrite an existing DART bundle path")
        os.replace(staging, output)
        return BundleReceipt(
            path=output,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            transition_count=len(records),
            image_count=manifest["image_count"],
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_file_receipt(root: Path, relative: str, receipt: Mapping[str, Any]) -> Path:
    if not isinstance(receipt, Mapping):
        raise RecoveryContractError("bundle file receipt must be an object")
    path = _safe_relative_path(root, relative)
    if path.is_symlink() or not path.is_file():
        raise RecoveryContractError(f"bundle file is missing or symlinked: {relative}")
    require_sha256(receipt.get("sha256"), field=f"files[{relative}].sha256")
    if type(receipt.get("bytes")) is not int or receipt["bytes"] < 0:
        raise RecoveryContractError(f"files[{relative}].bytes must be a nonnegative integer")
    payload = path.read_bytes()
    if len(payload) != receipt["bytes"] or _sha256_bytes(payload) != receipt["sha256"]:
        raise RecoveryContractError(f"bundle file bytes/SHA mismatch: {relative}")
    return path


def _validate_serialized_row(row: Mapping[str, Any], *, manifest: Mapping[str, Any]) -> None:
    if row.get("schema") != TRANSITION_SCHEMA:
        raise RecoveryContractError("bundle transition has an unknown schema")
    step_index = row.get("step_index")
    if type(step_index) is not int or step_index < 0:
        raise RecoveryContractError("bundle transition step_index is invalid")
    label_kind = row.get("label_kind")
    if label_kind not in {
        "dart_clean_supervisor_feedback",
        "dart_inspired_noisy_injection",
        "dart_inspired_actual_clean_recovery",
    }:
        raise RecoveryContractError("bundle transition label_kind is unknown")
    collection_mode = manifest.get("collection_mode")
    if collection_mode == COLLECTION_MODE_ORIGINAL_GAUSSIAN:
        if label_kind != "dart_clean_supervisor_feedback":
            raise RecoveryContractError("original Gaussian bundle contains a non-Gaussian transition")
    pre = row.get("pre_observation")
    post = row.get("post_observation")
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping):
        raise RecoveryContractError("bundle transition is missing pre/post observation receipts")
    pre_clock = pre.get("policy_clock")
    post_clock = post.get("policy_clock")
    if type(pre_clock) is not int or type(post_clock) is not int or post_clock <= pre_clock:
        raise RecoveryContractError("bundle transition policy clocks are invalid")
    for action_name in ("requested_noisy23", "sampled_noise23"):
        validate_raw23_action(row.get(action_name))
    applied = row.get("applied")
    if not isinstance(applied, Mapping) or applied.get("status") != "APPLIED":
        raise RecoveryContractError("bundle transition does not contain an APPLIED receipt")
    validate_raw23_action(applied.get("applied23"))
    if collection_mode == COLLECTION_MODE_ORIGINAL_GAUSSIAN and applied["applied23"] != row["requested_noisy23"]:
        raise RecoveryContractError("original Gaussian serialized transition records runtime clipping")
    teacher = row.get("teacher")
    if not isinstance(teacher, Mapping) or teacher.get("observed_policy_clock") != pre_clock:
        raise RecoveryContractError("serialized teacher receipt is not bound to pre-action clock")
    projection = row.get("actor_projection")
    if not isinstance(projection, Mapping) or projection.get("schema") != ACTOR_PROJECTION_SCHEMA:
        raise RecoveryContractError("serialized actor projection has an unknown schema")
    if projection.get("policy_clock") != pre_clock or projection.get("step_index") != step_index:
        raise RecoveryContractError("serialized actor projection is not bound to current pre-action state")
    if projection.get("future_excluded") is not True or projection.get("privileged_state_excluded") is not True:
        raise RecoveryContractError("serialized actor projection must explicitly exclude future/privileged state")
    _reject_forbidden_actor_content(projection)
    for phase in ("pre_actor_observation", "post_actor_observation"):
        actor = row.get(phase)
        if not isinstance(actor, Mapping) or actor.get("policy_clock") != (pre_clock if phase.startswith("pre") else post_clock):
            raise RecoveryContractError(f"serialized {phase} is not bound to its observation clock")
        rgb = actor.get("rgb")
        if not isinstance(rgb, Mapping) or set(rgb) != set(REQUIRED_VIEWS):
            raise RecoveryContractError(f"serialized {phase} must contain all canonical RGB views")
        camera_keys = []
        for view in REQUIRED_VIEWS:
            descriptor = rgb[view]
            if not isinstance(descriptor, Mapping) or descriptor.get("view") != view:
                raise RecoveryContractError("serialized RGB descriptor has a camera/view mismatch")
            camera_keys.append(descriptor.get("camera_key"))
            relative = descriptor.get("relative_path")
            receipt = manifest["files"].get(relative) if isinstance(relative, str) else None
            if receipt is None:
                raise RecoveryContractError("serialized RGB descriptor is absent from manifest files")
            if descriptor.get("sha256") != receipt.get("sha256") or descriptor.get("bytes") != receipt.get("bytes"):
                raise RecoveryContractError("serialized RGB descriptor disagrees with manifest file receipt")
        if len(set(camera_keys)) != len(camera_keys):
            raise RecoveryContractError("serialized RGB descriptors reuse a camera key")
    clock = row.get("clock")
    if not isinstance(clock, Mapping) or type(clock.get("simulation_tick_start")) is not int:
        raise RecoveryContractError("serialized transition is missing simulation tick clock")
    if type(clock.get("simulation_tick_end")) is not int or clock["simulation_tick_end"] <= clock["simulation_tick_start"]:
        raise RecoveryContractError("serialized simulation tick clock is invalid")
    expected_action_binding = canonical_sha256(
        {
            "requested_noisy23": row["requested_noisy23"],
            "sampled_noise23": row["sampled_noise23"],
            "applied23": applied["applied23"],
            "pre_policy_clock": pre_clock,
            "post_policy_clock": post_clock,
            "simulation_tick_start": clock["simulation_tick_start"],
            "simulation_tick_end": clock["simulation_tick_end"],
        }
    )
    if row.get("action_binding_sha256") != expected_action_binding:
        raise RecoveryContractError("serialized action binding SHA-256 does not match controls and clocks")


def load_bundle(path: Path, *, expected_manifest_sha256: str | None = None) -> LoadedDartEpisodeBundle:
    """Read and fail-closed validate a persisted private DART bundle."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise RecoveryContractError("DART bundle path must be a regular directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RecoveryContractError("DART bundle manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_sha256 = _sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 is not None and actual_manifest_sha256 != require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    ):
        raise RecoveryContractError("DART bundle manifest SHA-256 does not match expected pin")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise RecoveryContractError("DART bundle manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != BUNDLE_SCHEMA:
        raise RecoveryContractError("DART bundle manifest has an unknown schema")
    if (
        manifest.get("status") != PRIVATE_STATUS
        or manifest.get("candidate_only") is not True
        or manifest.get("authority_minted") is not False
        or manifest.get("authority_status") != "NO_AUTHORITY"
        or manifest.get("reviewed") is not False
        or manifest.get("ready_for_training") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("release_eligible") is not False
    ):
        raise RecoveryContractError("DART bundle release/training gates are not fail-closed")
    source = manifest.get("source")
    membership = manifest.get("verified_source_membership")
    if not isinstance(source, Mapping) or source.get("original_split") != "train":
        raise RecoveryContractError("DART bundle source is not an original TRAIN source")
    if not isinstance(membership, Mapping) or not isinstance(membership.get("candidate"), Mapping):
        raise RecoveryContractError("DART bundle is missing sealed source membership")
    candidate = membership["candidate"]
    if candidate.get("usage_role") != "student_candidate" or candidate.get("original_split") != "train":
        raise RecoveryContractError("DART bundle source role is not student_candidate TRAIN")
    for field in ("source_group_id", "source_release_sha256"):
        require_sha256(source.get(field), field=f"source.{field}")
    if (
        candidate.get("source_group_id") != source.get("source_group_id")
        or candidate.get("source_release_sha256") != source.get("source_release_sha256")
        or candidate.get("task_instance_id") != source.get("parent_task_instance_id")
    ):
        raise RecoveryContractError("DART bundle source and sealed membership identity disagree")
    if not isinstance(manifest.get("collection_mode"), str) or manifest["collection_mode"] not in _ALLOWED_COLLECTION_MODES:
        raise RecoveryContractError("DART bundle collection_mode is not an existing DART mode")
    if not isinstance(manifest.get("label_kind"), str) or manifest["label_kind"] not in {
        "dart_clean_supervisor_feedback",
        "dart_inspired_noisy_injection",
        "dart_inspired_actual_clean_recovery",
        "mixed",
    }:
        raise RecoveryContractError("DART bundle label_kind is not an existing DART candidate kind")
    run_provenance = manifest.get("run_provenance")
    if not isinstance(run_provenance, Mapping):
        raise RecoveryContractError("DART bundle is missing run provenance")
    _nonempty_text(run_provenance.get("run_id"), "run_provenance.run_id")
    _nonempty_text(run_provenance.get("source_episode_id"), "run_provenance.source_episode_id")
    if run_provenance.get("run_origin") not in {"live_runtime", "cpu_fixture"}:
        raise RecoveryContractError("DART bundle run_origin is invalid")
    for field in ("collector_code_sha256", "capture_adapter_sha256", "actor_observation_schema_sha256"):
        require_sha256(run_provenance.get(field), field=f"run_provenance.{field}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or "records.jsonl" not in files:
        raise RecoveryContractError("DART bundle manifest is missing records file receipt")
    for relative, receipt in files.items():
        if not isinstance(relative, str):
            raise RecoveryContractError("DART bundle file names must be text")
        _validate_file_receipt(root, relative, receipt)
    listed = {"manifest.json", *[str(relative) for relative in files]}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != listed:
        raise RecoveryContractError("DART bundle contains unlisted or missing files")
    records_path = _safe_relative_path(root, manifest.get("records_path"))
    if records_path.name != "records.jsonl":
        raise RecoveryContractError("DART bundle records_path is not records.jsonl")
    try:
        rows = tuple(json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line)
    except Exception as exc:
        raise RecoveryContractError("DART bundle records are not valid JSONL") from exc
    if len(rows) != manifest.get("transition_count") or not all(isinstance(row, Mapping) for row in rows):
        raise RecoveryContractError("DART bundle transition count is inconsistent")
    expected_image_count = 2 * len(rows) * len(REQUIRED_VIEWS)
    if manifest.get("image_count") != expected_image_count:
        raise RecoveryContractError("DART bundle image count is inconsistent")
    for row in rows:
        _validate_serialized_row(row, manifest=manifest)
        for phase in ("pre_actor_observation", "post_actor_observation"):
            actor = row[phase]
            for view in REQUIRED_VIEWS:
                descriptor = actor["rgb"][view]
                asset_path = _safe_relative_path(root, descriptor["relative_path"])
                width, height = _png_info(asset_path.read_bytes())
                if descriptor.get("width") != width or descriptor.get("height") != height:
                    raise RecoveryContractError("serialized RGB descriptor dimensions do not match PNG bytes")
                expected_clock = actor["policy_clock"]
                if descriptor.get("policy_clock") != expected_clock:
                    raise RecoveryContractError("serialized RGB descriptor clock does not match actor observation")
    for index, row in enumerate(rows):
        if row.get("step_index") != index:
            raise RecoveryContractError("DART bundle transition indices are not contiguous")
        if manifest["label_kind"] != "mixed" and row.get("label_kind") != manifest["label_kind"]:
            raise RecoveryContractError("DART bundle label_kind does not match transition")
        if index:
            previous = rows[index - 1]
            if (
                row["pre_observation"]["policy_clock"] != previous["post_observation"]["policy_clock"]
                or row["pre_observation"]["observation_sha256"] != previous["post_observation"]["observation_sha256"]
                or row["clock"]["simulation_tick_start"] != previous["clock"]["simulation_tick_end"]
            ):
                raise RecoveryContractError("DART bundle transitions are not causally linked")
    outcome_path = "private/outcome_evidence.json"
    if manifest.get("outcome_evidence_separate"):
        if outcome_path not in files:
            raise RecoveryContractError("manifest claims separate outcome evidence but file is missing")
        outcome = json.loads(_safe_relative_path(root, outcome_path).read_text(encoding="utf-8"))
        if not isinstance(outcome, Mapping) or outcome.get("actor_visible") is not False:
            raise RecoveryContractError("private outcome evidence must remain actor-invisible")
        if outcome.get("label") not in {"ACTUAL_FAULT", "SURVIVAL_NONFAILURE", "OUTCOME_UNKNOWN"}:
            raise RecoveryContractError("private outcome evidence has an unknown label")
        if type(outcome.get("observed_policy_clock")) is not int or type(outcome.get("available_policy_clock")) is not int:
            raise RecoveryContractError("private outcome evidence clocks must be integers")
        if outcome["available_policy_clock"] < outcome["observed_policy_clock"]:
            raise RecoveryContractError("private outcome evidence clocks are inconsistent")
        require_sha256(outcome.get("evidence_sha256"), field="private outcome evidence SHA-256")
    elif outcome_path in files:
        raise RecoveryContractError("private outcome evidence is unreferenced")
    _reject_forbidden_actor_content({"actor_projection": [row["actor_projection"] for row in rows]})
    return LoadedDartEpisodeBundle(root, manifest, tuple(rows))


def inspect_bundle(path: Path, *, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Return a small validated summary without exposing image bytes."""

    return load_bundle(path, expected_manifest_sha256=expected_manifest_sha256).inspect()


__all__ = [
    "ACTOR_PROJECTION_SCHEMA",
    "BUNDLE_SCHEMA",
    "BundleReceipt",
    "ActorObservation",
    "COLLECTION_MODE_BOUNDED_ACTUAL",
    "COLLECTION_MODE_ORIGINAL_GAUSSIAN",
    "DartEpisodeBundle",
    "DartEpisodeStep",
    "LoadedDartEpisodeBundle",
    "REQUIRED_VIEWS",
    "RGBAsset",
    "RunProvenance",
    "StepClock",
    "TRANSITION_SCHEMA",
    "inspect_bundle",
    "load_bundle",
    "write_episode_bundle",
]
