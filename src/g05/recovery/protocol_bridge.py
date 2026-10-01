"""Loose bridge to the data owner's canonical event protocol.

The recovery collector only emits an execution receipt.  This bridge is the
single integration point at which that receipt may be turned into a canonical
training view.  It intentionally does not recreate the protocol schema.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any, Mapping

from .branching import BranchPairReceipt
from .common import RecoveryContractError


PROTOCOL_MODULE = "g05.data.memlite_event_protocol"
PROTOCOL_SCHEMA_ID = "memlite-event-recovery-v1"


def load_protocol() -> Any:
    """Direct-load the stdlib-only contract without importing ``g05.data``.

    ``g05.data.__init__`` currently imports ML dataset dependencies.  The P107
    protocol is intentionally a no-dependency file, so a CPU collector must
    load that exact local source file directly.  The source location is stable
    within this repository; an installed/package deployment should register an
    equivalent direct protocol loader rather than silently importing datasets.
    """
    module_name = "_p107_memlite_event_protocol_direct"
    existing = sys.modules.get(module_name)
    if existing is not None:
        protocol = existing
    else:
        path = Path(__file__).resolve().parents[1] / "data" / "memlite_event_protocol.py"
        if not path.is_file():
            raise RecoveryContractError(f"Canonical memlite_event_protocol is absent: {path}")
        spec = spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RecoveryContractError("Could not direct-load canonical memlite_event_protocol")
        protocol = module_from_spec(spec)
        sys.modules[module_name] = protocol
        spec.loader.exec_module(protocol)
    if getattr(protocol, "SCHEMA_VERSION", None) != PROTOCOL_SCHEMA_ID:
        raise RecoveryContractError("Canonical memlite_event_protocol schema version disagrees with P107 bridge")
    return protocol


def validate_transport_receipt(
    receipt: BranchPairReceipt,
    *,
    source_ref: Mapping[str, Any],
    event: Mapping[str, Any],
    protocol: Any | None = None,
) -> dict[str, Any]:
    """Validate shared identities without inventing a final training-row format."""
    if not isinstance(receipt, BranchPairReceipt):
        raise RecoveryContractError("Expected paired recovery receipt")
    protocol = load_protocol() if protocol is None else protocol
    for name in ("validate_source_ref", "validate_event", "source_group_id", "event_id"):
        if not callable(getattr(protocol, name, None)):
            raise RecoveryContractError(f"Canonical protocol lacks {name}")
    protocol.validate_source_ref(source_ref)
    protocol.validate_event(event)
    event_source = event.get("source") if isinstance(event, Mapping) else None
    if not isinstance(event_source, Mapping) or dict(event_source) != dict(source_ref):
        raise RecoveryContractError("Execution receipt source must exactly match the canonical event source")
    source_group = protocol.source_group_id(source_ref)
    canonical_event_id = protocol.event_id(event)
    # A source group intentionally spans siblings from the same task instance;
    # it is not a per-episode/event binding.  Compare the entire canonical
    # objects and their content IDs so one sibling cannot envelope another.
    if dict(receipt.source_ref) != dict(source_ref):
        raise RecoveryContractError("Receipt source is not the canonical source bound to this envelope")
    if dict(receipt.event_ref) != dict(event):
        raise RecoveryContractError("Receipt event is not the canonical event bound to this envelope")
    receipt_event_id = receipt.event_ref.get("event_id") if isinstance(receipt.event_ref, Mapping) else None
    if receipt_event_id != canonical_event_id:
        raise RecoveryContractError("Receipt event ID does not match the canonical event identity")
    receipt_source_group = receipt.source_ref.get("source_group_id") if isinstance(receipt.source_ref, Mapping) else None
    if receipt_source_group != source_group:
        raise RecoveryContractError("Receipt source group does not match its full canonical source")
    if source_group != receipt.source_group_id:
        raise RecoveryContractError("Canonical source-group identity disagrees with execution receipt")
    return {
        "schema_id": PROTOCOL_SCHEMA_ID,
        "source_group_id": source_group,
        "event_id": canonical_event_id,
        "canonical_event_validated": True,
        "receipt": receipt.public(),
    }


def validate_corrective_action_view(view: Mapping[str, Any], *, protocol: Any | None = None) -> Any:
    """Reject obsolete caller-side corrective-action validation.

    Raw paired collection is only a candidate transport and has no authority
    to accept a caller-supplied ``PROPOSED`` mask or any positive action label.
    The reviewed data publisher must call its own externally sealed authority
    API after this transport has been frozen.  The retained name gives old
    callers a deterministic fail-closed error rather than silently delegating
    to a stale local protocol implementation.
    """
    del view, protocol
    raise RecoveryContractError(
        "Raw recovery bridge has no corrective-action publication authority; "
        "use the reviewed external publisher authority"
    )


def project_action_23_to_27(
    raw_actions: list[list[float]], actual_executed_length: int, *, protocol: Any | None = None
) -> dict[str, Any]:
    """Delegate fixed-shape model padding to the canonical protocol only.

    The collector intentionally exports raw 23-D actions.  This bridge does
    not recreate the 27-D mapping or action padding mask; it merely calls the
    data owner's contract after an actual-execution receipt has been retained.
    """
    protocol = load_protocol() if protocol is None else protocol
    projector = getattr(protocol, "project_action_23_to_27", None)
    if not callable(projector):
        raise RecoveryContractError("Canonical protocol lacks 23D-to-27D projection")
    projected = projector(raw_actions, actual_executed_length)
    if not isinstance(projected, dict):
        raise RecoveryContractError("Canonical action projection must return an object")
    return projected


def actor_evidence_projection(view: Mapping[str, Any], *, protocol: Any | None = None) -> dict[str, Any]:
    """Return only a recursively typed, causal actor-visible projection.

    The owner protocol is the source of the actor-evidence schema.  This
    bridge additionally verifies its returned object before crossing the SIM
    boundary.  That redundant fail-closed guard is deliberate: older local
    protocol snapshots only checked top-level actor-evidence keys and could
    return privileged fields hidden inside ``references``.  The guard matches
    the current canonical typed-reference subset and fails closed for future
    schema additions until this bridge is reviewed against them.
    """
    protocol = load_protocol() if protocol is None else protocol
    projector = getattr(protocol, "actor_evidence_projection", None)
    if not callable(projector):
        raise RecoveryContractError("Canonical protocol lacks actor-evidence projection")
    projected = projector(view)
    if not isinstance(projected, dict):
        raise RecoveryContractError("Canonical actor-evidence projection must return an object")
    if not isinstance(view, Mapping) or type(view.get("observation_frame")) is not int:
        raise RecoveryContractError("Actor-evidence projection requires an integer observation frame")
    _validate_actor_projection(projected, observation_frame=view["observation_frame"])
    return projected


_NON_PHYSICAL_EVIDENCE_KINDS = frozenset(
    {"SEGMENT_END", "ANNOTATION_END", "TIMEOUT", "GRIPPER_CLOSE", "MODEL_SELF_REPORT"}
)
_RGB_VIEWS = frozenset({"head", "left_wrist", "right_wrist"})


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_actor_projection(projected: Mapping[str, Any], *, observation_frame: int) -> None:
    """Verify the canonical recursive typed-reference actor boundary.

    This deliberately inspects every reachable actor value rather than using
    a spelling blacklist.  References are closed typed locators: RGB camera
    artifacts, named public proprioception artifacts, or command identifiers.
    No arbitrary nested mapping can cross this bridge.
    """
    required = {"kind", "evidence_end_frame", "available_frame", "references"}
    if set(projected) != required:
        raise RecoveryContractError("Actor evidence must use the closed typed-reference schema")
    kind = projected["kind"]
    references = projected["references"]
    if not isinstance(kind, str) or not isinstance(references, list):
        raise RecoveryContractError("Actor evidence kind and references must be typed")
    if kind == "MISSING":
        if (projected["evidence_end_frame"] is not None
                or projected["available_frame"] is not None or references):
            raise RecoveryContractError("MISSING actor evidence cannot carry references or clocks")
        return
    if not kind or kind in _NON_PHYSICAL_EVIDENCE_KINDS:
        raise RecoveryContractError("Actor evidence must name causal visual/physical/review evidence")
    evidence_end = projected["evidence_end_frame"]
    available = projected["available_frame"]
    if (type(evidence_end) is not int or type(available) is not int
            or evidence_end < 0 or available < evidence_end
            or available > observation_frame):
        raise RecoveryContractError("Actor evidence clocks must be causal at the observation frame")
    for index, reference in enumerate(references):
        _validate_actor_reference(
            reference,
            name=f"actor_evidence.references[{index}]",
            available_frame=available,
            observation_frame=observation_frame,
        )


def _validate_actor_reference(
    reference: Any, *, name: str, available_frame: int, observation_frame: int
) -> None:
    if not isinstance(reference, Mapping):
        raise RecoveryContractError(f"{name} must be an exact typed actor reference")
    kind = reference.get("kind")
    if kind == "rgb_frame":
        required = {"kind", "view", "frame", "artifact_sha256"}
        valid = (
            set(reference) == required
            and reference["view"] in _RGB_VIEWS
            and _is_sha256(reference["artifact_sha256"])
        )
        frame = reference.get("frame")
    elif kind == "proprio_observation":
        required = {"kind", "frame", "artifact_sha256", "field_names"}
        fields = reference.get("field_names")
        valid = (
            set(reference) == required
            and _is_sha256(reference["artifact_sha256"])
            and isinstance(fields, list)
            and bool(fields)
            and all(
                isinstance(field, str)
                and field
                and "state" not in field.lower()
                and "pose" not in field.lower()
                for field in fields
            )
        )
        frame = reference.get("frame")
    elif kind == "command_identifier":
        required = {"kind", "issued_frame", "command_id"}
        valid = (
            set(reference) == required
            and isinstance(reference["command_id"], str)
            and bool(reference["command_id"])
        )
        frame = reference.get("issued_frame")
    else:
        raise RecoveryContractError(
            f"{name} must be rgb_frame, proprio_observation, or command_identifier"
        )
    if not valid or type(frame) is not int or frame < 0:
        raise RecoveryContractError(f"{name} is not an exact typed actor locator")
    if frame > available_frame or frame > observation_frame:
        raise RecoveryContractError(f"{name} is not available at the observation frame")
