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
    if source_group != receipt.source_group_id:
        raise RecoveryContractError("Canonical source-group identity disagrees with execution receipt")
    return {
        "schema_id": PROTOCOL_SCHEMA_ID,
        "source_group_id": source_group,
        "event_id": protocol.event_id(event),
        "canonical_event_validated": True,
        "receipt": receipt.public(),
    }


def validate_corrective_action_view(view: Mapping[str, Any], *, protocol: Any | None = None) -> Any:
    """Delegate final action projection/masking to the shared data contract."""
    protocol = load_protocol() if protocol is None else protocol
    validator = getattr(protocol, "validate_corrective_action_view", None)
    if not callable(validator):
        raise RecoveryContractError("Canonical protocol lacks corrective-action validation")
    return validator(view)


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
    """Use the canonical actor-only projection; never derive one locally."""
    protocol = load_protocol() if protocol is None else protocol
    projector = getattr(protocol, "actor_evidence_projection", None)
    if not callable(projector):
        raise RecoveryContractError("Canonical protocol lacks actor-evidence projection")
    projected = projector(view)
    if not isinstance(projected, dict):
        raise RecoveryContractError("Canonical actor-evidence projection must return an object")
    return projected
