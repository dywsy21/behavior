"""Loose bridge to the data owner's canonical event protocol.

The recovery collector only emits an execution receipt.  This bridge is the
single integration point at which that receipt may be turned into a canonical
training view.  It intentionally does not recreate the protocol schema.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from .branching import BranchPairReceipt
from .common import RecoveryContractError


PROTOCOL_MODULE = "g05.data.memlite_event_protocol"
PROTOCOL_SCHEMA_ID = "memlite-event-recovery-v1"


def load_protocol() -> Any:
    try:
        protocol = import_module(PROTOCOL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == PROTOCOL_MODULE:
            raise RecoveryContractError("Canonical memlite_event_protocol is not installed in this worktree") from exc
        raise
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
    source_group = protocol.source_group_id(source_ref)
    if source_group != receipt.source_group_id:
        raise RecoveryContractError("Canonical source-group identity disagrees with execution receipt")
    return {
        "schema_id": PROTOCOL_SCHEMA_ID,
        "source_group_id": source_group,
        "event_id": protocol.event_id(event),
        "receipt": receipt.public(),
    }


def validate_corrective_action_view(view: Mapping[str, Any], *, protocol: Any | None = None) -> Any:
    """Delegate final action projection/masking to the shared data contract."""
    protocol = load_protocol() if protocol is None else protocol
    validator = getattr(protocol, "validate_corrective_action_view", None)
    if not callable(validator):
        raise RecoveryContractError("Canonical protocol lacks corrective-action validation")
    return validator(view)
