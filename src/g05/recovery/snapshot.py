"""Explicit snapshot inventory and round-trip receipts.

The old collector only called ``og.sim.dump_state``.  This module refuses to
equate that with a restorable world: every required component has a separately
auditable inventory entry.  The concrete simulator adapter is injected by a
live owner; this module itself is CPU-only and does not import OmniGibson.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .common import RecoveryContractError, canonical_sha256, require_sha256, validate_raw23_actions


REQUIRED_INVENTORY = (
    "world",
    "objects",
    "robot",
    "controller",
    "grasp",
    "particles",
    "rng",
    "clock",
    "observation_freshness",
)


@dataclass(frozen=True)
class LiveReadinessAttestation:
    """A reviewed live-session boundary, not a caller-selected ``live`` flag.

    This receipt is deliberately more specific than ``backend_kind``: it binds
    a known public OG adapter, one evaluator session, the inspected source and
    Isaac/OG version, a prior full-capability roundtrip receipt, and the only
    evidence providers that may contribute to a positive corrective label.
    It does not attempt to defend against arbitrary hostile Python in-process;
    it makes the supported integration boundary explicit and fail-closed.
    """

    adapter_id: str
    session_id: str
    source_revision: str
    omnigibson_version: str
    isaac_version: str
    validated_inventory: frozenset[str]
    roundtrip_receipt_sha256: str
    physical_evidence_provider_ids: frozenset[str]
    fault_evidence_provider_ids: frozenset[str]

    def __post_init__(self) -> None:
        for field in ("adapter_id", "session_id", "source_revision", "omnigibson_version", "isaac_version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise RecoveryContractError(f"Live readiness {field} must be explicit")
        if not self.isaac_version.startswith("5.1"):
            raise RecoveryContractError("P107 live readiness is restricted to a source-verified Isaac Sim 5.1 session")
        if not isinstance(self.validated_inventory, frozenset):
            raise RecoveryContractError("Live readiness inventory must be immutable")
        if self.validated_inventory != frozenset(REQUIRED_INVENTORY):
            raise RecoveryContractError("Live readiness must attest every required snapshot inventory component")
        require_sha256(self.roundtrip_receipt_sha256, field="live roundtrip receipt")
        for provider_ids, field in (
            (self.physical_evidence_provider_ids, "physical evidence provider IDs"),
            (self.fault_evidence_provider_ids, "fault evidence provider IDs"),
        ):
            if (
                not isinstance(provider_ids, frozenset)
                or not provider_ids
                or not all(isinstance(item, str) and item for item in provider_ids)
            ):
                raise RecoveryContractError(f"Live readiness {field} must be a nonempty immutable set")

    @property
    def attestation_id(self) -> str:
        return canonical_sha256({
            "adapter_id": self.adapter_id,
            "session_id": self.session_id,
            "source_revision": self.source_revision,
            "omnigibson_version": self.omnigibson_version,
            "isaac_version": self.isaac_version,
            "validated_inventory": sorted(self.validated_inventory),
            "roundtrip_receipt_sha256": self.roundtrip_receipt_sha256,
            "physical_evidence_provider_ids": sorted(self.physical_evidence_provider_ids),
            "fault_evidence_provider_ids": sorted(self.fault_evidence_provider_ids),
        })


@dataclass(frozen=True)
class InventoryEntry:
    """Fingerprint of one restorable component, never the privileged value."""

    status: str
    fingerprint: str | None
    note: str

    def __post_init__(self) -> None:
        if self.status not in {"captured", "unavailable"}:
            raise RecoveryContractError("Inventory status must be captured or unavailable")
        if self.status == "captured":
            require_sha256(self.fingerprint, field="inventory fingerprint")
        elif self.fingerprint is not None:
            raise RecoveryContractError("Unavailable inventory may not claim a fingerprint")
        if not isinstance(self.note, str) or not self.note:
            raise RecoveryContractError("Inventory entries require a nonempty audit note")

    def public(self) -> dict[str, Any]:
        return {"status": self.status, "fingerprint": self.fingerprint, "note": self.note}


class SnapshotBackend(Protocol):
    """The only simulator-facing surface expected by :class:`SnapshotAdapter`."""

    backend_kind: str  # ``live`` only after an owner has wired a real evaluator.
    adapter_id: str
    session_id: str

    def dump_state(self, *, serialized: bool) -> Any: ...

    def load_state(self, state: Any, *, serialized: bool) -> None: ...

    def inventory_component(self, component: str) -> Any | None: ...

    def state_fingerprint(self, state: Any) -> str | None: ...


class OmniGibsonPublicSnapshotBackend:
    """Narrow adapter for the one locally verified OmniGibson snapshot hook.

    The historical TRAIN collector calls ``og.sim.dump_state(serialized=False)``
    (``scripts/data/collect_memlite_recovery.py``).  No corresponding restore
    call is present in this checkout, so this adapter is deliberately
    ``unverified`` by default and refuses to invent one.  A live readiness
    ticket may pass an independently source-verified ``restore_public_state``
    callable and explicit component providers.  It can become ``live`` only by
    receiving a reviewed :class:`LiveReadinessAttestation`; an arbitrary
    ``backend_kind='live'`` argument is intentionally not accepted.

    It does not inspect private simulator, PhysX, controller, or robot fields.
    Component providers are explicit so unsupported grasp / particle / RNG
    coverage becomes ``unavailable`` in the receipt rather than implied.
    """

    def __init__(
        self,
        sim: Any,
        *,
        restore_public_state: Callable[[Any], None] | None = None,
        inventory_providers: Mapping[str, Callable[[], Any | None]] | None = None,
        state_fingerprint: Callable[[Any], str | None] | None = None,
        adapter_id: str = "omnigibson-public-unverified",
        session_id: str = "unverified-session",
        readiness_attestation: LiveReadinessAttestation | None = None,
    ) -> None:
        if not callable(getattr(sim, "dump_state", None)):
            raise RecoveryContractError("OmniGibson sim lacks the verified public dump_state hook")
        if restore_public_state is not None and not callable(restore_public_state):
            raise RecoveryContractError("restore_public_state must be a verified callable or None")
        providers = {} if inventory_providers is None else dict(inventory_providers)
        unknown = set(providers) - set(REQUIRED_INVENTORY)
        if unknown:
            raise RecoveryContractError(f"Unknown snapshot inventory component(s): {sorted(unknown)}")
        if not all(callable(provider) for provider in providers.values()):
            raise RecoveryContractError("Snapshot inventory providers must be callables")
        if state_fingerprint is not None and not callable(state_fingerprint):
            raise RecoveryContractError("state_fingerprint must be callable or None")
        if not isinstance(adapter_id, str) or not adapter_id or not isinstance(session_id, str) or not session_id:
            raise RecoveryContractError("OmniGibson adapter and session identities must be explicit")
        if readiness_attestation is not None:
            if not isinstance(readiness_attestation, LiveReadinessAttestation):
                raise RecoveryContractError("readiness_attestation must be a LiveReadinessAttestation or None")
            if (readiness_attestation.adapter_id != adapter_id or readiness_attestation.session_id != session_id):
                raise RecoveryContractError("Live readiness attestation does not bind this adapter/session")
        self._sim = sim
        self._restore_public_state = restore_public_state
        self._providers = providers
        self._fingerprint = state_fingerprint
        self.adapter_id = adapter_id
        self.session_id = session_id
        self.readiness_attestation = readiness_attestation
        self.backend_kind = "live" if readiness_attestation is not None else "unverified"

    @property
    def has_registered_restore(self) -> bool:
        """Whether this known adapter has an explicitly registered public restore hook."""
        return self._restore_public_state is not None

    def dump_state(self, *, serialized: bool) -> Any:
        # The call form is source-verifiable in the historical collector.
        return self._sim.dump_state(serialized=serialized)

    def load_state(self, state: Any, *, serialized: bool) -> None:
        if serialized:
            raise RecoveryContractError("P107 only admits the inspected serialized=False snapshot path")
        if self._restore_public_state is None:
            raise RecoveryContractError("No source-verified OmniGibson restore API has been registered")
        self._restore_public_state(state)

    def inventory_component(self, component: str) -> Any | None:
        provider = self._providers.get(component)
        return None if provider is None else provider()

    def state_fingerprint(self, state: Any) -> str | None:
        return None if self._fingerprint is None else self._fingerprint(state)


@dataclass
class SnapshotCapture:
    restore_identity: str
    state: Any = field(repr=False, compare=False)
    state_fingerprint: str | None
    inventory: dict[str, InventoryEntry]
    backend_kind: str
    adapter_id: str
    session_id: str
    readiness_attestation_id: str | None
    label: str

    @property
    def complete_inventory(self) -> bool:
        return all(self.inventory[name].status == "captured" for name in REQUIRED_INVENTORY)

    @property
    def live_candidate(self) -> bool:
        return bool(
            self.backend_kind == "live"
            and self.readiness_attestation_id is not None
            and self.complete_inventory
            and self.state_fingerprint is not None
        )

    def public(self) -> dict[str, Any]:
        return {
            "restore_identity": self.restore_identity,
            "state_fingerprint": self.state_fingerprint,
            "backend_kind": self.backend_kind,
            "adapter_id": self.adapter_id,
            "session_id": self.session_id,
            "readiness_attestation_id": self.readiness_attestation_id,
            "label": self.label,
            "inventory": {name: self.inventory[name].public() for name in REQUIRED_INVENTORY},
            "complete_inventory": self.complete_inventory,
            # A capture alone never proves restoration.
            "trainable": False,
        }


@dataclass(frozen=True)
class InventoryComparison:
    component: str
    status: str
    left_fingerprint: str | None
    right_fingerprint: str | None

    def public(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "left_fingerprint": self.left_fingerprint,
            "right_fingerprint": self.right_fingerprint,
        }


@dataclass
class SnapshotRoundTrip:
    initial: SnapshotCapture
    restored: SnapshotCapture
    first_after_actions: SnapshotCapture
    second_after_actions: SnapshotCapture
    actions_sha256: str
    actions_requested: int
    actions_first_executed: int
    actions_second_executed: int
    restore_comparison: list[InventoryComparison]
    replay_comparison: list[InventoryComparison]

    @property
    def restore_state_fingerprint_match(self) -> bool:
        return (
            self.initial.state_fingerprint is not None
            and self.initial.state_fingerprint == self.restored.state_fingerprint
        )

    @property
    def replay_state_fingerprint_match(self) -> bool:
        return (
            self.first_after_actions.state_fingerprint is not None
            and self.first_after_actions.state_fingerprint == self.second_after_actions.state_fingerprint
        )

    @property
    def passed(self) -> bool:
        comparisons = [*self.restore_comparison, *self.replay_comparison]
        return (
            self.initial.live_candidate
            and self.restored.live_candidate
            and self.first_after_actions.live_candidate
            and self.second_after_actions.live_candidate
            and self.actions_first_executed == self.actions_requested
            and self.actions_second_executed == self.actions_requested
            and self.restore_state_fingerprint_match
            and self.replay_state_fingerprint_match
            and all(item.status == "match" for item in comparisons)
        )

    def public(self) -> dict[str, Any]:
        return {
            "kind": "snapshot_roundtrip",
            "initial": self.initial.public(),
            "restored": self.restored.public(),
            "first_after_actions": self.first_after_actions.public(),
            "second_after_actions": self.second_after_actions.public(),
            "actions_sha256": self.actions_sha256,
            "actions_requested": self.actions_requested,
            "actions_first_executed": self.actions_first_executed,
            "actions_second_executed": self.actions_second_executed,
            "restore_state_fingerprint_match": self.restore_state_fingerprint_match,
            "replay_state_fingerprint_match": self.replay_state_fingerprint_match,
            "restore_comparison": [item.public() for item in self.restore_comparison],
            "replay_comparison": [item.public() for item in self.replay_comparison],
            "passed": self.passed,
            "trainable": False,
            "integration_status": "live-roundtrip-passed" if self.passed else "unverified-or-incomplete",
        }


class SnapshotAdapter:
    """Capture and restore opaque simulator state through explicit public hooks.

    ``inventory_component`` is deliberately supplied by the adapter owner.  No
    reflection over private OmniGibson/PhysX/controller attributes occurs here.
    A component without a documented provider is recorded as unavailable and
    prevents the receipt from becoming a branchable/live-ready snapshot.
    """

    def __init__(self, backend: SnapshotBackend):
        self.backend = backend
        self._sequence = 0
        if getattr(backend, "backend_kind", None) not in {"live", "fake", "unverified"}:
            raise RecoveryContractError("Snapshot backend_kind must be live, fake, or unverified")
        self.adapter_id = getattr(backend, "adapter_id", None)
        if not isinstance(self.adapter_id, str) or not self.adapter_id:
            self.adapter_id = f"untrusted:{type(backend).__module__}.{type(backend).__qualname__}:{id(backend)}"
        self.session_id = getattr(backend, "session_id", None)
        if not isinstance(self.session_id, str) or not self.session_id:
            self.session_id = f"untrusted-session:{id(backend)}"
        self.readiness_attestation: LiveReadinessAttestation | None = None
        if backend.backend_kind == "live":
            # Only the narrow, source-audited OG adapter may cross this boundary.
            if not isinstance(backend, OmniGibsonPublicSnapshotBackend):
                raise RecoveryContractError("Only the known OmniGibson public adapter may present live readiness")
            attestation = backend.readiness_attestation
            if not isinstance(attestation, LiveReadinessAttestation):
                raise RecoveryContractError("Live snapshot backend requires a full readiness attestation")
            if not backend.has_registered_restore:
                raise RecoveryContractError("Live snapshot backend requires a registered public restore hook")
            if attestation.adapter_id != self.adapter_id or attestation.session_id != self.session_id:
                raise RecoveryContractError("Live readiness attestation is not bound to this adapter/session")
            self.readiness_attestation = attestation

    @property
    def backend_kind(self) -> str:
        return self.backend.backend_kind

    def is_live_ready_capture(self, capture: SnapshotCapture) -> bool:
        attestation = self.readiness_attestation
        return bool(
            attestation is not None
            and capture.live_candidate
            and capture.adapter_id == self.adapter_id
            and capture.session_id == self.session_id
            and capture.readiness_attestation_id == attestation.attestation_id
        )

    def accepts_evidence_provider(self, provider_id: str, *, kind: str) -> bool:
        attestation = self.readiness_attestation
        if attestation is None or not isinstance(provider_id, str):
            return False
        if kind == "physical":
            return provider_id in attestation.physical_evidence_provider_ids
        if kind == "fault":
            return provider_id in attestation.fault_evidence_provider_ids
        raise RecoveryContractError("Unknown evidence provider kind")

    def _entry(self, component: str) -> InventoryEntry:
        try:
            value = self.backend.inventory_component(component)
        except Exception as exc:  # Provider errors are missing evidence, not proof.
            return InventoryEntry("unavailable", None, f"provider_error:{type(exc).__name__}")
        if value is None:
            return InventoryEntry("unavailable", None, "no_documented_provider")
        if isinstance(value, InventoryEntry):
            return value
        try:
            return InventoryEntry("captured", canonical_sha256(value), "provider_fingerprint")
        except RecoveryContractError:
            return InventoryEntry("unavailable", None, "provider_value_not_canonical")

    def capture(self, label: str) -> SnapshotCapture:
        if not isinstance(label, str) or not label:
            raise RecoveryContractError("Snapshot labels must be nonempty")
        state = self.backend.dump_state(serialized=False)
        fingerprint = self.backend.state_fingerprint(state)
        if fingerprint is not None:
            require_sha256(fingerprint, field="snapshot state_fingerprint")
        self._sequence += 1
        return SnapshotCapture(
            restore_identity=f"snapshot-{self._sequence:06d}",
            state=state,
            state_fingerprint=fingerprint,
            inventory={name: self._entry(name) for name in REQUIRED_INVENTORY},
            backend_kind=self.backend.backend_kind,
            adapter_id=self.adapter_id,
            session_id=self.session_id,
            readiness_attestation_id=(None if self.readiness_attestation is None else self.readiness_attestation.attestation_id),
            label=label,
        )

    def restore(self, capture: SnapshotCapture) -> None:
        if not isinstance(capture, SnapshotCapture):
            raise RecoveryContractError("Only a capture from this adapter may be restored")
        if capture.adapter_id != self.adapter_id or capture.session_id != self.session_id:
            raise RecoveryContractError("Cross-adapter or cross-session snapshot restoration is forbidden")
        attestation_id = None if self.readiness_attestation is None else self.readiness_attestation.attestation_id
        if capture.readiness_attestation_id != attestation_id:
            raise RecoveryContractError("Snapshot readiness attestation does not match this adapter/session")
        self.backend.load_state(capture.state, serialized=False)

    def restore_and_capture(self, capture: SnapshotCapture, *, label: str) -> SnapshotCapture:
        self.restore(capture)
        return self.capture(label)

    @staticmethod
    def compare(left: SnapshotCapture, right: SnapshotCapture) -> list[InventoryComparison]:
        comparisons = []
        for component in REQUIRED_INVENTORY:
            a, b = left.inventory[component], right.inventory[component]
            if a.status != "captured" or b.status != "captured":
                status = "unverified"
            elif a.fingerprint == b.fingerprint:
                status = "match"
            else:
                status = "mismatch"
            comparisons.append(InventoryComparison(component, status, a.fingerprint, b.fingerprint))
        return comparisons

    @classmethod
    def same_state(cls, left: SnapshotCapture, right: SnapshotCapture) -> bool:
        """Require both an opaque-state receipt and all required inventories."""
        return bool(
            left.adapter_id == right.adapter_id
            and left.session_id == right.session_id
            and left.readiness_attestation_id == right.readiness_attestation_id
            and left.state_fingerprint is not None
            and left.state_fingerprint == right.state_fingerprint
            and all(item.status == "match" for item in cls.compare(left, right))
        )

    def roundtrip(
        self,
        actions: Sequence[Any],
        execute: Callable[[list[list[float]]], int],
        *,
        label: str = "roundtrip",
    ) -> SnapshotRoundTrip:
        """Execute exactly the same official actions twice from one captured state.

        The caller owns ``execute`` and must route every action through its real
        evaluator ``env.step``.  Its returned count is checked; a fake can test
        control flow but cannot make the receipt live/trainable.
        """
        actions = validate_raw23_actions(actions)
        initial = self.capture(f"{label}:initial")
        first_count = int(execute(actions))
        first = self.capture(f"{label}:first_after_actions")
        restored = self.restore_and_capture(initial, label=f"{label}:after_restore")
        second_count = int(execute(actions))
        second = self.capture(f"{label}:second_after_actions")
        return SnapshotRoundTrip(
            initial=initial,
            restored=restored,
            first_after_actions=first,
            second_after_actions=second,
            actions_sha256=canonical_sha256(actions),
            actions_requested=len(actions),
            actions_first_executed=first_count,
            actions_second_executed=second_count,
            restore_comparison=self.compare(initial, restored),
            replay_comparison=self.compare(first, second),
        )
