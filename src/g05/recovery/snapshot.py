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
from .runtime import RuntimeCapability, same_runtime


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

    backend_kind: str  # Diagnostic only: raw collection never uses it as authority.
    adapter_id: str
    session_id: str
    runtime_capability: RuntimeCapability | None

    def dump_state(self, *, serialized: bool) -> Any: ...

    def load_state(self, state: Any, *, serialized: bool) -> None: ...

    def inventory_component(self, component: str) -> Any | None: ...

    def state_fingerprint(self, state: Any) -> str | None: ...


class OmniGibsonPublicSnapshotBackend:
    """Narrow adapter for the one locally verified OmniGibson snapshot hook.

    The historical TRAIN collector calls ``og.sim.dump_state(serialized=False)``
    (``scripts/data/collect_memlite_recovery.py``).  No corresponding restore
    call is present in this checkout, so this adapter is deliberately
    ``unverified`` and candidate-only by design.  A documented public restore
    callable may be supplied for a diagnostic restore attempt, but no callback,
    version string, or runtime capability confers training authority here.

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
        runtime_capability: RuntimeCapability | None = None,
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
        if runtime_capability is not None and not isinstance(runtime_capability, RuntimeCapability):
            raise RecoveryContractError("runtime_capability must be RuntimeCapability or None")
        self._sim = sim
        self._restore_public_state = restore_public_state
        self._providers = providers
        self._fingerprint = state_fingerprint
        self.adapter_id = adapter_id
        self.session_id = session_id
        self.runtime_capability = runtime_capability
        self.backend_kind = "unverified"

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
    runtime_capability: RuntimeCapability | None = field(repr=False, compare=False)
    adapter_instance_token: object = field(repr=False, compare=False)
    label: str

    @property
    def complete_inventory(self) -> bool:
        return all(self.inventory[name].status == "captured" for name in REQUIRED_INVENTORY)

    @property
    def runtime_bound(self) -> bool:
        """Local same-session diagnostic, never a readiness or training claim."""
        return self.runtime_capability is not None and self.state_fingerprint is not None

    def public(self) -> dict[str, Any]:
        return {
            "restore_identity": self.restore_identity,
            "state_fingerprint": self.state_fingerprint,
            "backend_kind": self.backend_kind,
            "adapter_id": self.adapter_id,
            "session_id": self.session_id,
            "runtime_bound": self.runtime_bound,
            "label": self.label,
            "inventory": {name: self.inventory[name].public() for name in REQUIRED_INVENTORY},
            "complete_inventory": self.complete_inventory,
            # A capture alone never proves restoration or training eligibility.
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
        """Always false: raw roundtrips are candidate diagnostics only.

        ``mechanically_consistent`` below retains the useful local replay
        signal without accidentally becoming a readiness approval.
        """
        return False

    @property
    def mechanically_consistent(self) -> bool:
        comparisons = [*self.restore_comparison, *self.replay_comparison]
        return (
            self.actions_first_executed == self.actions_requested
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
            "mechanically_consistent": self.mechanically_consistent,
            "trainable": False,
            "integration_status": "candidate-only-unverified",
        }


class SnapshotAdapter:
    """Capture and restore opaque simulator state through explicit public hooks.

    ``inventory_component`` is deliberately supplied by the adapter owner.  No
    reflection over private OmniGibson/PhysX/controller attributes occurs here.
    A component without a documented provider is recorded as unavailable.
    This adapter cannot assert simulator readiness or training eligibility.
    """

    def __init__(self, backend: SnapshotBackend):
        self.backend = backend
        self._sequence = 0
        if getattr(backend, "backend_kind", None) not in {"fake", "unverified"}:
            raise RecoveryContractError("Snapshot backend_kind must be fake or unverified; no caller live shortcut exists")
        self.adapter_id = getattr(backend, "adapter_id", None)
        if not isinstance(self.adapter_id, str) or not self.adapter_id:
            self.adapter_id = f"untrusted:{type(backend).__module__}.{type(backend).__qualname__}:{id(backend)}"
        self.session_id = getattr(backend, "session_id", None)
        if not isinstance(self.session_id, str) or not self.session_id:
            self.session_id = f"untrusted-session:{id(backend)}"
        runtime = getattr(backend, "runtime_capability", None)
        if runtime is not None and not isinstance(runtime, RuntimeCapability):
            raise RecoveryContractError("Snapshot runtime capability must be RuntimeCapability or None")
        self.runtime_capability: RuntimeCapability | None = runtime
        # An opaque per-adapter token closes the same adapter/session string
        # collision: only this adapter object may restore its captures.
        self._adapter_instance_token = object()

    @property
    def backend_kind(self) -> str:
        return self.backend.backend_kind

    def is_runtime_bound_capture(self, capture: SnapshotCapture) -> bool:
        return bool(
            capture.runtime_bound
            and same_runtime(capture.runtime_capability, self.runtime_capability)
            and capture.adapter_instance_token is self._adapter_instance_token
        )

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
            runtime_capability=self.runtime_capability,
            adapter_instance_token=self._adapter_instance_token,
            label=label,
        )

    def restore(self, capture: SnapshotCapture) -> None:
        if not isinstance(capture, SnapshotCapture):
            raise RecoveryContractError("Only a capture from this adapter may be restored")
        if capture.adapter_instance_token is not self._adapter_instance_token:
            raise RecoveryContractError("Cross-adapter snapshot restoration is forbidden")
        if capture.runtime_capability is not self.runtime_capability:
            raise RecoveryContractError("Cross-runtime snapshot restoration is forbidden")
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
            left.adapter_instance_token is right.adapter_instance_token
            and left.runtime_capability is right.runtime_capability
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
