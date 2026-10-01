"""Opaque same-runtime bindings for candidate-only recovery collection.

This is deliberately *not* a readiness attestation, approval, signature, or
training authority.  A session factory that owns an evaluator creates one
instance and passes that exact instance to every simulator-facing adapter.  We
then use object identity to reject accidental mixtures such as simulator A and
an action/observation provider from simulator B.  The downstream publisher,
not this raw collector, decides whether any candidate is eligible for a loss
mask using its independently reviewed authority registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import RecoveryContractError


@dataclass(frozen=True)
class RuntimeCapability:
    """One opaque process-local evaluator runtime binding.

    ``label`` is diagnostics only.  The private token is intentionally not
    serialized, and possession of a capability never makes candidate output
    trusted, live, or trainable.
    """

    label: str
    _token: object = field(default_factory=object, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise RecoveryContractError("Runtime capability requires a diagnostic label")


def same_runtime(left: RuntimeCapability | None, right: RuntimeCapability | None) -> bool:
    """Return true only for the exact capability object, never matching text."""
    return isinstance(left, RuntimeCapability) and left is right
