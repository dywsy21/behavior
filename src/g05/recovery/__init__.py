"""Conservative, simulator-agnostic building blocks for MEM-Lite recovery data.

Nothing in this package imports OmniGibson or starts a simulator.  A live
caller must supply a narrowly audited adapter around the evaluator it owns.
The CPU-only fakes used by the tests deliberately produce non-trainable
receipts; passing those tests is not simulator readiness.
"""

from .common import RecoveryContractError, validate_raw23_actions
from .evidence import EvidenceResult, PhysicalEvidenceProvider, PredicateSample
from .snapshot import OmniGibsonPublicSnapshotBackend, SnapshotAdapter, SnapshotCapture, SnapshotRoundTrip
from .branching import OmniGibsonEvaluatorActionBackend, PairedRecoveryCollector, BranchPairReceipt

__all__ = [
    "BranchPairReceipt",
    "EvidenceResult",
    "OmniGibsonEvaluatorActionBackend",
    "OmniGibsonPublicSnapshotBackend",
    "PairedRecoveryCollector",
    "PhysicalEvidenceProvider",
    "PredicateSample",
    "RecoveryContractError",
    "SnapshotAdapter",
    "SnapshotCapture",
    "SnapshotRoundTrip",
    "validate_raw23_actions",
]
