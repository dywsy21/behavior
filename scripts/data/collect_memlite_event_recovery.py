#!/usr/bin/env python3
"""Collect one paired recovery receipt from a supplied post-fault snapshot.

The factory is the only place permitted to initialise an evaluator.  It must
return a mapping with ``snapshot_adapter``, ``action_backend``,
``evidence_provider`` and ``post_fault_snapshot``.  This command records only
actions for which that backend returned from ``env.step``; it has no model,
teacher, reset, GPU, or OmniGibson defaults of its own.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from g05.recovery.branching import PairedRecoveryCollector
from g05.recovery.common import RecoveryContractError, canonical_json, validate_raw23_actions
from g05.recovery.evidence import FaultEvidenceProvider, PhysicalEvidenceProvider
from g05.recovery.protocol_bridge import validate_transport_receipt
from g05.recovery.snapshot import SnapshotAdapter, SnapshotCapture


def _load_factory(spec: str) -> Any:
    if ":" not in spec:
        raise RecoveryContractError("Factory must be module:callable")
    module_name, name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), name, None)
    if not callable(factory):
        raise RecoveryContractError("Collection factory is not callable")
    return factory()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:callable; registered live/fake session factory")
    parser.add_argument("--source-ref", type=Path, required=True)
    parser.add_argument("--source-group-id", required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--skill-binding", type=Path, required=True)
    parser.add_argument("--intent-bundle-id", required=True,
                        help="immutable intent identity; both collected branches must execute this bound intent")
    parser.add_argument("--prior-intent-bundle-id", required=True,
                        help="immutable intent whose pre-branch deviation made this a recovery candidate")
    parser.add_argument("--actor-evidence", type=Path, required=True)
    parser.add_argument("--no-intervention-actions", type=Path, required=True)
    parser.add_argument("--corrective-actions", type=Path, required=True)
    parser.add_argument("--branch-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fake", action="store_true", help="CPU test only; receipt remains non-trainable")
    args = parser.parse_args()
    parts = _load_factory(args.factory)
    if not isinstance(parts, dict):
        raise RecoveryContractError("Collection factory must return a mapping")
    snapshots = parts.get("snapshot_adapter")
    actions = parts.get("action_backend")
    evidence = parts.get("evidence_provider")
    fault_evidence = parts.get("fault_evidence_provider")
    post_fault = parts.get("post_fault_snapshot")
    if not isinstance(snapshots, SnapshotAdapter) or not isinstance(evidence, PhysicalEvidenceProvider):
        raise RecoveryContractError("Factory has invalid snapshot/evidence adapters")
    if not isinstance(post_fault, SnapshotCapture) or not all(
        callable(getattr(actions, name, None)) for name in ("observe", "step", "official_status")
    ):
        raise RecoveryContractError("Factory must expose post_fault_snapshot and action_backend observe/step/status")
    if post_fault.backend_kind == "fake" and not args.allow_fake:
        raise RecoveryContractError("Refusing fake collection backend without --allow-fake")
    if not snapshots.is_runtime_bound_capture(post_fault):
        raise RecoveryContractError("Factory must bind its snapshot to one explicit runtime capability")
    source_ref = _read_json(args.source_ref)
    event = _read_json(args.event)
    transport = PairedRecoveryCollector(snapshots, actions, evidence, fault_evidence).collect(
        source_ref=source_ref,
        source_group_id=args.source_group_id,
        event_ref=event,
        post_fault_snapshot=post_fault,
        no_intervention_actions23=validate_raw23_actions(_read_json(args.no_intervention_actions)),
        corrective_actions23=validate_raw23_actions(_read_json(args.corrective_actions)),
        skill_binding=_read_json(args.skill_binding),
        intent_bundle_id=args.intent_bundle_id,
        prior_intent_bundle_id=args.prior_intent_bundle_id,
        branch_seed=args.branch_seed,
        actor_evidence=_read_json(args.actor_evidence),
        context=parts.get("context", {}),
    )
    # The canonical protocol validates immutable event/source identities before
    # emitting this envelope.  It does *not* turn the transport receipt into a
    # final action view: only the data-owner projector may add 27-D padding and
    # low-action supervision fields.
    envelope = validate_transport_receipt(transport, source_ref=source_ref, event=event)
    envelope["factory"] = args.factory
    envelope["test_fake_allowed"] = post_fault.backend_kind == "fake" and bool(args.allow_fake)
    envelope["candidate_only"] = True
    args.output.write_text(canonical_json(envelope) + "\n")
    print(json.dumps({"canonical_event_validated": envelope["canonical_event_validated"],
                      "ready_for_training": envelope["receipt"]["ready_for_training"],
                      "positive_action": envelope["receipt"]["corrective_positive_action_mask"]}))


if __name__ == "__main__":
    main()
