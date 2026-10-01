#!/usr/bin/env python3
"""Run a registered snapshot round-trip through an owner-supplied adapter.

This script never imports OmniGibson, chooses a GPU, or creates an evaluator.
Its ``--factory`` is intentionally explicit: a future live ticket must provide
the frozen evaluator/worktree and a public-API adapter.  The test fake requires
``--allow-fake`` and always produces a non-trainable receipt.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from g05.recovery.common import RecoveryContractError, canonical_json, validate_raw23_actions
from g05.recovery.snapshot import SnapshotAdapter


def _factory(spec: str) -> Any:
    if ":" not in spec:
        raise RecoveryContractError("Factory must be module:callable")
    module_name, name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), name, None)
    if not callable(factory):
        raise RecoveryContractError("Snapshot factory is not callable")
    return factory()


def _parts(value: Any) -> tuple[SnapshotAdapter, Any]:
    if isinstance(value, dict):
        snapshots, actions = value.get("snapshot_adapter"), value.get("action_backend")
    else:
        snapshots, actions = value
    if not isinstance(snapshots, SnapshotAdapter) or not callable(getattr(actions, "step", None)):
        raise RecoveryContractError("Factory must return snapshot_adapter and action_backend.step")
    return snapshots, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:callable; owner-controlled evaluator adapter")
    parser.add_argument("--actions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="p107_snapshot_roundtrip")
    parser.add_argument("--allow-fake", action="store_true", help="CPU test only; result remains non-trainable")
    args = parser.parse_args()
    actions = validate_raw23_actions(json.loads(args.actions_json.read_text()))
    snapshots, backend = _parts(_factory(args.factory))
    backend_kind = snapshots.backend.backend_kind
    if backend_kind == "fake" and not args.allow_fake:
        raise RecoveryContractError("Refusing fake snapshot backend without --allow-fake")
    if backend_kind not in {"live", "fake"}:
        raise RecoveryContractError("Refusing an unverified snapshot integration; complete the live readiness gate first")

    def execute(chunk: list[list[float]]) -> int:
        count = 0
        for action in chunk:
            backend.step(action)
            count += 1
        return count

    receipt = snapshots.roundtrip(actions, execute, label=args.label).public()
    receipt["factory"] = args.factory
    receipt["test_fake_allowed"] = backend_kind == "fake" and bool(args.allow_fake)
    args.output.write_text(canonical_json(receipt) + "\n")
    print(json.dumps({"passed": receipt["passed"], "trainable": receipt["trainable"]}))


if __name__ == "__main__":
    main()
