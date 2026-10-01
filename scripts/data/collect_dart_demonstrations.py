#!/usr/bin/env python3
"""Run an explicitly supplied candidate-only DART collection factory.

This command is deliberately inert without ``--factory``: it never imports a
simulator, selects a GPU, opens SSH, or creates data authority.  The later
simulator owner supplies an adapter factory after independent review.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from g05.recovery.common import RecoveryContractError, canonical_json, canonical_sha256
from g05.recovery.dart_collection import CANDIDATE_ONLY_FLAGS, DartCollectionResult


def _load_factory(spec: str) -> Callable[[Mapping[str, Any]], DartCollectionResult]:
    if spec.count(":") != 1:
        raise RecoveryContractError("factory must be module_path:callable_name")
    module_name, callable_name = spec.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), callable_name)
    except (ImportError, AttributeError) as exc:
        raise RecoveryContractError(f"cannot import DART collection factory {spec!r}") from exc
    if not callable(factory):
        raise RecoveryContractError("DART collection factory must be callable")
    return factory


def _verify_candidate_result(result: DartCollectionResult) -> dict[str, Any]:
    if not isinstance(result, DartCollectionResult):
        raise RecoveryContractError("factory must return DartCollectionResult")
    public = result.public()
    _require_exact_candidate_flags(public, scope="collection result")
    for record in public["records"]:
        if not isinstance(record, Mapping):
            raise RecoveryContractError("DART CLI requires candidate records to be mappings")
        _require_exact_candidate_flags(record, scope="candidate record")
        expected = record.get("candidate_sha256")
        without_hash = {key: value for key, value in record.items() if key != "candidate_sha256"}
        if expected != canonical_sha256(without_hash):
            raise RecoveryContractError("DART candidate receipt hash does not match content")
    public["collection_result_sha256"] = canonical_sha256(public)
    return public


def _require_exact_candidate_flags(payload: Mapping[str, Any], *, scope: str) -> None:
    for field, expected in CANDIDATE_ONLY_FLAGS.items():
        actual = payload.get(field, object())
        if type(actual) is not type(expected) or actual != expected:
            raise RecoveryContractError(f"{scope} must keep {field}={expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path, help="JSON request consumed by the explicit adapter factory")
    parser.add_argument("--factory", required=True, help="adapter factory as module_path:callable_name")
    parser.add_argument("--output", required=True, type=Path, help="new candidate JSON output; must not already exist")
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise RecoveryContractError("DART request must be a JSON object")
        factory = _load_factory(args.factory)
        result = _verify_candidate_result(factory(request))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(result))
            handle.write("\n")
        print(json.dumps({"status": "candidate_written", "sha256": result["collection_result_sha256"], "output": str(args.output)}))
        return 0
    except (OSError, json.JSONDecodeError, RecoveryContractError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
