"""Additive, sealed actor-query records for P107 event views.

This module is intentionally standalone: it imports only the Python standard
library and does not import the P107 protocol, a dataset, torch, a simulator,
or a model.  It validates an optional package sidecar without turning a
preparation release into a training authorization.

The sidecar is deliberately separate from the canonical post-label view.  A
counterfactual goal is a desired query supplied by a frozen pre-label registry;
its answer is still a target and is never copied into ``actor_query``.  An
attempt query must bind an actual issued attempt at or before the observation.
Runtime intent is the only deployable variant.  Low-level BC binding is
checked by the dataset projection layer, where the executed action view is
available.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ACTOR_QUERY_SCHEMA = "p107-actor-query-sidecar-v1"
PRELABEL_QUERY_SCHEMA = "p107-prelabel-query-v1"
ACTOR_QUERY_MANIFEST_KEY = "actor_query_sidecar"
ACTOR_QUERY_KINDS = frozenset({
    "goal_satisfaction_counterfactual",
    "attempt_outcome_intent",
    "runtime_high_level_intent",
})
MAX_QUERY_TEXT_CHARS = 2048
MAX_IDENTIFIER_CHARS = 256

_SIDECAR_META_KEYS = frozenset({
    "schema_version", "path", "sha256", "row_count", "prelabel_query_registry",
})
_REGISTRY_META_KEYS = frozenset({"schema_version", "path", "sha256", "row_count"})
_SIDECAR_ROW_KEYS = frozenset({"schema_version", "view_id", "actor_query"})
_REGISTRY_ROW_KEYS = frozenset({
    "schema_version", "prelabel_query_id", "query_content_sha256", "text",
    "source_event_id", "source_skill_id",
})
_QUERY_COMMON_KEYS = frozenset({"kind", "query_content_sha256", "text"})
_QUERY_KEYS = {
    "goal_satisfaction_counterfactual": _QUERY_COMMON_KEYS | {"prelabel_query_id"},
    "attempt_outcome_intent": _QUERY_COMMON_KEYS | {"attempt_id", "issued_frame"},
    "runtime_high_level_intent": _QUERY_COMMON_KEYS | {"intent_id", "issued_frame"},
}

# These names are not accepted in any structured query payload.  Query text is
# intentionally not searched: a legitimate desired question may contain words
# such as "completed".  Exact variant allowlists above are the primary guard.
FORBIDDEN_QUERY_FIELDS = frozenset({
    "answer", "answers", "goal_satisfaction", "attempt_outcome", "result", "review",
    "evidence", "audit", "privileged", "oracle", "simulator", "ground_truth",
    "object_id", "object_instance_id", "segment_end_frame", "future_frame",
    "label_end_frame", "success", "failure", "recovery", "state", "pose",
})


class ActorQueryError(ValueError):
    """A sidecar/query record cannot safely cross the data boundary."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ActorQueryError(f"non-canonical actor-query JSON: {error}") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        raise ActorQueryError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ActorQueryError(f"{name} must be an integer >= {minimum}")
    return value


def _bounded_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ActorQueryError(f"{name} must be a non-empty string")
    if len(value) > MAX_QUERY_TEXT_CHARS:
        raise ActorQueryError(f"{name} exceeds the {MAX_QUERY_TEXT_CHARS}-character bound")
    if any(ord(char) < 0x20 for char in value):
        raise ActorQueryError(f"{name} contains a control character")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_IDENTIFIER_CHARS:
        raise ActorQueryError(f"{name} must be a bounded non-empty identifier")
    if any(ord(char) < 0x20 for char in value):
        raise ActorQueryError(f"{name} contains a control character")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ActorQueryError(f"{name} must be an object")
    keys = set(value)
    if keys != set(expected):
        unexpected = sorted(str(key) for key in keys - set(expected))
        missing = sorted(str(key) for key in set(expected) - keys)
        raise ActorQueryError(f"{name} has invalid fields; missing={missing}, unexpected={unexpected}")
    if any(not isinstance(key, str) or key in FORBIDDEN_QUERY_FIELDS for key in keys):
        raise ActorQueryError(f"{name} contains a forbidden structured field")


def query_content_sha256(kind: str, text: str) -> str:
    """Hash only the immutable query kind and text, never a post-label answer."""
    if kind not in ACTOR_QUERY_KINDS:
        raise ActorQueryError(f"unsupported actor-query kind: {kind!r}")
    _bounded_text(text, "query.text")
    return canonical_sha256({"schema_version": ACTOR_QUERY_SCHEMA, "kind": kind, "text": text})


def prelabel_query_id(source_event_id: str, source_skill_id: int, query_content_digest: str) -> str:
    """Derive a source-event/skill-bound pre-label identity.

    The post-label view identity is deliberately absent.  Content hashes stay
    independent so identical wording on two source events still gets distinct
    registry identities while preserving an auditable text digest.
    """
    source_event_id = _sha256(source_event_id, "source_event_id")
    source_skill_id = _integer(source_skill_id, "source_skill_id")
    _sha256(query_content_digest, "query_content_sha256")
    return canonical_sha256({
        "schema_version": PRELABEL_QUERY_SCHEMA,
        "kind": "goal_satisfaction_counterfactual",
        "source_event_id": source_event_id,
        "source_skill_id": source_skill_id,
        "query_content_sha256": query_content_digest,
    })


def validate_prelabel_registry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable desired-query registry row."""
    _exact_keys(row, _REGISTRY_ROW_KEYS, "prelabel_query_registry row")
    if row["schema_version"] != PRELABEL_QUERY_SCHEMA:
        raise ActorQueryError("prelabel query registry row has the wrong schema")
    text = _bounded_text(row["text"], "prelabel_query_registry.text")
    source_event_id = _sha256(row["source_event_id"], "prelabel_query_registry.source_event_id")
    source_skill_id = _integer(row["source_skill_id"], "prelabel_query_registry.source_skill_id")
    content_digest = _sha256(row["query_content_sha256"],
                             "prelabel_query_registry.query_content_sha256")
    expected_content = query_content_sha256("goal_satisfaction_counterfactual", text)
    if content_digest != expected_content:
        raise ActorQueryError("prelabel query content hash does not match its text")
    query_id = _sha256(row["prelabel_query_id"], "prelabel_query_registry.prelabel_query_id")
    if query_id != prelabel_query_id(source_event_id, source_skill_id, content_digest):
        raise ActorQueryError("prelabel query ID does not match its content hash")
    return json.loads(canonical_json(dict(row)))


def validate_prelabel_registry(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate a complete registry and return rows keyed by query ID."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        validated = validate_prelabel_registry_row(row)
        query_id = validated["prelabel_query_id"]
        if query_id in by_id:
            raise ActorQueryError("duplicate prelabel query ID")
        by_id[query_id] = validated
    return by_id


def validate_actor_query(query: Mapping[str, Any], *, observation_frame: int | None,
                         view_kind: str | None = None,
                         view: Mapping[str, Any] | None = None,
                         prelabel_registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Validate and canonicalize one actor query.

    ``view`` is optional for standalone validation.  Dataset projection passes
    it so goal/attempt bindings and the low-BC invariant can be checked without
    copying any target fields into the projected query.
    """
    if observation_frame is not None:
        _integer(observation_frame, "observation_frame")
    if not isinstance(query, Mapping):
        raise ActorQueryError("actor_query must be an object")
    kind = query.get("kind")
    if kind not in ACTOR_QUERY_KINDS:
        raise ActorQueryError("actor_query.kind is not an approved variant")
    _exact_keys(query, _QUERY_KEYS[kind], "actor_query")
    text = _bounded_text(query["text"], "actor_query.text")
    content_digest = _sha256(query["query_content_sha256"], "actor_query.query_content_sha256")
    if content_digest != query_content_sha256(kind, text):
        raise ActorQueryError("actor query content hash does not match kind/text")

    if kind == "goal_satisfaction_counterfactual":
        query_id = _sha256(query["prelabel_query_id"], "actor_query.prelabel_query_id")
        if view_kind is not None and view_kind != "goal_satisfaction_counterfactual":
            raise ActorQueryError("counterfactual goal query is bound to a non-goal view")
        if prelabel_registry is None:
            raise ActorQueryError("counterfactual goal query requires a verified prelabel registry")
        registry_row = prelabel_registry.get(query_id)
        if registry_row is None:
            raise ActorQueryError("actor query references an absent prelabel registry row")
        validate_prelabel_registry_row(registry_row)
        if (registry_row["query_content_sha256"] != content_digest or
                registry_row["text"] != text):
            raise ActorQueryError("actor query does not match its prelabel registry content")
        if (view is not None and
                registry_row["source_event_id"] != view.get("event_id")):
            raise ActorQueryError("actor query registry source event does not bind the view event")
    elif kind == "attempt_outcome_intent":
        attempt_id = _identifier(query["attempt_id"], "actor_query.attempt_id")
        issued_frame = _integer(query["issued_frame"], "actor_query.issued_frame")
        if observation_frame is not None and issued_frame > observation_frame:
            raise ActorQueryError("attempt intent is issued after the observation anchor")
        if view_kind is not None and view_kind != "attempt_outcome":
            raise ActorQueryError("attempt intent is bound to a non-attempt view")
        if view is not None and view.get("attempt_id") != attempt_id:
            raise ActorQueryError("attempt intent does not bind the view's actual attempt")
    else:
        intent_id = _identifier(query["intent_id"], "actor_query.intent_id")
        issued_frame = _integer(query["issued_frame"], "actor_query.issued_frame")
        if observation_frame is not None and issued_frame > observation_frame:
            raise ActorQueryError("runtime intent is issued after the observation anchor")
        if (view is not None and view_kind == "corrective_action" and
                view.get("low_action_supervision_mask") is True and
                view.get("action_intent_bundle_id") != intent_id):
            raise ActorQueryError("low-BC corrective view does not bind the runtime intent bundle")

    return json.loads(canonical_json(dict(query)))


def actor_query_projection(query: Mapping[str, Any], *, observation_frame: int,
                           view_kind: str | None = None,
                           view: Mapping[str, Any] | None = None,
                           prelabel_registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return only the validated, actor-visible query payload."""
    return validate_actor_query(query, observation_frame=observation_frame, view_kind=view_kind,
                                view=view, prelabel_registry=prelabel_registry)


def _safe_relative_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ActorQueryError(f"{name} must be a relative package path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ActorQueryError(f"{name} escapes the sealed package")
    candidate = root / path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ActorQueryError(f"{name} escapes the sealed package") from error
    # Preserve the unresolved candidate so _sealed_bytes can reject a symlink
    # even when it points to another file inside the package.
    return candidate


def _sealed_bytes(path: Path, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ActorQueryError(f"{name} is missing or symlinked")
    if path.stat().st_mode & 0o222:
        raise ActorQueryError(f"{name} is writable and not sealed")
    return path.read_bytes()


def _file_sha256(path: Path, name: str) -> str:
    return hashlib.sha256(_sealed_bytes(path, name)).hexdigest()


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _sealed_bytes(path, name).splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ActorQueryError(f"{name} contains invalid JSONL: {error}") from error
        if not isinstance(value, dict):
            raise ActorQueryError(f"{name} contains a non-object JSONL row")
        rows.append(value)
    return rows


def _validate_file_receipt(files: Mapping[str, Any], path_value: str, digest: str, name: str) -> None:
    receipt = files.get(path_value)
    if not isinstance(receipt, Mapping) or receipt.get("sha256") != digest:
        raise ActorQueryError(f"{name} is not registered in the sealed package file receipts")


def validate_sidecar_manifest(metadata: Mapping[str, Any], *, files: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact optional manifest registration, without reading files."""
    _exact_keys(metadata, _SIDECAR_META_KEYS, ACTOR_QUERY_MANIFEST_KEY)
    if metadata["schema_version"] != ACTOR_QUERY_SCHEMA:
        raise ActorQueryError("actor-query sidecar has the wrong schema")
    path_value = metadata["path"]
    if not isinstance(path_value, str) or Path(path_value).is_absolute() or ".." in Path(path_value).parts:
        raise ActorQueryError("actor-query sidecar path is not a safe relative path")
    digest = _sha256(metadata["sha256"], f"{ACTOR_QUERY_MANIFEST_KEY}.sha256")
    _integer(metadata["row_count"], f"{ACTOR_QUERY_MANIFEST_KEY}.row_count")
    _validate_file_receipt(files, path_value, digest, ACTOR_QUERY_MANIFEST_KEY)
    registry = metadata["prelabel_query_registry"]
    if registry is not None:
        _exact_keys(registry, _REGISTRY_META_KEYS, f"{ACTOR_QUERY_MANIFEST_KEY}.prelabel_query_registry")
        if registry["schema_version"] != PRELABEL_QUERY_SCHEMA:
            raise ActorQueryError("prelabel registry has the wrong schema")
        registry_path = registry["path"]
        if (not isinstance(registry_path, str) or Path(registry_path).is_absolute() or
                ".." in Path(registry_path).parts):
            raise ActorQueryError("prelabel registry path is not safe")
        registry_digest = _sha256(registry["sha256"], "prelabel_query_registry.sha256")
        _integer(registry["row_count"], "prelabel_query_registry.row_count")
        _validate_file_receipt(files, registry_path, registry_digest, "prelabel_query_registry")
    return json.loads(canonical_json(dict(metadata)))


def read_actor_query_sidecar(package_root: str | Path, metadata: Mapping[str, Any], *,
                             files: Mapping[str, Any], expected_view_ids: set[str] | None = None
                             ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read and verify the optional sealed sidecar and its prelabel registry.

    The return value is ``(query_by_view_id, prelabel_registry_by_id)``.  A goal
    query is rejected unless its registry is present, sealed, and content-bound;
    callers cannot silently fall back to a post-label goal relation.
    """
    root = Path(package_root)
    metadata = validate_sidecar_manifest(metadata, files=files)
    sidecar_path = _safe_relative_path(root, metadata["path"], ACTOR_QUERY_MANIFEST_KEY)
    if _file_sha256(sidecar_path, ACTOR_QUERY_MANIFEST_KEY) != metadata["sha256"]:
        raise ActorQueryError("actor-query sidecar receipt mismatch")
    sidecar_rows = _read_jsonl(sidecar_path, ACTOR_QUERY_MANIFEST_KEY)
    if len(sidecar_rows) != metadata["row_count"]:
        raise ActorQueryError("actor-query sidecar row count mismatch")

    registry_by_id: dict[str, dict[str, Any]] = {}
    registry_meta = metadata["prelabel_query_registry"]
    if registry_meta is not None:
        registry_path = _safe_relative_path(root, registry_meta["path"], "prelabel_query_registry")
        if _file_sha256(registry_path, "prelabel_query_registry") != registry_meta["sha256"]:
            raise ActorQueryError("prelabel registry receipt mismatch")
        registry_rows = _read_jsonl(registry_path, "prelabel_query_registry")
        if len(registry_rows) != registry_meta["row_count"]:
            raise ActorQueryError("prelabel registry row count mismatch")
        registry_by_id = validate_prelabel_registry(registry_rows)

    by_view_id: dict[str, dict[str, Any]] = {}
    for row in sidecar_rows:
        _exact_keys(row, _SIDECAR_ROW_KEYS, "actor-query sidecar row")
        if row["schema_version"] != ACTOR_QUERY_SCHEMA:
            raise ActorQueryError("actor-query sidecar row has the wrong schema")
        view_id = _sha256(row["view_id"], "actor-query sidecar.view_id")
        if expected_view_ids is not None and view_id not in expected_view_ids:
            raise ActorQueryError("actor-query sidecar references an unpackaged view")
        if view_id in by_view_id:
            raise ActorQueryError("duplicate actor-query binding for one postlabel view")
        query = row["actor_query"]
        # Sidecar-level validation has no view clock.  Goal registry binding is
        # still mandatory; frame/view/low-BC checks happen during projection.
        validated = validate_actor_query(query, observation_frame=None,
                                         prelabel_registry=registry_by_id)
        by_view_id[view_id] = validated
    if any(query["kind"] == "goal_satisfaction_counterfactual" for query in by_view_id.values()) and registry_meta is None:
        raise ActorQueryError("goal query sidecar requires a registered prelabel registry")
    return by_view_id, registry_by_id


__all__ = [
    "ACTOR_QUERY_KINDS", "ACTOR_QUERY_MANIFEST_KEY", "ACTOR_QUERY_SCHEMA", "ActorQueryError",
    "MAX_QUERY_TEXT_CHARS", "PRELABEL_QUERY_SCHEMA", "actor_query_projection", "canonical_json",
    "canonical_sha256", "prelabel_query_id", "query_content_sha256", "read_actor_query_sidecar",
    "validate_actor_query", "validate_prelabel_registry", "validate_prelabel_registry_row",
    "validate_sidecar_manifest",
]
