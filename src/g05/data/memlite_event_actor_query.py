"""Sealed, additive actor-query preparation records for P107.

The module is deliberately standalone (Python standard library only).  It is
data-preparation plumbing, not a training collator or an authorization gate.
The optional sidecar keeps a pre-label desired query separate from the
post-label answer and binds every sidecar row to a packaged view before a
dataset can project it.

The upstream producer registry is intentionally treated as an opaque,
versioned source.  Its identifiers and source pins are receipts, not values
that this module re-derives from a post-label view.  ``adapt_*`` adds an
explicit versioned normalized spelling for consumers while retaining the
complete upstream row.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTOR_QUERY_SCHEMA = "p107-actor-query-sidecar-v1"
UPSTREAM_PRELABEL_QUERY_SCHEMA = "p107.visual_relation_query_prelabel_registry.v1"
# Kept as the public name used by the first additive implementation; it now
# names the actual producer schema rather than an invented local registry.
PRELABEL_QUERY_SCHEMA = UPSTREAM_PRELABEL_QUERY_SCHEMA
PRELABEL_ADAPTER_SCHEMA = "p107-prelabel-query-adapter-v1"
ACTOR_QUERY_CONTENT_SCHEMA = "p107-actor-query-content-v1"
ACTOR_QUERY_MANIFEST_KEY = "actor_query_sidecar"
ACTOR_QUERY_KINDS = frozenset({
    "goal_satisfaction_counterfactual",
    "attempt_outcome_intent",
    "runtime_high_level_intent",
})
MAX_QUERY_TEXT_CHARS = 2048
MAX_IDENTIFIER_CHARS = 256
MAX_SOURCE_PIN_PATH_CHARS = 4096

_SIDECAR_META_KEYS = frozenset({
    "schema_version", "path", "sha256", "row_count", "prelabel_query_registry",
})
_REGISTRY_META_KEYS = frozenset({
    "schema_version", "adapter_schema_version", "path", "sha256", "row_count",
})
_SIDECAR_ROW_KEYS = frozenset({"schema_version", "view_id", "actor_query"})
_UPSTREAM_REGISTRY_ROW_KEYS = frozenset({
    "canonical_verb", "event_id", "observation_frame", "prelabel_query_id",
    "query_ordinal_within_event", "query_text", "relation_family", "schema_version",
    "skill_id", "source_group_id", "source_pin",
})
_SOURCE_PIN_KEYS = frozenset({
    "event_id", "phase_index_path", "phase_index_record_sha256", "phase_index_sha256",
    "queue_path", "queue_record_sha256", "queue_sha256", "selection_manifest_path",
    "selection_manifest_sha256",
})
_QUERY_COMMON_KEYS = frozenset({"kind", "query_content_sha256", "text"})
_QUERY_KEYS = {
    "goal_satisfaction_counterfactual": _QUERY_COMMON_KEYS | {"prelabel_query_id"},
    "attempt_outcome_intent": _QUERY_COMMON_KEYS | {"attempt_id", "issued_frame"},
    "runtime_high_level_intent": _QUERY_COMMON_KEYS | {"intent_id", "issued_frame"},
}

# Structured fields are denylisted; query text is intentionally not searched.
# A legitimate desired question can contain words such as "completed".
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


def _bounded_text(value: Any, name: str, *, maximum: int = MAX_QUERY_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ActorQueryError(f"{name} must be a non-empty string")
    if len(value) > maximum:
        raise ActorQueryError(f"{name} exceeds the {maximum}-character bound")
    if any(ord(char) < 0x20 for char in value):
        raise ActorQueryError(f"{name} contains a control character")
    return value


def _identifier(value: Any, name: str) -> str:
    return _bounded_text(value, name, maximum=MAX_IDENTIFIER_CHARS)


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
    """Hash only an actor-query variant and immutable text.

    The hash never includes a post-label answer, view identity, object ID,
    segment boundary, or any other target/provider field.
    """
    if kind not in ACTOR_QUERY_KINDS:
        raise ActorQueryError(f"unsupported actor-query kind: {kind!r}")
    _bounded_text(text, "query.text")
    return canonical_sha256({"schema_version": ACTOR_QUERY_CONTENT_SCHEMA,
                             "kind": kind, "text": text})


def _validate_source_pin(pin: Mapping[str, Any], event_id: str) -> dict[str, Any]:
    _exact_keys(pin, _SOURCE_PIN_KEYS, "prelabel_query_registry.source_pin")
    if pin["event_id"] != event_id:
        raise ActorQueryError("prelabel source pin event_id does not match its registry row")
    for key in ("phase_index_record_sha256", "phase_index_sha256", "queue_record_sha256",
                "queue_sha256", "selection_manifest_sha256"):
        _sha256(pin[key], f"prelabel source pin {key}")
    for key in ("phase_index_path", "queue_path", "selection_manifest_path"):
        _bounded_text(pin[key], f"prelabel source pin {key}", maximum=MAX_SOURCE_PIN_PATH_CHARS)
    return json.loads(canonical_json(dict(pin)))


def validate_upstream_prelabel_registry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one actual producer-registry row without rewriting its identity."""
    _exact_keys(row, _UPSTREAM_REGISTRY_ROW_KEYS, "upstream prelabel registry row")
    if row["schema_version"] != UPSTREAM_PRELABEL_QUERY_SCHEMA:
        raise ActorQueryError("prelabel query registry row has the wrong upstream schema")
    event_id = _sha256(row["event_id"], "prelabel registry.event_id")
    source_group_id = _sha256(row["source_group_id"], "prelabel registry.source_group_id")
    query_id = _identifier(row["prelabel_query_id"], "prelabel registry.prelabel_query_id")
    text = _bounded_text(row["query_text"], "prelabel registry.query_text")
    canonical_verb = _identifier(row["canonical_verb"], "prelabel registry.canonical_verb")
    relation_family = _identifier(row["relation_family"], "prelabel registry.relation_family")
    _integer(row["observation_frame"], "prelabel registry.observation_frame")
    _integer(row["query_ordinal_within_event"], "prelabel registry.query_ordinal_within_event")
    _integer(row["skill_id"], "prelabel registry.skill_id")
    pin = _validate_source_pin(row["source_pin"], event_id)
    validated = dict(row)
    validated["source_pin"] = pin
    validated["prelabel_query_id"] = query_id
    validated["source_group_id"] = source_group_id
    validated["event_id"] = event_id
    validated["canonical_verb"] = canonical_verb
    validated["relation_family"] = relation_family
    validated["query_text"] = text
    return json.loads(canonical_json(validated))


def adapt_prelabel_registry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a versioned normalized adapter view of an upstream row.

    ``event_id``/``skill_id``/``query_text`` are retained verbatim in the
    upstream portion and are additionally exposed as explicitly versioned
    ``source_event_id``/``source_skill_id``/``text`` aliases.  The opaque
    producer query ID and full source pin are never replaced or regenerated.
    """
    upstream = validate_upstream_prelabel_registry_row(row)
    adapted = dict(upstream)
    adapted.update({
        "adapter_schema_version": PRELABEL_ADAPTER_SCHEMA,
        "source_event_id": upstream["event_id"],
        "source_skill_id": upstream["skill_id"],
        "text": upstream["query_text"],
        "query_content_sha256": query_content_sha256(
            "goal_satisfaction_counterfactual", upstream["query_text"]),
    })
    return json.loads(canonical_json(adapted))


def adapt_prelabel_registry(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate/adapt a complete upstream registry keyed by opaque query ID."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        adapted = adapt_prelabel_registry_row(row)
        query_id = adapted["prelabel_query_id"]
        if query_id in by_id:
            raise ActorQueryError("duplicate prelabel query ID")
        by_id[query_id] = adapted
    return by_id


def _coerce_adapted_registry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate either a raw producer row or an already-adapted row."""
    if row.get("adapter_schema_version") == PRELABEL_ADAPTER_SCHEMA:
        base = {key: row[key] for key in _UPSTREAM_REGISTRY_ROW_KEYS if key in row}
        if set(base) != set(_UPSTREAM_REGISTRY_ROW_KEYS):
            raise ActorQueryError("adapted prelabel row does not retain the complete upstream row")
        adapted = adapt_prelabel_registry_row(base)
        for key in ("adapter_schema_version", "source_event_id", "source_skill_id", "text",
                    "query_content_sha256"):
            if row.get(key) != adapted[key]:
                raise ActorQueryError(f"adapted prelabel row has a tampered {key}")
        return adapted
    return adapt_prelabel_registry_row(row)


def validate_prelabel_registry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility name: validate and return the versioned adapter row."""
    return adapt_prelabel_registry_row(row)


def validate_prelabel_registry(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return adapt_prelabel_registry(rows)


def _event_skill_matches(registry_row: Mapping[str, Any], event: Mapping[str, Any]) -> None:
    event_id = event.get("event_id")
    if registry_row.get("event_id") != event_id:
        raise ActorQueryError("prelabel query source event does not bind the packaged event")
    source = event.get("source")
    if not isinstance(source, Mapping) or registry_row.get("source_group_id") != source.get("source_group_id"):
        raise ActorQueryError("prelabel query source group does not bind the packaged event")
    skills = event.get("skill_bundle")
    if not isinstance(skills, list):
        raise ActorQueryError("packaged event has no skill bundle for prelabel binding")
    matches = [skill for skill in skills
               if isinstance(skill, Mapping) and
               ((skill.get("skill_id") if skill.get("skill_id") is not None
                 else skill.get("skill_idx")) == registry_row.get("skill_id"))]
    if len(matches) != 1:
        raise ActorQueryError("prelabel query skill_id does not bind exactly one event skill")
    skill = matches[0]
    if str(skill.get("verb", "")).casefold() != str(registry_row.get("canonical_verb", "")).casefold():
        raise ActorQueryError("prelabel query canonical_verb does not bind the event skill")


def _validate_goal_relation_binding(registry_row: Mapping[str, Any], view: Mapping[str, Any]) -> None:
    """Require a frozen query-to-view relation binding.

    Equality is a valid explicit binding.  A producer may instead attach a
    versioned semantic-family receipt to a view in a future protocol extension;
    without that receipt, a differing post-label relation is ambiguous and is
    rejected.  In particular, this function never constructs actor text from
    ``view.goal_relation``.
    """
    relation = view.get("goal_relation")
    if not isinstance(relation, str) or not relation:
        raise ActorQueryError("goal view lacks a bounded goal_relation for query binding")
    if relation == registry_row.get("text"):
        return
    # This optional field is deliberately not accepted by the current canonical
    # protocol.  It exists only for a separately versioned future producer view;
    # if present, it must name the frozen registry family, not an answer/label.
    family = view.get("goal_relation_family")
    if family == registry_row.get("relation_family"):
        return
    raise ActorQueryError("prelabel query text/semantic family does not bind view.goal_relation")


def validate_goal_query_binding(registry_row: Mapping[str, Any], *, view: Mapping[str, Any],
                                event: Mapping[str, Any]) -> None:
    """Validate source identity, skill semantics, and goal relation binding."""
    adapted = _coerce_adapted_registry_row(registry_row)
    _event_skill_matches(adapted, event)
    if view.get("event_id") != adapted["event_id"]:
        raise ActorQueryError("actor query registry source event does not bind the view event")
    if view.get("source_group_id") != adapted["source_group_id"]:
        raise ActorQueryError("actor query registry source group does not bind the view")
    if view.get("label_kind") != "goal_satisfaction_counterfactual":
        raise ActorQueryError("counterfactual goal query is bound to a non-goal view")
    _validate_goal_relation_binding(adapted, view)


def validate_actor_query(query: Mapping[str, Any], *, observation_frame: int | None,
                         view_kind: str | None = None,
                         view: Mapping[str, Any] | None = None,
                         event: Mapping[str, Any] | None = None,
                         prelabel_registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Validate one actor query and, when supplied, its view/event binding."""
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
        query_id = _identifier(query["prelabel_query_id"], "actor_query.prelabel_query_id")
        if view_kind is not None and view_kind != "goal_satisfaction_counterfactual":
            raise ActorQueryError("counterfactual goal query is bound to a non-goal view")
        if prelabel_registry is None:
            raise ActorQueryError("counterfactual goal query requires a verified prelabel registry")
        registry_row = prelabel_registry.get(query_id)
        if registry_row is None:
            raise ActorQueryError("actor query references an absent prelabel registry row")
        adapted = _coerce_adapted_registry_row(registry_row)
        if (adapted["query_content_sha256"] != content_digest or adapted["text"] != text):
            raise ActorQueryError("actor query does not match its prelabel registry content")
        if view is not None:
            if event is None:
                raise ActorQueryError("goal query view binding requires its packaged event")
            validate_goal_query_binding(adapted, view=view, event=event)
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
                           event: Mapping[str, Any] | None = None,
                           prelabel_registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return only validated actor-visible query fields."""
    return validate_actor_query(query, observation_frame=observation_frame, view_kind=view_kind,
                                view=view, event=event, prelabel_registry=prelabel_registry)


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
    # Keep the unresolved path so _sealed_bytes rejects an in-package symlink.
    return candidate


def _sealed_bytes(path: Path, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ActorQueryError(f"{name} is missing or symlinked")
    if path.stat().st_mode & 0o222:
        raise ActorQueryError(f"{name} is writable and not sealed")
    return path.read_bytes()


def _read_jsonl(path: Path, name: str) -> tuple[bytes, list[dict[str, Any]]]:
    raw = _sealed_bytes(path, name)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ActorQueryError(f"{name} contains invalid JSONL: {error}") from error
        if not isinstance(value, dict):
            raise ActorQueryError(f"{name} contains a non-object JSONL row")
        rows.append(value)
    return raw, rows


def validate_sidecar_manifest(metadata: Mapping[str, Any], *, files: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact optional registration metadata without reading files."""
    _exact_keys(metadata, _SIDECAR_META_KEYS, ACTOR_QUERY_MANIFEST_KEY)
    if metadata["schema_version"] != ACTOR_QUERY_SCHEMA:
        raise ActorQueryError("actor-query sidecar has the wrong schema")
    path_value = metadata["path"]
    if (not isinstance(path_value, str) or Path(path_value).is_absolute() or
            ".." in Path(path_value).parts):
        raise ActorQueryError("actor-query sidecar path is not a safe relative path")
    digest = _sha256(metadata["sha256"], f"{ACTOR_QUERY_MANIFEST_KEY}.sha256")
    row_count = _integer(metadata["row_count"], f"{ACTOR_QUERY_MANIFEST_KEY}.row_count")
    receipt = files.get(path_value)
    if not isinstance(receipt, Mapping) or receipt.get("sha256") != digest:
        raise ActorQueryError(f"{ACTOR_QUERY_MANIFEST_KEY} is not registered in sealed file receipts")
    if receipt.get("rows") != row_count:
        raise ActorQueryError(f"{ACTOR_QUERY_MANIFEST_KEY} manifest/file row count mismatch")
    registry = metadata["prelabel_query_registry"]
    if registry is not None:
        _exact_keys(registry, _REGISTRY_META_KEYS, f"{ACTOR_QUERY_MANIFEST_KEY}.prelabel_query_registry")
        if registry["schema_version"] != UPSTREAM_PRELABEL_QUERY_SCHEMA:
            raise ActorQueryError("prelabel registry has the wrong upstream schema")
        if registry["adapter_schema_version"] != PRELABEL_ADAPTER_SCHEMA:
            raise ActorQueryError("prelabel registry has the wrong adapter schema")
        registry_path = registry["path"]
        if (not isinstance(registry_path, str) or Path(registry_path).is_absolute() or
                ".." in Path(registry_path).parts):
            raise ActorQueryError("prelabel registry path is not safe")
        registry_digest = _sha256(registry["sha256"], "prelabel_query_registry.sha256")
        registry_rows = _integer(registry["row_count"], "prelabel_query_registry.row_count")
        registry_receipt = files.get(registry_path)
        if (not isinstance(registry_receipt, Mapping) or
                registry_receipt.get("sha256") != registry_digest or
                registry_receipt.get("rows") != registry_rows):
            raise ActorQueryError("prelabel registry is not fully registered in sealed file receipts")
    return json.loads(canonical_json(dict(metadata)))


def validate_sidecar_bindings(rows: Mapping[str, Mapping[str, Any]], *,
                              prelabel_registry: Mapping[str, Mapping[str, Any]],
                              views_by_id: Mapping[str, Mapping[str, Any]],
                              events_by_id: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate every sidecar row against its packaged view/event.

    This is intentionally independent of a selected dataset ``view_kind``;
    callers must run it before filtering, otherwise a malformed OTHER-kind
    binding could remain hidden from the selected dataset.
    """
    for view_id, query in rows.items():
        view = views_by_id.get(view_id)
        if view is None:
            raise ActorQueryError("actor-query sidecar references an unpackaged view")
        event = events_by_id.get(view.get("event_id"))
        if event is None:
            raise ActorQueryError("actor-query view references an unpackaged event")
        validate_actor_query(query, observation_frame=view.get("observation_frame"),
                             view_kind=view.get("label_kind"), view=view, event=event,
                             prelabel_registry=prelabel_registry)


def read_actor_query_sidecar(package_root: str | Path, metadata: Mapping[str, Any], *,
                             files: Mapping[str, Any], expected_view_ids: set[str] | None = None,
                             views_by_id: Mapping[str, Mapping[str, Any]] | None = None,
                             events_by_id: Mapping[str, Mapping[str, Any]] | None = None
                             ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read the sealed sidecar and optionally validate all view bindings."""
    root = Path(package_root)
    metadata = validate_sidecar_manifest(metadata, files=files)
    sidecar_path = _safe_relative_path(root, metadata["path"], ACTOR_QUERY_MANIFEST_KEY)
    raw_sidecar, sidecar_rows = _read_jsonl(sidecar_path, ACTOR_QUERY_MANIFEST_KEY)
    if hashlib.sha256(raw_sidecar).hexdigest() != metadata["sha256"]:
        raise ActorQueryError("actor-query sidecar receipt mismatch")
    if len(sidecar_rows) != metadata["row_count"]:
        raise ActorQueryError("actor-query sidecar row count mismatch")
    sidecar_receipt = files.get(metadata["path"])
    if not isinstance(sidecar_receipt, Mapping) or sidecar_receipt.get("bytes") != len(raw_sidecar):
        raise ActorQueryError("actor-query sidecar byte receipt mismatch")

    registry_by_id: dict[str, dict[str, Any]] = {}
    registry_meta = metadata["prelabel_query_registry"]
    if registry_meta is not None:
        registry_path = _safe_relative_path(root, registry_meta["path"], "prelabel_query_registry")
        raw_registry, registry_rows = _read_jsonl(registry_path, "prelabel_query_registry")
        if hashlib.sha256(raw_registry).hexdigest() != registry_meta["sha256"]:
            raise ActorQueryError("prelabel registry receipt mismatch")
        if len(registry_rows) != registry_meta["row_count"]:
            raise ActorQueryError("prelabel registry row count mismatch")
        registry_receipt = files.get(registry_meta["path"])
        if not isinstance(registry_receipt, Mapping) or registry_receipt.get("bytes") != len(raw_registry):
            raise ActorQueryError("prelabel registry byte receipt mismatch")
        registry_by_id = adapt_prelabel_registry(registry_rows)

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
        validated = validate_actor_query(query, observation_frame=None,
                                         prelabel_registry=registry_by_id)
        by_view_id[view_id] = validated
    if any(query["kind"] == "goal_satisfaction_counterfactual" for query in by_view_id.values()) and registry_meta is None:
        raise ActorQueryError("goal query sidecar requires a registered prelabel registry")
    if views_by_id is not None or events_by_id is not None:
        if views_by_id is None or events_by_id is None:
            raise ActorQueryError("sidecar binding validation requires both views and events")
        validate_sidecar_bindings(by_view_id, prelabel_registry=registry_by_id,
                                  views_by_id=views_by_id, events_by_id=events_by_id)
    return by_view_id, registry_by_id


__all__ = [
    "ACTOR_QUERY_CONTENT_SCHEMA", "ACTOR_QUERY_KINDS", "ACTOR_QUERY_MANIFEST_KEY",
    "ACTOR_QUERY_SCHEMA", "ActorQueryError", "MAX_QUERY_TEXT_CHARS", "PRELABEL_ADAPTER_SCHEMA",
    "PRELABEL_QUERY_SCHEMA", "UPSTREAM_PRELABEL_QUERY_SCHEMA", "actor_query_projection",
    "adapt_prelabel_registry", "adapt_prelabel_registry_row", "canonical_json", "canonical_sha256",
    "query_content_sha256", "read_actor_query_sidecar", "validate_actor_query",
    "validate_goal_query_binding", "validate_prelabel_registry", "validate_prelabel_registry_row",
    "validate_sidecar_bindings", "validate_sidecar_manifest", "validate_upstream_prelabel_registry_row",
]
