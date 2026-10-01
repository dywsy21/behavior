"""Seal P107 event labels without turning candidates into a training release.

This publisher deliberately sits after the immutable event index and before any
future training integration.  It copies only compact metadata/actual controls
into a new directory, verifies every canonical id again, and always publishes
``ready_for_training=false``.  A later, independently reviewed authorization
is intentionally outside this tool's scope.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]
RELEASE_SCHEMA = "memlite-event-label-release-v1"
ANNOTATION_SCHEMA = "memlite-event-label-annotations-v1"
USAGE_ROLES = frozenset(("student_candidate", "annotation_calibration", "evaluation_only"))
PRIVATE_TOKENS = frozenset((
    "private", "pose", "predicate", "snapshot", "teacher", "future", "truth", "oracle",
    "simulator", "provider", "contact_state", "robot_state",
))


def _load_protocol() -> Any:
    path = REPO / "src/g05/data/memlite_event_protocol.py"
    spec = importlib.util.spec_from_file_location("p107_memlite_event_protocol_pack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P107 protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_protocol = _load_protocol()
SCHEMA_VERSION = _protocol.SCHEMA_VERSION
ACTION_HORIZON = _protocol.ACTION_HORIZON
LABEL_KINDS = _protocol.LABEL_KINDS
canonical_json = _protocol.canonical_json
canonical_sha256 = _protocol.canonical_sha256
validate_event = _protocol.validate_event
validate_goal_satisfaction_view = _protocol.validate_goal_satisfaction_view
validate_attempt_outcome_view = _protocol.validate_attempt_outcome_view
validate_recovery_decision_view = _protocol.validate_recovery_decision_view
validate_corrective_action_view = _protocol.validate_corrective_action_view
actor_evidence_projection = _protocol.actor_evidence_projection
project_action_23_to_27 = _protocol.project_action_23_to_27
action_payload_sha256 = _protocol.action_payload_sha256
raw_action_payload_sha256 = _protocol.raw_action_payload_sha256
load_corrective_action_authority = _protocol.load_corrective_action_authority
authority_raw_action_payload = _protocol.authority_raw_action_payload


def strict_json_bytes(data: bytes, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number in {name}: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def read_json(path: Path) -> Any:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular input file is missing: {path}")
    return strict_json_bytes(path.read_bytes(), name=str(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular input file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        value = strict_json_bytes(line, name=f"{path}:{number}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {path}:{number} is not an object")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_json(row).encode("utf-8") + b"\n")
            count += 1
    return {"sha256": sha256_file(path), "rows": count, "bytes": path.stat().st_size}


def _assert_no_sidecar(value: Any, path: str = "annotations") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "sidecar" in str(key).casefold():
                raise ValueError(f"old MemLiteSidecar field is forbidden in P107 package: {path}.{key}")
            _assert_no_sidecar(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_sidecar(child, f"{path}[{index}]")


def _assert_actor_projection_safe(view: Mapping[str, Any]) -> dict[str, Any]:
    """Reject provider/private facts even when nested in an evidence reference."""
    evidence = actor_evidence_projection(view)

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key).casefold()
                if any(bad in token for bad in PRIVATE_TOKENS):
                    raise ValueError(f"actor evidence leaks private/provider field at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            token = value.casefold()
            if any(bad in token for bad in ("private pose", "oracle", "future truth", "simulator snapshot", "teacher action")):
                raise ValueError(f"actor evidence contains private/provider text at {path}")

    walk(evidence, "actor_evidence")
    return evidence


def _validate_view(view: Mapping[str, Any], event: Mapping[str, Any], *, authority: Any | None = None) -> None:
    kind = view.get("label_kind")
    validators = {
        "goal_satisfaction_counterfactual": validate_goal_satisfaction_view,
        "attempt_outcome": validate_attempt_outcome_view,
        "recovery_decision": validate_recovery_decision_view,
        "corrective_action": validate_corrective_action_view,
    }
    if kind not in validators:
        raise ValueError(f"unknown P107 label view: {kind}")
    if kind == "corrective_action":
        validators[kind](view, event, authority=authority)
    else:
        validators[kind](view, event)
    _assert_actor_projection_safe(view)


def _read_sealed_index(index_dir: Path, *, expected_inventory_seal_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    index_dir = Path(index_dir)
    if index_dir.is_symlink() or not index_dir.is_dir():
        raise ValueError("event index must be a regular sealed directory")
    manifest = read_json(index_dir / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "memlite-event-index-v1":
        raise ValueError("not a P107 immutable event index")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("event index lacks sealed file receipts")
    paths = {"source_groups.jsonl": index_dir / "source_groups.jsonl",
             "event_candidates.jsonl": index_dir / "event_candidates.jsonl"}
    for name, path in paths.items():
        receipt = files.get(name)
        if not isinstance(receipt, Mapping) or receipt.get("sha256") != sha256_file(path):
            raise ValueError(f"event index receipt mismatch: {name}")
    seal_path = index_dir / "inventory_seal.json"
    if not _is_sha(expected_inventory_seal_sha256):
        raise ValueError("publisher needs an externally recorded immutable index inventory seal SHA-256")
    actual_inventory_seal_sha256 = sha256_file(seal_path)
    if actual_inventory_seal_sha256 != expected_inventory_seal_sha256:
        raise ValueError("externally recorded immutable index inventory seal SHA-256 does not match input")
    seal = read_json(seal_path)
    expected_seal = {"schema_version": "memlite-event-inventory-seal-v1",
                     "index_manifest_sha256": sha256_file(index_dir / "manifest.json"),
                     "source_release_manifest_sha256": manifest.get("source_release_manifest_sha256"),
                     "coverage_expectations_sha256": manifest.get("coverage_expectations_sha256"),
                     "expected_payload_files": sorted(paths), "payload_files": files}
    if seal != expected_seal:
        raise ValueError("event index inventory seal does not bind exact manifest/payload files")
    groups, events = read_jsonl(paths["source_groups.jsonl"]), read_jsonl(paths["event_candidates.jsonl"])
    if not groups or not events:
        raise ValueError("event index must contain source groups and event candidates")
    return groups, events, manifest, actual_inventory_seal_sha256


def _validate_group_inventory(groups: list[Mapping[str, Any]], events: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        group_id, role, split = group.get("source_group_id"), group.get("usage_role"), group.get("original_split")
        if not _is_sha(group_id) or role not in USAGE_ROLES or split not in {"train", "eval"}:
            raise ValueError("malformed immutable source-group inventory")
        if group_id in by_id:
            raise ValueError("duplicate source-group inventory row")
        if (split == "eval") != (role == "evaluation_only"):
            raise ValueError("source group attempts to change its inherited original split")
        if int(group.get("task_instance_id", 0)) >= 301 and role != "evaluation_only":
            raise ValueError("public_test source group cannot enter train or calibration")
        by_id[group_id] = dict(group)
    seen_events: set[str] = set()
    published_episode_identity: dict[int, tuple[Any, ...]] = {}
    for event in events:
        validate_event(event)
        event_id, source = event["event_id"], event["source"]
        if event_id in seen_events:
            raise ValueError("duplicate event_id in package source")
        seen_events.add(event_id)
        episode_identity = (source["source_release_manifest_sha256"], source["task_index"],
                            source["task_instance_id"], source["raw_episode_id"])
        prior = published_episode_identity.setdefault(source["episode_index"], episode_identity)
        if prior != episode_identity:
            raise ValueError("published episode_index maps to conflicting immutable source episodes")
        group = by_id.get(source["source_group_id"])
        if group is None:
            raise ValueError("event has no immutable source-group inventory row")
        if event.get("usage_role") != group["usage_role"]:
            raise ValueError("event usage role differs from immutable source-group policy")
        for key in ("source_release_manifest_sha256", "task_index", "task_instance_id", "original_split"):
            if source[key] != group[key]:
                raise ValueError(f"event source {key} conflicts with inherited source-group provenance")
    return by_id


def _validate_review(review: Mapping[str, Any], view: Mapping[str, Any]) -> dict[str, Any]:
    required = ("view_id", "candidate_annotated", "parent_reviewed", "accepted_auxiliary",
                "outcome_validated", "reviewer")
    if any(key not in review for key in required):
        raise ValueError("each P107 view needs an explicit review/provenance record")
    if review["view_id"] != view["view_id"]:
        raise ValueError("review is bound to a different view_id")
    if any(type(review[key]) is not bool for key in required[1:5]):
        raise ValueError("review state booleans must be explicit")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, Mapping) or reviewer.get("kind") not in {"agent", "human", "root_model"}:
        raise ValueError("reviewer provenance must identify agent, human, or root_model")
    provenance = view["annotation_provenance"].casefold()
    if provenance == "agent":
        if reviewer.get("kind") != "agent" or not isinstance(reviewer.get("model"), str) or not reviewer["model"]:
            raise ValueError("agent annotation must record its actual agent model, never human provenance")
    if provenance == "human" and reviewer.get("kind") != "human":
        raise ValueError("human annotation provenance must not be substituted by an agent")
    if review["outcome_validated"] and view["label_kind"] != "attempt_outcome":
        raise ValueError("only attempt_outcome can be marked outcome-validated")
    if view["review_status"] == "PARENT_APPROVED" and not review["parent_reviewed"]:
        raise ValueError("PARENT_APPROVED view needs a matching parent-review status record")
    return dict(review)


def _validate_action_payload(payload: Mapping[str, Any], view: Mapping[str, Any], *, authority: Any | None,
                             accept_packaged_positive_payload: bool = False) -> dict[str, Any]:
    """Copy controls from a pinned parent root for positives, never annotation input."""
    if view["low_action_supervision_mask"]:
        if payload and not accept_packaged_positive_payload:
            raise ValueError("positive corrective controls must come only from the sealed parent publisher root")
        if authority is None:
            raise ValueError("positive corrective controls require a sealed parent publisher capability")
        sealed_payload = authority_raw_action_payload(authority, view["action_payload_sha256"])
        raw = sealed_payload["raw_actions_23"]
        raw_sha = sealed_payload["raw_action_sha256"]
        raw_artifact_sha = sealed_payload["raw_action_artifact_sha256"]
        if payload and ({"view_id": view["view_id"], "raw_actions_23": raw, "raw_action_sha256": raw_sha,
                         "raw_action_artifact_sha256": raw_artifact_sha,
                         "action_payload_sha256": sealed_payload["action_payload_sha256"]} !=
                        {key: payload.get(key) for key in ("view_id", "raw_actions_23", "raw_action_sha256",
                                                           "raw_action_artifact_sha256", "action_payload_sha256")}):
            raise ValueError("packaged positive action differs from the sealed parent publisher payload")
    else:
        if payload.get("view_id") != view["view_id"] or not isinstance(payload.get("raw_actions_23"), list):
            raise ValueError("non-qualifying corrective action payload must bind a view_id and actual raw 23D actions")
        raw = payload["raw_actions_23"]
        raw_artifact_sha = payload.get("raw_action_artifact_sha256")
        if not _is_sha(raw_artifact_sha):
            raise ValueError("non-qualifying corrective action payload must preserve its raw action artifact digest")
        raw_sha = raw_action_payload_sha256(raw, raw_action_artifact_sha256=raw_artifact_sha)
        if payload.get("raw_action_sha256") != raw_sha:
            raise ValueError("non-qualifying actual action payload hash is not bound to its source artifact")
    if len(raw) != view["actual_executed_length"]:
        raise ValueError("actual 23D payload length disagrees with corrective action receipt")
    # This owns no action mapping: it asks the protocol to enforce the R1Pro
    # 23D->27D placement and fixed 32-step shape/padding contract.
    projection = project_action_23_to_27(raw, view["actual_executed_length"])
    if not _is_sha(raw_sha) or raw_sha != view["executed_action_receipt"]["raw_action_sha256"]:
        raise ValueError("actual action payload hash disagrees with executed-action receipt")
    payload_sha = action_payload_sha256(raw)
    if view["action_payload_sha256"] != payload_sha:
        raise ValueError("actual action payload digest disagrees with corrective action view")
    result = {"view_id": view["view_id"], "raw_actions_23": raw, "raw_action_sha256": raw_sha,
              "raw_action_artifact_sha256": raw_artifact_sha, "action_payload_sha256": payload_sha,
              "actions_27": projection["actions_27"],
              "action_is_pad": projection["action_is_pad"],
              "action_dim_is_pad": projection["action_dim_is_pad"]}
    if len(result["actions_27"]) != ACTION_HORIZON or any(result["action_is_pad"][:len(raw)]):
        raise ValueError("projected corrective action lost its actual-vs-pad proof")
    return result


def _root_review_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"completed": False, "reason": "root visual-review receipt missing"}
    completed = value.get("completed") is True
    if completed and (value.get("reviewer_kind") != "root_model" or not isinstance(value.get("artifact"), str) or not value["artifact"]):
        raise ValueError("a completed root visual review needs root_model provenance and an artifact")
    return {"completed": completed, "artifact": value.get("artifact"), "reviewer_kind": value.get("reviewer_kind"),
            "model": value.get("model")}


def _authority_capability(value: Any, *, index_inventory_seal_sha256: str) -> dict[str, Any] | None:
    """Persist only externally pinned authority roots, never receipt rows."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("corrective authority capability must be a separately supplied publisher-root descriptor")
    required = {"publisher_manifest_sha256", "index_inventory_seal_sha256",
                "accepted_live_runtime_acceptance_root_sha256"}
    if set(value) != required or not _is_sha(value.get("publisher_manifest_sha256")):
        raise ValueError("corrective authority capability has an invalid externally pinned publisher manifest SHA")
    if value["index_inventory_seal_sha256"] != index_inventory_seal_sha256:
        raise ValueError("corrective authority capability must bind this immutable index inventory seal")
    live_root = value["accepted_live_runtime_acceptance_root_sha256"]
    if live_root is not None and not _is_sha(live_root):
        raise ValueError("corrective authority capability has an invalid live-runtime acceptance root SHA")
    return dict(value)


def _pilot_receipt(path: Path) -> dict[str, Any]:
    """Preserve the local 24-row visual pilot as calibration-only, never labels."""
    pilot = read_json(path)
    if not isinstance(pilot, Mapping) or pilot.get("schema_version") != "p107.visual_label.v1":
        raise ValueError("auxiliary pilot must be p107.visual_label.v1")
    if pilot.get("accepted_scope") != "visual_relation_calibration_only":
        raise ValueError("only the explicitly auxiliary visual-relation pilot may be attached")
    if pilot.get("human_reviewed") is not False or not isinstance(pilot.get("annotator_model"), str):
        raise ValueError("pilot must retain agent model provenance and human_reviewed=false")
    records = pilot.get("records")
    if not isinstance(records, list):
        raise ValueError("pilot records missing")
    for record in records:
        if record.get("split") != "train":
            raise ValueError("pilot calibration must not contain eval/public sources")
        for query in record.get("queries", []):
            if query.get("outcome_loss_mask") is not False or query.get("action_loss_mask") is not False:
                raise ValueError("visual calibration pilot must preserve zero outcome/action masks")
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "records": len(records),
            "annotator_model": pilot["annotator_model"], "accepted_scope": pilot["accepted_scope"],
            "release_eligibility": "AUXILIARY_CALIBRATION_ONLY", "stage3_training_authorized": False,
            "ambiguous_legacy_record_review_fields": sum(record.get("parent_review") == "PENDING" for record in records)}


def build_release(groups: list[Mapping[str, Any]], events: list[Mapping[str, Any]], annotations: Mapping[str, Any], *,
                  index_manifest: Mapping[str, Any], release_role: str, minimum_scale: Mapping[str, int] | None,
                  request_dataset_quality_eligibility: bool, auxiliary_pilots: Iterable[Path] = (),
                  require_complete_source_index: bool = False,
                  index_inventory_seal_sha256: str | None = None,
                  corrective_action_authority: Any | None = None,
                  corrective_authority_capability: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Validate a sealed package in memory; callers write it atomically afterwards."""
    if release_role not in USAGE_ROLES:
        raise ValueError("release role must be explicit")
    if not isinstance(annotations, Mapping) or annotations.get("schema_version") != ANNOTATION_SCHEMA:
        raise ValueError(f"annotations must declare {ANNOTATION_SCHEMA}")
    _assert_no_sidecar(annotations)
    if "corrective_action_authority" in annotations:
        raise ValueError("parent corrective-action authority must be supplied separately from agent annotations")
    group_by_id = _validate_group_inventory(list(groups), list(events))
    if not _is_sha(index_inventory_seal_sha256):
        raise ValueError("publisher must receive the sealed immutable index inventory digest")
    authority_capability = _authority_capability(corrective_authority_capability,
                                                 index_inventory_seal_sha256=index_inventory_seal_sha256)
    if (corrective_action_authority is None) != (authority_capability is None):
        raise ValueError("sealed parent authority and its externally pinned capability must be supplied together")
    views = annotations.get("views")
    reviews = annotations.get("view_reviews")
    payloads = annotations.get("action_payloads", [])
    if not isinstance(views, list) or not isinstance(reviews, list) or not isinstance(payloads, list):
        raise ValueError("annotations must have views, view_reviews, and action_payloads lists")
    event_by_id = {event["event_id"]: event for event in events}
    review_by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, Mapping) or review.get("view_id") in review_by_id:
            raise ValueError("duplicate or malformed view review")
        review_by_id[review["view_id"]] = dict(review)
    payload_by_id: dict[str, Mapping[str, Any]] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping) or payload.get("view_id") in payload_by_id:
            raise ValueError("duplicate or malformed corrective action payload")
        payload_by_id[payload["view_id"]] = payload
    sealed_views, sealed_actions, status = [], [], Counter()
    seen_views: set[str] = set()
    coverage: dict[str, dict[str, Any]] = {}
    for view in sorted(views, key=lambda row: str(row.get("view_id", ""))):
        if not isinstance(view, Mapping):
            raise ValueError("label view must be an object")
        event = event_by_id.get(view.get("event_id"))
        if event is None or view.get("view_id") in seen_views:
            raise ValueError("view must have one unique event-bound view_id")
        seen_views.add(view["view_id"])
        _validate_view(view, event, authority=corrective_action_authority)
        group = group_by_id[view["source_group_id"]]
        if group["usage_role"] != event["usage_role"]:
            raise ValueError("view cannot override its source-group usage role")
        review = _validate_review(review_by_id.pop(view["view_id"], {}), view)
        kind = view["label_kind"]
        quality_gates = {"goal_calibration": False, "outcome": False, "decision": False, "corrective_fm": False}
        if kind == "goal_satisfaction_counterfactual":
            quality_gates["goal_calibration"] = bool(view["valid_goal_mask"] and review["accepted_auxiliary"])
        elif kind == "attempt_outcome":
            quality_gates["outcome"] = bool(view["valid_result_mask"] and review["outcome_validated"])
        elif kind == "recovery_decision":
            quality_gates["decision"] = bool(view["decision_supervision_mask"] and review["accepted_auxiliary"])
        else:
            payload = _validate_action_payload(payload_by_id.pop(view["view_id"], {}), view,
                                               authority=corrective_action_authority)
            sealed_actions.append(payload)
            # A failed correction is retained in the package, but protocol
            # validation and this gate make it impossible to become FM BC.
            quality_gates["corrective_fm"] = bool(view["low_action_supervision_mask"] and group["usage_role"] == "student_candidate")
            status["non_qualifying_corrective_actions"] += int(not quality_gates["corrective_fm"])
        status["candidate_annotated"] += int(review["candidate_annotated"])
        status["parent_reviewed"] += int(review["parent_reviewed"])
        status["accepted_auxiliary"] += int(review["accepted_auxiliary"])
        status["outcome_validated"] += int(review["outcome_validated"])
        # Recovery-action verification is a protocol/authority conclusion,
        # never an annotation review checkbox.
        status["verified_recovery_actions"] += int(quality_gates["corrective_fm"])
        task = str(event["source"]["task_index"])
        local = coverage.setdefault(task, {"skills": set(), "instances": set(), "episodes": set(),
                                           "source_groups": set(), "candidate_views": 0,
                                           "quality_accepted": Counter()})
        local["skills"].update(skill.get("verb") for skill in event["skill_bundle"])
        local["instances"].add(event["source"]["task_instance_id"])
        local["episodes"].add(event["source"]["episode_index"])
        local["source_groups"].add(event["source"]["source_group_id"])
        local["candidate_views"] += 1
        for gate_name, enabled in quality_gates.items():
            local["quality_accepted"][gate_name] += int(enabled)
        sealed_views.append({"view": dict(view), "review": review, "usage_role": group["usage_role"],
                             "original_split": group["original_split"], "dataset_quality_gates": quality_gates})
    if review_by_id:
        raise ValueError("review refers to no packaged view")
    if payload_by_id:
        raise ValueError("action payload refers to no packaged corrective view")
    labeled_event_ids = {row["view"]["event_id"] for row in sealed_views}
    source_episodes = {(event_by_id[event_id]["source"]["source_group_id"],
                        event_by_id[event_id]["source"]["episode_index"]) for event_id in labeled_event_ids}
    root_gate = _root_review_gate(annotations.get("root_visual_review"))
    minimum_scale = dict(minimum_scale or {})
    if request_dataset_quality_eligibility:
        required_minima = {"source_episodes", "goal_outcome_windows", "corrective_action_windows"}
        if set(minimum_scale) != required_minima or any(type(v) is not int or v < 1 for v in minimum_scale.values()):
            raise ValueError("dataset-quality eligibility requires explicit positive --minimum-source-episodes, --minimum-goal-outcome-windows, and --minimum-corrective-action-windows")
    actual = {"source_episodes": len(source_episodes),
              "goal_outcome_windows": sum(row["dataset_quality_gates"]["goal_calibration"] or
                                          row["dataset_quality_gates"]["outcome"] for row in sealed_views),
              "corrective_action_windows": int(status["verified_recovery_actions"])}
    scale_ok = bool(minimum_scale) and all(actual[name] >= value for name, value in minimum_scale.items())
    index_coverage = index_manifest.get("coverage")
    if not isinstance(index_coverage, Mapping):
        raise ValueError("sealed event index lacks an explicit coverage/exclusion report")
    missing_grid = index_coverage.get("missing_grid")
    if not isinstance(missing_grid, list) or type(index_coverage.get("coverage_complete")) is not bool:
        raise ValueError("sealed event index has no valid official task/skill coverage report")
    source_index_complete = bool(require_complete_source_index and index_coverage["coverage_complete"] and
                                 not index_manifest.get("partial_source_coverage", True))
    # A source group cannot try to smuggle an otherwise valid action positive
    # through a later quality gate.  The attempt remains retained as
    # non-student evidence, never a corrective FM target.
    forbidden_student_positive = any(
        row["view"]["label_kind"] == "corrective_action" and row["view"]["low_action_supervision_mask"] and
        (row["usage_role"] != "student_candidate" or row["original_split"] != "train") for row in sealed_views)
    no_protected_student = not forbidden_student_positive and all(
        row["usage_role"] == "student_candidate" and row["original_split"] == "train"
        for row in sealed_views if row["dataset_quality_gates"].get("corrective_fm") or
        row["dataset_quality_gates"].get("outcome") or row["dataset_quality_gates"].get("decision"))
    dataset_quality_eligibility = bool(request_dataset_quality_eligibility and require_complete_source_index and
                                       source_index_complete and root_gate["completed"] and scale_ok and no_protected_student)
    pilots = [_pilot_receipt(Path(path)) for path in auxiliary_pilots]
    policy = index_manifest.get("calibration_selection")
    if not isinstance(policy, Mapping):
        raise ValueError("sealed event index has no calibration selection policy")
    policy_core = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy.get("policy_sha256") != canonical_sha256(policy_core):
        raise ValueError("sealed event index calibration policy hash is invalid")
    manifest = {
        "schema_version": RELEASE_SCHEMA, "protocol_schema_version": SCHEMA_VERSION,
        "release_role": release_role, "immutable_index_manifest_sha256": canonical_sha256(index_manifest),
        "immutable_index_inventory_seal_sha256": index_inventory_seal_sha256,
        "calibration_policy": policy, "calibration_policy_sha256": policy["policy_sha256"],
        "source_group_policy_sha256": canonical_sha256(sorted(groups, key=lambda row: row["source_group_id"])),
        "root_visual_review": root_gate, "minimum_scale_requested": minimum_scale,
        "actual_scale": actual, "require_complete_source_index": require_complete_source_index,
        "source_index_complete": source_index_complete, "source_index_coverage": index_coverage,
        "corrective_authority_capability": authority_capability,
        "dataset_quality_eligibility": dataset_quality_eligibility,
        "release_eligibility": ("DATASET_QUALITY_QUALIFIED_PENDING_INDEPENDENT_REVIEW"
                                if dataset_quality_eligibility else "CANDIDATE_ONLY"),
        # No caller can turn preparation into stage3 authorization by copying files.
        "ready_for_training": False, "training_run_authorized": False, "stage3_training_authorized": False,
        "status_report": {"source_indexed": len(events), "candidate_annotated": int(status["candidate_annotated"]),
                          "parent_reviewed": int(status["parent_reviewed"]), "accepted_auxiliary": int(status["accepted_auxiliary"]),
                          "outcome_validated": int(status["outcome_validated"]),
                          "verified_recovery_actions": int(status["verified_recovery_actions"]),
                          "non_qualifying_corrective_actions": int(status["non_qualifying_corrective_actions"]),
                          "auxiliary_visual_calibration_pilots": pilots},
        "coverage": {task: {"skills": sorted(values["skills"]), "source_instances": sorted(values["instances"]),
                             "source_episodes": sorted(values["episodes"]), "source_episode_count": len(values["episodes"]),
                             "source_group_count": len(values["source_groups"]), "candidate_view_count": values["candidate_views"],
                             "quality_accepted_view_counts": dict(sorted(values["quality_accepted"].items()))}
                     for task, values in sorted(coverage.items(), key=lambda item: int(item[0]))},
        "exclusions": {"protected_roles": ["annotation_calibration", "evaluation_only"],
                       "public_test": "never student", "source_group_atomic": True,
                       "old_memlite_sidecar": "not imported or accepted", "pilot": "auxiliary calibration only",
                       "partial_index": bool(index_manifest.get("partial_source_coverage", True)),
                       "official_coverage_missing_grid": missing_grid},
    }
    return manifest, {"source_groups": [dict(row) for row in groups], "events": [dict(row) for row in events],
                      "views": sealed_views, "actions": sealed_actions}


def publish(index_dir: Path, annotations_path: Path, output: Path, *, release_role: str,
            minimum_scale: Mapping[str, int] | None, request_dataset_quality_eligibility: bool,
            expected_index_inventory_seal_sha256: str,
            corrective_publisher_root: Path | None = None,
            expected_corrective_publisher_manifest_sha256: str | None = None,
            expected_live_runtime_acceptance_root_sha256: str | None = None,
            auxiliary_pilots: Iterable[Path] = (), require_complete_source_index: bool = False) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError("output exists; immutable packages are never appended or overwritten")
    if Path(index_dir).resolve() in output.resolve(strict=False).parents:
        raise ValueError("package output may not be inside its immutable source index")
    groups, events, index_manifest, index_inventory_seal_sha256 = _read_sealed_index(
        index_dir, expected_inventory_seal_sha256=expected_index_inventory_seal_sha256)
    if (corrective_publisher_root is None) != (expected_corrective_publisher_manifest_sha256 is None):
        raise ValueError("corrective publisher root and its externally pinned manifest SHA must be supplied together")
    corrective_action_authority = None
    corrective_authority_capability = None
    if corrective_publisher_root is not None:
        corrective_action_authority = load_corrective_action_authority(
            corrective_publisher_root,
            expected_publisher_manifest_sha256=expected_corrective_publisher_manifest_sha256,
            expected_index_inventory_seal_sha256=index_inventory_seal_sha256,
            expected_live_runtime_acceptance_root_sha256=expected_live_runtime_acceptance_root_sha256)
        corrective_authority_capability = {
            "publisher_manifest_sha256": expected_corrective_publisher_manifest_sha256,
            "index_inventory_seal_sha256": index_inventory_seal_sha256,
            "accepted_live_runtime_acceptance_root_sha256": expected_live_runtime_acceptance_root_sha256,
        }
    manifest, records = build_release(groups, events, read_json(annotations_path), index_manifest=index_manifest,
                                      release_role=release_role, minimum_scale=minimum_scale,
                                      request_dataset_quality_eligibility=request_dataset_quality_eligibility,
                                      auxiliary_pilots=auxiliary_pilots,
                                      require_complete_source_index=require_complete_source_index,
                                      index_inventory_seal_sha256=index_inventory_seal_sha256,
                                      corrective_action_authority=corrective_action_authority,
                                      corrective_authority_capability=corrective_authority_capability)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    try:
        files = {"source_groups.jsonl": _write_jsonl(staging / "source_groups.jsonl", records["source_groups"]),
                 "events.jsonl": _write_jsonl(staging / "events.jsonl", records["events"]),
                 "views.jsonl": _write_jsonl(staging / "views.jsonl", records["views"]),
                 "actions.jsonl": _write_jsonl(staging / "actions.jsonl", records["actions"])}
        manifest["files"] = files
        manifest["publisher_sha256"] = sha256_file(Path(__file__))
        manifest_path = staging / "release_manifest.json"
        manifest_path.write_bytes(canonical_json(manifest).encode("utf-8") + b"\n")
        seal = {"schema_version": "memlite-event-label-release-seal-v1",
                "release_manifest_sha256": sha256_file(manifest_path), "files": files,
                "publisher_sha256": manifest["publisher_sha256"]}
        seal_path = staging / "release_seal.json"
        seal_path.write_bytes(canonical_json(seal).encode("utf-8") + b"\n")
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**manifest, "release_seal_sha256": sha256_file(output / "release_seal.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-index", type=Path, required=True, help="sealed build_memlite_event_index output")
    parser.add_argument("--annotations", type=Path, required=True, help="small P107 label/review JSON")
    parser.add_argument("--output", type=Path, required=True, help="new external immutable package directory")
    parser.add_argument("--release-role", choices=sorted(USAGE_ROLES), required=True)
    parser.add_argument("--expected-index-inventory-seal-sha256", required=True,
                        help="externally recorded inventory seal SHA-256; self-consistent input is insufficient")
    parser.add_argument("--corrective-publisher-root", type=Path,
                        help="separate parent-owned sealed publisher root; required for a positive corrective FM mask")
    parser.add_argument("--expected-corrective-publisher-manifest-sha256",
                        help="externally pinned parent publisher-manifest SHA-256")
    parser.add_argument("--expected-live-runtime-acceptance-root-sha256",
                        help="externally accepted live-runtime root for GOLD, omitted for SILVER-only authority")
    parser.add_argument("--request-dataset-quality-eligibility", action="store_true",
                        help="request an evidence-quality gate only; it never authorizes a training run")
    parser.add_argument("--require-complete-source-index", action="store_true",
                        help="require the index's explicit official coverage contract to have no gaps")
    parser.add_argument("--minimum-source-episodes", type=int)
    parser.add_argument("--minimum-goal-outcome-windows", type=int)
    parser.add_argument("--minimum-corrective-action-windows", type=int)
    parser.add_argument("--auxiliary-pilot", type=Path, action="append", default=[],
                        help="agent visual pilot, retained only as auxiliary calibration provenance")
    args = parser.parse_args()
    supplied = (args.minimum_source_episodes, args.minimum_goal_outcome_windows, args.minimum_corrective_action_windows)
    if any(value is not None for value in supplied) and any(value is None for value in supplied):
        parser.error("all three --minimum-* arguments must be supplied together")
    minima = None if supplied[0] is None else {"source_episodes": supplied[0], "goal_outcome_windows": supplied[1],
                                                "corrective_action_windows": supplied[2]}
    result = publish(args.event_index, args.annotations, args.output, release_role=args.release_role,
                     minimum_scale=minima, request_dataset_quality_eligibility=args.request_dataset_quality_eligibility,
                     expected_index_inventory_seal_sha256=args.expected_index_inventory_seal_sha256,
                     corrective_publisher_root=args.corrective_publisher_root,
                     expected_corrective_publisher_manifest_sha256=args.expected_corrective_publisher_manifest_sha256,
                     expected_live_runtime_acceptance_root_sha256=args.expected_live_runtime_acceptance_root_sha256,
                     auxiliary_pilots=args.auxiliary_pilot,
                     require_complete_source_index=args.require_complete_source_index)
    print(canonical_json({"release_eligibility": result["release_eligibility"], "ready_for_training": False,
                          "release_seal_sha256": result["release_seal_sha256"],
                          "status_report": result["status_report"]}), flush=True)


if __name__ == "__main__":
    main()
