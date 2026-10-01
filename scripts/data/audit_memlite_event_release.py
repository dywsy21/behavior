"""Independently verify a sealed P107 event-label preparation package.

The audit is intentionally read-only with respect to the package.  Its report
is a separate new file and distinguishes the evidence status of each view from
any hypothetical future training authorization.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]


def _load_packer() -> Any:
    path = REPO / "scripts/data/pack_memlite_event_labels.py"
    spec = importlib.util.spec_from_file_location("p107_memlite_event_packer_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P107 packer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_pack = _load_packer()
RELEASE_SCHEMA = _pack.RELEASE_SCHEMA
SCHEMA_VERSION = _pack.SCHEMA_VERSION
canonical_json = _pack.canonical_json
canonical_sha256 = _pack.canonical_sha256
read_json = _pack.read_json
read_jsonl = _pack.read_jsonl
sha256_file = _pack.sha256_file
_validate_group_inventory = _pack._validate_group_inventory
_validate_view = _pack._validate_view
_validate_review = _pack._validate_review
_validate_action_payload = _pack._validate_action_payload
load_corrective_action_authority = _pack.load_corrective_action_authority


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"package contains a missing or symlinked file: {path}")
    if path.stat().st_mode & 0o222:
        raise ValueError(f"immutable package file remains writable: {path}")


def audit(package: Path, *, expected_release_seal_sha256: str,
          corrective_publisher_root: Path | None = None) -> dict[str, Any]:
    package = Path(package)
    if package.is_symlink() or not package.is_dir():
        raise ValueError("package must be a regular immutable directory")
    if package.stat().st_mode & 0o222:
        raise ValueError("immutable package directory remains writable")
    required = ("release_manifest.json", "release_seal.json", "source_groups.jsonl", "events.jsonl", "views.jsonl", "actions.jsonl")
    for name in required:
        _regular(package / name)
    if not _pack._is_sha(expected_release_seal_sha256):
        raise ValueError("audit needs an externally recorded release seal SHA-256")
    if sha256_file(package / "release_seal.json") != expected_release_seal_sha256:
        raise ValueError("externally recorded release seal SHA-256 does not match package")
    manifest, seal = read_json(package / "release_manifest.json"), read_json(package / "release_seal.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError("invalid package release manifest")
    if not isinstance(seal, Mapping) or seal.get("schema_version") != "memlite-event-label-release-seal-v1":
        raise ValueError("invalid package release seal")
    if seal.get("release_manifest_sha256") != sha256_file(package / "release_manifest.json"):
        raise ValueError("release manifest does not match immutable seal")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or seal.get("files") != files:
        raise ValueError("release manifest/seal disagree on packaged file receipts")
    for name in ("source_groups.jsonl", "events.jsonl", "views.jsonl", "actions.jsonl"):
        receipt = files.get(name)
        if not isinstance(receipt, Mapping) or receipt.get("sha256") != sha256_file(package / name):
            raise ValueError(f"packaged record receipt mismatch: {name}")
    if (manifest.get("ready_for_training") is not False or manifest.get("training_run_authorized") is not False or
            manifest.get("stage3_training_authorized") is not False):
        raise ValueError("P107 preparation package may never self-authorize stage3 training")
    policy = manifest.get("calibration_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("missing calibration policy")
    policy_core = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy.get("policy_sha256") != canonical_sha256(policy_core) or manifest.get("calibration_policy_sha256") != policy["policy_sha256"]:
        raise ValueError("calibration policy digest mismatch")
    groups, events = read_jsonl(package / "source_groups.jsonl"), read_jsonl(package / "events.jsonl")
    group_by_id = _validate_group_inventory(groups, events)
    if manifest.get("source_group_policy_sha256") != canonical_sha256(
            sorted(groups, key=lambda row: row["source_group_id"])):
        raise ValueError("source-group policy digest does not match packaged immutable inventory")
    capability = _pack._authority_capability(
        manifest.get("corrective_authority_capability"),
        index_inventory_seal_sha256=manifest.get("immutable_index_inventory_seal_sha256"))
    if capability is None:
        if corrective_publisher_root is not None:
            raise ValueError("package has no corrective authority capability but a publisher root was supplied")
        authority = None
    else:
        if corrective_publisher_root is None:
            raise ValueError("package with corrective authority capability requires its separate publisher root for audit")
        authority = load_corrective_action_authority(
            corrective_publisher_root,
            expected_publisher_manifest_sha256=capability["publisher_manifest_sha256"],
            expected_index_inventory_seal_sha256=capability["index_inventory_seal_sha256"],
            expected_live_runtime_acceptance_root_sha256=capability["accepted_live_runtime_acceptance_root_sha256"])
    views, actions = read_jsonl(package / "views.jsonl"), read_jsonl(package / "actions.jsonl")
    event_by_id = {row["event_id"]: row for row in events}
    action_by_id = {row.get("view_id"): row for row in actions}
    if len(action_by_id) != len(actions):
        raise ValueError("duplicate corrective action payload in package")
    view_ids: set[str] = set()
    status = {"source_indexed": len(events), "candidate_annotated": 0, "parent_reviewed": 0,
              "accepted_auxiliary": 0, "outcome_validated": 0, "verified_recovery_actions": 0,
              "non_qualifying_corrective_actions": 0}
    coverage: dict[str, dict[str, Any]] = {}
    for wrapper in views:
        if not isinstance(wrapper, Mapping) or not isinstance(wrapper.get("view"), Mapping) or not isinstance(wrapper.get("review"), Mapping):
            raise ValueError("views.jsonl must retain view and review objects")
        view, review = wrapper["view"], wrapper["review"]
        if view.get("view_id") in view_ids:
            raise ValueError("duplicate view_id in package")
        view_ids.add(view["view_id"])
        event = event_by_id.get(view.get("event_id"))
        if event is None:
            raise ValueError("view references no packaged event")
        _validate_view(view, event, authority=authority)
        _validate_review(review, view)
        group = group_by_id[view["source_group_id"]]
        if wrapper.get("usage_role") != group["usage_role"] or wrapper.get("original_split") != group["original_split"]:
            raise ValueError("view wrapper changed immutable source-group role or split")
        expected = {"goal_calibration": False, "outcome": False, "decision": False, "corrective_fm": False}
        kind = view["label_kind"]
        if kind == "goal_satisfaction_counterfactual":
            expected["goal_calibration"] = bool(view["valid_goal_mask"] and review["accepted_auxiliary"])
        elif kind == "attempt_outcome":
            expected["outcome"] = bool(view["valid_result_mask"] and review["outcome_validated"])
        elif kind == "recovery_decision":
            expected["decision"] = bool(view["decision_supervision_mask"] and review["accepted_auxiliary"])
        else:
            payload = action_by_id.pop(view["view_id"], None)
            if payload is None:
                raise ValueError("corrective view lacks preserved actual 23D action payload")
            _validate_action_payload(payload, view, authority=authority, accept_packaged_positive_payload=True)
            expected["corrective_fm"] = bool(view["low_action_supervision_mask"] and group["usage_role"] == "student_candidate")
            status["non_qualifying_corrective_actions"] += int(not expected["corrective_fm"])
        if wrapper.get("dataset_quality_gates") != expected:
            raise ValueError("stored view dataset-quality gate is inconsistent")
        if (expected["outcome"] or expected["decision"] or expected["corrective_fm"]) and (
                group["usage_role"] != "student_candidate" or group["original_split"] != "train"):
            raise ValueError("calibration/eval/public source leaked into student-supervision evidence")
        status["candidate_annotated"] += int(review["candidate_annotated"])
        status["parent_reviewed"] += int(review["parent_reviewed"])
        status["accepted_auxiliary"] += int(review["accepted_auxiliary"])
        status["outcome_validated"] += int(review["outcome_validated"])
        status["verified_recovery_actions"] += int(expected["corrective_fm"])
        if kind in {"goal_satisfaction_counterfactual", "attempt_outcome"}:
            task = str(event["source"]["task_index"])
        task = str(event["source"]["task_index"])
        local = coverage.setdefault(task, {"skills": set(), "instances": set(), "episodes": set(),
                                           "source_groups": set(), "candidate_views": 0,
                                           "quality_accepted": {name: 0 for name in expected}})
        local["skills"].update(item.get("verb") for item in event["skill_bundle"])
        local["instances"].add(event["source"]["task_instance_id"])
        local["episodes"].add(event["source"]["episode_index"])
        local["source_groups"].add(event["source"]["source_group_id"])
        local["candidate_views"] += 1
        for gate_name, enabled in expected.items():
            local["quality_accepted"][gate_name] += int(enabled)
    if action_by_id:
        raise ValueError("orphan action payload in package")
    labeled_event_ids = {row["view"]["event_id"] for row in views}
    actual_scale = {"source_episodes": len({(event_by_id[event_id]["source"]["source_group_id"],
                                               event_by_id[event_id]["source"]["episode_index"])
                                          for event_id in labeled_event_ids}),
                    "goal_outcome_windows": sum(row["dataset_quality_gates"]["goal_calibration"] or
                                                row["dataset_quality_gates"]["outcome"] for row in views),
                    "corrective_action_windows": status["verified_recovery_actions"]}
    if manifest.get("actual_scale") != actual_scale:
        raise ValueError("stored scale report is not reproducible from package rows")
    reported = manifest.get("status_report")
    if not isinstance(reported, Mapping) or any(reported.get(key) != value for key, value in status.items()):
        raise ValueError("stored status report is not reproducible from package rows")
    expected_coverage = {task: {"skills": sorted(value["skills"]), "source_instances": sorted(value["instances"]),
                                "source_episodes": sorted(value["episodes"]), "source_episode_count": len(value["episodes"]),
                                "source_group_count": len(value["source_groups"]), "candidate_view_count": value["candidate_views"],
                                "quality_accepted_view_counts": dict(sorted(value["quality_accepted"].items()))}
                         for task, value in sorted(coverage.items(), key=lambda item: int(item[0]))}
    if manifest.get("coverage") != expected_coverage:
        raise ValueError("stored task/family/instance coverage is not reproducible")
    root_review = manifest.get("root_visual_review")
    if manifest.get("dataset_quality_eligibility") and (not isinstance(root_review, Mapping) or root_review.get("completed") is not True):
        raise ValueError("dataset-quality eligibility lacks the required root visual-review gate")
    if manifest.get("dataset_quality_eligibility") and not (manifest.get("require_complete_source_index") and manifest.get("source_index_complete")):
        raise ValueError("dataset-quality eligibility cannot bypass incomplete source-index coverage")
    return {"schema_version": "memlite-event-release-audit-v1", "status": "PASS_WITH_STAGE3_TRAINING_BLOCKED",
            "ready_for_training": False, "training_run_authorized": False, "stage3_training_authorized": False, "actual_scale": actual_scale,
            "status_report": status, "coverage": expected_coverage,
            "release_eligibility": manifest.get("release_eligibility"),
            "warnings": ["accepted_scope/review_decision do not override CANDIDATE_ONLY release eligibility",
                         "auxiliary visual calibration pilots are not outcome/recovery/action labels"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-release-seal-sha256", required=True,
                        help="externally recorded package seal SHA-256; self-consistent package files are insufficient")
    parser.add_argument("--corrective-publisher-root", type=Path,
                        help="separate sealed parent publisher root required when package contains positive corrective evidence")
    parser.add_argument("--report", type=Path, required=True, help="new audit report outside the immutable package")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("audit report already exists; do not overwrite evidence")
    if args.package.resolve() in args.report.resolve(strict=False).parents:
        raise ValueError("audit report must be outside the immutable package")
    result = audit(args.package, expected_release_seal_sha256=args.expected_release_seal_sha256,
                   corrective_publisher_root=args.corrective_publisher_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json(result).encode("utf-8") + b"\n")
    args.report.chmod(0o444)
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
