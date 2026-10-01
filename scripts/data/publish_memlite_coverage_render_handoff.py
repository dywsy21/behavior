"""Publish sealed render handoffs derived from a P107 coverage selection.

The coverage selector deliberately emits a diagnostic manifest rather than the
canonical queue-manifest contract consumed by the renderer.  This module is a
small, explicit adapter:

* TRAIN is re-sealed through the existing queue builder using a tiny sealed
  subset index containing exactly the selected TRAIN events.  The selector
  report and prior-window provenance remain outside the canonical queue files.
* EVAL is authorized by a separate receipt.  It never creates a TRAIN queue or
  a TRAIN queue seal; the evaluation renderer binds its packets to that
  receipt and keeps ``evaluation_only``/``eval``/``training_eligible=false``.

No RGB, action, outcome, recovery, or label data is produced here.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import render_memlite_event_packets as renderer
import select_memlite_coverage_cohort as cohort

base = cohort.base


TRAIN_PROVENANCE_SCHEMA = "p107-coverage-train-render-provenance-v1"
EVAL_RECEIPT_SCHEMA = "p107-evaluation-render-authorization-v1"
INDEX_INVENTORY_SEAL_SCHEMA = "memlite-event-inventory-seal-v1"
EVAL_RECEIPT_FIELDS = frozenset({
    "schema_version", "status", "training_eligible", "usage_role", "immutable_split",
    "selector_manifest_sha256", "selector_selection_seal_sha256", "selector_request_filename",
    "selector_request_sha256", "index_manifest_sha256", "index_event_file_sha256",
    "inventory_seal_sha256", "source_release_manifest_sha256", "canonical_protocol_sha256",
    "packet_manifest_sha256", "packet_manifest_queue_authority_sha256", "packet_count",
    "request_bindings", "no_outcome_or_action_labels",
})
TRAIN_HARNESS_SEED = "p107-coverage-train-render-handoff-20261002"


def _sha256(path: Path) -> str:
    return base.sha256_file(Path(path))


def _regular(path: Path, name: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular file")
    return path


def _safe_child(root: Path, relative: str, name: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"{name} contains an unsafe relative path")
    path = (root / candidate).resolve(strict=False)
    if root.resolve(strict=True) not in path.parents:
        raise ValueError(f"{name} escapes its selector directory")
    return path


def _receipt(path: Path, expected: Mapping[str, Any], *, name: str) -> None:
    path = _regular(path, name)
    if (not isinstance(expected, Mapping) or set(expected) != {"sha256", "rows", "bytes"} or
            not base.is_sha256(expected.get("sha256")) or type(expected.get("rows")) is not int or
            type(expected.get("bytes")) is not int or expected["rows"] < 0 or expected["bytes"] < 0 or
            path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"] or
            len([line for line in path.read_bytes().splitlines() if line.strip()]) != expected["rows"]):
        raise ValueError(f"{name} bytes/row receipt does not match the selector handoff")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    value = base.read_json(_regular(path, name))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows = base.read_jsonl(_regular(path, name))
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{name} must contain JSON objects")
    return rows


@dataclass(frozen=True)
class SelectorHandoff:
    root: Path
    role: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    selection_seal_sha256: str
    queue_seal_sha256: str | None
    selected_rows: tuple[Mapping[str, Any], ...]
    jobs: tuple[Mapping[str, Any], ...]
    requests: tuple[Mapping[str, Any], ...]
    selected_rows_path: Path
    request_path: Path

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(str(row["event_id"]) for row in self.selected_rows)


def read_selector_handoff(root: Path, *, role: str, expected_manifest_sha256: str,
                          expected_selection_seal_sha256: str,
                          expected_queue_seal_sha256: str | None = None) -> SelectorHandoff:
    """Verify one selector role without treating its custom manifest as canonical."""
    if role not in {"train", "eval"}:
        raise ValueError("selector role must be train or eval")
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("selector handoff must be an existing regular directory")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, "selector manifest")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("selector manifest bytes do not match the externally pinned SHA-256")
    if (manifest.get("schema_version") != cohort.COVERAGE_SCHEMA or
            manifest.get("selection_role") != role or
            manifest.get("usage_role") != ("annotation_calibration" if role == "train" else "evaluation_only") or
            manifest.get("immutable_split") != ("train" if role == "train" else "eval") or
            manifest.get("training_eligible") is not False):
        raise ValueError("selector manifest has the wrong immutable role or candidate-only status")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("selector manifest has no file receipt map")
    for filename, receipt in files.items():
        path = _safe_child(root, str(filename), f"selector file {filename}")
        _receipt(path, receipt, name=f"selector file {filename}")

    selection_seal_path = root / "selection_seal.json"
    selection_seal = _read_json(selection_seal_path, "selector selection seal")
    if _sha256(selection_seal_path) != expected_selection_seal_sha256:
        raise ValueError("selector selection-seal bytes do not match the externally pinned SHA-256")
    required_seal = {"schema_version", "manifest_sha256", "policy_sha256", "training_eligible", "files"}
    allowed_seal = required_seal | ({"queue_seal_sha256"} if role == "train" else set())
    if (set(selection_seal) != allowed_seal or selection_seal.get("schema_version") != cohort.COVERAGE_SCHEMA or
            selection_seal.get("manifest_sha256") != expected_manifest_sha256 or
            not base.is_sha256(selection_seal.get("policy_sha256")) or
            selection_seal.get("training_eligible") is not False or selection_seal.get("files") != files):
        raise ValueError("selector selection seal is malformed or does not bind its manifest")

    queue_seal_sha256 = None
    if role == "train":
        if expected_queue_seal_sha256 is None:
            raise ValueError("TRAIN selector handoff requires an externally pinned queue seal")
        queue_path = root / "queue_seal.json"
        request_path = root / "camera_native_render_requests.jsonl"
        if _sha256(queue_path) != expected_queue_seal_sha256:
            raise ValueError("TRAIN selector queue-seal bytes do not match the externally pinned SHA-256")
        queue_seal_sha256, source_release = renderer._read_queue_seal(
            queue_path, expected_queue_seal_sha256, render_requests_path=request_path,
            render_requests_sha256=_sha256(request_path))
        manifest_release = manifest.get("source_release_manifest_sha256")
        if source_release != manifest_release:
            raise ValueError("TRAIN selector queue seal source release differs from its manifest")
        if selection_seal.get("queue_seal_sha256") != queue_seal_sha256:
            raise ValueError("TRAIN selector selection seal does not bind its queue seal")
        request_name = "camera_native_render_requests.jsonl"
        job_name = "annotation_calibration_queue.jsonl"
    else:
        if expected_queue_seal_sha256 is not None or (root / "queue_seal.json").exists():
            raise ValueError("EVAL selector handoff must not carry a TRAIN queue seal")
        request_name = "evaluation_only_render_requests.jsonl"
        job_name = "evaluation_only_selection.jsonl"

    selected_rows_path = root / "selected_rows.jsonl"
    request_path = root / request_name
    jobs_path = root / job_name
    selected_rows = tuple(_read_jsonl(selected_rows_path, "selected_rows.jsonl"))
    jobs = tuple(_read_jsonl(jobs_path, job_name))
    requests = tuple(_read_jsonl(request_path, request_name))
    if len(selected_rows) != len(jobs) or len(jobs) != len(requests):
        raise ValueError("selector rows/jobs/requests have inconsistent counts")
    selected_ids = [row.get("event_id") for row in selected_rows]
    if (any(not base.is_sha256(event_id) for event_id in selected_ids) or
            len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("selector rows contain missing or duplicate event IDs")
    expected_role = "annotation_calibration" if role == "train" else "evaluation_only"
    expected_split = "train" if role == "train" else "eval"
    for row, job, request in zip(selected_rows, jobs, requests):
        if (row.get("usage_role") != expected_role or row.get("immutable_split") != expected_split or
                row.get("training_eligible") is not False or job.get("event_id") != row.get("event_id") or
                job.get("usage_role") != expected_role or job.get("immutable_split") != expected_split or
                job.get("training_eligible") is not False or request.get("event_id") != row.get("event_id") or
                request.get("request_id") != job.get("camera_native_render_request_id")):
            raise ValueError("selector role rows are not consistently bound to the requested immutable role")
    return SelectorHandoff(
        root=root, role=role, manifest=manifest, manifest_sha256=expected_manifest_sha256,
        selection_seal_sha256=expected_selection_seal_sha256, queue_seal_sha256=queue_seal_sha256,
        selected_rows=selected_rows, jobs=jobs, requests=requests,
        selected_rows_path=selected_rows_path, request_path=request_path,
    )


@dataclass(frozen=True)
class ParentIndex:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    coverage_sha256: str
    events: Mapping[str, Mapping[str, Any]]
    source_groups: Mapping[str, Mapping[str, Any]]


def _load_parent_index(index: Path, selector: SelectorHandoff, *,
                       expected_source_release_sha256: str, expected_inventory_sha256: str,
                       coverage_path: Path, expected_coverage_sha256: str,
                       protocol_path: Path, expected_protocol_sha256: str) -> ParentIndex:
    index = Path(index)
    protocol = base.load_canonical_protocol(protocol_path, expected_sha256=expected_protocol_sha256)
    manifest, events_path, _groups = base.validate_index(
        index, expected_source_manifest_sha256=expected_source_release_sha256,
        canonical_protocol=protocol, expected_protocol_sha256=expected_protocol_sha256)
    inventory, inventory_sha256 = base.validate_inventory_seal(
        index, expected_inventory_seal_sha256=expected_inventory_sha256)
    coverage = base.read_coverage_expectations(Path(coverage_path), expected_sha256=inventory["coverage_expectations_sha256"])
    coverage_sha256 = base.canonical_sha256(coverage)
    if coverage_sha256 != expected_coverage_sha256 or manifest.get("coverage_expectations_sha256") != coverage_sha256:
        raise ValueError("coverage expectations do not match the pinned full index")
    selected_ids = selector.event_ids
    events: dict[str, Mapping[str, Any]] = {}
    for event in base.iter_jsonl_verified(events_path, expected=manifest["files"]["event_candidates.jsonl"]):
        event_id = event.get("event_id")
        if event_id not in selected_ids:
            continue
        protocol.validate_event(event)
        if event_id in events:
            raise ValueError("selected event IDs are duplicated in the full index")
        events[str(event_id)] = event
    if set(events) != set(selected_ids):
        raise ValueError("selector refers to an event absent from the immutable full index")

    source_rows = base.read_jsonl_verified(index / "source_groups.jsonl", expected=manifest["files"]["source_groups.jsonl"])
    source_groups = {str(row["source_group_id"]): row for row in source_rows}
    expected_role = "annotation_calibration" if selector.role == "train" else "evaluation_only"
    expected_split = "train" if selector.role == "train" else "eval"
    for row in selector.selected_rows:
        event = events[row["event_id"]]
        source = event["source"]
        identity = row.get("source_identity")
        if (not isinstance(identity, Mapping) or
                any(identity.get(key) != source.get(key) for key in (
                    "source_release_manifest_sha256", "source_annotation_sha256", "source_group_id",
                    "task_index", "task_instance_id", "raw_episode_id", "episode_index")) or
                row.get("observation_frame") != event["observation"]["frame"] or
                row.get("usage_role") != expected_role or row.get("immutable_split") != expected_split or
                event.get("usage_role") != expected_role or source.get("original_split") != expected_split):
            raise ValueError("selector row does not bind its exact full-index event/source identity")
        group_id = str(source["source_group_id"])
        group = source_groups.get(group_id)
        if group is None or group.get("usage_role") != expected_role or group.get("original_split") != expected_split:
            raise ValueError("selector row does not bind an immutable source-group role")
    return ParentIndex(
        root=index, manifest=manifest, manifest_sha256=base.sha256_file(index / "manifest.json"),
        inventory_sha256=inventory_sha256, protocol_sha256=expected_protocol_sha256,
        coverage_sha256=coverage_sha256, events=events,
        source_groups={str(key): source_groups[str(key)] for key in {
            str(event["source"]["source_group_id"]) for event in events.values()
        }},
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.write_bytes(base.canonical_json(value).encode("utf-8") + b"\n")
    return {"sha256": _sha256(path), "rows": 1, "bytes": path.stat().st_size}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(base.canonical_json(row).encode("utf-8") + b"\n")
            count += 1
    return {"sha256": _sha256(path), "rows": count, "bytes": path.stat().st_size}


def _write_subset_index(output: Path, parent: ParentIndex, selector: SelectorHandoff) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"subset index already exists: {output}")
    output.mkdir(parents=True)
    event_rows = [parent.events[event_id] for event_id in sorted(selector.event_ids)]
    group_rows = [parent.source_groups[group_id] for group_id in sorted(parent.source_groups)]
    group_receipt = _write_jsonl(output / "source_groups.jsonl", group_rows)
    event_receipt = _write_jsonl(output / "event_candidates.jsonl", event_rows)
    manifest = {
        "schema_version": base.INDEX_SCHEMA,
        "status": base.INDEX_STATUS,
        "training_eligible": False,
        "source_release_manifest_sha256": parent.manifest["source_release_manifest_sha256"],
        "protocol_sha256": parent.protocol_sha256,
        "coverage_expectations_sha256": parent.coverage_sha256,
        "parent_index_manifest_sha256": parent.manifest_sha256,
        "parent_inventory_seal_sha256": parent.inventory_sha256,
        "subset_kind": "P107_COVERAGE_TRAIN_RENDER_HANDOFF",
        "files": {"source_groups.jsonl": group_receipt, "event_candidates.jsonl": event_receipt},
    }
    _write_json(output / "manifest.json", manifest)
    inventory = {
        "schema_version": INDEX_INVENTORY_SEAL_SCHEMA,
        "index_manifest_sha256": _sha256(output / "manifest.json"),
        "source_release_manifest_sha256": parent.manifest["source_release_manifest_sha256"],
        "coverage_expectations_sha256": parent.coverage_sha256,
        "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
        "payload_files": {"event_candidates.jsonl": event_receipt, "source_groups.jsonl": group_receipt},
    }
    _write_json(output / "inventory_seal.json", inventory)
    return {
        "manifest": manifest,
        "manifest_sha256": _sha256(output / "manifest.json"),
        "inventory_sha256": _sha256(output / "inventory_seal.json"),
        "event_receipt": event_receipt,
        "source_group_receipt": group_receipt,
    }


def _canonical_train_args(index: Path, output: Path, coverage_path: Path, *,
                          inventory_sha256: str, source_release_sha256: str,
                          protocol_path: Path, protocol_sha256: str,
                          calibration_budget: int) -> SimpleNamespace:
    return SimpleNamespace(
        index=Path(index), output=Path(output), protocol_path=Path(protocol_path),
        expected_source_release_manifest_sha256=source_release_sha256,
        expected_inventory_seal_sha256=inventory_sha256, coverage_expectations=Path(coverage_path),
        expected_protocol_sha256=protocol_sha256, expected_queue_seal_sha256=None,
        seed=TRAIN_HARNESS_SEED, candidate_budget=0, calibration_budget=calibration_budget,
        max_per_episode=1, max_per_source_group=1, min_separation_frames=0,
        boundary_gap_frames=1, long_interval_frames=360, normal_control_fraction=0.0,
        actor_history_frames=60, review_before_frames=60, review_after_frames=60,
        review_sample_stride_frames=15, max_retained_candidates=50000,
    )


def publish_canonical_train_handoff(*, index: Path, train_selector: Path, output_root: Path,
                                   coverage_path: Path, expected_source_release_sha256: str,
                                   expected_inventory_sha256: str, expected_coverage_sha256: str,
                                   protocol_path: Path, expected_protocol_sha256: str,
                                   expected_selector_manifest_sha256: str,
                                   expected_selector_selection_seal_sha256: str,
                                   expected_selector_queue_seal_sha256: str) -> dict[str, Any]:
    """Publish one canonical TRAIN render queue plus external coverage provenance."""
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"canonical TRAIN handoff root already exists: {output_root}")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise ValueError("canonical TRAIN handoff parent must be an existing regular directory")
    selector = read_selector_handoff(
        train_selector, role="train", expected_manifest_sha256=expected_selector_manifest_sha256,
        expected_selection_seal_sha256=expected_selector_selection_seal_sha256,
        expected_queue_seal_sha256=expected_selector_queue_seal_sha256)
    parent = _load_parent_index(
        index, selector, expected_source_release_sha256=expected_source_release_sha256,
        expected_inventory_sha256=expected_inventory_sha256, coverage_path=coverage_path,
        expected_coverage_sha256=expected_coverage_sha256, protocol_path=protocol_path,
        expected_protocol_sha256=expected_protocol_sha256)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=str(output_root.parent)))
    try:
        subset = _write_subset_index(staging / "train_source_index", parent, selector)
        train_output = staging / "train"
        args = _canonical_train_args(
            staging / "train_source_index", train_output, coverage_path,
            inventory_sha256=subset["inventory_sha256"], source_release_sha256=expected_source_release_sha256,
            protocol_path=protocol_path, protocol_sha256=expected_protocol_sha256,
            calibration_budget=len(selector.selected_rows))
        result = base.build_queue(staging / "train_source_index", train_output, args=args)
        manifest = base.read_json(train_output / "manifest.json")
        queue_seal_sha256 = base.validate_queue_seal(
            train_output, manifest, expected_queue_seal_sha256=result["queue_seal_sha256"],
            expected_inventory_seal_sha256=subset["inventory_sha256"],
            expected_source_manifest_sha256=expected_source_release_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_coverage_expectations_sha256=expected_coverage_sha256)
        args.expected_queue_seal_sha256 = queue_seal_sha256
        resume = base.resume_queue(staging / "train_source_index", train_output, args=args)
        jobs = base.read_jsonl(train_output / "annotation_calibration_queue.jsonl")
        if ({job.get("event_id") for job in jobs} != set(selector.event_ids) or
                len(jobs) != len(selector.selected_rows) or
                any(job.get("usage_role") != "annotation_calibration" or job.get("immutable_split") != "train" or
                    job.get("training_eligible") is not False for job in jobs)):
            raise ValueError("canonical TRAIN queue does not preserve the selected event role set")
        request_path = train_output / "camera_native_render_requests.jsonl"
        provenance = {
            "schema_version": TRAIN_PROVENANCE_SCHEMA,
            "status": "CANONICAL_TRAIN_RENDER_HANDOFF_READY",
            "training_eligible": False,
            "usage_role": "annotation_calibration",
            "immutable_split": "train",
            "parent_index_manifest_sha256": parent.manifest_sha256,
            "parent_inventory_seal_sha256": parent.inventory_sha256,
            "parent_event_file_sha256": parent.manifest["files"]["event_candidates.jsonl"]["sha256"],
            "source_release_manifest_sha256": expected_source_release_sha256,
            "canonical_protocol_sha256": expected_protocol_sha256,
            "coverage_expectations_sha256": expected_coverage_sha256,
            "selector_manifest_sha256": selector.manifest_sha256,
            "selector_selection_seal_sha256": selector.selection_seal_sha256,
            "selector_queue_seal_sha256": selector.queue_seal_sha256,
            "selector_rows_sha256": _sha256(selector.selected_rows_path),
            "selector_prior_source_windows_sha256": selector.manifest.get("prior_source_windows_sha256"),
            "subset_index_manifest_sha256": subset["manifest_sha256"],
            "subset_inventory_seal_sha256": subset["inventory_sha256"],
            "canonical_queue_manifest_sha256": _sha256(train_output / "manifest.json"),
            "canonical_queue_seal_sha256": queue_seal_sha256,
            "canonical_request_sha256": _sha256(request_path),
            "selected_event_count": len(jobs),
            "resume_validation_status": resume["status"],
            "no_outcome_or_action_labels": True,
        }
        _write_json(staging / "train_coverage_provenance.json", provenance)
        os.rename(staging, output_root)
        return {
            "status": "CANONICAL_TRAIN_RENDER_HANDOFF_READY",
            "output_root": str(output_root),
            "train_output": str(output_root / "train"),
            "train_source_index": str(output_root / "train_source_index"),
            "queue_seal_sha256": queue_seal_sha256,
            "request_sha256": _sha256(output_root / "train" / "camera_native_render_requests.jsonl"),
            "provenance_sha256": _sha256(output_root / "train_coverage_provenance.json"),
            "selected": len(jobs),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_evaluation_render_receipt(*, selector: SelectorHandoff, parent: ParentIndex,
                                    packet_dir: Path, packet_manifest_sha256: str) -> dict[str, Any]:
    """Build the authenticated EVAL receipt consumed by the review helper."""
    if selector.role != "eval":
        raise ValueError("EVAL receipt requires an evaluation selector handoff")
    manifest = _read_json(Path(packet_dir) / "manifest.json", "packet manifest")
    packets = _read_jsonl(Path(packet_dir) / "packets.jsonl", "packet payload")
    if (_sha256(Path(packet_dir) / "manifest.json") != packet_manifest_sha256 or
            manifest.get("training_eligible") is not False or
            manifest.get("render_requests_sha256") != _sha256(selector.request_path) or
            manifest.get("queue_seal_sha256") != selector.selection_seal_sha256 or
            manifest.get("protocol_sha256") != parent.protocol_sha256 or
            manifest.get("index_manifest_sha256") != parent.manifest_sha256):
        raise ValueError("EVAL packet manifest does not bind its selector/index authority")
    bindings = []
    for packet in packets:
        audit = packet.get("audit")
        if (not isinstance(audit, Mapping) or audit.get("usage_role") != "evaluation_only" or
                packet.get("training_eligible") is not False):
            raise ValueError("EVAL packet crossed the evaluation-only role boundary")
        bindings.append({
            "event_id": audit["event_id"],
            "request_id": audit.get("render_request_id"),
            "packet_id": packet["packet_id"],
        })
    bindings.sort(key=lambda row: row["event_id"])
    receipt = {
        "schema_version": EVAL_RECEIPT_SCHEMA,
        "status": "EVALUATION_ONLY_RENDER_RECEIPT",
        "training_eligible": False,
        "usage_role": "evaluation_only",
        "immutable_split": "eval",
        "selector_manifest_sha256": selector.manifest_sha256,
        "selector_selection_seal_sha256": selector.selection_seal_sha256,
        "selector_request_filename": selector.request_path.name,
        "selector_request_sha256": _sha256(selector.request_path),
        "index_manifest_sha256": parent.manifest_sha256,
        "index_event_file_sha256": parent.manifest["files"]["event_candidates.jsonl"]["sha256"],
        "inventory_seal_sha256": parent.inventory_sha256,
        "source_release_manifest_sha256": parent.manifest["source_release_manifest_sha256"],
        "canonical_protocol_sha256": parent.protocol_sha256,
        "packet_manifest_sha256": packet_manifest_sha256,
        "packet_manifest_queue_authority_sha256": selector.selection_seal_sha256,
        "packet_count": len(packets),
        "request_bindings": bindings,
        "no_outcome_or_action_labels": True,
    }
    if set(receipt) != EVAL_RECEIPT_FIELDS:
        raise AssertionError("EVAL receipt schema drift")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--train-selector", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coverage-expectations", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-release-manifest-sha256", required=True)
    parser.add_argument("--expected-inventory-seal-sha256", required=True)
    parser.add_argument("--expected-coverage-expectations-sha256", required=True)
    parser.add_argument("--expected-selector-manifest-sha256", required=True)
    parser.add_argument("--expected-selector-selection-seal-sha256", required=True)
    parser.add_argument("--expected-selector-queue-seal-sha256", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = publish_canonical_train_handoff(
        index=args.index, train_selector=args.train_selector, output_root=args.output_root,
        coverage_path=args.coverage_expectations,
        expected_source_release_sha256=args.expected_source_release_manifest_sha256,
        expected_inventory_sha256=args.expected_inventory_seal_sha256,
        expected_coverage_sha256=args.expected_coverage_expectations_sha256,
        protocol_path=args.protocol_path, expected_protocol_sha256=args.expected_protocol_sha256,
        expected_selector_manifest_sha256=args.expected_selector_manifest_sha256,
        expected_selector_selection_seal_sha256=args.expected_selector_selection_seal_sha256,
        expected_selector_queue_seal_sha256=args.expected_selector_queue_seal_sha256,
    )
    print(base.canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
