from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import build_memlite_causal_review_pages as builder  # noqa: E402


CAMERAS = ("head", "left_wrist", "right_wrist")
SIZES = {"head": (720, 720), "left_wrist": (480, 480), "right_wrist": (480, 480)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _fixture(root: Path, *, reordered_queue: bool = False, request_backed: bool = False,
             canonical_source_group_shape: str = "top_level") -> dict[str, Path]:
    sealed = root / "sealed"
    index = root / "phase_candidate_index"
    assets = sealed / "assets"
    assets.mkdir(parents=True)
    index.mkdir()
    # Event A has a longer causal history; event B represents frame-zero
    # clamping, with one causal anchor and two future audit samples.
    specs = ([
        ("a" * 64, "1" * 64, 10, [(0, -4, 6), (1, -3, 7), (2, -2, 8), (3, -1, 9), (4, 0, 10)],
         [(5, 1, 11), (6, 2, 12), (7, 3, 13), (8, 4, 14), (9, 5, 15)]),
        ("b" * 64, "2" * 64, 0, [(0, 0, 0)], [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]),
    ] if request_backed else [
        ("a" * 64, "1" * 64, 10, [(0, -2, 8), (1, -1, 9), (2, 0, 10)], [(3, 1, 11), (4, 16, 26)]),
        ("b" * 64, "2" * 64, 0, [(0, 0, 0)], [(1, 1, 1), (2, 16, 16)]),
    ])
    events: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for event_id, packet_id, anchor, causal_spec, future_spec in specs:
        source = {
            "source_release_manifest_sha256": "3" * 64,
            "source_annotation_sha256": "4" * 64,
            "source_group_id": ("5" if event_id == "a" * 64 else "6") * 64,
            "task_index": 4,
            "task_instance_id": 7 if event_id == "a" * 64 else 8,
            "raw_episode_id": 9 if event_id == "a" * 64 else 10,
            "episode_index": 11 if event_id == "a" * 64 else 12,
        }
        event = {
            "event_id": event_id,
            "source": source,
            "observation": {"frame": anchor, "timestamp_s": anchor / 30},
            "task_name": "fixture task",
            "video_locators": [{"view": camera} for camera in CAMERAS],
        }
        if request_backed:
            event["skill_bundle"] = [{"skill_id": 4 if event_id == "a" * 64 else 5,
                                       "skill_idx": 0, "verb": "PRESS"}]
        events.append(event)
        causal: list[dict[str, Any]] = []
        future: list[dict[str, Any]] = []
        schedule: list[dict[str, Any]] = []
        all_specs = [(frame_index, offset, frame, "ACTOR_CAUSAL") for frame_index, offset, frame in causal_spec]
        all_specs += [(frame_index, offset, frame, "OFFLINE_FUTURE_AUDIT") for frame_index, offset, frame in future_spec]
        requested_count = 10 if request_backed else (5 if anchor == 0 else len(all_specs))
        clamped_count = requested_count - len(all_specs)
        for sample_index, offset, frame, role in all_specs:
            images: dict[str, str] = {}
            locators = [{"view": camera} for camera in CAMERAS]
            schedule.append({
                "sample_index": sample_index,
                "offset_frames": offset,
                "sample_frame": frame,
                "timestamp_s": frame / 30,
                "temporal_role": role,
                "video_locators": locators,
            })
            for camera in CAMERAS:
                filename = f"{packet_id}_t{sample_index:02d}_{camera}.png"
                relative = f"assets/{filename}"
                path = assets / filename
                color = ((sample_index * 53) % 255, (frame * 17) % 255, (len(camera) * 29) % 255)
                Image.new("RGB", SIZES[camera], color=color).save(path, format="PNG", optimize=False)
                images[camera] = relative
                receipts.append({
                    "asset_kind": "camera_native_rgb_png",
                    "bytes": path.stat().st_size,
                    "packet_id": packet_id,
                    "relative_path": relative,
                    "sample_frame": frame,
                    "sample_index": sample_index,
                    "schema_version": "fixture",
                    "sha256": _sha(path),
                    "temporal_role": role,
                    "view": camera,
                })
            sample = {
                "images": images,
                "offset_frames": offset,
                "sample_frame": frame,
                "sample_index": sample_index,
                "temporal_role": role,
                "timestamp_s": frame / 30,
            }
            (causal if role == "ACTOR_CAUSAL" else future).append(sample)
        actor_anchor = next(sample for sample in causal if sample["offset_frames"] == 0)
        packet = {
            "packet_id": packet_id,
            "schema_version": "fixture",
            "status": "DECODED_PENDING_LABELS",
            "training_eligible": False,
            "actor_packet": {
                "actor_instruction": "fixture",
                "causal_temporal_rgb": causal,
                "event_id": event_id,
                "images": actor_anchor["images"],
                "observation_frame": anchor,
                "packet_id": packet_id,
                "question_context": {},
            },
            "audit": {
                "clamped_duplicate_sample_frame_count": clamped_count,
                "decoded_pts_receipts": [],
                "distinct_sample_frame_count": len(schedule),
                "event_id": event_id,
                "offline_future_rgb": future,
                "render_request_id": ("7" if event_id == "a" * 64 else "8") * 64 if request_backed else None,
                "requested_temporal_slot_count": requested_count,
                "review_only_contact_sheet": None,
                "source": source,
                "temporal_schedule": schedule,
                "temporal_slot_count": len(schedule),
                "usage_role": "annotation_calibration",
                "video_locators": [{"view": camera} for camera in CAMERAS],
            },
        }
        packets.append(packet)
    _write_jsonl(index / "event_candidates.jsonl", events)
    source_groups = [{
        "original_split": "train",
        "schema_version": "fixture",
        "source_episode_count": 1,
        "source_episode_ids": [event["source"]["raw_episode_id"]],
        "source_group_id": event["source"]["source_group_id"],
        "source_release_manifest_sha256": event["source"]["source_release_manifest_sha256"],
        "task_index": event["source"]["task_index"],
        "task_instance_id": event["source"]["task_instance_id"],
        "usage_role": "annotation_calibration",
    } for event in events]
    _write_jsonl(index / "source_groups.jsonl", source_groups)
    index_manifest = {
        "schema_version": "memlite-event-index-v1",
        "status": "METADATA_ONLY_READY_FOR_PACKET_RENDER",
        "training_eligible": False,
        "source_release_manifest_sha256": "3" * 64,
        "protocol_sha256": "a" * 64,
        "files": {
            "event_candidates.jsonl": {"sha256": _sha(index / "event_candidates.jsonl"),
                                       "rows": len(events), "bytes": (index / "event_candidates.jsonl").stat().st_size},
            "source_groups.jsonl": {"sha256": _sha(index / "source_groups.jsonl"),
                                    "rows": len(source_groups), "bytes": (index / "source_groups.jsonl").stat().st_size},
        },
    }
    (index / "manifest.json").write_text(json.dumps(index_manifest, sort_keys=True) + "\n", encoding="utf-8")
    inventory = {
        "schema_version": "memlite-event-inventory-seal-v1",
        "index_manifest_sha256": _sha(index / "manifest.json"),
        "source_release_manifest_sha256": "3" * 64,
        "coverage_expectations_sha256": "e" * 64,
        "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
        "payload_files": index_manifest["files"],
    }
    (index / "inventory_seal.json").write_text(json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8")
    if request_backed:
        queue_sources = []
        for event in events:
            queue_source = dict(event["source"])
            queue_source.pop("source_group_id")
            queue_sources.append(queue_source)
        queue_rows = [
            {"event_id": "a" * 64, "job_id": "7" * 64, "queue_kind": "ANNOTATION_CALIBRATION",
             "status": "CANDIDATE_MISSING_EVIDENCE", "training_eligible": False,
             "schema_version": "p107-metadata-annotation-queue-v1", "immutable_split": "train",
             "usage_role": "annotation_calibration", "source_identity": queue_sources[0],
             "source_group_id": events[0]["source"]["source_group_id"], "skill_ids": [4]},
            {"event_id": "b" * 64, "job_id": "8" * 64, "queue_kind": "ANNOTATION_CALIBRATION",
             "status": "CANDIDATE_MISSING_EVIDENCE", "training_eligible": False,
             "schema_version": "p107-metadata-annotation-queue-v1", "immutable_split": "train",
             "usage_role": "annotation_calibration", "source_identity": queue_sources[1],
             "source_group_id": events[1]["source"]["source_group_id"], "skill_ids": [5]},
        ]
        if canonical_source_group_shape == "nested_equal":
            for row, event in zip(queue_rows, events):
                row["source_identity"]["source_group_id"] = event["source"]["source_group_id"]
        elif canonical_source_group_shape == "nested_conflict":
            queue_rows[0]["source_identity"]["source_group_id"] = "f" * 64
        elif canonical_source_group_shape == "missing":
            for row in queue_rows:
                del row["source_group_id"]
        elif canonical_source_group_shape != "top_level":
            raise ValueError(f"unknown canonical_source_group_shape: {canonical_source_group_shape}")
    else:
        queue_rows = [
            {"event_id": "a" * 64, "observation_frame": 10, "selection_order": 0,
             "source_identity": events[0]["source"], "schema_version": "p107-phase-balanced-calibration-queue-v1",
             "training_eligible": False, "immutable_split": "train", "usage_role": "annotation_calibration"},
            {"event_id": "b" * 64, "observation_frame": 0, "selection_order": 1,
             "source_identity": events[1]["source"], "schema_version": "p107-phase-balanced-calibration-queue-v1",
             "training_eligible": False, "immutable_split": "train", "usage_role": "annotation_calibration"},
        ]
    if reordered_queue and not request_backed:
        queue_rows = [
            {"event_id": "b" * 64, "observation_frame": 0, "selection_order": 0,
             "source_identity": events[1]["source"], "schema_version": "p107-phase-balanced-calibration-queue-v1",
             "training_eligible": False, "immutable_split": "train", "usage_role": "annotation_calibration"},
            {"event_id": "a" * 64, "observation_frame": 10, "selection_order": 1,
             "source_identity": events[0]["source"], "schema_version": "p107-phase-balanced-calibration-queue-v1",
             "training_eligible": False, "immutable_split": "train", "usage_role": "annotation_calibration"},
        ]
    if not request_backed:
        queue = root / "phase_balanced_queue.jsonl"
        _write_jsonl(queue, queue_rows)
        payload_files = {
            "phase_balanced_queue.jsonl": {"sha256": _sha(queue), "rows": len(queue_rows), "bytes": queue.stat().st_size},
            "phase_candidate_index/event_candidates.jsonl": index_manifest["files"]["event_candidates.jsonl"],
            "phase_candidate_index/source_groups.jsonl": index_manifest["files"]["source_groups.jsonl"],
            "phase_candidate_index/manifest.json": {"sha256": _sha(index / "manifest.json"), "rows": 1,
                                                     "bytes": (index / "manifest.json").stat().st_size},
            "phase_candidate_index/inventory_seal.json": {"sha256": _sha(index / "inventory_seal.json"), "rows": 1,
                                                           "bytes": (index / "inventory_seal.json").stat().st_size},
        }
        selection_manifest = {
            "schema_version": "p107-phase-balanced-calibration-manifest-v1",
            "status": "METADATA_CANDIDATES_READY_FOR_INDEPENDENT_RENDER_REVIEW",
            "training_eligible": False,
            "source_release_manifest_sha256": "3" * 64,
            "files": payload_files,
            "policy": {"source_release_manifest_sha256": "3" * 64, "protocol_sha256": "a" * 64},
            "mini_index": {"manifest_sha256": _sha(index / "manifest.json"),
                           "inventory_seal_sha256": _sha(index / "inventory_seal.json")},
            "parent_index": {"inventory_seal_sha256": "f" * 64},
            "selection": {"exact_budget": len(queue_rows), "all_usage_role_annotation_calibration": True,
                          "all_immutable_split_train": True, "all_evidence_missing": True},
        }
        selection_path = root / "phase_selection_manifest.json"
        selection_path.write_text(json.dumps(selection_manifest, sort_keys=True) + "\n", encoding="utf-8")
        queue_seal = {
            "schema_version": "p107-phase-balanced-calibration-seal-v1",
            "selection_manifest_sha256": _sha(selection_path),
            "parent_inventory_seal_sha256": "f" * 64,
            "source_release_manifest_sha256": "3" * 64,
            "protocol_sha256": "a" * 64,
            "payload_files": payload_files,
            "training_eligible": False,
        }
        queue_seal_path = root / "phase_queue_seal.json"
        queue_seal_path.write_text(json.dumps(queue_seal, sort_keys=True) + "\n", encoding="utf-8")
    packets_path = sealed / "packets.jsonl"
    receipts_path = sealed / "rendered_asset_receipts.jsonl"
    _write_jsonl(packets_path, packets)
    _write_jsonl(receipts_path, receipts)
    if request_backed:
        requests_path = root / "camera_native_render_requests.jsonl"
        requests = []
        for packet in packets:
            event_id = packet["audit"]["event_id"]
            causal = packet["actor_packet"]["causal_temporal_rgb"]
            future = packet["audit"]["offline_future_rgb"]
            actor_frames = [sample["sample_frame"] for sample in causal]
            after_frames = [sample["sample_frame"] for sample in future]
            requests.append({
                "schema_version": "p107-camera-native-temporal-request-v1",
                "request_id": packet["audit"]["render_request_id"],
                "status": "PENDING_RENDERER_HANDSHAKE_NO_RGB_DECODED",
                "event_id": event_id,
                "source_identity": next(event["source"] for event in events if event["event_id"] == event_id),
                "requested_frame_indices": actor_frames + after_frames,
                "actor_available_frame_indices": actor_frames,
                "offline_review_before_frame_indices": actor_frames[:-1],
                "offline_review_after_frame_indices": after_frames,
                "anchor_camera_locators": next(event["video_locators"] for event in events if event["event_id"] == event_id),
                "frame_locator_resolution": "RENDERER_MUST_LOOK_UP_EACH_REQUESTED_FRAME_FROM_FROZEN_SOURCE_METADATA",
                "camera_delivery": {
                    "camera_native_only": True, "include_footer": False,
                    "allow_contact_sheet_in_actor_input": False, "decoded_by_selector": False,
                    "forbidden_footer_fields": ["outcome", "success", "failure", "recovery", "label", "evidence", "review"],
                },
            })
        _write_jsonl(requests_path, requests)
        queue = root / "annotation_calibration_queue.jsonl"
        _write_jsonl(queue, queue_rows)
        for name, contents in (("counts.json", "{}\n"), ("student_candidate_queue.jsonl", "")):
            (root / name).write_text(contents, encoding="utf-8")
        payload_files = {
            "annotation_calibration_queue.jsonl": {"sha256": _sha(queue), "rows": len(queue_rows), "bytes": queue.stat().st_size},
            "camera_native_render_requests.jsonl": {"sha256": _sha(requests_path), "rows": len(requests), "bytes": requests_path.stat().st_size},
            "counts.json": {"sha256": _sha(root / "counts.json"), "rows": 1, "bytes": (root / "counts.json").stat().st_size},
            "student_candidate_queue.jsonl": {"sha256": _sha(root / "student_candidate_queue.jsonl"), "rows": 0, "bytes": (root / "student_candidate_queue.jsonl").stat().st_size},
        }
        policy = {
            "frozen_source_release_manifest_sha256": "3" * 64,
            "sealed_index_manifest_sha256": _sha(index / "manifest.json"),
            "canonical_protocol_sha256": "a" * 64,
            "all_jobs_status": "CANDIDATE_MISSING_EVIDENCE",
            "policy_sha256": "9" * 64,
        }
        selection_manifest = {
            "schema_version": "p107-metadata-annotation-queue-manifest-v1",
            "status": "METADATA_CANDIDATES_READY_FOR_RENDERER_HANDSHAKE",
            "training_eligible": False,
            "all_jobs_status": "CANDIDATE_MISSING_EVIDENCE",
            "source_release_manifest_sha256": "3" * 64,
            "index_manifest_sha256": _sha(index / "manifest.json"),
            "inventory_seal_sha256": _sha(index / "inventory_seal.json"),
            "canonical_protocol_sha256": "a" * 64,
            "coverage_expectations_sha256": "e" * 64,
            "index_event_file_sha256": index_manifest["files"]["event_candidates.jsonl"]["sha256"],
            "index_source_group_file_sha256": index_manifest["files"]["source_groups.jsonl"]["sha256"],
            "policy": policy,
            "files": payload_files,
            "renderer_handshake": {"status": "PENDING_INTERFACE_OWNER_CONFIRMATION",
                                   "schema_version": "p107-camera-native-temporal-request-v1",
                                   "selector_decoded_rgb": False, "temporal_sequence_required": True,
                                   "no_footer_truth": True},
            "design_handoff": {},
        }
        selection_path = root / "manifest.json"
        selection_path.write_text(json.dumps(selection_manifest, sort_keys=True) + "\n", encoding="utf-8")
        queue_seal = {
            "schema_version": "p107-metadata-annotation-queue-seal-v1",
            "queue_manifest_sha256": _sha(selection_path),
            "inventory_seal_sha256": _sha(index / "inventory_seal.json"),
            "source_release_manifest_sha256": "3" * 64,
            "canonical_protocol_sha256": "a" * 64,
            "coverage_expectations_sha256": "e" * 64,
            "policy_sha256": policy["policy_sha256"],
            "expected_payload_files": sorted(payload_files),
            "payload_files": payload_files,
        }
        queue_seal_path = root / "queue_seal.json"
        queue_seal_path.write_text(json.dumps(queue_seal, sort_keys=True) + "\n", encoding="utf-8")
    else:
        requests_path = None
    offsets = sorted({sample["offset_frames"] for packet in packets
                      for sample in packet["audit"]["temporal_schedule"]})
    temporal_slots = sum(packet["audit"]["requested_temporal_slot_count"] for packet in packets)
    distinct_frames = sum(packet["audit"]["distinct_sample_frame_count"] for packet in packets)
    clamped_frames = sum(packet["audit"]["clamped_duplicate_sample_frame_count"] for packet in packets)
    packet_manifest = {"schema_version": "memlite-event-packet-index-v1",
        "status": "CPU_READY_LABELS_PENDING",
        "training_eligible": False,
        "index_manifest_sha256": _sha(index / "manifest.json"),
        "index_event_file_sha256": _sha(index / "event_candidates.jsonl"),
        "files": {
            "packets.jsonl": {"sha256": _sha(packets_path), "rows": len(packets), "bytes": packets_path.stat().st_size},
            "rendered_asset_receipts.jsonl": {"sha256": _sha(receipts_path), "rows": len(receipts), "bytes": receipts_path.stat().st_size},
        },
        "packets": len(packets), "rendered_asset_receipts": len(receipts), "decoded_camera_native_rgb": True,
        "temporal_sample_offsets_frames": offsets, "temporal_slots": temporal_slots,
        "distinct_temporal_sample_frames": distinct_frames, "clamped_duplicate_temporal_samples": clamped_frames,
        "render_requests_sha256": _sha(requests_path) if request_backed else None,
        "queue_seal_sha256": _sha(queue_seal_path) if request_backed else None, "review_only_contact_sheets": False,
        "full_video_hashing": False, "renderer_sha256": "b" * 64, "protocol_sha256": "a" * 64,
        "wall_seconds": 0.0,
    }
    packet_manifest_path = sealed / "manifest.json"
    packet_manifest_path.write_text(json.dumps(packet_manifest, sort_keys=True) + "\n", encoding="utf-8")
    query_registry = root / "query_registry.jsonl"
    _write_jsonl(query_registry, [
        {"event_id": "a" * 64, "prelabel_query_id": "c" * 64, "query_text": "Can A be established?"},
        {"event_id": "b" * 64, "prelabel_query_id": "d" * 64, "query_text": "Can B be established?"},
    ])
    result = {"sealed": sealed, "index": index, "queue": queue, "query": query_registry,
            "packets": packets_path, "receipts": receipts_path, "queue_seal": queue_seal_path,
            "packet_manifest": packet_manifest_path}
    if request_backed:
        result["render_requests"] = requests_path
    return result


def _build(paths: dict[str, Path], output: Path, **kwargs):
    if kwargs.get("query_registry") is not None:
        kwargs.setdefault("expected_query_registry_sha256", _sha(paths["query"]))
    if kwargs.get("coverage_event_bindings_path") is not None:
        kwargs.setdefault("expected_coverage_event_bindings_sha256", _sha(paths["coverage_bindings"]))
    return builder.build(
        sealed_root=paths["sealed"], index_root=paths["index"], queue_path=paths["queue"],
        queue_seal_path=paths["queue_seal"], expected_queue_seal_sha256=_sha(paths["queue_seal"]),
        expected_packet_manifest_sha256=_sha(paths["packet_manifest"]), output=output, **kwargs,
    )


def _coverage_registry(paths: dict[str, Path], *, schema: str = builder.COVERAGE_REGISTRY_SCHEMA) -> Path:
    """Project the fixture's canonical TRAIN handoff into afbf874's registry contract."""
    queue_rows = [json.loads(line) for line in paths["queue"].read_text().splitlines() if line.strip()]
    request_rows = [json.loads(line) for line in paths["render_requests"].read_text().splitlines() if line.strip()]
    events = [json.loads(line) for line in (paths["index"] / "event_candidates.jsonl").read_text().splitlines() if line.strip()]
    events_by_id = {row["event_id"]: row for row in events}
    requests_by_id = {row["event_id"]: row for row in request_rows}
    queue_seal_sha = _sha(paths["queue_seal"])
    queue_manifest = paths["queue_seal"].parent / "manifest.json"
    index_manifest = paths["index"] / "manifest.json"
    inventory = paths["index"] / "inventory_seal.json"
    index_manifest_row = json.loads(index_manifest.read_text())
    registry: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        event_id = queue_row["event_id"]
        event = events_by_id[event_id]
        source = event["source"]
        request = requests_by_id[event_id]
        skill_id = queue_row["skill_ids"][0]
        skill = next(skill for skill in event["skill_bundle"] if skill["skill_id"] == skill_id)
        pin = {
            "schema_version": builder.COVERAGE_SOURCE_PIN_SCHEMA,
            "binding_kind": "coverage_selector_event_query",
            "selection_role": "train",
            "usage_role": "annotation_calibration",
            "immutable_split": "train",
            "event_id": event_id,
            "selector_manifest_sha256": _sha(queue_manifest),
            "selector_selection_seal_sha256": queue_seal_sha,
            "selector_job_filename": paths["queue"].name,
            "selector_job_sha256": _sha(paths["queue"]),
            "selector_record_sha256": _canonical_digest(queue_row),
            "render_request_filename": paths["render_requests"].name,
            "render_request_sha256": _sha(paths["render_requests"]),
            "render_request_record_sha256": _canonical_digest(request),
            "index_manifest_sha256": _sha(index_manifest),
            "index_event_file_sha256": index_manifest_row["files"]["event_candidates.jsonl"]["sha256"],
            "index_event_record_sha256": _canonical_digest(event),
            "inventory_seal_sha256": _sha(inventory),
            "source_release_manifest_sha256": source["source_release_manifest_sha256"],
            "source_annotation_sha256": source["source_annotation_sha256"],
            "source_group_id": source["source_group_id"],
            "raw_episode_id": source["raw_episode_id"],
            "episode_index": source["episode_index"],
            "observation_frame": event["observation"]["frame"],
        }
        text = f"Is the fixture {skill['verb'].lower()} relation visibly established?"
        row = {
            "schema_version": schema,
            "status": "PRELABEL_QUERY_CANDIDATE_ONLY",
            "prelabel_query_id": "0" * 64,
            "event_id": event_id,
            "source_group_id": source["source_group_id"],
            "query_ordinal_within_event": 0,
            "observation_frame": event["observation"]["frame"],
            "skill_id": skill_id,
            "canonical_verb": skill["verb"],
            "query_text": text,
            "query_content_sha256": _canonical_digest({
                "schema_version": builder.COVERAGE_QUERY_CONTENT_SCHEMA,
                "kind": "goal_satisfaction_counterfactual", "text": text,
            }),
            "relation_family": builder._coverage_relation_family(skill["verb"]),
            "source_pin": pin,
            "usage_role": "annotation_calibration",
            "immutable_split": "train",
            "training_eligible": False,
            "no_outcome_or_action_labels": True,
        }
        row["prelabel_query_id"] = _canonical_digest({
            "schema_version": builder.COVERAGE_QUERY_SCHEMA,
            "selection_role": "train", "usage_role": row["usage_role"],
            "immutable_split": row["immutable_split"], "event_id": event_id,
            "source_group_id": row["source_group_id"],
            "observation_frame": row["observation_frame"],
            "query_ordinal_within_event": row["query_ordinal_within_event"],
            "skill_id": row["skill_id"], "canonical_verb": row["canonical_verb"],
            "query_text": row["query_text"], "relation_family": row["relation_family"],
            "goal_scope": "CURRENT_VISIBLE_RELATION_AT_ANCHOR", "source_pin": pin,
        })
        registry.append(row)
    _write_jsonl(paths["query"], registry)
    return paths["query"]


def _coverage_event_bindings(paths: dict[str, Path], *, unbound_event_ids: set[str] | None = None) -> Path:
    """Write the producer's explicit selected_event_binding receipt."""
    unbound_event_ids = unbound_event_ids or set()
    queue_rows = [json.loads(line) for line in paths["queue"].read_text().splitlines() if line.strip()]
    registry_rows = [json.loads(line) for line in paths["query"].read_text().splitlines() if line.strip()]
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in registry_rows:
        by_event.setdefault(row["event_id"], []).append(row)
    rows: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        event_id = queue_row["event_id"]
        eligible = [] if event_id in unbound_event_ids else by_event.get(event_id, [])
        eligible_ids = [row["prelabel_query_id"] for row in eligible]
        quarantined_ids = ["e" * 64] if event_id in unbound_event_ids else []
        status = ("CATEGORY_GROUNDING_QUARANTINED_NO_ELIGIBLE_QUERY"
                  if quarantined_ids else "ELIGIBLE_QUERIES")
        rows.append({
            "schema_version": builder.COVERAGE_EVENT_BINDING_SCHEMA,
            "status": status,
            "event_id": event_id,
            "source_group_id": queue_row["source_group_id"],
            "observation_frame": next(json.loads(line)["observation"]["frame"] for line in
                                       (paths["index"] / "event_candidates.jsonl").read_text().splitlines()
                                       if json.loads(line)["event_id"] == event_id),
            "selected_skill_ids": list(queue_row["skill_ids"]),
            "candidate_query_count": 1,
            "eligible_query_count": len(eligible_ids),
            "quarantined_query_count": len(quarantined_ids),
            "unsupported_skill_count": 0,
            "eligible_prelabel_query_ids": eligible_ids,
            "quarantined_prelabel_query_ids": quarantined_ids,
            "unsupported_skill_ids": [],
            "selector_record_sha256": _canonical_digest(queue_row),
            "usage_role": "annotation_calibration",
            "immutable_split": "train",
            "training_eligible": False,
            "no_outcome_or_action_labels": True,
        })
    binding_path = paths["queue"].parent / "selected_event_bindings.jsonl"
    _write_jsonl(binding_path, rows)
    paths["coverage_bindings"] = binding_path
    return binding_path


class CausalReviewPageBuilderTests(unittest.TestCase):
    def test_default_pages_have_no_future_or_second_row_future_leak_and_preserve_clamp(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            output = root / "actor-pages"
            result = _build(paths, output, query_registry=paths["query"])
            self.assertEqual(result["events"], 2)
            self.assertEqual(result["actor_pages"], 3)
            pages = [json.loads(line) for line in (output / "pages.jsonl").read_text().splitlines()]
            assets = [asset for page in pages for asset in page["source_assets"]]
            self.assertTrue(assets)
            self.assertTrue(all(asset["temporal_role"] == "ACTOR_CAUSAL" for asset in assets))
            for page in pages:
                anchor = 10 if page["event_id"] == "a" * 64 else 0
                self.assertTrue(all(asset["sample_frame"] <= anchor for asset in page["source_assets"]))
            self.assertFalse(any("OFFLINE_FUTURE_AUDIT" in json.dumps(page) for page in pages))
            # The source asset rows are the authoritative per-cell provenance;
            # there is no lower-row future sample hidden in a composite.
            self.assertTrue(all(asset["canvas_xy"][1] in (0, 720) for asset in assets))
            ordered = [json.loads(line) for line in (output / "ordered_events.jsonl").read_text().splitlines()]
            frame_zero = next(row for row in ordered if row["event_id"] == "b" * 64)
            self.assertEqual(frame_zero["temporal_provenance"]["clamped_duplicate_sample_frame_count"], 2)
            self.assertEqual(frame_zero["temporal_provenance"]["actor_causal_sample_indices"], [0])
            self.assertNotIn('"query_text":', json.dumps(ordered))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["future_pages_in_default_output"])
            self.assertNotIn("audit_output", manifest)

    def test_reordered_queue_binds_exact_event_and_query_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, reordered_queue=True)
            output = root / "actor-pages"
            _build(paths, output, query_registry=paths["query"])
            rows = [json.loads(line) for line in (output / "ordered_events.jsonl").read_text().splitlines()]
            self.assertEqual([row["event_id"] for row in rows], ["b" * 64, "a" * 64])
            self.assertEqual([row["prelabel_query_id"] for row in rows], ["d" * 64, "c" * 64])
            self.assertEqual([row["helper_id"] for row in rows], ["q000", "q001"])

    def test_re_pinned_registry_with_noncanonical_query_id_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            rows = [json.loads(line) for line in paths["query"].read_text().splitlines()]
            rows[0]["prelabel_query_id"] = "not-a-canonical-query-id"
            _write_jsonl(paths["query"], rows)
            # _build re-pins the mutated registry bytes, so this exercises the
            # schema gate rather than the external SHA mismatch gate.
            with self.assertRaisesRegex(ValueError, "query registry query ID"):
                _build(paths, root / "actor-pages", query_registry=paths["query"])
            self.assertFalse((root / "actor-pages").exists())

    def test_optional_future_audit_is_separate_and_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            output = root / "actor-pages"
            audit = root / "future-audit-pages"
            _build(paths, output, audit_output=audit)
            actor_manifest = json.loads((output / "manifest.json").read_text())
            audit_manifest = json.loads((audit / "manifest.json").read_text())
            self.assertFalse(actor_manifest["future_pages_in_default_output"])
            self.assertTrue(audit_manifest["future_pages_in_default_output"])
            self.assertEqual(audit_manifest["temporal_view"], "OFFLINE_FUTURE_AUDIT_ONLY")
            audit_pages = [json.loads(line) for line in (audit / "pages.jsonl").read_text().splitlines()]
            self.assertTrue(any(page["source_assets"] for page in audit_pages))
            self.assertTrue(all(asset["temporal_role"] == "OFFLINE_FUTURE_AUDIT"
                                for page in audit_pages for asset in page["source_assets"]))

    def test_authenticated_canonical_train_render_requests_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            output = root / "actor-pages"
            _build(paths, output, render_requests_path=paths["render_requests"],
                   expected_render_requests_sha256=_sha(paths["render_requests"]))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["source_selection_manifest_sha256"], _sha(root / "manifest.json"))
            self.assertEqual(manifest["source_protocol_sha256"], "a" * 64)

    def test_authenticated_coverage_train_registry_handshake(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            registry = _coverage_registry(paths)
            bindings = _coverage_event_bindings(paths)
            output = root / "actor-pages"
            _build(paths, output, render_requests_path=paths["render_requests"],
                   expected_render_requests_sha256=_sha(paths["render_requests"]),
                   query_registry=registry, coverage_event_bindings_path=bindings)
            rows = [json.loads(line) for line in (output / "ordered_events.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(len(row["prelabel_query_id"]) == 64 for row in rows))
            self.assertTrue(all(row["query_ids"] == [row["prelabel_query_id"]] for row in rows))

    def test_coverage_eval_registry_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            registry = _coverage_registry(paths, schema=builder.COVERAGE_EVAL_REGISTRY_SCHEMA)
            output = root / "actor-pages"
            with self.assertRaisesRegex(ValueError, "coverage EVAL"):
                _build(paths, output, render_requests_path=paths["render_requests"],
                       expected_render_requests_sha256=_sha(paths["render_requests"]),
                       query_registry=registry)
            self.assertFalse(output.exists())

    def test_coverage_registry_content_pin_query_tamper_is_rejected_before_output(self):
        for field in ("query_content_sha256", "prelabel_query_id", "source_pin"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                paths = _fixture(root, request_backed=True)
                registry = _coverage_registry(paths)
                bindings = _coverage_event_bindings(paths)
                rows = [json.loads(line) for line in registry.read_text().splitlines()]
                if field == "query_content_sha256":
                    rows[0][field] = "0" * 64
                elif field == "prelabel_query_id":
                    rows[0][field] = "0" * 64
                else:
                    rows[0][field]["index_manifest_sha256"] = "0" * 64
                _write_jsonl(registry, rows)
                output = root / "actor-pages"
                with self.assertRaises(ValueError):
                    _build(paths, output, render_requests_path=paths["render_requests"],
                           expected_render_requests_sha256=_sha(paths["render_requests"]),
                           query_registry=registry, coverage_event_bindings_path=bindings)
                self.assertFalse(output.exists())

    def test_coverage_partial_registry_requires_explicit_unbound_event(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            registry = _coverage_registry(paths)
            registry_rows = [json.loads(line) for line in registry.read_text().splitlines()]
            registry_rows = [row for row in registry_rows if row["event_id"] != "b" * 64]
            _write_jsonl(registry, registry_rows)
            output = root / "implicit-unbound"
            with self.assertRaisesRegex(ValueError, "explicit event-binding receipt"):
                _build(paths, output, render_requests_path=paths["render_requests"],
                       expected_render_requests_sha256=_sha(paths["render_requests"]),
                       query_registry=registry)
            self.assertFalse(output.exists())

            bindings = _coverage_event_bindings(paths, unbound_event_ids={"b" * 64})
            output = root / "explicit-unbound"
            result = _build(paths, output, render_requests_path=paths["render_requests"],
                            expected_render_requests_sha256=_sha(paths["render_requests"]),
                            query_registry=registry, coverage_event_bindings_path=bindings)
            self.assertEqual(result["query_registry_rows"], 1)
            rows = [json.loads(line) for line in (output / "ordered_events.jsonl").read_text().splitlines()]
            unbound = next(row for row in rows if row["event_id"] == "b" * 64)
            self.assertEqual(unbound["query_ids"], [])
            self.assertEqual(unbound["query_binding_status"],
                             "CATEGORY_GROUNDING_QUARANTINED_NO_ELIGIBLE_QUERY")
            self.assertNotIn("prelabel_query_id", unbound)

    def test_coverage_event_binding_tamper_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            registry = _coverage_registry(paths)
            bindings = _coverage_event_bindings(paths)
            rows = [json.loads(line) for line in bindings.read_text().splitlines()]
            rows[0]["eligible_prelabel_query_ids"] = []
            _write_jsonl(bindings, rows)
            output = root / "actor-pages"
            with self.assertRaises(ValueError):
                _build(paths, output, render_requests_path=paths["render_requests"],
                       expected_render_requests_sha256=_sha(paths["render_requests"]),
                       query_registry=registry, coverage_event_bindings_path=bindings)
            self.assertFalse(output.exists())

    def test_canonical_nested_source_group_copy_must_match_top_level(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True, canonical_source_group_shape="nested_equal")
            _build(paths, root / "actor-pages", render_requests_path=paths["render_requests"],
                   expected_render_requests_sha256=_sha(paths["render_requests"]))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True, canonical_source_group_shape="nested_conflict")
            output = root / "actor-pages"
            with self.assertRaisesRegex(ValueError, "nested source_group_id"):
                _build(paths, output, render_requests_path=paths["render_requests"],
                       expected_render_requests_sha256=_sha(paths["render_requests"]))
            self.assertFalse(output.exists())

    def test_canonical_top_level_source_group_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True, canonical_source_group_shape="missing")
            output = root / "actor-pages"
            with self.assertRaisesRegex(ValueError, "canonical queue source_group_id"):
                _build(paths, output, render_requests_path=paths["render_requests"],
                       expected_render_requests_sha256=_sha(paths["render_requests"]))
            self.assertFalse(output.exists())

    def test_request_provenance_requires_explicit_path_and_rejects_unclaimed_legacy_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            with self.assertRaisesRegex(ValueError, "--render-requests is required"):
                _build(paths, root / "actor-pages")
            self.assertFalse((root / "actor-pages").exists())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            fake = root / "unclaimed-render-requests.jsonl"
            fake.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "phase queue seal"):
                _build(paths, root / "actor-pages", render_requests_path=fake)
            self.assertFalse((root / "actor-pages").exists())

    def test_render_request_wrong_hash_and_altered_schedule_fail_before_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            with self.assertRaisesRegex(ValueError, "expected render-request SHA"):
                _build(paths, root / "actor-pages", render_requests_path=paths["render_requests"],
                       expected_render_requests_sha256="0" * 64)
            self.assertFalse((root / "actor-pages").exists())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root, request_backed=True)
            rows = [json.loads(line) for line in paths["render_requests"].read_text().splitlines()]
            rows[0]["requested_frame_indices"][-1] += 1
            _write_jsonl(paths["render_requests"], rows)
            with self.assertRaises(ValueError):
                _build(paths, root / "actor-pages", render_requests_path=paths["render_requests"])
            self.assertFalse((root / "actor-pages").exists())

    def test_authenticated_queue_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            rows = [json.loads(line) for line in paths["queue"].read_text().splitlines()]
            rows[0]["source_identity"]["source_group_id"] = "z" * 64
            _write_jsonl(paths["queue"], rows)
            with self.assertRaises(ValueError):
                _build(paths, root / "actor-pages")

    def test_packet_audit_source_mutation_is_rejected_even_with_resealed_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            packets = [json.loads(line) for line in paths["packets"].read_text().splitlines()]
            packets[0]["audit"]["source"]["source_group_id"] = "z" * 64
            _write_jsonl(paths["packets"], packets)
            manifest = json.loads(paths["packet_manifest"].read_text())
            manifest["files"]["packets.jsonl"] = {"sha256": _sha(paths["packets"]),
                                                    "rows": len(packets), "bytes": paths["packets"].stat().st_size}
            paths["packet_manifest"].write_text(json.dumps(manifest, sort_keys=True) + "\n")
            with self.assertRaises(ValueError):
                _build(paths, root / "actor-pages")

    def test_packet_manifest_missing_receipt_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            manifest = json.loads(paths["packet_manifest"].read_text())
            del manifest["files"]["packets.jsonl"]["bytes"]
            paths["packet_manifest"].write_text(json.dumps(manifest, sort_keys=True) + "\n")
            with self.assertRaises(ValueError):
                _build(paths, root / "actor-pages")

    def test_nested_actor_audit_targets_are_rejected_before_mkdir(self):
        for actor_name, audit_name in (("actor", "actor/future"), ("actor/causal", "actor")):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                paths = _fixture(root)
                actor = root / actor_name
                audit = root / audit_name
                with self.assertRaises(ValueError):
                    _build(paths, actor, audit_output=audit)
                self.assertFalse(actor.exists())
                self.assertFalse(audit.exists())

    def test_second_publish_rename_failure_rolls_back_only_own_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            actor = root / "actor-pages"
            audit = root / "future-audit-pages"
            real_rename = builder.os.rename
            calls = 0

            def fail_second(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected second publish failure")
                return real_rename(source, destination)

            with patch.object(builder.os, "rename", side_effect=fail_second), self.assertRaises(OSError):
                _build(paths, actor, audit_output=audit)
            self.assertFalse(actor.exists())
            self.assertFalse(audit.exists())
            self.assertEqual(list(root.glob(".actor-pages.staging-*")), [])
            self.assertEqual(list(root.glob(".future-audit-pages.staging-*")), [])

    def test_contradictory_role_is_rejected(self):
        self._assert_mutation_rejected(lambda packet: packet["actor_packet"]["causal_temporal_rgb"][0].__setitem__(
            "temporal_role", "OFFLINE_FUTURE_AUDIT"))

    def test_missing_camera_is_rejected(self):
        self._assert_mutation_rejected(lambda packet: packet["actor_packet"]["causal_temporal_rgb"][0]["images"].pop(
            "right_wrist"))

    def test_wrong_hash_is_rejected(self):
        def mutate(_packet: dict[str, Any], receipts: list[dict[str, Any]]) -> None:
            receipts[0]["sha256"] = "0" * 64

        self._assert_mutation_rejected(mutate, mutate_receipts=True)

    def test_path_traversal_is_rejected(self):
        def mutate(packet: dict[str, Any]) -> None:
            packet["actor_packet"]["causal_temporal_rgb"][0]["images"]["head"] = "assets/../escape.png"

        self._assert_mutation_rejected(mutate)

    def _assert_mutation_rejected(self, mutate, *, mutate_receipts: bool = False) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = _fixture(root)
            packets = [json.loads(line) for line in paths["packets"].read_text().splitlines()]
            receipts = [json.loads(line) for line in paths["receipts"].read_text().splitlines()]
            if mutate_receipts:
                mutate(packets[0], receipts)
            else:
                mutate(packets[0])
            _write_jsonl(paths["packets"], packets)
            _write_jsonl(paths["receipts"], receipts)
            # Keep the sealed file claims consistent so the test reaches the
            # temporal/path/asset gate under test rather than the outer hash gate.
            manifest = json.loads((paths["sealed"] / "manifest.json").read_text())
            manifest["files"]["packets.jsonl"] = {"sha256": _sha(paths["packets"]),
                                                    "rows": len(packets), "bytes": paths["packets"].stat().st_size}
            manifest["files"]["rendered_asset_receipts.jsonl"] = {"sha256": _sha(paths["receipts"]),
                                                                    "rows": len(receipts), "bytes": paths["receipts"].stat().st_size}
            (paths["sealed"] / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
            with self.assertRaises(ValueError):
                _build(paths, root / "actor-pages")


if __name__ == "__main__":
    unittest.main()
