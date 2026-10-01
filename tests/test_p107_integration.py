"""Minimal stdlib integration smoke for accepted P107 candidate components.

This creates synthetic sealed metadata only. It is not a DART trajectory,
label package, authority root, or data-release test.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


DATA_PROTOCOL_SHA256 = "efdd20642fed24241f38bbdeb4abff6cf4faf1c72a86fe7c2acb32ba7496193b"


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _write_source_membership_fixture(root: Path) -> tuple[str, str, str, str, str]:
    """Write the DATA API's sealed group boundary, intentionally no event file."""

    candidate_group = _digest("integrated-candidate-group")
    calibration_group = _digest("integrated-calibration-group")
    source_release = _digest("integrated-source-release")
    rows = [
        {
            "source_group_id": candidate_group,
            "source_release_manifest_sha256": source_release,
            "task_index": 42,
            "task_instance_id": 9,
            "original_split": "train",
            "usage_role": "student_candidate",
        },
        {
            "source_group_id": calibration_group,
            "source_release_manifest_sha256": source_release,
            "task_index": 42,
            "task_instance_id": 9,
            "original_split": "train",
            "usage_role": "annotation_calibration",
        },
    ]
    source_groups_bytes = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    (root / "source_groups.jsonl").write_bytes(source_groups_bytes)
    files = {
        "source_groups.jsonl": {
            "sha256": sha256(source_groups_bytes).hexdigest(),
            "bytes": len(source_groups_bytes),
            "rows": len(rows),
        },
        # Omission is intentional: source-membership must not scan events.
        "event_candidates.jsonl": {
            "sha256": sha256(b"").hexdigest(),
            "bytes": 0,
            "rows": 0,
        },
    }
    manifest = {
        "schema_version": "memlite-event-index-v1",
        "source_release_manifest_sha256": source_release,
        "files": files,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    seal = {
        "schema_version": "memlite-event-inventory-seal-v1",
        "index_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "source_release_manifest_sha256": source_release,
        "coverage_expectations_sha256": _digest("integrated-coverage"),
        "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
        "payload_files": files,
    }
    seal_bytes = _canonical_json_bytes(seal)
    (root / "inventory_seal.json").write_bytes(seal_bytes)
    return (
        candidate_group,
        calibration_group,
        source_release,
        sha256(seal_bytes).hexdigest(),
        sha256(manifest_bytes).hexdigest(),
    )


class P107IntegrationTests(unittest.TestCase):
    def test_default_dart_reader_direct_loads_current_data_protocol_without_g05_data(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        protocol_path = project_root / "src/g05/data/memlite_event_protocol.py"
        self.assertEqual(sha256(protocol_path.read_bytes()).hexdigest(), DATA_PROTOCOL_SHA256)
        with TemporaryDirectory() as folder:
            index_root = Path(folder)
            candidate, calibration, source_release, inventory_seal, manifest = _write_source_membership_fixture(index_root)
            child = f"""
import json
import sys
from pathlib import Path
from g05.recovery.dart_collection import CalibrationReceipt, CandidateSourceReceipt, DataSealedSourceGroupIndexReader
source = CandidateSourceReceipt(
    source_group_id={candidate!r}, original_split='train', source_release_sha256={source_release!r},
    parent_task_id='task-42', parent_task_index=42, parent_task_instance_id=9, parent_task_seed=9,
    trajectory_source_kind='fresh_dart_trajectory', collection_run_id='synthetic-integration-run')
calibration = CalibrationReceipt(
    source_group_ids=({calibration!r},), calibration_trajectory_sha256={'a' * 64!r},
    learner_checkpoint_sha256={'b' * 64!r}, teacher_checkpoint_sha256={'c' * 64!r})
reader = DataSealedSourceGroupIndexReader(Path({str(index_root)!r}), expected_inventory_seal_sha256={inventory_seal!r})
# A second source-groups scan would now fail; the loaded capability is reused.
Path({str(index_root / 'source_groups.jsonl')!r}).unlink()
first = reader.verify_dart_membership(source, calibration)
second = reader.verify_dart_membership(source, calibration)
assert 'g05.data' not in sys.modules
print(json.dumps({{'candidate': first.candidate.usage_role, 'calibration': second.calibration_groups[0].usage_role, 'manifest': first.source_group_index_manifest_sha256, 'protocol': first.source_membership_protocol_sha256}}))
"""
            completed = subprocess.run(
                [sys.executable, "-S", "-c", child],
                capture_output=True,
                text=True,
                check=True,
                env={"PATH": os.environ["PATH"], "PYTHONPATH": str(project_root / "src")},
            )
            self.assertEqual(json.loads(completed.stdout), {
                "candidate": "student_candidate",
                "calibration": "annotation_calibration",
                "manifest": manifest,
                "protocol": DATA_PROTOCOL_SHA256,
            })


if __name__ == "__main__":
    unittest.main()
