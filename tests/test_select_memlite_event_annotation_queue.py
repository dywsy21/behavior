import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "data"))
import select_memlite_event_annotation_queue as queue  # noqa: E402


SOURCE_SHA = "e" * 64


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def receipt(path):
    return {"sha256": queue.sha256_file(path), "bytes": path.stat().st_size,
            "rows": len(read_lines(path))}


def event(*, group, role, task, instance, raw_episode, episode, start, end, verb, target, length=900):
    event_id = digest(f"{group}:{raw_episode}:{start}:{end}:{verb}:{target}")
    return {
        "event_id": event_id,
        "usage_role": role,
        "source": {
            "source_release_manifest_sha256": SOURCE_SHA,
            "source_annotation_sha256": digest(f"annotation:{raw_episode}"),
            "source_group_id": group,
            "task_index": task,
            "task_instance_id": instance,
            "raw_episode_id": raw_episode,
            "episode_index": episode,
            "episode_length": length,
            "original_split": "eval" if role == "evaluation_only" else "train",
        },
        "observation": {"frame": start, "timestamp_s": start / 30.0},
        "event_interval": {"start_frame": start, "end_frame": end},
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "action": {"actual_executed_length": None},
        "skill_bundle": [{"verb": verb, "target": target, "source": "", "destination": "",
                          "target_part": "", "arm": "UNSPECIFIED"}],
        "video_locators": [{"view": "head", "relative_path": "videos/head.mp4"}],
    }


def write_index(root):
    root.mkdir()
    student0 = digest("student-task-0")
    student1 = digest("student-task-1")
    calibration0 = digest("calibration-task-0")
    calibration1 = digest("calibration-task-1")
    evaluation2 = digest("eval-task-2")
    groups = [
        {"source_group_id": student0, "task_index": 0, "task_instance_id": 1,
         "original_split": "train", "usage_role": "student_candidate"},
        {"source_group_id": student1, "task_index": 1, "task_instance_id": 2,
         "original_split": "train", "usage_role": "student_candidate"},
        {"source_group_id": calibration0, "task_index": 0, "task_instance_id": 3,
         "original_split": "train", "usage_role": "annotation_calibration"},
        {"source_group_id": calibration1, "task_index": 1, "task_instance_id": 4,
         "original_split": "train", "usage_role": "annotation_calibration"},
        {"source_group_id": evaluation2, "task_index": 2, "task_instance_id": 5,
         "original_split": "eval", "usage_role": "evaluation_only"},
    ]
    events = [
        # The first two events share a boundary, and the GRASP repeats in a
        # distinct source episode.  The selector must cap the adjacent window.
        event(group=student0, role="student_candidate", task=0, instance=1,
              raw_episode=10, episode=10, start=0, end=100, verb="GRASP", target="cup"),
        event(group=student0, role="student_candidate", task=0, instance=1,
              raw_episode=10, episode=10, start=100, end=200, verb="PLACE", target="cup"),
        event(group=student0, role="student_candidate", task=0, instance=1,
              raw_episode=11, episode=11, start=300, end=400, verb="GRASP", target="cup"),
        # This is deliberately a signal-free normal-control candidate.
        event(group=student1, role="student_candidate", task=1, instance=2,
              raw_episode=20, episode=20, start=250, end=300, verb="OPEN", target="drawer"),
        event(group=student1, role="student_candidate", task=1, instance=2,
              raw_episode=21, episode=21, start=0, end=500, verb="WASH", target="plate"),
        event(group=calibration0, role="annotation_calibration", task=0, instance=3,
              raw_episode=30, episode=30, start=0, end=100, verb="GRASP", target="cup"),
        event(group=calibration1, role="annotation_calibration", task=1, instance=4,
              raw_episode=40, episode=40, start=0, end=100, verb="OPEN", target="drawer"),
        event(group=evaluation2, role="evaluation_only", task=2, instance=5,
              raw_episode=50, episode=50, start=0, end=100, verb="PRESS", target="button"),
    ]
    groups_path, events_path = root / "source_groups.jsonl", root / "event_candidates.jsonl"
    groups_path.write_text("".join(queue.canonical_json(row) + "\n" for row in groups))
    # Reverse input ordering to prove the sampler does not accept the first N
    # rows as its diversity policy.
    events_path.write_text("".join(queue.canonical_json(row) + "\n" for row in reversed(events)))
    manifest = {
        "schema_version": queue.INDEX_SCHEMA,
        "status": queue.INDEX_STATUS,
        "training_eligible": False,
        "source_release_manifest_sha256": SOURCE_SHA,
        "files": {"source_groups.jsonl": receipt(groups_path), "event_candidates.jsonl": receipt(events_path)},
    }
    (root / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
    (root / "inventory_seal.json").write_text(queue.canonical_json({
        "schema_version": "memlite-event-inventory-seal-v1",
        "index_manifest_sha256": queue.sha256_file(root / "manifest.json"),
        "source_release_manifest_sha256": SOURCE_SHA,
        "coverage_expectations_sha256": digest("fixture-coverage-expectations"),
        "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
        "payload_files": {
            "event_candidates.jsonl": manifest["files"]["event_candidates.jsonl"],
            "source_groups.jsonl": manifest["files"]["source_groups.jsonl"],
        },
    }) + "\n")


def reseal(root, manifest):
    seal = json.loads((root / "inventory_seal.json").read_text())
    seal["index_manifest_sha256"] = queue.sha256_file(root / "manifest.json")
    seal["payload_files"] = {
        "event_candidates.jsonl": manifest["files"]["event_candidates.jsonl"],
        "source_groups.jsonl": manifest["files"]["source_groups.jsonl"],
    }
    (root / "inventory_seal.json").write_text(queue.canonical_json(seal) + "\n")


def args(index, output):
    return queue.parser().parse_args([
        "--index", str(index), "--output", str(output),
        "--expected-source-release-manifest-sha256", SOURCE_SHA,
        "--expected-inventory-seal-sha256", queue.sha256_file(index / "inventory_seal.json"),
        "--candidate-budget", "5", "--calibration-budget", "4",
        "--max-per-episode", "2", "--max-per-source-group", "4",
        "--min-separation-frames", "120", "--normal-control-fraction", "0.4",
        "--review-before-frames", "30", "--review-after-frames", "30",
        "--review-sample-stride-frames", "15",
    ])


class SelectMemliteEventAnnotationQueueTests(unittest.TestCase):
    def test_sealed_deterministic_diverse_candidate_and_calibration_queues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "sealed-index"
            write_index(index)
            first = root / "queue-a"
            result = queue.build_queue(index, first, args=args(index, first))
            self.assertFalse(result["training_eligible"])
            self.assertEqual(result["all_jobs_status"], queue.STATUS)
            self.assertEqual(result["renderer_handshake"]["status"], "PENDING_INTERFACE_OWNER_CONFIRMATION")
            student = read_lines(first / "student_candidate_queue.jsonl")
            calibration = read_lines(first / "annotation_calibration_queue.jsonl")
            requests = read_lines(first / "camera_native_render_requests.jsonl")
            self.assertEqual({job["usage_role"] for job in student}, {"student_candidate"})
            self.assertEqual({job["usage_role"] for job in calibration}, {"annotation_calibration"})
            self.assertNotIn("evaluation_only", str(student + calibration))
            self.assertTrue(all(job["status"] == queue.STATUS for job in student + calibration))
            self.assertTrue(all(job["current_actor_evidence"]["kind"] == "MISSING" for job in student + calibration))
            self.assertTrue(any("REPEATED_SKILL_TARGET_CANDIDATE" in job["candidate_signals"] for job in student))
            self.assertTrue(any(job["selection_stratum"] == "NORMAL_CONTROL_CANDIDATE" for job in student))
            self.assertEqual({job["source_identity"]["task_index"] for job in student}, {0, 1})
            self.assertEqual({job["source_identity"]["task_index"] for job in calibration}, {0, 1})
            self.assertTrue(all(request["camera_delivery"]["decoded_by_selector"] is False for request in requests))
            self.assertTrue(all(request["camera_delivery"]["include_footer"] is False for request in requests))
            for job in student + calibration:
                windows = job["temporal_windows"]
                anchor = windows["anchor_frame"]
                self.assertLessEqual(max(windows["actor_available_window"]["sampled_frames"]), anchor)
                self.assertTrue(all(frame > anchor for frame in windows["offline_review_after_window"]["sampled_frames"]))
                self.assertTrue(job["review_constraints"]["actor_may_only_use_actor_available_window"])
            counts = json.loads((first / "counts.json").read_text())
            self.assertEqual(counts["events_excluded_by_role"]["evaluation_only"], 1)
            self.assertGreaterEqual(counts["student_candidate_queue"]["near_duplicate"]["adjacent_window_cap_rejected"], 1)
            self.assertIn(99, counts["student_candidate_queue"]["coverage"]["missing_task_ids_if_expected_100"])
            self.assertEqual(queue.resume_queue(
                index, first, expected_source_manifest_sha256=SOURCE_SHA,
                expected_inventory_seal_sha256=queue.sha256_file(index / "inventory_seal.json"),
            )["status"], "RESUME_VALIDATED")

            second = root / "queue-b"
            queue.build_queue(index, second, args=args(index, second))
            for name in ("student_candidate_queue.jsonl", "annotation_calibration_queue.jsonl",
                         "camera_native_render_requests.jsonl", "counts.json", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_index_evidence_or_role_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "sealed-index"
            write_index(index)
            events_path = index / "event_candidates.jsonl"
            records = read_lines(events_path)
            next(row for row in records if row["usage_role"] == "student_candidate")["evidence"] = {
                "kind": "SOME_FUTURE_TRUTH", "evidence_end_frame": 1, "available_frame": 1
            }
            events_path.write_text("".join(queue.canonical_json(row) + "\n" for row in records))
            manifest = json.loads((index / "manifest.json").read_text())
            manifest["files"]["event_candidates.jsonl"] = receipt(events_path)
            (index / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            reseal(index, manifest)
            output = root / "bad-evidence"
            with self.assertRaisesRegex(ValueError, "non-missing evidence"):
                queue.build_queue(index, output, args=args(index, output))
            self.assertFalse(output.exists())

            write_index(root / "second-index")
            bad_index = root / "second-index"
            groups_path = bad_index / "source_groups.jsonl"
            groups = read_lines(groups_path)
            groups[-1]["usage_role"] = "student_candidate"
            groups_path.write_text("".join(queue.canonical_json(row) + "\n" for row in groups))
            manifest = json.loads((bad_index / "manifest.json").read_text())
            manifest["files"]["source_groups.jsonl"] = receipt(groups_path)
            (bad_index / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            reseal(bad_index, manifest)
            output = root / "bad-role"
            with self.assertRaisesRegex(ValueError, "eval role"):
                queue.build_queue(bad_index, output, args=args(bad_index, output))
            self.assertFalse(output.exists())

    def test_external_inventory_seal_hash_is_required_and_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "sealed-index"
            write_index(index)
            recorded_hash = queue.sha256_file(index / "inventory_seal.json")
            seal = json.loads((index / "inventory_seal.json").read_text())
            seal["coverage_expectations_sha256"] = digest("different-coverage-contract")
            (index / "inventory_seal.json").write_text(queue.canonical_json(seal) + "\n")
            output = root / "bad-seal"
            parsed = args(index, output)
            parsed.expected_inventory_seal_sha256 = recorded_hash
            with self.assertRaisesRegex(ValueError, "inventory seal hash"):
                queue.build_queue(index, output, args=parsed)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
