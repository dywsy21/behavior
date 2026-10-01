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
OWNER_PROTOCOL = REPO / "tests" / "fixtures" / "p107_memlite_event_protocol_fixture.py"
OFFICIAL_SKILLS = (
    (10, "grasp an object"), (20, "open a receptacle"), (30, "place an object"),
    (40, "press a control"), (50, "wash an object"),
)
SKILL_ID_BY_VERB = {"GRASP": 10, "OPEN": 20, "PLACE": 30, "PRESS": 40, "WASH": 50}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol():
    if not OWNER_PROTOCOL.is_file():
        raise RuntimeError(f"missing pinned in-repository protocol fixture: {OWNER_PROTOCOL}")
    return queue.load_canonical_protocol(OWNER_PROTOCOL, expected_sha256=queue.sha256_file(OWNER_PROTOCOL))


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def receipt(path):
    return {"sha256": queue.sha256_file(path), "bytes": path.stat().st_size,
            "rows": len(read_lines(path))}


def source_group_id(owner_protocol, task, instance):
    return owner_protocol.source_group_id({
        "source_release_manifest_sha256": SOURCE_SHA,
        "task_index": task,
        "task_instance_id": instance,
    })


def event(owner_protocol, *, group, role, task, instance, raw_episode, episode, start, end, verb, target,
          length=900, skill_id=None):
    skill_id = SKILL_ID_BY_VERB[verb] if skill_id is None else skill_id
    row = {
        "schema_version": owner_protocol.SCHEMA_VERSION,
        "record_kind": "event_candidate",
        "event_id": "",
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
        "event_kind": "ANNOTATED_SKILL_SEGMENT",
        "event_interval": {"start_frame": start, "end_frame": end},
        "observation": {"frame": start, "timestamp_s": start / 30.0},
        "action": {"start_frame": start, "actual_executed_length": None, "raw_action_dim": 23,
                   "model_action_dim": 27, "model_padding_indices": [7, 8, 17, 18]},
        "bundle_id": digest(f"bundle:{verb}:{target}"),
        "skill_bundle": [{"skill_id": skill_id, "verb": verb, "target": target, "source": "", "destination": "",
                          "target_part": "", "arm": "UNSPECIFIED"}],
        "parallel_bundle": False,
        "binding": [],
        "evidence": {"kind": "MISSING", "evidence_end_frame": None, "available_frame": None},
        "video_locators": [{"view": "head", "camera_key": "observation.rgb.zed_link_camera_0",
                            "relative_path": "videos/head.mp4", "episode_start_timestamp_s": 0.0,
                            "requested_timestamp_s": start / 30.0, "expected_fps": 30,
                            "locator_status": "METADATA_ONLY_UNRESOLVED"}],
        "task_name": f"fixture-task-{task}",
        "usage_role": role,
    }
    row["event_id"] = owner_protocol.event_id(row)
    owner_protocol.validate_event(row)
    return row


def coverage_expectations(root, *, tasks=(0, 1, 2), skills=OFFICIAL_SKILLS):
    value = {
        "schema_version": "p107-official-coverage-expectations-v3",
        "official_task_metadata_sha256": digest("fixture-official-tasks"),
        "official_skill_vocabulary_sha256": digest("fixture-official-skills"),
        "expected_task_ids": list(tasks),
        "expected_skill_vocabulary": [
            {"skill_id": skill_id, "skill_description": description} for skill_id, description in skills
        ],
        "required_task_skill_pairs": None,
    }
    path = root / "coverage-expectations.json"
    path.write_text(queue.canonical_json(value) + "\n")
    return path, value


def write_index(root, *, extra_student_tasks=()):
    root.mkdir()
    owner_protocol = protocol()
    student0 = source_group_id(owner_protocol, 0, 1)
    student1 = source_group_id(owner_protocol, 1, 2)
    calibration0 = source_group_id(owner_protocol, 0, 3)
    calibration1 = source_group_id(owner_protocol, 1, 4)
    evaluation2 = source_group_id(owner_protocol, 2, 5)
    events = [
        # These source rows share an exact boundary and have a repeated GRASP
        # in distinct episodes.  The selector must cap the adjacent window.
        event(owner_protocol, group=student0, role="student_candidate", task=0, instance=1,
              raw_episode=10, episode=10, start=0, end=100, verb="GRASP", target="cup"),
        event(owner_protocol, group=student0, role="student_candidate", task=0, instance=1,
              raw_episode=10, episode=10, start=100, end=200, verb="PLACE", target="cup"),
        event(owner_protocol, group=student0, role="student_candidate", task=0, instance=1,
              raw_episode=11, episode=11, start=300, end=400, verb="GRASP", target="cup"),
        event(owner_protocol, group=student1, role="student_candidate", task=1, instance=2,
              raw_episode=20, episode=20, start=250, end=300, verb="OPEN", target="drawer"),
        event(owner_protocol, group=student1, role="student_candidate", task=1, instance=2,
              raw_episode=21, episode=21, start=0, end=500, verb="WASH", target="plate"),
        event(owner_protocol, group=calibration0, role="annotation_calibration", task=0, instance=3,
              raw_episode=30, episode=30, start=0, end=100, verb="GRASP", target="cup"),
        event(owner_protocol, group=calibration1, role="annotation_calibration", task=1, instance=4,
              raw_episode=40, episode=40, start=0, end=100, verb="OPEN", target="drawer"),
        event(owner_protocol, group=evaluation2, role="evaluation_only", task=2, instance=5,
              raw_episode=50, episode=50, start=0, end=100, verb="PRESS", target="button"),
    ]
    for task in extra_student_tasks:
        group = source_group_id(owner_protocol, task, task + 100)
        events.append(event(owner_protocol, group=group, role="student_candidate", task=task, instance=task + 100,
                            raw_episode=1000 + task, episode=1000 + task, start=0, end=90,
                            verb="OPEN", target=f"object-{task}"))
    groups = []
    for group_id, task, instance, split, role in (
        (student0, 0, 1, "train", "student_candidate"),
        (student1, 1, 2, "train", "student_candidate"),
        (calibration0, 0, 3, "train", "annotation_calibration"),
        (calibration1, 1, 4, "train", "annotation_calibration"),
        (evaluation2, 2, 5, "eval", "evaluation_only"),
        *[(source_group_id(owner_protocol, task, task + 100), task, task + 100, "train", "student_candidate")
          for task in extra_student_tasks],
    ):
        episode_ids = sorted({row["source"]["episode_index"] for row in events
                              if row["source"]["source_group_id"] == group_id})
        groups.append({
            "schema_version": owner_protocol.SCHEMA_VERSION,
            "source_group_id": group_id,
            "source_release_manifest_sha256": SOURCE_SHA,
            "task_index": task,
            "task_instance_id": instance,
            "original_split": split,
            "usage_role": role,
            "source_episode_ids": episode_ids,
            "source_episode_count": len(episode_ids),
        })
    groups_path, events_path = root / "source_groups.jsonl", root / "event_candidates.jsonl"
    groups_path.write_text("".join(queue.canonical_json(row) + "\n" for row in groups))
    # Reverse source order to ensure queue order is a policy decision, not the
    # first sorted/physical metadata records.
    events_path.write_text("".join(queue.canonical_json(row) + "\n" for row in reversed(events)))
    coverage_path, coverage = coverage_expectations(root, tasks=tuple(range(max((2, *extra_student_tasks)) + 1)))
    manifest = {
        "schema_version": queue.INDEX_SCHEMA,
        "status": queue.INDEX_STATUS,
        "training_eligible": False,
        "source_release_manifest_sha256": SOURCE_SHA,
        "protocol_sha256": queue.sha256_file(OWNER_PROTOCOL),
        "coverage_expectations_sha256": queue.canonical_sha256(coverage),
        "files": {"source_groups.jsonl": receipt(groups_path), "event_candidates.jsonl": receipt(events_path)},
    }
    (root / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
    (root / "inventory_seal.json").write_text(queue.canonical_json({
        "schema_version": "memlite-event-inventory-seal-v1",
        "index_manifest_sha256": queue.sha256_file(root / "manifest.json"),
        "source_release_manifest_sha256": SOURCE_SHA,
        "coverage_expectations_sha256": queue.canonical_sha256(coverage),
        "expected_payload_files": ["event_candidates.jsonl", "source_groups.jsonl"],
        "payload_files": {
            "event_candidates.jsonl": manifest["files"]["event_candidates.jsonl"],
            "source_groups.jsonl": manifest["files"]["source_groups.jsonl"],
        },
    }) + "\n")
    return coverage_path


def reseal_index(root, manifest):
    seal = json.loads((root / "inventory_seal.json").read_text())
    seal["index_manifest_sha256"] = queue.sha256_file(root / "manifest.json")
    seal["payload_files"] = {
        "event_candidates.jsonl": manifest["files"]["event_candidates.jsonl"],
        "source_groups.jsonl": manifest["files"]["source_groups.jsonl"],
    }
    (root / "inventory_seal.json").write_text(queue.canonical_json(seal) + "\n")


def reseal_queue(root):
    manifest = json.loads((root / "manifest.json").read_text())
    seal = json.loads((root / "queue_seal.json").read_text())
    seal["queue_manifest_sha256"] = queue.sha256_file(root / "manifest.json")
    seal["payload_files"] = manifest["files"]
    (root / "queue_seal.json").write_text(queue.canonical_json(seal) + "\n")
    return queue.sha256_file(root / "queue_seal.json")


def args(index, output, coverage_path, *, seed="p107-metadata-candidate-selection-20261001"):
    return queue.parser().parse_args([
        "--index", str(index), "--output", str(output),
        "--expected-source-release-manifest-sha256", SOURCE_SHA,
        "--expected-inventory-seal-sha256", queue.sha256_file(index / "inventory_seal.json"),
        "--coverage-expectations", str(coverage_path),
        "--protocol-path", str(OWNER_PROTOCOL), "--expected-protocol-sha256", queue.sha256_file(OWNER_PROTOCOL),
        "--seed", seed, "--candidate-budget", "5", "--calibration-budget", "4",
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
            coverage = write_index(index)
            first = root / "queue-a"
            result = queue.build_queue(index, first, args=args(index, first, coverage))
            self.assertFalse(result["training_eligible"])
            self.assertTrue(queue.is_sha256(result["queue_seal_sha256"]))
            student = read_lines(first / "student_candidate_queue.jsonl")
            calibration = read_lines(first / "annotation_calibration_queue.jsonl")
            requests = read_lines(first / "camera_native_render_requests.jsonl")
            self.assertEqual({job["usage_role"] for job in student}, {"student_candidate"})
            self.assertEqual({job["usage_role"] for job in calibration}, {"annotation_calibration"})
            self.assertTrue(all(job["status"] == queue.STATUS and not job["training_eligible"] for job in student + calibration))
            self.assertTrue(all(job["source_kind"] == "LOGGED_EXPERT_DEMONSTRATION_METADATA_CANDIDATE"
                                for job in student + calibration))
            self.assertTrue(any("REPEATED_SKILL_TARGET_CANDIDATE" in job["candidate_signals"] for job in student))
            self.assertTrue(any(job["selection_stratum"] == "NORMAL_CONTROL_CANDIDATE" for job in student))
            self.assertTrue(all("GRIPPER" not in str(job["candidate_signals"]) for job in student + calibration))
            for job in student + calibration:
                windows = job["temporal_windows"]
                anchor = windows["anchor_frame"]
                self.assertLessEqual(max(windows["actor_available_window"]["sampled_frames"]), anchor)
                self.assertTrue(all(frame > anchor for frame in windows["offline_review_after_window"]["sampled_frames"]))
                self.assertEqual(job["action_heuristic_status"], "DISABLED_NO_TRUSTED_EXECUTED_ACTION_PROOF_IN_METADATA_QUEUE")
            self.assertTrue(all(request["camera_delivery"]["decoded_by_selector"] is False and
                                request["camera_delivery"]["include_footer"] is False for request in requests))
            counts = json.loads((first / "counts.json").read_text())
            self.assertEqual(counts["events_excluded_by_role"]["evaluation_only"], 1)
            self.assertEqual(counts["student_candidate_queue"]["coverage"]["expected_task_count"], 3)
            self.assertEqual(counts["student_candidate_queue"]["coverage"]["expected_skill_ids"], [10, 20, 30, 40, 50])
            self.assertEqual(counts["student_candidate_queue"]["coverage"]["missing_expected_task_ids_from_pool"], [2])
            self.assertIn("GRASP", counts["student_candidate_queue"]["coverage"]["raw_source_verbs_diagnostic"])
            self.assertEqual(student[0]["skill_ids"], sorted(student[0]["skill_ids"]))
            self.assertEqual(counts["student_candidate_queue"]["coverage"]["required_task_skill_pair_queue_status"],
                             "NOT_DECLARED_DESCRIPTIVE_ONLY")
            resume_args = args(index, first, coverage)
            resume_args.expected_queue_seal_sha256 = result["queue_seal_sha256"]
            self.assertEqual(queue.resume_queue(index, first, args=resume_args)["status"], "RESUME_VALIDATED")
            second = root / "queue-b"
            second_result = queue.build_queue(index, second, args=args(index, second, coverage))
            self.assertEqual(result["queue_seal_sha256"], second_result["queue_seal_sha256"])
            for name in ("student_candidate_queue.jsonl", "annotation_calibration_queue.jsonl",
                         "camera_native_render_requests.jsonl", "counts.json", "manifest.json", "queue_seal.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_resume_rejects_resealed_manifest_job_and_counts_escalation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "sealed-index"
            coverage = write_index(index)
            output = root / "queue"
            result = queue.build_queue(index, output, args=args(index, output, coverage))
            original_seal = result["queue_seal_sha256"]
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["training_eligible"] = True
            manifest["all_jobs_status"] = "SUCCEEDED"
            (output / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            attacker_seal = reseal_queue(output)
            external_args = args(index, output, coverage)
            external_args.expected_queue_seal_sha256 = original_seal
            with self.assertRaisesRegex(ValueError, "queue seal"):
                queue.resume_queue(index, output, args=external_args)
            external_args.expected_queue_seal_sha256 = attacker_seal
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                queue.resume_queue(index, output, args=external_args)

            caps_output = root / "caps-queue"
            queue.build_queue(index, caps_output, args=args(index, caps_output, coverage))
            manifest = json.loads((caps_output / "manifest.json").read_text())
            manifest["policy"]["max_per_source_group"] = 999
            policy_core = dict(manifest["policy"])
            policy_core.pop("policy_sha256")
            manifest["policy"]["policy_sha256"] = queue.canonical_sha256(policy_core)
            (caps_output / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            attacker_seal = reseal_queue(caps_output)
            resume_args = args(index, caps_output, coverage)
            resume_args.expected_queue_seal_sha256 = attacker_seal
            with self.assertRaisesRegex(ValueError, "queue policy differs"):
                queue.resume_queue(index, caps_output, args=resume_args)

            jobs_output = root / "jobs-queue"
            queue.build_queue(index, jobs_output, args=args(index, jobs_output, coverage))
            jobs_path = jobs_output / "student_candidate_queue.jsonl"
            jobs = read_lines(jobs_path)
            jobs[0]["low_action_supervision_mask"] = True
            jobs[0]["forced_answer"] = "RECOVERY"
            jobs[0]["candidate_signals"] = ["FAILED_AND_RECOVERY"]
            jobs[0]["selection_stratum"] = "PHYSICAL_FAILURE"
            jobs_path.write_text("".join(queue.canonical_json(row) + "\n" for row in jobs))
            manifest = json.loads((jobs_output / "manifest.json").read_text())
            manifest["files"]["student_candidate_queue.jsonl"] = receipt(jobs_path)
            (jobs_output / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            attacker_seal = reseal_queue(jobs_output)
            resume_args = args(index, jobs_output, coverage)
            resume_args.expected_queue_seal_sha256 = attacker_seal
            with self.assertRaisesRegex(ValueError, "queue job schema"):
                queue.resume_queue(index, jobs_output, args=resume_args)

            semantic_jobs_output = root / "semantic-jobs-queue"
            queue.build_queue(index, semantic_jobs_output, args=args(index, semantic_jobs_output, coverage))
            jobs_path = semantic_jobs_output / "student_candidate_queue.jsonl"
            jobs = read_lines(jobs_path)
            jobs[0]["candidate_signals"] = ["FAILED_AND_RECOVERY"]
            jobs[0]["selection_stratum"] = "PHYSICAL_FAILURE"
            jobs_path.write_text("".join(queue.canonical_json(row) + "\n" for row in jobs))
            manifest = json.loads((semantic_jobs_output / "manifest.json").read_text())
            manifest["files"]["student_candidate_queue.jsonl"] = receipt(jobs_path)
            (semantic_jobs_output / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            attacker_seal = reseal_queue(semantic_jobs_output)
            resume_args = args(index, semantic_jobs_output, coverage)
            resume_args.expected_queue_seal_sha256 = attacker_seal
            with self.assertRaisesRegex(ValueError, "queue job differs"):
                queue.resume_queue(index, semantic_jobs_output, args=resume_args)

            counts_output = root / "counts-queue"
            queue.build_queue(index, counts_output, args=args(index, counts_output, coverage))
            counts_path = counts_output / "counts.json"
            counts = json.loads(counts_path.read_text())
            counts["training_eligible"] = True
            counts["student_candidate_queue"]["budget"] = 999999
            counts_path.write_text(queue.canonical_json(counts) + "\n")
            manifest = json.loads((counts_output / "manifest.json").read_text())
            manifest["files"]["counts.json"] = receipt(counts_path)
            (counts_output / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            attacker_seal = reseal_queue(counts_output)
            resume_args = args(index, counts_output, coverage)
            resume_args.expected_queue_seal_sha256 = attacker_seal
            with self.assertRaisesRegex(ValueError, "counts differ"):
                queue.resume_queue(index, counts_output, args=resume_args)

            timing_output = root / "timing-queue"
            queue.build_queue(index, timing_output, args=args(index, timing_output, coverage))
            requests_path = timing_output / "camera_native_render_requests.jsonl"
            requests = read_lines(requests_path)
            requests[0]["actor_available_frame_indices"].append(999)
            requests_path.write_text("".join(queue.canonical_json(row) + "\n" for row in requests))
            manifest = json.loads((timing_output / "manifest.json").read_text())
            manifest["files"]["camera_native_render_requests.jsonl"] = receipt(requests_path)
            (timing_output / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            attacker_seal = reseal_queue(timing_output)
            resume_args = args(index, timing_output, coverage)
            resume_args.expected_queue_seal_sha256 = attacker_seal
            with self.assertRaisesRegex(ValueError, "render request differs"):
                queue.resume_queue(index, timing_output, args=resume_args)

    def test_canonical_source_group_blocks_cap_bypass_and_caps_real_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "sealed-index"
            coverage = write_index(index)
            output = root / "cap"
            parsed = args(index, output, coverage)
            parsed.max_per_source_group = 1
            queue.build_queue(index, output, args=parsed)
            self.assertLessEqual(len(read_lines(output / "student_candidate_queue.jsonl")), 2)

            fake = root / "fake-index"
            coverage = write_index(fake)
            events_path = fake / "event_candidates.jsonl"
            records = read_lines(events_path)
            target = next(row for row in records if row["usage_role"] == "student_candidate")
            target["source"]["source_group_id"] = digest("two-fake-groups-same-task-instance")
            target["event_id"] = protocol().event_id(target)
            events_path.write_text("".join(queue.canonical_json(row) + "\n" for row in records))
            manifest = json.loads((fake / "manifest.json").read_text())
            manifest["files"]["event_candidates.jsonl"] = receipt(events_path)
            (fake / "manifest.json").write_text(queue.canonical_json(manifest) + "\n")
            reseal_index(fake, manifest)
            with self.assertRaisesRegex(ValueError, "source identity"):
                queue.build_queue(fake, root / "fake-output", args=args(fake, root / "fake-output", coverage))

    def test_budget_40_joint_reserve_covers_rare_official_skills(self):
        def candidate(task, sequence, skills):
            return queue.Candidate(
                event={}, event_id=f"{sequence:064x}", source_group_id=f"group-{sequence}", task_id=task,
                task_instance_id=sequence + 1, episode_key=(f"group-{sequence}", sequence, sequence),
                episode_index=sequence, episode_length=1000, anchor_frame=0, interval_start=0, interval_end=5,
                skill_ids=tuple(skills), raw_source_verbs=("RAW_SOURCE_VERB",), skill_keys=(str(sequence),),
                repeat_attempt_count=0, boundary_before=False, boundary_after=False, long_interval=False,
                candidate_signals=("NORMAL_CONTROL_CANDIDATE",), selection_stratum="NORMAL_CONTROL_CANDIDATE")

        pool = [candidate(task, task, (1,)) for task in range(100)]
        pool.extend(candidate(99, 100 + skill, (skill,)) for skill in range(100, 136))
        selected, diagnostics = queue.select_diverse(
            pool, budget=40, seed="coverage-adversary", max_per_episode=2, min_separation_frames=0,
            max_per_source_group=2, normal_control_fraction=0.0, priority_skill_ids=[1, *range(100, 136)])
        selected_skills = {skill for item, _ in selected for skill in item.skill_ids}
        self.assertEqual(len(selected), 40)
        self.assertTrue(set(range(100, 136)) <= selected_skills)
        self.assertEqual(diagnostics["priority_skill_ids_present_but_not_queued"], [])
        self.assertGreaterEqual(sum(phase == "JOINT_RARE_SKILL_RESERVE" for _, phase in selected), 36)

    def test_v3_fixture_and_runtime_protocol_default_are_hermetic(self):
        self.assertTrue(OWNER_PROTOCOL.is_file())
        self.assertEqual(OWNER_PROTOCOL, REPO / "tests" / "fixtures" / "p107_memlite_event_protocol_fixture.py")
        self.assertEqual(queue.DEFAULT_PROTOCOL_PATH, REPO / "src" / "g05" / "data" / "memlite_event_protocol.py")


if __name__ == "__main__":
    unittest.main()
