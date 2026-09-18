import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/semantic_robot"))
from contact_replay_contract import load_contract


class ContactReplayTests(unittest.TestCase):
    def test_native_teardown_failure_preserves_original_without_repeated_hold(self):
        from replay_contact_audit import preserved_session
        original = RuntimeError("original simulation failure")
        first = [None]
        calls = []
        def record(exc):
            first[0] = first[0] or exc
        class Session:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                raise OSError("native teardown failed")
        try:
            with preserved_session(Session, record, lambda: first[0]):
                record(original)
                calls.append("hold and cleanup attempted")
                raise original
        except BaseException as caught:
            self.assertIs(caught, original)
        self.assertEqual(len(calls), 1)

    def test_column_only_support_and_visual_only_link_coverage(self):
        from replay_contact_audit import audit_pairs
        api = Mock()
        def query(scene, rows, cols, current):
            if cols is None:
                return {("robot", "floor")}
            self.assertEqual(cols, {"robot", "table"})
            return {("radio", "table"), ("floor", "robot")}
        api.get_contact_pairs.side_effect = query
        pairs = audit_pairs(api, 0, {"entity"}, {"robot", "radio"},
                            {"robot", "table", "visual_only"}, {"robot", "table", "floor"}, True)
        self.assertEqual(pairs, [("floor", "robot"), ("radio", "table")])

    def test_all_reporting_writes_fail_without_replacing_primary_exception(self):
        from replay_contact_audit import report_failure
        primary = RuntimeError("original")
        broken = Mock()
        broken.write.side_effect = OSError("broken stderr")
        broken.flush.side_effect = OSError("broken stderr flush")
        with patch("replay_contact_audit.write", side_effect=OSError("disk")), patch("sys.stderr", broken):
            try:
                raise primary
            except RuntimeError as exc:
                for _ in range(3):
                    report_failure(Path("unused"), exc, 4, None)
                self.assertIs(exc, primary)

    def fixture(self, root, change=None):
        manifest = {"code_commit": "original", "task": 0, "split": "train", "seed": 0, "instance": 138,
                    "args": {"prefix": 0, "odometry_substep_controls": 6}}
        actions = [{"control": i + 1, "decision": i // 7, "action23": [0.] * 23, "terminal": False} for i in range(14)]
        commands = [{"decision": i, "action": {"part": "all", "move": "hold"}, "feedback": {"control_ticks": 6},
                     "control_start": i * 7, "control_end": (i + 1) * 7} for i in range(2)]
        # Deliberately ticks=6 but seven actual commands, as abort-held H13 d67.
        rows = actions[:7] + commands[:1] + actions[7:] + commands[1:] + [{"control": 15, "safety_stop": True}]
        result = {"prefix_controls": 0, "diagnostic_replay_controls": 0, "controls": 15, "decisions": commands}
        labels = ["after_prefix", "decision_0", "motion_0_6", "motion_0_7", "motion_1_13", "motion_1_14"]
        sensors = [{"label": x, "cameras": {v: {"render_barrier_updates": 4} for v in ("head", "left_wrist", "right_wrist")}} for x in labels]
        state = {"q": [0.] * 18, "gripper": [.05, .05]}
        if change:
            change(manifest, rows, result, state, sensors)
        files = {"manifest.json": json.dumps(manifest), "steps.jsonl": "\n".join(map(json.dumps, rows)),
                 "result.json": json.dumps(result), "sensor_checks.json": json.dumps(sensors),
                 **{f"decision_{i:03d}/proprio.json": json.dumps(state) for i in range(2)}}
        hashes = {}
        for name, value in files.items():
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(value)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = {"schema": "contact_replay_audit_v1", "source_run": str(root), "source_code_commit": "original",
                "autonomous_policy": False, "decisions": 2, "replay_controls": 14, "sha256": hashes}
        path = root / "spec.json"
        path.write_text(json.dumps(spec))
        return path

    def test_actual_control_clock_not_servo_ticks_and_cached_barriers(self):
        with tempfile.TemporaryDirectory() as d:
            replay = load_contract(self.fixture(Path(d)))
            self.assertEqual(replay["actions"].shape, (14, 23))
            self.assertEqual(replay["end_controls"], {7, 14})
            self.assertEqual(replay["sample_controls"], {6, 7, 13, 14})
            self.assertEqual(set(replay["boundaries"]), {0, 7})

    def test_hash_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.fixture(Path(d))
            (Path(d) / "steps.jsonl").write_text("{}")
            with self.assertRaises(ValueError):
                load_contract(p)

    def test_changed_task_controls_state_and_barriers_fail(self):
        changes = [lambda m,r,x,s,c: m.update(task=3),
                   lambda m,r,x,s,c: m["args"].update(prefix=448),
                   lambda m,r,x,s,c: r[0].update(control=2),
                   lambda m,r,x,s,c: r[0].update(terminal=True),
                   lambda m,r,x,s,c: r[0]["action23"].__setitem__(0, float("nan")),
                   lambda m,r,x,s,c: r[0]["action23"].__setitem__(14, 2.),
                   lambda m,r,x,s,c: r[-1].update(control=16),
                   lambda m,r,x,s,c: s["q"].pop(),
                   lambda m,r,x,s,c: c[2].update(label="motion_0_5"),
                   lambda m,r,x,s,c: c[2]["cameras"]["head"].update(render_barrier_updates=3)]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    load_contract(self.fixture(Path(d), change))


if __name__ == "__main__":
    unittest.main()
