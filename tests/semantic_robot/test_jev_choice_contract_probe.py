import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/semantic_robot"))
import probe_jev_choice_contract as probe
from semantic_robot.v2.jev_client import JevChoiceMismatch, MODEL


class ChoiceContractProbeTests(unittest.TestCase):
    def test_registered_task1_search_is_not_rewritten_to_recover(self):
        state = {"harness": {"stage": "SEARCH"}, "facts": {"target_visible": False},
                 "commands": {"command_000": {"command": {"move": "hold"}, "role": "hold", "current_preflight": {}}}}
        receipt = {"request_without_pixel_duplicates": {"model": MODEL, "images": [],
            "questions": {"intent": {}}, "state": state},
            "result": {"model": MODEL, "answers": {"intent": {"choice": "search"}}}}
        before = copy.deepcopy(receipt)
        with self.assertRaises(ValueError):
            probe.rebuild_command(receipt)
        request = probe.rebuild_command(receipt, "SEARCH")
        self.assertEqual(receipt, before)
        self.assertEqual(request["state"], {**state, "chosen_tactic_not_new_evidence": "search"})
        raw = json.dumps(receipt).encode()
        size = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode())
        case = {"source_sha256": hashlib.sha256(raw).hexdigest(), "request_bytes": size, "stage": "SEARCH"}
        with patch.dict(probe.CASES, {"task1a": case}):
            rebuilt, encoded = probe.registered_request(raw, "task1a")
            self.assertEqual(rebuilt, request)
            self.assertEqual(len(encoded), size)
            with self.assertRaises(ValueError):
                probe.registered_request(raw, "task0a")
            with self.assertRaises(ValueError):
                probe.registered_request(raw + b" ", "task1a")
            with patch.dict(case, request_bytes=size + 1), self.assertRaises(ValueError):
                probe.registered_request(raw, "task1a")

    def test_reconstruction_only_adds_existing_intent_and_same_rubric(self):
        state = {"harness": {"stage": "RECOVER"}, "facts": {"target_visible": False},
                 "commands": {"command_000": {"command": {"move": "hold"}, "role": "hold", "current_preflight": {}}}}
        receipt = {"request_without_pixel_duplicates": {"model": MODEL, "images": [],
            "questions": {"intent": {}}, "state": state},
            "result": {"model": MODEL, "answers": {"intent": {"choice": "search"}}}}
        before = copy.deepcopy(receipt)
        request = probe.rebuild_command(receipt)
        self.assertEqual(receipt, before)
        self.assertEqual(request["state"], {**state, "chosen_tactic_not_new_evidence": "search"})
        self.assertEqual(set(request["questions"]["command"]["criteria"]), {"command_000", "abstain"})
        receipt["result"]["answers"]["intent"]["choice"] = "approach"
        with self.assertRaises(ValueError):
            probe.rebuild_command(receipt)

    def test_bad_response_kept_not_replaced_by_successful_repeat(self):
        class API:
            calls = 0
            def evaluate(self, state, questions):
                self.calls += 1
                if self.calls == 2:
                    raise JevChoiceMismatch({"probability_gap": .05})
                return {"answers": {"command": {"choice": "abstain"}}}, {"safe": True}
        api, recorded = API(), {}
        with patch.object(probe, "REPETITIONS", 3):
            rows = probe.run_trials(api, {"state": {}, "questions": {}}, lambda key, value: recorded.update({key: copy.deepcopy(value)}))
        self.assertEqual(api.calls, 3)
        self.assertEqual([row["repeat"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[1]["status"], "rejected_or_transport_failure")
        self.assertEqual(rows[1]["choice_mismatch_diagnostic"]["probability_gap"], .05)
        self.assertEqual(recorded["progress.json"]["completed_repetitions"], 3)
        self.assertNotIn("response_02.json", recorded)
        self.assertTrue(all(row["new_controls"] == 0 for row in rows))

    def test_unexpected_stop_keeps_completed_prefix(self):
        class API:
            calls = 0
            def evaluate(self, state, questions):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("test interruption")
                return {"answers": {"command": {"choice": "abstain"}}}, {"safe": True}
        rows = []
        with patch.object(probe, "REPETITIONS", 3), self.assertRaises(RuntimeError):
            probe.run_trials(API(), {"state": {}, "questions": {}}, lambda *args: None, rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["repeat"], 1)


if __name__ == "__main__":
    unittest.main()
