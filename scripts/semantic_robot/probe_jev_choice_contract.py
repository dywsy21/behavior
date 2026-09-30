"""Thirty non-actuating repetitions of the reconstructed JEV-05 failed request.

All responses, including contract violations, remain in the denominator. This
does not retry a live robot command, modify its choice, or run a simulator.
The original rejected response was not persisted; these are NEW responses.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.jev_client import JevClient, JevError, MODEL, choice
from semantic_robot.v2.jev_policy import CONTROL_RULES, command_rubric

SOURCE_SHA = "1f1b0a47e22ed5c7e361c9d77af1ac0bb2977dfdfb24e29587b0e75343c220ec"
COMMAND_BYTES = 20184
REPETITIONS = 30


def rebuild_command(intent_receipt):
    """Use the saved validated intent, not another model request or new fact."""
    request = intent_receipt["request_without_pixel_duplicates"]
    result = intent_receipt["result"]
    if (request["model"] != MODEL or request["images"] or set(request["questions"]) != {"intent"}
            or result["model"] != MODEL or result["answers"]["intent"]["choice"] != "search"
            or request["state"]["harness"]["stage"] != "RECOVER"
            or request["state"]["facts"]["target_visible"] is not False):
        raise ValueError("Expected the saved unseen-target RECOVER state with search intent")
    state = copy.deepcopy(request["state"])
    state["chosen_tactic_not_new_evidence"] = result["answers"]["intent"]["choice"]
    options = {key: command_rubric(row) for key, row in state["commands"].items()}
    options["abstain"] = "None of the offered commands is justified. Stop without issuing a new robot command."
    question = choice(CONTROL_RULES + " Choose one command consistent with the tactic and current feedback, or abstain. An unsafe tactic does not override evidence or constraints.", options)
    return {"model": MODEL, "state": state, "questions": {"command": question}}


def run_trials(api, request, record, rows=None):
    rows = [] if rows is None else rows
    for index in range(REPETITIONS):
        row = {"repeat": index + 1, "new_controls": 0}
        try:
            answer, receipt = api.evaluate(request["state"], request["questions"])
            row.update(status="validated", answer=answer["answers"]["command"])
            record(f"response_{index + 1:02d}.json", receipt)
        except JevError as error:
            # JevError messages are already credential-free; numeric mismatch
            # detail is durable in the independent per-attempt client journal.
            row.update(status="rejected_or_transport_failure", error=str(error))
            if hasattr(error, "diagnostic"):
                row["choice_mismatch_diagnostic"] = error.diagnostic
        rows.append(row)
        record("progress.json", {"rows": rows, "completed_repetitions": len(rows)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-file", required=True)
    args = parser.parse_args()
    if "TYPESAFE_API_KEY" in os.environ:
        raise ValueError("Private key file only")
    raw = Path(args.source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Only the frozen JEV-05 task0a intent receipt")
    request = rebuild_command(json.loads(raw))
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    if len(encoded) != COMMAND_BYTES:
        raise ValueError("Reconstructed request differs from original attempt byte count")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Clean independent frozen source required")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    result = dict(schema="jev-choice-contract-probe-v1", source_sha256=SOURCE_SHA,
        source_commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        reconstructed_request_sha256=hashlib.sha256(encoded).hexdigest(), request_bytes=len(encoded),
        model=MODEL, repetitions=REPETITIONS, wall_seconds=1200, new_controls=0, new_resets=0,
        training_updates=0, original_rejected_response_unavailable=True,
        original_request_reconstructed_not_directly_archived=True, completed=False, rows=[])
    write("request.json", request)
    api = None
    with (folder / "calls.jsonl").open("x") as stream:
        def journal(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        try:
            api = JevClient(key_file=args.key_file, max_calls=REPETITIONS, journal=journal)
            api.deadline = time.perf_counter() + 1200
            run_trials(api, request, write, result["rows"])
            result["completed"] = True
        finally:
            if api:
                result.update(api_requests=api.calls, validated_responses=api.validated_responses,
                    input_tokens=api.input_tokens, output_tokens=api.output_tokens,
                    token_usage_scope="validated_responses_only")
            write("result.json", result)
    print(json.dumps({key: result[key] for key in ("completed", "api_requests", "validated_responses")}), flush=True)


if __name__ == "__main__":
    main()
