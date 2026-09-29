"""JEV-04 exact live planning entry, old failure replay and conflict controls.

No GPU, simulator, perception or motor commands. Per-batch bounds prevent code
bugs; the user authorized further evidence-driven API batches without a global
request cap. No response is retried within a trial and all trials are recorded.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.jev_client import JevAbstained, JevClient, MODEL
from semantic_robot.v2.jev_policy import JevGroundedPolicy, load_task_plan, planning_request

BASELINE_SHA = "4cb60701d374b0e77ce566984e7ff9ee508c230e208a9f4ed3be551b80bc3f43"
PLAN = REPO / "configs/semantic_robot/jev_task0_plan.json"
CONFLICTS = (
    "Turn on the radio without picking it up. Never move or lift the radio.",
    "Do not touch the radio and do not turn it on.",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def baseline_request(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError("Only the hash-pinned JEV-03 actual planning failure is a baseline")
    request = json.loads(raw)["request_without_pixel_duplicates"]
    if request["model"] != MODEL or request["images"]:
        raise ValueError("Expected text-only pinned Jev baseline")
    return request["state"], request["questions"]


def qualification(rows, repeats):
    planning = [r for r in rows if r["kind"] == "exact_live_plan"]
    negative = [r for r in rows if r["kind"] == "contradictory_requirements_control"]
    return (len(planning) == repeats and len(negative) == len(CONFLICTS)
            and all(r["status"] == "complete" and r["selected_goal_ids"] == ["goal_0", "goal_1"] for r in planning)
            and all(r["choice"] == "abstain" for r in negative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--key-file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "TYPESAFE_API_KEY" in os.environ:
        raise ValueError("Inherited TypeSafe environment forbidden")
    if (not args.dry_run and not args.key_file) or (args.dry_run and args.key_file):
        raise ValueError("Live requires private --key-file; dry must not read a credential")
    if not 1 <= args.repetitions <= 100:
        raise ValueError("Choose a finite 1..100-trial batch; further batches need distinct run directories")
    old_state, old_questions = baseline_request(args.baseline)
    goals, identity = load_task_plan(PLAN, 0)
    code = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Freeze clean source before API validation")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    templates = [planning_request(goals, identity, identity["official_instruction"], prefix) for prefix in ([], [0])]
    result = dict(schema="jev04-planning-validation-v1", code_commit=code, model=MODEL,
                  policy_sha256=hashlib.sha256((REPO/"src/semantic_robot/v2/jev_policy.py").read_bytes()).hexdigest(),
                  client_sha256=hashlib.sha256((REPO/"src/semantic_robot/v2/jev_client.py").read_bytes()).hexdigest(),
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  contract_sha256=identity["sha256"], official_instruction=identity["official_instruction"],
                  official_instruction_sha256=identity["official_instruction_sha256"],
                  request_template_sha256=[digest(dict(state=s, questions=q)) for s, q in templates],
                  baseline_sha256=BASELINE_SHA, repetitions=args.repetitions,
                  dry_run=args.dry_run, qualified=False, rows=[], new_controls=0, new_resets=0, training_updates=0,
                  batch_request_bound=3+2*args.repetitions+len(CONFLICTS), wall_seconds=900,
                  global_api_cap=None, not_a_success_rate_evaluation=True)
    write("request_templates.json", [dict(state=s, questions=q) for s, q in templates])
    api = None
    with (out / "calls.jsonl").open("x") as journal, (out / "trials.jsonl").open("x") as trials:
        def record(value):
            journal.write(json.dumps(value, allow_nan=False)+"\n"); journal.flush()
        def trial(row, receipt):
            result["rows"].append(row)
            trials.write(json.dumps(dict(summary=row, receipt=receipt), allow_nan=False)+"\n"); trials.flush()
            print(json.dumps(row), flush=True)
        try:
            if args.dry_run:
                result["input_validation_only"] = True
                return
            api = JevClient(key_file=args.key_file, max_calls=result["batch_request_bound"], journal=record)
            api.deadline = time.perf_counter()+900
            for index in range(3):
                answer, receipt = api.evaluate(old_state, old_questions)
                trial(dict(kind="old_exact_second_question", index=index,
                           choice=answer["answers"]["next_goal"]["choice"]), receipt)
            for index in range(args.repetitions):
                policy = JevGroundedPolicy(SimpleNamespace(calls=0, identity={}), api, task_id=0, task_plan=PLAN)
                policy.deadline = api.deadline
                try:
                    selected, receipt = policy.plan(0, identity["official_instruction"], None)
                    row = dict(kind="exact_live_plan", index=index, status="complete",
                               selected_goal_ids=receipt["result"]["selected_goal_ids"],
                               goals=[asdict(g) for g in selected])
                except JevAbstained:
                    receipt = policy.last_call
                    row = dict(kind="exact_live_plan", index=index, status="abstained_no_action",
                               selected_goal_ids=receipt["result"]["selected_goal_ids"])
                trial(row, receipt)
            for index, instruction in enumerate(CONFLICTS):
                # Synthetic contradictory TASK TEXT, not fabricated sensor or physical labels.
                control_identity = copy.deepcopy(identity)
                control_identity["official_instruction"] = instruction
                state, questions = planning_request(goals, control_identity, instruction, [])
                answer, receipt = api.evaluate(state, questions)
                trial(dict(kind="contradictory_requirements_control", index=index, synthetic_task_text=True,
                           choice=answer["answers"]["next_goal"]["choice"]), receipt)
            result["qualified"] = qualification(result["rows"], args.repetitions)
        except Exception as error:
            result["error"] = str(error)
            raise
        finally:
            if api:
                result.update(api_requests=api.calls, validated_responses=api.validated_responses,
                              input_tokens=api.input_tokens, output_tokens=api.output_tokens)
            journal.flush(); trials.flush()
            result["calls_sha256"] = hashlib.sha256((out/"calls.jsonl").read_bytes()).hexdigest()
            result["trials_sha256"] = hashlib.sha256((out/"trials.jsonl").read_bytes()).hexdigest()
            write("result.json", result)
            print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
