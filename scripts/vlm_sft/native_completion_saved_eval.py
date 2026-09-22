"""Eight preregistered DEVELOPMENT snapshots; never training data or new SR.

Public actor construction and private same-clock scoring are separate. Only
actor + exact RGB bytes enter the existing9de status/motion service.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from common import CAMERAS, sha
from native_actor_protocol import from_capture, actor_input
from native_completion_protocol import VERSION, VARIANTS, canonical, request_payload, snapshot_sha
from native_completion_serve import validate_decision
from native_completion_runtime import clean_source, INITIAL_ADAPTER_SHA, CompletionStorage
from native_execution import CARRY_PROFILE, completed
from native_motion_codec import ACTOR_VERSION
from native_reference_profile import ROOT
from native_storage import WALL2100_SPEC, NVME, runtime_environment, tree_bytes
from native_teacher_artifacts import load_calibration

SERVICE_CODE = "9deafb1b65cc9e0c3c91371af0eb2f815b063774"
ORIGINAL_CODE = "6f3288770f72b727a114d596cd5b98f577adc8c1"
TASK = "Put the three cans of soda from the living room inside the trash can in the kitchen."
INSTRUCTION = "verb=GRASP; target=trash can; source=floors"
CALIBRATION = "815e5e789d28166e5396666cf38ad139d80957df0f9877366871afb3634ac7ca"
RUNS = {
    "eval_t1_i71_finetuned_v1": "5c8416d8b6f8648c2bf0ea84de5b4cd57fdf842a49b8f74f5e241a9822412fcc",
    "eval_t1_i71_proprio_history_nn_v1": "23358f8abb4059acee008a374732354b6d0314f6739dd343f9d6178d53584ce9"}
SELECTION = [(list(RUNS)[0], "decision_00/before")] + [
    (list(RUNS)[0], f"decision_{i:02d}/after_settle") for i in range(1, 6)] + [
    (list(RUNS)[1], f"decision_{i:02d}/after_settle") for i in (4, 5)]
PUBLIC_REPORT_SHA = "e4a670319e3d5b6f99d1c2935abc12459ae57ffc8aa360f7d9237300bc309b1d"
SCORER_SHA = "85ed4400bc1dd712a2e8b8e4413ed93656dde2956d8a1e4b5ff9e87eeb16c261"
CONFIG = {"version": "h09ac-eight-saved-completion-v1", "service_code_commit": SERVICE_CODE,
    "variants": list(VARIANTS), "selection": [list(x) for x in SELECTION], "inventories": RUNS,
    "public_saved_report_sha256": PUBLIC_REPORT_SHA, "max_decisions": 16, "max_queries": 32,
    "service_max_calls": 64, "wall_seconds": 600, "output_MiB": 16,
    "new_resets": 0, "new_training_updates": 0, "development_diagnostic_not_SR": True}


def read(path): return json.loads(Path(path).read_text())


def checked_inventory(root, name):
    root = Path(root).resolve(); path = root/(name+"_hashes.json")
    if sha(path) != RUNS[name]: raise ValueError("Original full-run inventory changed")
    inventory = read(path)
    for relative, entry in inventory["files"].items():
        rel = Path(relative); file = root/rel
        if rel.is_absolute() or ".." in rel.parts or file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError("Escaping/link inventory member")
        if file.stat().st_size != entry["bytes"] or sha(file) != entry["sha256"]:
            raise ValueError("Original full-run evidence changed: "+relative)
    return inventory


def original_scores(run):
    """Private posthoc only; no result or spec is an actor input."""
    from native_teacher_outcomes import LocalOutcome
    if sha(Path(__file__).with_name("native_teacher_outcomes.py")) != SCORER_SHA:
        raise ValueError("Original physical outcome implementation changed")
    initial = read(run/"PRIVATE_initial.json")
    frames = [json.loads(line) for line in (run/"PRIVATE_measurements.jsonl").read_text().splitlines()]
    tokens = {}
    for path in sorted(run.glob("decision_*/execution.json")):
        value = read(path)
        if type(value.get("control_end")) is int:
            for tick in range(value["control_start"]+1, value["control_end"]+1):
                tokens[1038+tick] = value["token"]
    oracles = {arm: LocalOutcome({**initial["spec"], "hand": arm, "support_hand": None}) for arm in ("left", "right")}
    scores = {}
    for frame in [initial["frame"], *frames]:
        scores[frame["tick"]] = {arm: oracle.update(frame, tokens.get(frame["tick"])) for arm, oracle in oracles.items()}
    return scores


def expected_status(scores, tick):
    current = scores.get(tick)
    if not current: return "UNKNOWN"
    values = [current[arm]["outcome"] for arm in ("left", "right")]
    if "SUCCEEDED" in values: return "REQUEST_VERIFY"
    return "CONTINUE" if all(x == "IN_PROGRESS" for x in values) else "UNKNOWN"


def public_saved_flags(path):
    if sha(Path(path)) != PUBLIC_REPORT_SHA: raise ValueError("Exact reviewed31b saved-response report required")
    report = read(path)
    if (report["actual_model_calls"] != 0 or report["physical_controls"] != 0 or
            report["module_sha256"] != "be133733fc221b6edc5726cc460f1514577040fa9565495099a5ea4322170fa2"):
        raise ValueError("Not the reviewed zero-new-call public replay")
    flags = {}
    for result in report["results"]:
        if result["case"] not in ("ft_i71", "nn_i71"): continue
        for frame in result["frames"]:
            flags[frame["capture_sha256"]] = frame["grasp"]["verified"]
    return flags


def load_cases(root, public_report):
    root = Path(root); flags = public_saved_flags(public_report); data = {}
    for name in RUNS:
        inventory = checked_inventory(root, name); run = root/name
        manifest = read(run/"manifest.json")
        prep = manifest["preparation"]
        if (manifest["code_commit"] != ORIGINAL_CODE or manifest["protocol"] != ACTOR_VERSION or
                manifest["oracle_actor_feedback"] is not False or prep["source"] != [1, 247, 71] or
                prep["prefix_controls"] != 1038 or prep["training_eligible"] is not False or
                prep["active_instruction"] != INSTRUCTION or prep["specified_hand"] is not None):
            raise ValueError("Fixed development run/instruction mismatch")
        model = load_calibration(run)
        if model.sha != CALIBRATION: raise ValueError("Original actual robot calibration changed")
        trace = [json.loads(x) for x in (run/"native_trace.jsonl").read_text().splitlines()]
        lookup = {(x["prefix_control"], x["native_control"]): x for x in trace if x["phase"] != "prefix"}
        scores = original_scores(run); history = []; snapshots = {}
        for index in range(6):
            directory = run/f"decision_{index:02d}"
            before = read(directory/"before/capture.json"); execution = read(directory/"execution.json")
            req = read(directory/"request.json")
            if req["capture_sha256"] != sha(directory/"before/capture.json"):
                raise ValueError("Original query snapshot drift")
            p, hashes, _ = from_capture(directory/"before", model,
                capture_sha256=req["capture_sha256"], expected_clock=before["clock"],
                calibration_sha256=model.sha, protocol=ACTOR_VERSION)
            actor = actor_input(TASK, INSTRUCTION, p, hashes, history[-5:], protocol=ACTOR_VERSION)
            if canonical(actor) != canonical(req["actor"]): raise ValueError("Original actual query/history mismatch")
            snapshots[f"decision_{index:02d}/before"] = (actor, before)
            start, end = execution["control_start"], execution["control_end"]
            if (before["clock"] != {"prefix_control": 1038, "native_control": start} or
                    not completed(execution["token"], execution["feedback"], profile=CARRY_PROFILE) or
                    execution["feedback"]["control_ticks"] != end-start):
                raise ValueError("History requires original fully completed execution")
            if any(lookup.get((1038, tick), {}).get("phase") != "candidate" for tick in range(start+1, end+1)):
                raise ValueError("Missing actual issued/completed macro controls")
            if any(lookup.get((1038, tick), {}).get("phase") != "settle" for tick in range(end+1, end+13)):
                raise ValueError("Incomplete actual12-control settle")
            history.append(execution["token"])
            after = read(directory/"after_settle/capture.json")
            completion = read(directory/"completed.json")
            if (after["clock"] != {"prefix_control": 1038, "native_control": end+12} or
                    completion != {"clock": after["clock"], "history": history[-5:]}):
                raise ValueError("Actual settled history/clock mismatch")
            p, hashes, _ = from_capture(directory/"after_settle", model,
                capture_sha256=sha(directory/"after_settle/capture.json"), expected_clock=after["clock"],
                calibration_sha256=model.sha, protocol=ACTOR_VERSION)
            snapshots[f"decision_{index:02d}/after_settle"] = (
                actor_input(TASK, INSTRUCTION, p, hashes, history[-5:], protocol=ACTOR_VERSION), after)
        data[name] = (snapshots, scores)
    cases = []
    for index, (name, relative) in enumerate(SELECTION):
        actor, capture = data[name][0][relative]; tick = sum(capture["clock"].values())
        directory = root/name/relative; capture_sha = sha(directory/"capture.json")
        cases.append({"id": f"state_{index:02d}", "run": name, "relative_capture": relative,
            "capture_sha256": capture_sha, "clock": capture["clock"], "inventory_sha256": RUNS[name],
            "actor": actor, "snapshot_sha256": snapshot_sha(actor),
            "images": {v: str(directory/(v+".png")) for v in CAMERAS},
            "expected_status": expected_status(data[name][1], tick),
            "private_same_tick_score": data[name][1].get(tick),
            "saved_public_holding_verified": flags.get(capture_sha),
            "public_verifier_new_calls": 0, "not_training_eligible": True})
    return cases


def inputs_sha(cases):
    return hashlib.sha256(canonical([{k: c[k] for k in ("id", "capture_sha256", "snapshot_sha256", "inventory_sha256")} for c in cases]).encode()).hexdigest()


def confusion(cases, predictions):
    if len(cases) != len(predictions): raise ValueError("One prediction per fixed case")
    result = dict(true_request=0, false_early_request=0, missed_late_request=0, true_continue=0, unknown_label=0,
                  public_verified_but_strictly_early_request=0)
    for case, prediction in zip(cases, predictions):
        if prediction not in ("CONTINUE", "REQUEST_VERIFY"): raise ValueError("Invalid status")
        truth = case["expected_status"]
        if truth == "UNKNOWN": result["unknown_label"] += 1; continue
        key = {("REQUEST_VERIFY", "REQUEST_VERIFY"): "true_request", ("CONTINUE", "REQUEST_VERIFY"): "false_early_request",
               ("REQUEST_VERIFY", "CONTINUE"): "missed_late_request", ("CONTINUE", "CONTINUE"): "true_continue"}[(truth, prediction)]
        result[key] += 1
        if key == "false_early_request" and case["saved_public_holding_verified"] is True:
            result["public_verified_but_strictly_early_request"] += 1
    return result


def evaluate(cases, identity, identity_sha, *, call, check, record, deadline, initial_calls):
    """Bounded16 transactions; the injected transport never receives scores/paths."""
    decisions = []; calls = initial_calls
    if len(cases) != 8 or len({c["id"] for c in cases}) != 8: raise ValueError("Exact eight unique snapshots")
    if type(calls) is not int or not 0 <= calls <= 32: raise ValueError("Reserve32 of original64 calls")
    for case in cases:
        for variant in VARIANTS:
            check()
            if time.monotonic() >= deadline: raise TimeoutError("No request after600s")
            images = {v: Path(case["images"][v]).read_bytes() for v in CAMERAS}
            request = request_payload(copy.deepcopy(case["actor"]), images, variant,
                "h09ac_"+case["id"]+"_"+variant, identity_sha)
            record({"event": "REQUEST_ISSUED", "case": case["id"], "request": {k: v for k, v in request.items() if k != "images"}})
            response = call(request, max(.001, deadline-time.monotonic()))
            validate_decision(response, request, identity, identity_sha)
            if response["queries"][0]["call"] != calls+1: raise ValueError("Concurrent/hidden service calls")
            calls = response["queries"][-1]["call"]
            if calls-initial_calls > 32: raise RuntimeError("Block query cap")
            check()
            if time.monotonic() > deadline: raise TimeoutError("Late response beyond600s")
            decisions.append({"case": case["id"], "variant": variant, "response": response})
            record({"event": "DECISION_COMPLETED", **decisions[-1]})
    return {"decisions": decisions, "actual_queries": calls-initial_calls,
        "confusion": {v: confusion(cases, [d["response"]["skill_status"] for d in decisions if d["variant"] == v]) for v in VARIANTS},
        "baselines": {"always_request": confusion(cases, ["REQUEST_VERIFY"]*8), "never_request": confusion(cases, ["CONTINUE"]*8)},
        "new_resets": 0, "new_training_updates": 0, "new_simulator_success_rate": None,
        "original_trial_outcomes_unchanged": True, "development_saved_state_diagnostic_only": True}


def checked_health(value, identity, digest, calls):
    if canonical(value) != canonical({**identity, "identity_sha256": digest, "calls": calls}):
        raise ValueError("Exact service identity/call position changed")


def main():
    p = argparse.ArgumentParser()
    for key in ("evidence-root", "public-report", "config", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--authorization", type=Path); p.add_argument("--service-identity", type=Path)
    args = p.parse_args(); cfg = read(args.config)
    if canonical(cfg) != canonical(CONFIG): raise ValueError("Exact H09AC fixed selection and limits")
    code = clean_source(); cases = load_cases(args.evidence_root, args.public_report)
    binding = inputs_sha(cases)
    if args.prepare_only:
        with args.output.open("x") as stream:
            json.dump({"code_commit": code, "inputs_sha256": binding, "cases": cases,
                "config_sha256": sha(args.config), "new_queries": 0, "new_resets": 0}, stream, indent=2, allow_nan=False)
        return
    auth = read(args.authorization); identity = read(args.service_identity); identity_sha = sha(args.service_identity)
    expected = {"code_commit": code, "config_sha256": sha(args.config), "inputs_sha256": binding,
        "service_code_commit": SERVICE_CODE, "service_identity_sha256": identity_sha,
        "candidate_adapter_sha256": identity["adapter_sha256"][VARIANTS[1]],
        "training_result_sha256": identity["training_result_sha256"],
        "initial_calls": auth.get("initial_calls"),
        "authorize_saved_eval": True, "authorize_training": False, "authorize_physical": False,
        "reviewer": "Codex-parent", "max_queries": 32, "wall_seconds": 600, "output_MiB": 16,
        "output": str(ROOT/"completion_saved_eval_v1")}
    if type(expected["initial_calls"]) is not int or not 0 <= expected["initial_calls"] <= 32:
        raise ValueError("Typed exact initial query counter required")
    parent = read(auth["parent_release"])
    if sha(Path(auth["parent_release"])) != auth["parent_release_sha256"]: raise ValueError("Parent release bytes changed")
    for key, value in expected.items():
        if canonical(auth.get(key)) != canonical(value) or canonical(parent.get(key)) != canonical(value):
            raise ValueError("Exact independent authorization required: "+key)
    if (identity["code_commit"] != SERVICE_CODE or identity["protocol"] != VERSION or identity["max_calls"] != 64 or
            identity["variants"] != list(VARIANTS) or identity["adapter_sha256"][VARIANTS[0]] != INITIAL_ADAPTER_SHA or
            identity["adapter_sha256"][VARIANTS[1]] != auth["candidate_adapter_sha256"] or
            identity["training_result_sha256"] != auth["training_result_sha256"] or
            identity["physical_gpu"] != 3 or identity["port"] != 8931):
        raise ValueError("Exact original/new adapters on reviewed service required")
    if args.output.resolve() != ROOT/"completion_saved_eval_v1" or args.output.exists() or args.output.is_symlink():
        raise ValueError("Unique fixed diagnostic output required")
    os.environ.update(runtime_environment(ROOT/"service_completion_v1"))
    storage = CompletionStorage({"storage": WALL2100_SPEC}, ROOT/"service_completion_v1")
    def check():
        storage.check()
        if args.output.exists() and tree_bytes(args.output, NVME.stat().st_dev) >= 16*1024**2:
            raise RuntimeError("Diagnostic16MiB output cap")
    def health():
        with urllib.request.urlopen("http://127.0.0.1:8931/health", timeout=10) as response:return json.load(response)
    checked_health(health(), identity, identity_sha, auth["initial_calls"]); check()
    args.output.mkdir(); started = time.monotonic(); deadline = started+600
    with (args.output/"inputs.json").open("x") as stream:json.dump({"binding": binding, "cases": cases}, stream, indent=2, allow_nan=False)
    with (args.output/"decisions.jsonl").open("x", buffering=1) as ledger:
        def record(value):
            line = json.dumps(value, allow_nan=False)+"\n"
            if len(line) > 128*1024: raise RuntimeError("Bounded response evidence")
            ledger.write(line)
        def call(value, remaining):
            request = urllib.request.Request("http://127.0.0.1:8931/decision", data=json.dumps(value).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=min(90., remaining)) as response:return json.load(response)
        try:
            result = evaluate(cases, identity, identity_sha, call=call, check=check, record=record,
                deadline=deadline, initial_calls=auth["initial_calls"])
            checked_health(health(), identity, identity_sha, auth["initial_calls"]+result["actual_queries"])
            check()
            if time.monotonic() > deadline: raise TimeoutError("Final validation beyond600s")
            result.update(status="COMPLETE", code_commit=code, inputs_sha256=binding,
                service_identity_sha256=identity_sha, authorization_sha256=sha(args.authorization), seconds=time.monotonic()-started)
        except Exception as exc:
            result = {"status": "FAILED_NO_RETRY", "error": repr(exc), "seconds": time.monotonic()-started,
                "query_count_requires_actual_service_ledger": True, "new_resets": 0}
        with (args.output/"result.json").open("x") as stream:json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__": main()
