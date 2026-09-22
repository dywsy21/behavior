"""Separate status→motion service, two exact adapters, one64-call ledger.

No simulator, early stopping, retry, oracle or public-verifier invocation.
REQUEST_VERIFY is handed back without a motion or success claim.
"""
import argparse
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import time

from common import sha, write_json
from native_completion_protocol import (VERSION, VARIANTS, canonical, snapshot_sha,
    query_row, parse_request, vocabulary)
from native_completion_modeling import load_model, encode, decode
from native_completion_training_data import load_reviewed, DATA_SHA, STATES_SHA
from native_completion_runtime import (validate_config, clean_source, require_authorization,
    initial_checkpoint, completed_checkpoint, CompletionStorage, BASE, base_identity,
    INITIAL_ADAPTER_SHA, EXECUTOR_DIGEST, CORE_COMMIT, execution_metadata, CARRY_PROFILE)


def checked_prediction(value, kind):
    keys = {"prediction", "latency_s", "input_tokens", "output_tokens", "input_ids_sha256"}
    from native_actor_protocol import check_sha
    if (type(value) is not dict or set(value) != keys or type(value["prediction"]) is not str or
            value["prediction"] not in vocabulary(kind) or type(value["latency_s"]) not in (float, int) or
            not math.isfinite(value["latency_s"]) or value["latency_s"] < 0 or
            any(type(value[k]) is not int or value[k] < 1 for k in ("input_tokens", "output_tokens"))):
        raise ValueError("Invalid typed model-query receipt")
    check_sha(value["input_ids_sha256"])
    return value


class DecisionEngine:
    def __init__(self, identity, identity_sha256, predict, *, check, record):
        if type(identity.get("max_calls")) is not int or identity["max_calls"] != 64 or identity.get("variants") != list(VARIANTS):
            raise ValueError("Exact two-adapter64-call service required")
        self.identity = copy.deepcopy(identity); self.identity_sha = identity_sha256
        self.predict = predict; self.check = check; self.record = record; self.calls = 0; self.seen = set()
    def decide(self, value):
        actor, images = parse_request(value, instructions=self.identity["instructions"], identity_sha256=self.identity_sha)
        request_id = value["request_id"]
        if request_id in self.seen: raise ValueError("No replay or implicit retry of a request id")
        self.check(); self.seen.add(request_id); frozen = canonical(actor); receipts = []
        def query(kind):
            self.check()
            if self.calls >= self.identity["max_calls"]: raise RuntimeError("Total model-call budget exhausted")
            if canonical(actor) != frozen: raise RuntimeError("Snapshot changed between queries")
            self.calls += 1
            entry = {"event": "QUERY_ISSUED", "call": self.calls, "query_kind": kind,
                "variant": value["variant"], "request_id": request_id, "snapshot_sha256": snapshot_sha(actor),
                "identity_sha256": self.identity_sha, "actor": copy.deepcopy(actor), "unix": time.time()}
            self.record(entry)
            # Independent copies protect the frozen snapshot even from a backend
            # mutating its inputs. Both queries receive identical pixels/actor.
            result = checked_prediction(self.predict(value["variant"], query_row(actor, kind),
                {k: image.copy() for k, image in images.items()}), kind)
            self.check(); result = {**result, "call": self.calls, "query_kind": kind}
            self.record({**entry, "event": "QUERY_COMPLETED", "result": result}); receipts.append(result)
            return result["prediction"]
        status = query("status")
        motion = query("motion") if status == "CONTINUE" else None
        result = {"protocol": VERSION, "request_id": request_id, "identity_sha256": self.identity_sha,
            "snapshot_sha256": snapshot_sha(actor), "variant": value["variant"],
            "adapter_sha256": self.identity["adapter_sha256"][value["variant"]],
            "skill_status": status, "motion": motion, "success_claim": False, "queries": receipts}
        validate_decision(result, value, self.identity, self.identity_sha)
        return result


def validate_decision(result, request, identity, identity_sha256):
    keys = {"protocol", "request_id", "identity_sha256", "snapshot_sha256", "variant", "adapter_sha256",
        "skill_status", "motion", "success_claim", "queries"}
    if type(result) is not dict or set(result) != keys: raise ValueError("Exact decision schema required")
    for key, value in {"protocol": VERSION, "request_id": request["request_id"], "identity_sha256": identity_sha256,
            "snapshot_sha256": snapshot_sha(request["actor"]), "variant": request["variant"],
            "adapter_sha256": identity["adapter_sha256"][request["variant"]], "success_claim": False}.items():
        if canonical(result[key]) != canonical(value): raise ValueError("Decision identity mismatch: "+key)
    status = result["skill_status"]
    if status not in vocabulary("status"): raise ValueError("Unknown decision status")
    if (status == "REQUEST_VERIFY" and result["motion"] is not None or
            status == "CONTINUE" and result["motion"] not in vocabulary("motion")):
        raise ValueError("Status/motion mismatch")
    queries = result["queries"]; kinds = ["status", "motion"] if status == "CONTINUE" else ["status"]
    if type(queries) is not list or len(queries) != len(kinds): raise ValueError("Incorrect actual query count")
    previous = None
    for query, kind in zip(queries, kinds):
        if set(query) != {"prediction", "latency_s", "input_tokens", "output_tokens", "input_ids_sha256", "call", "query_kind"}:
            raise ValueError("Unexpected query receipt")
        checked_prediction({k: v for k, v in query.items() if k not in ("call", "query_kind")}, kind)
        if (query["query_kind"] != kind or type(query["call"]) is not int or not 1 <= query["call"] <= identity["max_calls"] or
                previous is not None and query["call"] != previous+1): raise ValueError("Call ledger mismatch")
        if query["prediction"] != (status if kind == "status" else result["motion"]): raise ValueError("Query prediction mismatch")
        previous = query["call"]
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "motion-data", "data-review", "initial-training", "training", "training-config", "config", "authorization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); validate_config(cfg, service=True)
    training_cfg = json.loads(args.training_config.read_text()); validate_config(training_cfg)
    code = clean_source(); auth = json.loads(args.authorization.read_text())
    require_authorization(auth, stage="service", code=code, config_sha=sha(args.config), output=args.output)
    if auth.get("training_result_sha256") != sha(args.training/"result.json"):
        raise ValueError("Exact new training-result authorization required")
    examples, _ = load_reviewed(args.data, args.motion_data, args.data_review)
    initial, _ = initial_checkpoint(args.initial_training)
    adapter, training, checkpoint = completed_checkpoint(args.training, code=code, config_sha=sha(args.training_config))
    base_files = base_identity()
    if any(training.get(k) != v for k, v in base_files.items()): raise ValueError("Base/config/tokenizer changed since training")
    storage = CompletionStorage(auth, args.output); storage.create()
    # Bind the exact port BEFORE loading weights. Collision fails, never kills
    # an existing service and never searches for an unreviewed replacement port.
    server = HTTPServer(("127.0.0.1", cfg["port"]), BaseHTTPRequestHandler)
    args.output.mkdir(parents=True, exist_ok=False)
    ledger = None
    try:
        import torch
        torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(41)
        model, processor = load_model(BASE, adapter=adapter)
        model.load_adapter(str(initial), adapter_name="motion_only_0120", is_trainable=False)
        model.eval(); storage.check()
        identity = {**cfg, "code_commit": code, "dataset_sha256": DATA_SHA, "states_sha256": STATES_SHA,
            "training_result_sha256": sha(args.training/"result.json"), "training_config_sha256": sha(args.training_config),
            "config_sha256": sha(args.config), "authorization_sha256": sha(args.authorization), **base_files,
            "adapter_sha256": {VARIANTS[0]: INITIAL_ADAPTER_SHA, VARIANTS[1]: checkpoint["adapter_sha256"]},
            "executor_digest": EXECUTOR_DIGEST, "carry_duration_core_commit": CORE_COMMIT, **execution_metadata(CARRY_PROFILE),
            "instructions": sorted({e["query"]["actor"]["active_instruction"] for e in examples}),
            "storage": storage.check(), "request_verify_is_success": False, "new_physical_actions": 0}
        write_json(args.output/"identity.json", identity); identity_sha = sha(args.output/"identity.json")
        ledger = (args.output/"calls.jsonl").open("x", buffering=1)
        def record(value): ledger.write(json.dumps(value, allow_nan=False)+"\n")
        def predict(variant, row, images):
            model.set_adapter("motion_only_0120" if variant == VARIANTS[0] else "default"); model.eval()
            return decode(model, processor, encode(processor, row, images, supervised=False), row["query_kind"])
        engine = DecisionEngine(identity, identity_sha, predict, check=storage.check, record=record)
        class Handler(BaseHTTPRequestHandler):
            def send(self, status, value):
                data = json.dumps(value, allow_nan=False).encode(); self.send_response(status)
                self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
            def do_GET(self):
                self.send(200 if self.path == "/health" else 404,
                    {**identity, "identity_sha256": identity_sha, "calls": engine.calls} if self.path == "/health" else {"error": "Unknown endpoint"})
            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/decision" or not 1 <= size <= 13*1024**2: raise ValueError("Endpoint/request size")
                    value = json.loads(self.rfile.read(size)); result = engine.decide(value)
                    record({"event": "DECISION_COMPLETED", "unix": time.time(), "result": result}); self.send(200, result)
                except Exception as exc:
                    record({"event": "ERROR_NO_RETRY", "unix": time.time(), "error": repr(exc), "calls": engine.calls})
                    self.send(400, {"error": str(exc), "calls": engine.calls})
        server.RequestHandlerClass = Handler
        print(json.dumps({"ready": True, "identity_sha256": identity_sha, **identity}), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        if ledger is not None: ledger.close()


if __name__ == "__main__": main()
