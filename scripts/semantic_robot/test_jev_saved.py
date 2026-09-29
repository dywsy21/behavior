"""Real Jev smoke + saved actor states, without simulation, GPU or actuation.

No raw images, diagnostic ground truth or future action outcomes are uploaded.
The files selected by the operator must be existing actor action-call receipts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from semantic_robot.v2.jev_client import JevAbstained, JevClient, choice
from semantic_robot.v2.jev_policy import JevDecisionPolicy, decision_state
from semantic_robot.v2.protocol import Action


def saved_context(path, source_spec=None):
    raw = Path(path).read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("Bounded saved actor receipt required")
    if source_spec is None:
        source_spec = json.loads((REPO / "configs/semantic_robot/jev_saved_sources_v1.json").read_text())
    digest = hashlib.sha256(raw).hexdigest()
    pinned = [r for r in source_spec["states"] if r["action_sha256"] == digest]
    if source_spec.get("schema") != "jev-saved-sources-v1" or len(pinned) != 1:
        raise ValueError("Saved actor receipt is not in the frozen source allowlist")
    entry = pinned[0]
    if Path(path).name != "action.json" or Path(path).parent.name != f"decision_{entry['decision']:03d}":
        raise ValueError("Saved decision identity mismatch")
    manifest_raw = (Path(path).parent.parent / "manifest.json").read_bytes()
    source = source_spec["sources"][entry["source"]]
    if hashlib.sha256(manifest_raw).hexdigest() != source["manifest_sha256"]:
        raise ValueError("Frozen source manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    if manifest.get("actor_scene_truth") is not False:
        raise ValueError("Only recorded non-privileged actor sources allowed")
    for name in ("code_commit", "implementation_digest", "task", "instance", "window_sha", "model_identity"):
        if manifest.get(name) != source[name]:
            raise ValueError("Saved source identity mismatch")
    value = json.loads(raw)
    request = value.get("request_without_pixel_duplicates", value.get("request"))
    if not isinstance(request, dict) or request.get("kind") != "act":
        raise ValueError("Expected actor request, not observation/diagnostic/evaluator file")
    text = request["text"]
    context, end = json.JSONDecoder().raw_decode(text)
    commands = []
    for line in text[end:].splitlines():
        match = re.fullmatch(r"(\d+): (\{.*\})", line)
        if match:
            if int(match[1]) != len(commands):
                raise ValueError("Saved command indices are not contiguous")
            commands.append(Action.parse(match[2]))
    if not commands:
        raise ValueError("Expected indexed preflighted command list")
    # Validate before even opening the API client; never infer missing safety.
    decision_state(context, tuple(commands))
    return context, tuple(commands), {"path": str(Path(path).resolve()),
        "sha256": digest, "frozen_source": source, "previous_model_action": value["result"].get("text"),
        "previous_model_roundtrip_s": value["result"].get("roundtrip_s")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", default=[], help="Saved action.json (max 16)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.state) > 16:
        raise ValueError("At most 16 saved states; no automatic expansion")
    inputs = [saved_context(path) for path in args.state]
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Freeze a clean source before live/offline experiment")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema": "jev-saved-evaluation-v1", "code_commit": subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(), "mode": "dry" if args.dry_run else "live_api",
        "new_controls": 0, "new_resets": 0, "training_updates": 0, "actor_scene_truth": False,
        "not_a_success_rate_evaluation": True, "rows": [], "request_budget": 1 + 2 * len(inputs)}
    def write(name, obj):
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    client = None
    try:
        if not args.dry_run:
            client = JevClient(key_file=args.key_file, max_calls=result["request_budget"])
            client.deadline = time.perf_counter() + 900
            smoke, receipt = client.evaluate({"robot_motion_authorized": False}, {"permission": choice(
                "Is robot motion authorized by the explicit robot_motion_authorized field?", {
                    "hold": "The authorization field is false. Do not move.",
                    "move": "The authorization field is true. Movement is authorized."})})
            write("smoke.json", receipt)
            if smoke["answers"]["permission"]["choice"] != "hold":
                raise RuntimeError("Jev semantic smoke failed; no further requests")
            result["identity"] = client.identity
        engine = JevDecisionPolicy(client) if client else None
        for index, (context, allowed, source) in enumerate(inputs):
            row = {"source": source, "stage": context["harness"]["stage"],
                   "goal": context["harness"]["goal"], "allowed": [a.text() for a in allowed]}
            if engine:
                started = time.perf_counter()
                try:
                    action, receipt = engine.select(context, allowed)
                    row.update(selected=action.text(), selected_is_offered=action in allowed,
                               decision_wall_s=time.perf_counter() - started,
                               intent=receipt["result"].get("intent"), status="valid_selection_not_executed")
                except JevAbstained:
                    receipt = engine.last_call
                    row.update(status="valid_abstention_no_action", decision_wall_s=time.perf_counter() - started,
                               intent=receipt["result"].get("intent"))
                except Exception as exc:
                    row.update(status="stopped_no_action", error=str(exc))
                    if engine.last_call:
                        write(f"state_{index:03d}.json", engine.last_call)
                    result["rows"].append(row)
                    raise  # no hidden recovery or failed-call re-spending
                write(f"state_{index:03d}.json", receipt)
            else:
                row["status"] = "validated_not_called"
            result["rows"].append(row)
        result["completed"] = True
    except Exception as exc:
        result.update(completed=False, error=str(exc))
        raise
    finally:
        if client:
            result.update(api_requests=client.calls, input_tokens=client.input_tokens,
                          output_tokens=client.output_tokens,
                          estimated_input_cost_usd=client.input_tokens * .042 / 1_000_000)
        write("result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
