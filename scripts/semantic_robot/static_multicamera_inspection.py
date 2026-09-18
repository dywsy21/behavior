"""Two hash-checked saved-image choices, no physics or fabricated camera view."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from static_inspection_budget import REPO, restore, GroundedPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--uri", required=True)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("Pinned clean source required")
    a.output.mkdir(exist_ok=False)
    started = time.monotonic()
    policy = GroundedPolicy(a.uri, a.revision, max_calls=2)
    rows = []
    try:
        for index in (4, 20):
            if time.monotonic() - started > 300: raise TimeoutError("Static gate wall budget")
            h, state, bundle, allowed, original = restore(a.run, index, multicamera=True)
            if h.stop_reason: raise ValueError(h.stop_reason)
            action, call = policy.act_feasible(h, state, bundle, allowed)
            if call["result"]["images"] != original["result"]["images"]:
                raise ValueError("Saved image pixels changed")
            chosen = next((r for r in h.candidate_receipt["tested"] if r["action"] == asdict(action)), None)
            row = {"decision": index, "action": asdict(action), "chosen": chosen, "call": call,
                   "context": h.held_inspection, "candidates": h.candidate_receipt,
                   "allowed": [asdict(x) for x in allowed], "same_original_images": True}
            (a.output / f"decision_{index:03d}.json").write_text(json.dumps(row, indent=2, allow_nan=False))
            rows.append(row)
            print(json.dumps({"decision": index, "action": row["action"], "after": (chosen or {}).get("inspection_after")}), flush=True)
        if time.monotonic() - started > 300: raise TimeoutError("Static gate wall budget")
        (a.output / "result.json").write_text(json.dumps({"status": "complete", "controls": 0,
            "model_calls": policy.calls, "wall_seconds": time.monotonic() - started,
            "decisions": [r["decision"] for r in rows], "not_task_success": True}, indent=2))
    except BaseException as error:
        (a.output / "failure.json").write_text(json.dumps({"error": repr(error), "model_calls": policy.calls, "controls": 0}))
        raise


if __name__ == "__main__":
    main()
