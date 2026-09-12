"""Real separated checkpoints through the official bridge, WITHOUT simulation.

Use recorded heldout observations for five full model/processor/23D wire probes.
This checks interface integration and reset semantics, not closed-loop success.
"""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from memlite_fm_v11_common import RECOVERY, sha, source_hashes
from preflight_memlite_fm_model import forbid_action_codec
from serve_policy_memlite_fm import load_separated, load_bridge, behavior_handler_for, TASKS_PATH
from serve_policy_mem import ChunkedPolicyWrapper
from g05.data.memlite_recovery_dataset import recovery_raw_sample


class RecordedConnection:
    remote_address = ("offline-preflight", 0)

    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.messages)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, value):
        self.sent.append(value)


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    bridge = load_bridge()
    inferencer, processor = load_separated(args.checkpoint, device="cuda:0", bridge=bridge)
    assert inferencer.high.policy is not inferencer.low.policy
    assert not any(p.requires_grad for p in inferencer.high.policy.parameters())
    instructions = bridge.load_task_instructions(TASKS_PATH)
    handler = behavior_handler_for(inferencer, processor, instructions, bridge=bridge,
                                  trace_root=args.output.with_suffix(".traces"))
    manifest = json.loads((RECOVERY/"recovery_manifest.json").read_text())
    rows = []
    high_processor = processor["galaxea_r1pro"]
    for task in range(5):
        entry = next(r for r in manifest["samples"]["eval"] if int(r["task_id"]) == task and r["memlite_branch"] == "low")
        entry = {**entry, "index": 0}
        root = Path(entry["trajectory"])
        obs = recovery_raw_sample(entry, root, high_processor, npz(root/"trajectory.npz"), npz)
        obs.pop("action", None)
        obs.pop("action_is_pad", None)
        # Ensure the deployed BF16 serving path, real inverse transform and
        # admission rules accept an FM proposal independently of high status.
        with forbid_action_codec(inferencer.low.policy):
            proposal = inferencer.infer_low_level_action([deepcopy(obs)], [entry["intent"]])[0]
        wrapper = ChunkedPolicyWrapper(inferencer, processor, action_steps=16, strict_memlite=True)
        wrapper._admit_action(proposal)
        assert "_absent_keys" not in proposal
        payload = npz(root/"obs_00000.npz")
        payload = {k:v for k,v in payload.items() if k.endswith("::rgb") or k.endswith("::proprio")}
        payload["task_id"] = np.asarray([task], dtype=np.int64)
        message = bridge.official_packb(payload)
        # Explicit reset + one 16-action chunk, followed by a second reset and
        # new first action. A fresh high call after reset is required.
        reset = bridge.official_packb({"reset": True})
        connection = RecordedConnection([reset, *([message]*16), reset, message])
        with patch.object(inferencer, "infer_low_level_action", wraps=inferencer.infer_low_level_action) as low_call, \
             patch.object(inferencer, "infer_high_level", wraps=inferencer.infer_high_level) as high_call:
            asyncio.run(handler(connection))
        decoded = [bridge.unpackb(v) for v in connection.sent]
        assert decoded[0]["action_dim"] == 23 and decoded[0]["action_steps"] == 16
        assert len(decoded) == 18 and high_call.call_count == 2
        vectors = [v["action"] for v in decoded[1:]]
        assert all(isinstance(v,np.ndarray) and v.shape == (23,) and np.isfinite(v).all() for v in vectors)
        rows.append({"task_id":task, "direct_fm_admitted":True, "wire_vectors":len(vectors),
                     "high_calls":high_call.call_count, "low_calls":low_call.call_count,
                     "reset_replanned":True, "planner_held_instead_of_fm":low_call.call_count < 2})
        print(json.dumps(rows[-1]), flush=True)
    result = {"passed": True, "checkpoint":str(Path(args.checkpoint).resolve()), "checkpoint_sha256":sha(args.checkpoint),
        "source_hashes":source_hashes(Path(__file__).resolve().parents[1]),
        "bridge_sha256":sha(bridge.__file__), "tasks":rows, "simulation_run":False, "success_rate_measured":False}
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
