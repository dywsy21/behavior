"""Actual delivered sample identities, not just planned sampler counts."""
import json
from pathlib import Path

import torch


def record_batch_receipt(batch, output_dir, *, rank, step, batch_index):
    samples = batch["samples"]
    n = len(samples)
    def item(key, i):
        values = batch.get(key)
        if values is None:
            return None
        value = values[i]
        return value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value
    rows = []
    for i, sample in enumerate(samples):
        index = item("memlite_requested_index", i)
        if index is None:
            raise ValueError("Missing actual sampler source receipt")
        rows.append({"requested_index": int(index), "actual_raw_index": int(item("idx", i)),
                     "source_kind": item("memlite_source_kind", i), "branch": sample["memlite_branch"],
                     "locator": item("dataset_locator", i), "task": item("task", i)})
    path = Path(output_dir) / f"sample_receipts_rank{rank}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"step": int(step), "batch_index": int(batch_index), "samples": rows}) + "\n")
