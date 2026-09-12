import json
from types import SimpleNamespace

import pytest
import torch

from g05.utils.training.memlite_receipts import record_batch_receipt


def test_receipt_records_actual_branch_source_and_raw_identity(tmp_path):
    b = {"samples": [{"memlite_branch": "high"}, {"memlite_branch": "low"}],
         "memlite_requested_index": torch.tensor([100, 999]), "idx": torch.tensor([120, 999]),
         "memlite_source_kind": ["original", "recovery"], "dataset_locator": ["episode=1 frame=20", "recovery=1 step=96"],
         "task": ["radio", "radio"]}
    record_batch_receipt(b, tmp_path, rank=3, step=5, batch_index=4)
    r = json.loads((tmp_path / "sample_receipts_rank3.jsonl").read_text())
    assert r["step"] == 5 and r["samples"][0]["requested_index"] == 100
    assert r["samples"][0]["actual_raw_index"] == 120
    assert r["samples"][1]["source_kind"] == "recovery"
    b.pop("memlite_requested_index")
    with pytest.raises(ValueError, match="Missing"):
        record_batch_receipt(b, tmp_path, rank=3, step=6, batch_index=5)
