from copy import deepcopy
from pathlib import Path

import pytest

def test_exact_sampler_delivery_and_source_substitution_rejection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_memlite_fm_v11 import compare_delivery
    planned = [[2,7], [5,9]]
    delivered = [{"step": s+1, "batch_index": s, "samples": [
        {"requested_index": i, "actual_raw_index": i+100, "source_kind": "original",
         "branch": "low", "locator": f"episode/frame {i}"} for i in batch]}
        for s, batch in enumerate(planned)]
    resolve = lambda i: ("original", i+100)
    compare_delivery(delivered, planned, resolve)
    for key, value in (("requested_index", 999), ("actual_raw_index", 777),
                       ("source_kind", "recovery"), ("branch", "high"), ("locator", "")):
        changed = deepcopy(delivered)
        changed[0]["samples"][0][key] = value
        with pytest.raises(ValueError):
            compare_delivery(changed, planned, resolve)
    with pytest.raises(ValueError):
        compare_delivery(delivered[:-1], planned, resolve)
    delivered[1]["batch_index"] = 0
    with pytest.raises(ValueError):
        compare_delivery(delivered, planned, resolve)
