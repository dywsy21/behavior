from copy import deepcopy

import pytest

from probe_ar_schema_reference import validate_paired_sources


def fixture():
    candidates = [(f"{split}_task{task}", None, dict(task=task, split=split, episode=10 + task))
                  for split in ("train", "heldout") for task in range(5)]
    reference = dict(complete=True, actual_generations=10, complete_generations=10,
        rows=[dict(label=label, source=deepcopy(source), valid=True) for label, _, source in candidates])
    return candidates, reference


def test_pairing_requires_exact_ten_original_sources():
    validate_paired_sources(*fixture())


@pytest.mark.parametrize("damage", ["source", "split", "order", "count", "invalid"])
def test_wrong_window_or_incomplete_run_cannot_be_a_paired_reference(damage):
    candidates, reference = fixture()
    if damage == "source":
        reference["rows"][0]["source"]["episode"] += 1
    elif damage == "split":
        reference["rows"][0]["source"]["split"] = "heldout"
    elif damage == "order":
        candidates[0], candidates[1] = candidates[1], candidates[0]
    elif damage == "count":
        reference["actual_generations"] = 9
    else:
        reference["rows"][0]["valid"] = False
    with pytest.raises(ValueError):
        validate_paired_sources(candidates, reference)
