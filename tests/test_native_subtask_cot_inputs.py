import pytest
import torch

from probe_native_subtask_cot_inputs import observed_prefix_receipts


def rows():
    ids = torch.tensor([[0, 0, 10, 11, 20, 30], [12, 13, 21, 22, 23, 30]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[0, :4] = -100
    labels[1, :2] = -100
    prefix = torch.tensor([[10, 11], [12, 13]])
    return ids, labels, mask, prefix, torch.ones_like(prefix)


def test_variable_cot_output_padding_does_not_change_the_observed_prefix():
    result = observed_prefix_receipts(*rows())
    assert all(r["same_tokens"] and r["same_types"] and r["observation_labels_masked"] for r in result)
    assert result[0]["training_left_padding"] == 2 and result[0]["inference_left_padding"] == 0


@pytest.mark.parametrize("change", ["token", "mask", "label"])
def test_real_prefix_or_label_mismatch_is_not_hidden_by_unpadding(change):
    ids, labels, mask, prefix, pm = rows()
    if change == "token":
        ids[0, 3] = 99
    elif change == "mask":
        mask[0, 3] = 2
    else:
        labels[0, 3] = ids[0, 3]
    r = observed_prefix_receipts(ids, labels, mask, prefix, pm)[0]
    assert not (r["same_tokens"] and r["same_types"] and r["observation_labels_masked"])
