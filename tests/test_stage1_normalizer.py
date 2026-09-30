import torch
from g05.utils.data.normalizer import SingleFieldLinearNormalizer


def test_action_unclipping_preserves_coordinates_and_inverse_safety_bounds():
    stats = {k:torch.tensor([v]) for k,v in dict(mean=0.,std=.01,q01=-1.,q99=1.,min=-.2,max=.2).items()}
    legacy = SingleFieldLinearNormalizer(stats, mode="z-score-tail-bounded")
    extended = SingleFieldLinearNormalizer(stats, mode="z-score-tail-bounded")
    extended.forward_clip = None
    inside = torch.tensor([[.02]])
    outside = torch.tensor([[.08]])
    assert torch.equal(legacy.forward(inside),extended.forward(inside))
    assert legacy.forward(outside).item() == 5.
    assert extended.forward(outside).item() == 8.
    assert torch.allclose(extended.backward(extended.forward(outside)),outside)
    assert extended.backward(torch.tensor([[1000.]])).item() <= .200001
    assert torch.equal(legacy.scale,extended.scale)
    assert torch.equal(legacy.offset,extended.offset)
