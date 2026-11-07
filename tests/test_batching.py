import pytest
import torch

from utils.data import get_batch


def test_batch_shapes_and_shift():
    data = torch.arange(200)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y, x + 1)  # targets are inputs shifted by one


def test_batch_too_short_raises():
    with pytest.raises(ValueError):
        get_batch(torch.arange(5), block_size=8, batch_size=2)