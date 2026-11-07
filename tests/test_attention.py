import pytest
import torch

from models.attention import (
    Head,
    MultiHeadAttention,
    causal_average_demo,
    scaled_dot_product_attention,
)


def test_causal_average_is_running_mean():
    x = torch.tensor([[[1.0], [3.0], [5.0]]])  # (1, 3, 1)
    out = causal_average_demo(x)
    assert torch.allclose(out.flatten(), torch.tensor([1.0, 2.0, 3.0]))


def test_sdpa_shapes_and_rows_sum_to_one():
    q = k = v = torch.randn(2, 6, 8)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 6, 8)
    assert w.shape == (2, 6, 6)
    assert torch.allclose(w.sum(-1), torch.ones(2, 6), atol=1e-5)


def test_causal_mask_zeroes_future_weights():
    q = k = v = torch.randn(3, 5, 4)
    _, w = scaled_dot_product_attention(q, k, v, causal=True)
    upper = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.all(w[:, upper] == 0)


def test_future_tokens_do_not_change_past_outputs():
    torch.manual_seed(0)
    mha = MultiHeadAttention(n_embd=16, n_head=4).eval()
    x = torch.randn(1, 6, 16)
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn(1, 2, 16)  # change only the last two positions
    assert torch.allclose(mha(x)[:, :4], mha(x2)[:, :4], atol=1e-5)


def test_head_output_shape():
    head = Head(n_embd=16, head_size=4)
    assert head(torch.randn(2, 7, 16)).shape == (2, 7, 4)


def test_multihead_shapes():
    mha = MultiHeadAttention(n_embd=16, n_head=4)
    out, w = mha(torch.randn(2, 7, 16), return_weights=True)
    assert out.shape == (2, 7, 16)
    assert w.shape == (2, 4, 7, 7)


def test_multihead_rejects_bad_divisibility():
    with pytest.raises(ValueError):
        MultiHeadAttention(n_embd=10, n_head=3)