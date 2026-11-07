import torch

from models.transformer import Block, FeedForward


def test_feedforward_shape_and_hidden_width():
    ffn = FeedForward(n_embd=16)
    assert ffn(torch.randn(2, 5, 16)).shape == (2, 5, 16)
    assert ffn.net[0].out_features == 64  # 4x expansion


def test_block_forward_shape():
    block = Block(n_embd=16, n_head=4)
    assert block(torch.randn(2, 7, 16)).shape == (2, 7, 16)


def test_block_returns_attention_weights():
    block = Block(n_embd=16, n_head=4)
    out, w = block(torch.randn(2, 7, 16), return_weights=True)
    assert out.shape == (2, 7, 16)
    assert w.shape == (2, 4, 7, 7)


def test_block_is_identity_when_sublayer_outputs_are_zero():
    """Residual connections: if both sublayers output 0, the block returns x."""
    block = Block(n_embd=16, n_head=4).eval()
    with torch.no_grad():
        for layer in (block.attn.proj, block.ffn.net[2]):
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(2, 6, 16)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_block_is_causal():
    """Changing future positions must not change earlier outputs."""
    torch.manual_seed(0)
    block = Block(n_embd=16, n_head=4).eval()
    x = torch.randn(1, 6, 16)
    x2 = x.clone()
    x2[:, 4:, :] = torch.randn(1, 2, 16)
    assert torch.allclose(block(x)[:, :4], block(x2)[:, :4], atol=1e-5)


def test_gradients_reach_input():
    block = Block(n_embd=16, n_head=4)
    x = torch.randn(2, 5, 16, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()  