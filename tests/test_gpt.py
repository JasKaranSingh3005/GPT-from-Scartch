import math

import pytest
import torch

from models.gpt import GPT
from utils.config import ModelConfig

V, C, H, L, BS = 11, 16, 4, 2, 8


def make_model(**overrides) -> GPT:
    kwargs = dict(vocab_size=V, n_embd=C, n_head=H, n_layer=L, block_size=BS, dropout=0.0)
    kwargs.update(overrides)
    return GPT(**kwargs)


def test_output_shapes_and_loss():
    model = make_model()
    idx = torch.randint(V, (3, 5))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 5, V)
    assert loss is not None and loss.ndim == 0


def test_no_targets_means_no_loss():
    _, loss = make_model()(torch.randint(V, (2, 4)))
    assert loss is None


def test_initial_loss_is_near_log_vocab_size():
    torch.manual_seed(0)
    model = make_model()
    idx = torch.randint(V, (16, BS))
    _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(V)) < 0.3


def test_sequence_longer_than_block_size_raises():
    with pytest.raises(ValueError):
        make_model()(torch.randint(V, (1, BS + 1)))


def test_parameter_count_matches_formula():
    # tok emb + pos emb + L blocks + final LN + LM head (with bias)
    # per block: 12*C^2 + 10*C (qkv 3C^2, proj C^2+C, FFN 8C^2+5C, two LNs 4C)
    expected = V * C + BS * C + L * (12 * C**2 + 10 * C) + 2 * C + (C * V + V)
    assert make_model().num_parameters() == expected


def test_single_head_is_supported():
    model = make_model(n_head=1)
    logits, _, weights = model(torch.randint(V, (2, 5)), return_weights=True)
    assert logits.shape == (2, 5, V)
    assert weights[0].shape == (2, 1, 5, 5)


def test_return_weights_one_entry_per_layer():
    _, _, weights = make_model()(torch.randint(V, (2, 6)), return_weights=True)
    assert len(weights) == L
    assert all(w.shape == (2, H, 6, 6) for w in weights)


def test_model_is_causal():
    torch.manual_seed(0)
    model = make_model().eval()
    idx = torch.randint(V, (1, 6))
    idx2 = idx.clone()
    idx2[:, 4:] = (idx2[:, 4:] + 1) % V  # change only the last two tokens
    out1, _ = model(idx)
    out2, _ = model(idx2)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)


def test_can_overfit_a_single_batch():
    torch.manual_seed(0)
    model = make_model()
    x = torch.randint(V, (4, BS))
    y = torch.randint(V, (4, BS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = model(x, y)[1].item()
    for _ in range(60):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.9 * first


def test_from_config():
    cfg = ModelConfig(n_embd=16, n_head=2, n_layer=3, block_size=10, dropout=0.0)
    model = GPT.from_config(cfg, vocab_size=V)
    assert len(model.blocks) == 3 and model.block_size == 10


def test_bad_head_count_raises():
    with pytest.raises(ValueError):
        make_model(n_embd=10, n_head=3)