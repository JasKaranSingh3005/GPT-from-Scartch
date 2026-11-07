import pytest
import torch

from models.gpt import GPT
from models.sampling import generate_stream, sample_next_token

KW = dict(vocab_size=7, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0)


def model() -> GPT:
    torch.manual_seed(0)
    return GPT(**KW).eval()


def test_stream_matches_generate_under_the_same_seed():
    m = model()
    start = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(5)
    expected = m.generate(start, 20, temperature=0.9, top_k=4)[0, 1:].tolist()
    torch.manual_seed(5)
    assert list(generate_stream(m, start, 20, temperature=0.9, top_k=4)) == expected


def test_stream_crops_long_prompts_like_generate():
    m = model()
    prompt = torch.randint(7, (1, 20))
    torch.manual_seed(1)
    expected = m.generate(prompt, 10, top_k=1)[0, 20:].tolist()
    assert list(generate_stream(m, prompt, 10, top_k=1)) == expected


def test_private_generator_is_reproducible_and_leaves_global_rng_alone():
    m = model()
    start = torch.zeros((1, 1), dtype=torch.long)
    a = list(generate_stream(m, start, 15, generator=torch.Generator().manual_seed(3)))
    torch.manual_seed(999)
    before = torch.get_rng_state()
    b = list(generate_stream(m, start, 15, generator=torch.Generator().manual_seed(3)))
    assert a == b
    assert torch.equal(before, torch.get_rng_state())


def test_different_seeds_differ():
    m = model()
    start = torch.zeros((1, 1), dtype=torch.long)
    a = list(generate_stream(m, start, 30, generator=torch.Generator().manual_seed(1)))
    b = list(generate_stream(m, start, 30, generator=torch.Generator().manual_seed(2)))
    assert a != b


def test_stream_validates_eagerly_and_restores_mode():
    m = model()
    start = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ValueError):
        generate_stream(m, start, 5, temperature=0.0)  # raises before iterating
    with pytest.raises(ValueError):
        generate_stream(m, torch.zeros((2, 1), dtype=torch.long), 5)
    m.train()
    stream = generate_stream(m, start, 5)
    next(stream)
    stream.close()
    assert m.training  # restored even when closed early


def test_sample_next_token_default_behaviour_unchanged():
    logits = torch.tensor([[0.0, 1.0, 3.0]])
    assert sample_next_token(logits, top_k=1).item() == 2