import pytest
import torch
from models.bigram import BigramLanguageModel
from models.gpt import GPT
from models.sampling import generate_text, sample_next_token
from models.tokenizer import CharTokenizer
from training.checkpoint import load_model, save_checkpoint

V, BS = 7, 8  # vocab matches CharTokenizer.from_text("abcdef\n")
GPT_KWARGS = dict(vocab_size=V, n_embd=16, n_head=2, n_layer=2, block_size=BS, dropout=0.0)
TOKENIZER = CharTokenizer.from_text("abcdef\n")
LOGITS = torch.tensor([[0.0, 1.0, 3.0], [5.0, 0.0, 1.0]])  # argmax = [2, 0]


def make_gpt(dropout: float = 0.0) -> GPT:
    torch.manual_seed(0)
    return GPT(**{**GPT_KWARGS, "dropout": dropout})


# ---- sampler ----------------------------------------------------------------

def test_top_k_never_samples_outside_top_k():
    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0, -1.0]]).repeat(500, 1)
    torch.manual_seed(0)
    picks = sample_next_token(logits, top_k=2)
    assert picks.shape == (500, 1)
    assert set(picks.flatten().tolist()) == {0, 1}


def test_top_k_one_is_argmax():
    picks = sample_next_token(LOGITS, top_k=1)
    assert picks.flatten().tolist() == [2, 0]


def test_very_low_temperature_is_greedy():
    torch.manual_seed(0)
    picks = sample_next_token(LOGITS, temperature=0.01)
    assert picks.flatten().tolist() == [2, 0]


def test_top_k_larger_than_vocab_is_allowed():
    assert sample_next_token(LOGITS, top_k=1000).shape == (2, 1)


def test_invalid_sampling_arguments_raise():
    with pytest.raises(ValueError):
        sample_next_token(LOGITS, temperature=0.0)
    with pytest.raises(ValueError):
        sample_next_token(LOGITS, top_k=0)


# ---- GPT.generate -----------------------------------------------------------

def test_generate_length_and_token_range():
    out = make_gpt().generate(torch.zeros((2, 1), dtype=torch.long), 12, top_k=3)
    assert out.shape == (2, 13)
    assert out.min() >= 0 and out.max() < V


def test_prompt_longer_than_block_size_is_cropped():
    model = make_gpt()
    prompt = torch.randint(V, (1, 20))  # 20 > block_size (8)
    n = 5
    long_out = model.generate(prompt, n, top_k=1)
    short_out = model.generate(prompt[:, -BS:], n, top_k=1)
    assert long_out.shape == (1, 25)
    assert torch.equal(long_out[:, :20], prompt)  # the prompt is kept intact
    # Greedy decoding only ever sees the last block_size tokens, so both agree.
    assert torch.equal(long_out[:, -n:], short_out[:, -n:])


def test_greedy_generation_is_deterministic():
    model = make_gpt()
    start = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(1)
    a = model.generate(start, 10, top_k=1)
    torch.manual_seed(2)
    b = model.generate(start, 10, top_k=1)
    assert torch.equal(a, b)


def test_dropout_is_disabled_and_mode_restored():
    model = make_gpt(dropout=0.5)
    model.train()
    start = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(1)
    a = model.generate(start, 10, top_k=1)
    torch.manual_seed(2)
    b = model.generate(start, 10, top_k=1)
    assert torch.equal(a, b)  # active dropout would make greedy outputs differ
    assert model.training  # previous mode restored


def test_empty_prompt_tensor_raises():
    with pytest.raises(ValueError):
        make_gpt().generate(torch.zeros((1, 0), dtype=torch.long), 5)


# ---- generate_text ----------------------------------------------------------

def test_generate_text_is_reproducible_with_seed():
    model = make_gpt()
    a = generate_text(model, TOKENIZER, "ab", 10, seed=3)
    b = generate_text(model, TOKENIZER, "ab", 10, seed=3)
    assert a == b and len(a) == 12 and a.startswith("ab")
    assert set(a) <= set(TOKENIZER.chars)


def test_generate_text_empty_prompt_starts_from_newline():
    text = generate_text(make_gpt(), TOKENIZER, "", 10, seed=0)
    assert len(text) == 11 and text[0] == "\n"


def test_generate_text_rejects_unknown_prompt_characters():
    with pytest.raises(ValueError):
        generate_text(make_gpt(), TOKENIZER, "xyz", 5)


def test_generate_text_works_with_bigram():
    model = BigramLanguageModel(vocab_size=V)
    text = generate_text(model, TOKENIZER, "a", 8, top_k=3, seed=0)
    assert len(text) == 9


def test_checkpoint_to_text(tmp_path):
    model = make_gpt()
    path = save_checkpoint(
        tmp_path / "g.pt", model, model_type="gpt", model_kwargs=GPT_KWARGS,
        chars=TOKENIZER.chars, iteration=1,
    )
    loaded, tokenizer, _ = load_model(path)
    text = generate_text(loaded, tokenizer, "ab", 15, temperature=0.8, top_k=5, seed=0)
    assert len(text) == 17 and text.startswith("ab")
    # The loaded model produces the same text as the original under the same seed.
    assert text == generate_text(model, TOKENIZER, "ab", 15, temperature=0.8, top_k=5, seed=0)