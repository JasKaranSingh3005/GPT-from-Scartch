import pytest
import torch

from models.bigram import BigramLanguageModel
from models.gpt import GPT
from training.checkpoint import load_checkpoint, load_model, save_checkpoint

GPT_KWARGS = dict(vocab_size=5, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.1)
CHARS = ["a", "b", "c", "d", "e"]


def test_gpt_roundtrip_gives_identical_outputs(tmp_path):
    torch.manual_seed(0)
    model = GPT(**GPT_KWARGS).eval()
    path = save_checkpoint(
        tmp_path / "m.pt", model, model_type="gpt", model_kwargs=GPT_KWARGS,
        chars=CHARS, iteration=7, history=[{"iter": 0, "train_loss": 1.0}],
    )
    loaded, tokenizer, ckpt = load_model(path)
    idx = torch.randint(5, (2, 6))
    assert torch.allclose(model(idx)[0], loaded(idx)[0], atol=1e-6)
    assert tokenizer.chars == CHARS
    assert ckpt["iteration"] == 7 and ckpt["history"][0]["iter"] == 0
    assert not loaded.training


def test_bigram_roundtrip(tmp_path):
    model = BigramLanguageModel(vocab_size=5).eval()
    path = save_checkpoint(
        tmp_path / "b.pt", model, model_type="bigram", model_kwargs={"vocab_size": 5},
        chars=CHARS, iteration=0,
    )
    loaded, _, _ = load_model(path)
    idx = torch.randint(5, (2, 6))
    assert torch.allclose(model(idx)[0], loaded(idx)[0])


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_corrupt_checkpoint_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError):
        load_checkpoint(bad)