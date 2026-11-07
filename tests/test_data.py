import pytest

from utils.config import ModelConfig, load_config
from utils.data import split_text


def test_split_is_contiguous_and_complete():
    text = "abcdefghij" * 10
    train, val = split_text(text, 0.9)
    assert train + val == text
    assert len(train) == 90 and len(val) == 10


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_text("abc", 1.5)


def test_model_config_validates_heads():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=10, n_head=3)


def test_default_config_loads():
    cfg = load_config("config/config.yaml")
    assert cfg.model.n_embd % cfg.model.n_head == 0