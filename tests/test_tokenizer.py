import pytest

from models.tokenizer import CharTokenizer


def test_roundtrip():
    text = "Hello, world!\nSecond line."
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size_and_order():
    tok = CharTokenizer.from_text("baca")
    assert tok.chars == ["a", "b", "c"]
    assert tok.vocab_size == 3
    assert tok.encode("cab") == [2, 0, 1]


def test_unknown_character_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("abz")


def test_unknown_id_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.decode([0, 99])


def test_empty_text_raises():
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")