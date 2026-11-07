from __future__ import annotations

from typing import Iterable


class CharTokenizer:
    """Character-level tokenizer: every distinct character is one token."""

    def __init__(self, chars: Iterable[str]) -> None:
        self.chars: list[str] = sorted(set(chars))
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("Cannot build a vocabulary from empty text")
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """String -> list of token IDs."""
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            raise ValueError(f"Characters not in vocabulary: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        """List of token IDs -> string."""
        try:
            return "".join(self.itos[int(i)] for i in ids)
        except KeyError as exc:
            raise ValueError(f"Token ID {exc.args[0]} is not in vocabulary") from exc