from __future__ import annotations
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tokenizer import CharTokenizer


def next_token_probs(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> torch.Tensor:
    """Turn next-token logits (B, V) into the sampling distribution (B, V).

    Temperature rescales the logits before the softmax:
        p_i = exp(z_i / T) / sum_j exp(z_j / T)
    T < 1 sharpens the distribution (safer, more repetitive); T > 1 flattens
    it (more surprising, more errors). Top-k keeps only the k highest logits
    and sets the rest to -inf, so they get probability exactly 0.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 (or None to disable)")

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k).values[:, [-1]]  # (B, 1) smallest kept logit
        # Strict '<' keeps ties with the k-th value, so very rarely more than k survive.
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    return F.softmax(logits, dim=-1)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token ID per row from next-token logits.

    logits: (B, V) -> returns (B, 1) integer tensor. Pass a `generator` for a private
    random stream; with None, the global PyTorch RNG is used (the original behaviour).
    """
    probs = next_token_probs(logits, temperature, top_k)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def encode_prompt(tokenizer: CharTokenizer, prompt: str) -> list[int]:
    """Prompt -> token IDs. An empty prompt becomes a newline (or token 0 if
    the vocabulary has no newline). Raises ValueError on unknown characters."""
    if prompt:
        return tokenizer.encode(prompt)
    return tokenizer.encode("\n") if "\n" in tokenizer.stoi else [0]


def generate_text(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str = "",
    max_new_tokens: int = 300,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> str:
    """Prompt string in, prompt + continuation out."""
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.tensor([encode_prompt(tokenizer, prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())

@torch.no_grad()
def generate_stream(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[int]:
    """Yield each newly sampled token ID (single sequence, idx of shape (1, T)).

    Same algorithm as GPT.generate: crop the context to block_size (if the model has
    one), take the last position's logits, sample, append. The model is switched to
    eval mode and its previous mode restored when the stream ends or is closed.
    """
    if idx.ndim != 2 or idx.size(0) != 1 or idx.size(1) < 1:
        raise ValueError("idx must have shape (1, T) with T >= 1")
    # Validate eagerly (a generator function would otherwise only fail on first next()).
    next_token_probs(torch.zeros(1, 2), temperature, top_k)
    return _stream(model, idx, max_new_tokens, temperature, top_k, generator)


def _stream(model, idx, max_new_tokens, temperature, top_k, generator) -> Iterator[int]:
    block_size = getattr(model, "block_size", None)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            cond = idx[:, -block_size:] if block_size else idx
            logits, _ = model(cond)
            nxt = sample_next_token(logits[:, -1, :], temperature, top_k, generator)
            idx = torch.cat([idx, nxt], dim=1)
            yield int(nxt.item())
    finally:
        model.train(was_training)