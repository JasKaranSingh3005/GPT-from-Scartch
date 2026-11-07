from __future__ import annotations

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
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
) -> torch.Tensor:
    """Sample one token ID per row from next-token logits.

    logits: (B, V) -> returns (B, 1) integer tensor.
    """
    probs = next_token_probs(logits, temperature, top_k)
    return torch.multinomial(probs, num_samples=1)


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