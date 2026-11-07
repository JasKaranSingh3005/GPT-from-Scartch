from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_average_demo(x: torch.Tensor) -> torch.Tensor:
    """The 'mathematical trick': each position averages itself and all earlier ones.

    A lower-triangular matrix of ones, row-normalized, multiplied by x gives a
    running mean. Self-attention generalizes this by LEARNING the weights.
    x: (B, T, C) -> (B, T, C)
    """
    T = x.size(1)
    tril = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
    weights = tril / tril.sum(dim=1, keepdim=True)  # each row sums to 1
    return weights @ x  # (T, T) @ (B, T, C) -> (B, T, C)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + M) V

    q, k, v: (..., T, d_k). Returns (output (..., T, d_k), weights (..., T, T)).
    M is 0 where a position may attend (j <= i) and -inf where it may not (j > i).
    Written manually; does not call F.scaled_dot_product_attention.
    """
    d_k = q.size(-1)
    T = q.size(-2)
    # Scaling by sqrt(d_k) keeps score variance ~1 so softmax doesn't saturate.
    scores = (q @ k.transpose(-2, -1)) / (d_k**0.5)  # (..., T, T)
    if causal:
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd: int, head_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.last_weights: torch.Tensor | None = None  # kept for the visualizer

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # x: (B, T, n_embd) -> out: (B, T, head_size)
        q, k, v = self.query(x), self.key(x), self.value(x)
        out, weights = scaled_dot_product_attention(q, k, v, causal=True, dropout=self.dropout)
        self.last_weights = weights.detach()
        return (out, weights) if return_weights else out


class MultiHeadAttention(nn.Module):
    """Several independent heads run in parallel, concatenated, then projected.

    Implemented as a loop over Head modules for clarity (educational), not
    speed. head_size = n_embd // n_head, so concatenation restores n_embd.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # x: (B, T, n_embd)
        outs, all_weights = [], []
        for head in self.heads:
            out, w = head(x, return_weights=True)
            outs.append(out)
            all_weights.append(w)
        out = torch.cat(outs, dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))
        if return_weights:
            return out, torch.stack(all_weights, dim=1)  # (B, n_head, T, T)
        return out