from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise MLP: FFN(x) = Dropout(W2 · ReLU(W1 · x + b1) + b2).

    Applied independently to every position. The hidden layer is 4x wider
    than n_embd (the ratio used in the original Transformer and GPT-2).
    Attention mixes information ACROSS positions; this layer processes each
    position on its own.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0, expansion: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion * n_embd),  # (B, T, n_embd) -> (B, T, 4*n_embd)
            nn.ReLU(),
            nn.Linear(expansion * n_embd, n_embd),  # back to (B, T, n_embd)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block (pre-norm): communication, then computation.

        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    LayerNorm normalizes each position's feature vector:
        LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    The residual (skip) connections give gradients a direct path to earlier
    layers, which is what makes deep stacks trainable.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # x: (B, T, n_embd) -> (B, T, n_embd)
        if return_weights:
            attn_out, weights = self.attn(self.ln1(x), return_weights=True)
        else:
            attn_out, weights = self.attn(self.ln1(x)), None
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return (x, weights) if return_weights else x