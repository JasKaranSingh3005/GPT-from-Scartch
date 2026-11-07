from __future__ import annotations
from models.sampling import sample_next_token
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import Block
from utils.config import ModelConfig


class GPT(nn.Module):
    """Decoder-only Transformer language model.

    Pipeline (B = batch, T = sequence length, C = n_embd, V = vocab_size):
        idx (B, T)
          -> token embedding (B, T, C) + positional embedding (T, C)
          -> N x Block
          -> final LayerNorm
          -> linear head -> logits (B, T, V)
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        if min(vocab_size, n_layer, block_size) < 1:
            raise ValueError("vocab_size, n_layer and block_size must be >= 1")

        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)  # learned, one row per position
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, cfg: ModelConfig, vocab_size: int) -> "GPT":
        return cls(
            vocab_size=vocab_size,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # N(0, 0.02) is the GPT-2 convention. Small weights make the initial
        # logits near 0, so the initial loss is close to ln(vocab_size).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_weights: bool = False,
    ):
        """Returns (logits, loss), plus a list of per-layer attention weights
        (each (B, n_head, T, T)) when return_weights=True."""
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        positions = torch.arange(T, device=idx.device)  # (T,)
        x = self.token_embedding(idx) + self.position_embedding(positions)  # (B, T, C)
        x = self.drop(x)

        all_weights = []
        for block in self.blocks:
            if return_weights:
                x, w = block(x, return_weights=True)
                all_weights.append(w)
            else:
                x = block(x)

        logits = self.lm_head(self.ln_f(x))  # (B, T, V)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))

        if return_weights:
            return logits, loss, all_weights
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend idx (B, T) by max_new_tokens tokens.

        Each step feeds the model the LAST block_size tokens only (the
        positional embedding table has no rows beyond that), takes the logits
        at the final position, and samples the next token. Dropout is switched
        off during generation and the previous train/eval mode is restored.
        """
        if idx.ndim != 2 or idx.size(1) < 1:
            raise ValueError("idx must have shape (B, T) with T >= 1")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size :]  # crop context: (B, <=block_size)
                logits, _ = self(idx_cond)
                idx_next = sample_next_token(logits[:, -1, :], temperature, top_k)
                idx = torch.cat([idx, idx_next], dim=1)
        finally:
            self.train(was_training)
        return idx