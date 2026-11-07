from __future__ import annotations


def gpt_parameter_breakdown(
    vocab_size: int, n_embd: int, n_layer: int, block_size: int
) -> dict[str, int]:
    """Parameter count per component, derived by hand from the architecture.

    Per layer: attention q,k,v (3C^2, no bias) + output projection (C^2 + C)
    = 4C^2 + C; feed-forward C*4C + 4C + 4C*C + C = 8C^2 + 5C; two LayerNorms 4C.
    The number of heads does not change the count: head_size = C / n_head.
    """
    C, V = n_embd, vocab_size
    return {
        "Token embedding": V * C,
        "Positional embedding": block_size * C,
        "Attention (all layers)": n_layer * (4 * C * C + C),
        "Feed-forward (all layers)": n_layer * (8 * C * C + 5 * C),
        "LayerNorm (all)": n_layer * 4 * C + 2 * C,
        "LM head": C * V + V,
    }


def gpt_parameter_count(vocab_size: int, n_embd: int, n_layer: int, block_size: int) -> int:
    return sum(gpt_parameter_breakdown(vocab_size, n_embd, n_layer, block_size).values())


def training_memory_estimate_mb(
    n_params: int, batch_size: int, block_size: int, n_layer: int, n_head: int
) -> dict[str, float]:
    """Rough LOWER BOUND of training memory in MB (float32).

    Weights (4 B) + gradients (4 B) + AdamW's two moment buffers (8 B) = 16 B per
    parameter. Attention matrices cost B*L*H*T^2*4 bytes for one copy; real runs keep
    several copies of these and many other activations for backprop, so treat the
    total as a floor, not a prediction.
    """
    mb = 1024**2
    weights = n_params * 4 / mb
    return {
        "Weights": weights,
        "Gradients": weights,
        "AdamW state": 2 * weights,
        "Attention matrices (one copy)": batch_size * n_layer * n_head * block_size**2 * 4 / mb,
    }


def valid_head_counts(n_embd: int, candidates: list[int]) -> list[int]:
    """Candidates that divide n_embd evenly (always includes 1 if offered)."""
    return sorted(h for h in set(candidates) if h >= 1 and n_embd % h == 0)