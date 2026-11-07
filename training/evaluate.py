from __future__ import annotations

import torch
import torch.nn as nn

from utils.data import get_batch


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device | str,
) -> dict[str, float]:
    """Average loss over `eval_iters` random batches for each split.

    Averaging many batches gives a much less noisy estimate than the loss of
    a single training batch. Dropout is disabled during evaluation, and the
    model's previous train/eval mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    try:
        out: dict[str, float] = {}
        for name, data in splits.items():
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(data, block_size, batch_size, device)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[name] = losses.mean().item()
    finally:
        model.train(was_training)
    return out