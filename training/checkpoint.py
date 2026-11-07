from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.bigram import BigramLanguageModel
from models.gpt import GPT
from models.tokenizer import CharTokenizer
from utils.config import ModelConfig

MODEL_TYPES = ("gpt", "bigram")
_REQUIRED_KEYS = {"model_type", "model_kwargs", "chars", "iteration", "state_dict"}


def model_kwargs_from_config(
    model_type: str, model_cfg: ModelConfig, vocab_size: int
) -> dict[str, Any]:
    """Constructor arguments for a model type, as plain Python values."""
    if model_type == "gpt":
        return {
            "vocab_size": vocab_size,
            "n_embd": model_cfg.n_embd,
            "n_head": model_cfg.n_head,
            "n_layer": model_cfg.n_layer,
            "block_size": model_cfg.block_size,
            "dropout": model_cfg.dropout,
        }
    if model_type == "bigram":
        return {"vocab_size": vocab_size}
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")


def create_model(model_type: str, model_kwargs: dict[str, Any]) -> nn.Module:
    if model_type == "gpt":
        return GPT(**model_kwargs)
    if model_type == "bigram":
        return BigramLanguageModel(**model_kwargs)
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_type: str,
    model_kwargs: dict[str, Any],
    chars: list[str],
    iteration: int,
    history: list[dict[str, float]] | None = None,
) -> Path:
    """Save weights plus everything needed to rebuild the model and tokenizer.

    Written to a temporary file first and then renamed, so an interrupted
    save can't leave a corrupted checkpoint behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "chars": list(chars),
        "iteration": iteration,
        "history": history or [],
        "state_dict": model.state_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        # weights_only=True refuses to unpickle arbitrary code from the file
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint {path}: {exc}") from exc
    missing = _REQUIRED_KEYS - set(ckpt)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
    return ckpt


def load_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, CharTokenizer, dict[str, Any]]:
    """Rebuild a model and its tokenizer from a checkpoint (model is in eval mode)."""
    ckpt = load_checkpoint(path, device)
    model = create_model(ckpt["model_type"], ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, CharTokenizer(ckpt["chars"]), ckpt