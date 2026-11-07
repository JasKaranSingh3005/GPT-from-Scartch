from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from utils.device import get_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEVICE_CHOICES = ("auto", "cpu", "cuda")


def _int_env(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Backend settings. Environment variables (all optional):

    GPTLAB_DATA_DIR         artifact root containing runs/ (default: <project>/outputs)
    GPTLAB_DEVICE           auto | cpu | cuda (default: auto)
    GPTLAB_CORS_ORIGINS     comma-separated allowed origins
    GPTLAB_MAX_NEW_TOKENS   per-request generation cap (default 1000)
    GPTLAB_MAX_LOADED       models kept in memory at once (default 3)
    """

    data_dir: Path
    device: str = "auto"
    cors_origins: tuple[str, ...] = DEFAULT_ORIGINS
    max_new_tokens: int = 1000
    max_loaded_models: int = 3

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        data_dir = Path(env.get("GPTLAB_DATA_DIR") or PROJECT_ROOT / "outputs")
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir

        device = (env.get("GPTLAB_DEVICE") or "auto").strip().lower()
        if device not in DEVICE_CHOICES:
            raise ValueError(f"GPTLAB_DEVICE must be one of {DEVICE_CHOICES}, got {device!r}")

        raw_origins = env.get("GPTLAB_CORS_ORIGINS")
        origins = (
            tuple(o.strip() for o in raw_origins.split(",") if o.strip())
            if raw_origins
            else DEFAULT_ORIGINS
        )
        return cls(
            data_dir=data_dir,
            device=device,
            cors_origins=origins,
            max_new_tokens=_int_env(env, "GPTLAB_MAX_NEW_TOKENS", 1000, 1),
            max_loaded_models=_int_env(env, "GPTLAB_MAX_LOADED", 3, 1),
        )


def resolve_device(preference: str) -> torch.device:
    """Turn a setting into a real device. Never claims CUDA unless torch can use it."""
    pref = preference.lower()
    if pref == "auto":
        return get_device()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "GPTLAB_DEVICE=cuda but this PyTorch build cannot use CUDA "
                f"(torch {torch.__version__})"
            )
        return torch.device("cuda")
    raise ValueError(f"Unknown device preference {preference!r}")