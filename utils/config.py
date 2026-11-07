from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    url: str
    path: str = "data/input.txt"
    train_fraction: float = 0.9


@dataclass
class ModelConfig:
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_iters: int = 3000
    eval_interval: int = 300
    eval_iters: int = 100
    seed: int = 1337


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _build(cls: type, raw: dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path = "config/config.yaml") -> Config:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        data=_build(DataConfig, raw["data"]),
        model=_build(ModelConfig, raw["model"]),
        training=_build(TrainingConfig, raw["training"]),
    )