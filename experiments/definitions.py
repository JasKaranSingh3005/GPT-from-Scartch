from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from utils.config import Config, ModelConfig

# Every GPT variant starts from THIS model, not from config.yaml, so results stay
# reproducible even if you edit the config. Each variant changes exactly one thing.
BASE_MODEL: dict[str, Any] = dict(n_embd=64, n_head=4, n_layer=4, block_size=64, dropout=0.1)


@dataclass
class Variant:
    name: str  # run folder name
    label: str  # shown in plots and tables
    model_type: str = "gpt"
    overrides: dict[str, Any] = field(default_factory=dict)  # changes to BASE_MODEL


@dataclass
class Experiment:
    key: str
    title: str
    variants: list[str]  # variant names, in plot order


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant("base", "GPT base (4 heads, 4 layers, n_embd 64, ctx 64, dropout 0.1)"),
        Variant("bigram", "Bigram", model_type="bigram"),
        Variant("heads_1", "1 head", overrides={"n_head": 1}),
        Variant("heads_2", "2 heads", overrides={"n_head": 2}),
        Variant("heads_8", "8 heads", overrides={"n_head": 8}),
        Variant("embd_32", "n_embd 32", overrides={"n_embd": 32}),
        Variant("embd_128", "n_embd 128", overrides={"n_embd": 128}),
        Variant("layers_1", "1 layer", overrides={"n_layer": 1}),
        Variant("layers_2", "2 layers", overrides={"n_layer": 2}),
        Variant("ctx_16", "context 16", overrides={"block_size": 16}),
        Variant("ctx_32", "context 32", overrides={"block_size": 32}),
        Variant("ctx_128", "context 128", overrides={"block_size": 128}),
        Variant("drop_0", "dropout 0.0", overrides={"dropout": 0.0}),
        Variant("drop_30", "dropout 0.3", overrides={"dropout": 0.3}),
    ]
}

# "base" appears in every GPT experiment but is trained only once.
EXPERIMENTS: dict[str, Experiment] = {
    e.key: e
    for e in [
        Experiment("bigram_vs_gpt", "Bigram vs GPT", ["bigram", "base"]),
        Experiment("heads", "Number of attention heads (single vs multi-head)",
                   ["heads_1", "heads_2", "base", "heads_8"]),
        Experiment("embedding", "Embedding dimension", ["embd_32", "base", "embd_128"]),
        Experiment("layers", "Number of Transformer layers", ["layers_1", "layers_2", "base"]),
        Experiment("context", "Context length (block_size)",
                   ["ctx_16", "ctx_32", "base", "ctx_128"]),
        Experiment("dropout", "Dropout", ["drop_0", "base", "drop_30"]),
    ]
}


def selected_variants(experiment_keys: list[str] | None = None) -> list[str]:
    """Variant names for the chosen experiments (all if None), in order, without duplicates."""
    keys = experiment_keys or list(EXPERIMENTS)
    unknown = [k for k in keys if k not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiment(s) {unknown}; choose from {list(EXPERIMENTS)}")
    names: list[str] = []
    for key in keys:
        for name in EXPERIMENTS[key].variants:
            if name not in names:
                names.append(name)
    return names


def build_config(
    base: Config, variant: Variant, max_iters: int, eval_iters: int = 50
) -> Config:
    """Full config for one variant. Data, batch size, seed come from config.yaml.

    ModelConfig validates n_embd % n_head == 0, so a bad variant fails here, early.
    The bigram uses lr 1e-2 (same choice as train.py); GPTs use the config's learning rate.
    """
    model = ModelConfig(**{**BASE_MODEL, **variant.overrides})
    lr = 1e-2 if variant.model_type == "bigram" else base.training.learning_rate
    training = replace(
        base.training,
        max_iters=max_iters,
        eval_interval=max(1, max_iters // 10),
        eval_iters=eval_iters,
        learning_rate=lr,
    )
    return replace(base, model=model, training=training)