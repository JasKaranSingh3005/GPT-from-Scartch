from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml

from app.components.common import current_config
from models.gpt import GPT
from utils.config import ModelConfig, TrainingConfig
from utils.estimate import (
    gpt_parameter_breakdown,
    gpt_parameter_count,
    training_memory_estimate_mb,
    valid_head_counts,
)

N_EMBD_OPTIONS = [16, 32, 64, 96, 128, 192, 256, 384, 512]
HEAD_OPTIONS = [1, 2, 3, 4, 6, 8, 12, 16]
BLOCK_OPTIONS = [16, 32, 64, 128, 256, 512]
BATCH_OPTIONS = [8, 16, 32, 64, 128]
LR_OPTIONS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def _options(base: list, value) -> list:
    return sorted(set(base) | {value})


def build_train_command(name: str, model: ModelConfig, training: TrainingConfig) -> str:
    """The terminal command that reproduces this configuration."""
    return " ".join(
        [
            "python train.py --model gpt",
            f"--name {name}",
            f"--n-embd {model.n_embd}",
            f"--n-head {model.n_head}",
            f"--n-layer {model.n_layer}",
            f"--block-size {model.block_size}",
            f"--dropout {model.dropout:g}",
            f"--batch-size {training.batch_size}",
            f"--lr {training.learning_rate:g}",
            f"--max-iters {training.max_iters}",
        ]
    )


def render_model_configuration() -> None:
    st.title("Model Configuration")
    st.markdown(
        "Adjust the hyperparameters and see what they cost. The values you choose here are "
        "remembered by the **Training Dashboard** for short in-app runs, and the command at "
        "the bottom reproduces them from the terminal."
    )
    try:
        base = current_config()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load config: {exc}")
        return
    m, t = base.model, base.training

    vocab = int(st.number_input("Vocabulary size (Tiny Shakespeare has 65)", 2, 5000, 65, key="cfg_vocab"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model")
        embd_opts = _options(N_EMBD_OPTIONS, m.n_embd)
        n_embd = st.selectbox("Embedding dimension (n_embd)", embd_opts,
                              index=embd_opts.index(m.n_embd), key="cfg_n_embd")
        heads = valid_head_counts(n_embd, _options(HEAD_OPTIONS, m.n_head))
        head_default = heads.index(m.n_head) if m.n_head in heads else 0
        # the key includes n_embd so a stale choice can never be invalid for a new n_embd
        n_head = st.selectbox("Attention heads (n_head)", heads, index=head_default,
                              key=f"cfg_n_head_{n_embd}")
        n_layer = st.slider("Transformer layers (n_layer)", 1, max(12, m.n_layer), m.n_layer,
                            key="cfg_n_layer")
        block_opts = _options(BLOCK_OPTIONS, m.block_size)
        block_size = st.selectbox("Context length (block_size)", block_opts,
                                  index=block_opts.index(m.block_size), key="cfg_block")
        dropout = st.slider("Dropout", 0.0, 0.5, min(m.dropout, 0.5), step=0.05, key="cfg_dropout")
    with col2:
        st.subheader("Training")
        batch_opts = _options(BATCH_OPTIONS, t.batch_size)
        batch_size = st.selectbox("Batch size", batch_opts, index=batch_opts.index(t.batch_size),
                                  key="cfg_batch")
        lr = st.select_slider("Learning rate", options=_options(LR_OPTIONS, t.learning_rate),
                              value=t.learning_rate, format_func=lambda v: f"{v:g}", key="cfg_lr")
        max_iters = int(st.number_input("Training iterations", 100, 50000, t.max_iters,
                                        step=100, key="cfg_iters"))

    model_cfg = ModelConfig(n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                            block_size=block_size, dropout=dropout)
    train_cfg = TrainingConfig(batch_size=batch_size, learning_rate=lr, max_iters=max_iters,
                               eval_interval=t.eval_interval, eval_iters=t.eval_iters, seed=t.seed)
    st.session_state["cfg_overrides"] = {
        "model": {"n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
                  "block_size": block_size, "dropout": dropout},
        "training": {"batch_size": batch_size, "learning_rate": lr, "max_iters": max_iters},
    }

    breakdown = gpt_parameter_breakdown(vocab, n_embd, n_layer, block_size)
    n_params = gpt_parameter_count(vocab, n_embd, n_layer, block_size)
    mem = training_memory_estimate_mb(n_params, batch_size, block_size, n_layer, n_head)

    st.subheader("What this configuration costs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Parameters", f"{n_params:,}")
    c2.metric("Head size (n_embd / n_head)", n_embd // n_head)
    c3.metric("Memory floor (MB)", f"{sum(mem.values()):,.0f}")

    left, right = st.columns(2)
    with left:
        df = pd.DataFrame({"component": list(breakdown), "parameters": list(breakdown.values())})
        df["share"] = (100 * df["parameters"] / n_params).round(1).astype(str) + " %"
        st.dataframe(df, hide_index=True)
        st.bar_chart(df.set_index("component")["parameters"])
    with right:
        st.dataframe(pd.DataFrame({"part": list(mem), "MB": [round(v, 1) for v in mem.values()]}),
                     hide_index=True)
        st.caption("Memory is a lower bound: backpropagation keeps many more activations than "
                   "the single attention copy counted here.")

    if st.button("Verify by building the model", key="cfg_verify"):
        real = GPT.from_config(model_cfg, vocab).num_parameters()
        if real == n_params:
            st.success(f"A real GPT with this configuration has {real:,} parameters, "
                       "matching the estimate.")
        else:
            st.error(f"Mismatch: estimate {n_params:,} vs real model {real:,}.")

    if n_params > 5_000_000:
        st.warning("Over 5M parameters: training on a CPU will be slow. A GPU is recommended.")
    if block_size ** 2 * n_layer * n_head * batch_size > 2e8:
        st.warning("Attention memory grows with block_size squared; this setting may not fit "
                   "on a small GPU.")
    st.info("Tiny Shakespeare is only about 1.1M characters, so large models tend to overfit. "
            "Compare train and validation loss on the Training Dashboard.")

    st.subheader("Reproduce from the terminal")
    st.code(build_train_command("my_run", model_cfg, train_cfg), language="bash")
    st.caption("Or put these values in config/config.yaml:")
    st.code(
        yaml.safe_dump(
            {"model": st.session_state["cfg_overrides"]["model"],
             "training": {"batch_size": batch_size, "learning_rate": lr, "max_iters": max_iters}},
            sort_keys=False,
        ),
        language="yaml",
    )