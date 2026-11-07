from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from app.components.common import (  # noqa: E402
    load_checkpoint_cached,
    model_device,
    pick_checkpoint,
)
from app.components.tokenizer_page import visible_char  # noqa: E402
from models.tokenizer import CharTokenizer  # noqa: E402

DEFAULT_PROMPT = "ROMEO:\nBut soft, what"
AVERAGE = "Average of all heads"


@dataclass
class AttentionResult:
    chars: list[str]  # the characters actually fed to the model
    weights: list[torch.Tensor]  # one (n_head, T, T) tensor per layer
    cropped: bool  # True if the prompt was longer than block_size


def attention_for_prompt(model: torch.nn.Module, tokenizer: CharTokenizer, prompt: str) -> AttentionResult:
    """Run the GPT on `prompt` and collect every layer's attention weights.

    Prompts longer than block_size are cropped to their LAST block_size
    characters (the positional embedding table has no rows beyond that).
    Raises ValueError for an empty prompt or unknown characters.
    """
    if not prompt:
        raise ValueError("Enter a prompt")
    cropped = len(prompt) > model.block_size
    used = prompt[-model.block_size :]
    ids = tokenizer.encode(used)  # ValueError on unknown characters
    idx = torch.tensor([ids], dtype=torch.long, device=model_device(model))
    with torch.no_grad():
        _, _, weights = model(idx, return_weights=True)  # each (1, n_head, T, T)
    return AttentionResult(list(used), [w[0].cpu() for w in weights], cropped)


def draw_heatmap(chars: list[str], matrix: torch.Tensor, title: str):
    """Row i shows how much position i attends to each earlier position j <= i.
    The empty upper triangle is the causal mask (future positions get weight 0)."""
    T = len(chars)
    labels = [visible_char(c) for c in chars]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    side = min(max(5.0, T * 0.32), 12.0)
    fig, ax = plt.subplots(figsize=(side + 1.5, side))
    sns.heatmap(
        matrix.numpy(), mask=mask, xticklabels=labels, yticklabels=labels,
        cmap="viridis", square=True, vmin=0.0, vmax=1.0, ax=ax,
        annot=T <= 16, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("attended-to character (key)")
    ax.set_ylabel("current character (query)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def draw_row(chars: list[str], row: torch.Tensor, query_pos: int):
    """Bar chart of one query position's attention over the characters it can see."""
    labels = [f"{visible_char(c)}\n{i}" for i, c in enumerate(chars[: query_pos + 1])]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.35 * len(labels)), 3))
    ax.bar(range(len(labels)), row[: query_pos + 1].numpy(), color="#4f6bed")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("attention weight")
    ax.set_title(f"What position {query_pos} ({visible_char(chars[query_pos])!r}) attends to")
    fig.tight_layout()
    return fig


def render_attention_visualizer() -> None:
    st.title("Attention Visualizer")
    st.markdown(
        "Each attention head computes, for every character, a probability distribution "
        "over the characters **before it (and itself)**. Brighter cells mean more weight. "
        "The mask guarantees nothing is ever attended to from the future."
    )

    info = pick_checkpoint("GPT checkpoint", "attn_ckpt", model_type="gpt")
    if info is None:
        return
    try:
        model, tokenizer, ckpt = load_checkpoint_cached(info.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Could not load checkpoint: {exc}")
        return

    prompt = st.text_area(
        f"Prompt (the model sees at most {model.block_size} characters)",
        value=DEFAULT_PROMPT, height=90, key="attn_prompt",
    )
    try:
        result = attention_for_prompt(model, tokenizer, prompt)
    except ValueError as exc:
        st.error(str(exc))
        return
    if result.cropped:
        st.warning(f"Prompt longer than block_size ({model.block_size}); "
                   "using its last characters.")

    n_layer = len(result.weights)
    n_head = result.weights[0].shape[0]
    col1, col2 = st.columns(2)
    layer = col1.selectbox("Layer", list(range(n_layer)), key="attn_layer")
    head_choice = col2.selectbox(
        "Head", [AVERAGE] + [f"Head {h}" for h in range(n_head)], key="attn_head"
    )

    layer_w = result.weights[layer]  # (n_head, T, T)
    if head_choice == AVERAGE:
        matrix = layer_w.mean(dim=0)
    else:
        matrix = layer_w[int(head_choice.split()[1])]

    fig = draw_heatmap(result.chars, matrix, f"Layer {layer} | {head_choice}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Look at one position")
    T = len(result.chars)
    query_pos = st.slider("Query position", 0, T - 1, T - 1, key="attn_query") if T > 1 else 0
    fig = draw_row(result.chars, matrix[query_pos], query_pos)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("How to read this"):
        st.markdown(
            f"""
- Every row sums to 1 (it's a softmax over the allowed positions). Position 0 can
  only attend to itself, so its weight is always 1.
- A model with random or barely trained weights attends roughly **uniformly**: row `t`
  gives each of the `t+1` visible characters about `1/(t+1)`. Structure that departs
  from that is something training added. Comparing a barely trained checkpoint (like a
  short smoke run) with a fully trained one is a good way to see it.
- Different heads in the same layer can learn different patterns, which is the point of
  having several. Single-head models (`--n-head 1`) have one pattern per layer.
- This checkpoint: layer count {n_layer}, heads per layer {n_head}, trained for
  {ckpt['iteration']} iterations.
"""
        )