from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from app.components.common import (
    CONFIG_PATH,
    PROJECT_ROOT,
    dataset_path,
    discover_checkpoints,
)
from utils.config import load_config

ARCHITECTURE_DOT = r"""
digraph GPT {
  rankdir=TB;
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#4f6bed",
        fontname="Helvetica", fontsize=11];
  text   [label="Text"];
  tok    [label="CharTokenizer\ntext -> token IDs"];
  emb    [label="Token embedding + Positional embedding"];
  blocks [label="Transformer Block  x N\n(LayerNorm -> Multi-Head Causal Attention -> + residual\nLayerNorm -> Feed-Forward -> + residual)", fillcolor="#fff4e5", color="#e08a00"];
  ln     [label="Final LayerNorm"];
  head   [label="Linear LM head -> logits over vocabulary"];
  out    [label="Softmax -> sample next character", fillcolor="#e8f7ee", color="#2e9d5b"];
  text -> tok -> emb -> blocks -> ln -> head -> out;
}
"""


def render_home() -> None:
    st.title("GPT from Scratch")
    st.caption(
        "A decoder-only Transformer implemented manually in PyTorch and trained on "
        "Tiny Shakespeare, with no pretrained models and no Hugging Face."
    )

    st.markdown(
        """
A language model learns one thing: **given the text so far, predict the next
token**. Generating text is just doing that repeatedly, feeding each prediction
back in as input. This project builds that machine up in stages:

1. A **bigram model** that sees only the previous character (the baseline).
2. **Self-attention**, which lets every position look at earlier positions.
3. **Multi-head attention, feed-forward layers, residuals, LayerNorm**, stacked
   into Transformer blocks.
4. The complete **GPT**, trained and sampled with temperature and top-k.
"""
    )

    st.subheader("Architecture")
    st.graphviz_chart(ARCHITECTURE_DOT)

    st.subheader("The three equations that matter most")
    st.markdown("**Causal scaled dot-product attention.** The mask `M` is 0 for allowed "
                "positions and −∞ for future ones:")
    st.latex(
        r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left("
        r"\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V"
    )
    st.markdown("**Layer normalization,** applied per position:")
    st.latex(r"\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta")
    st.markdown("**Training objective,** the average negative log-likelihood of the next character:")
    st.latex(r"\mathcal{L}=-\frac{1}{N}\sum_{t}\log p_\theta\!\left(x_{t+1}\mid x_{\le t}\right)")

    st.subheader("What's in this app")
    st.markdown(
        """
- **Tokenizer Explorer:** turn text into token IDs and back.
- **Attention Visualizer:** see which characters each head looks at.
- **Model Configuration:** adjust the hyperparameters.
- **Training Dashboard:** loss curves from your runs.
- **Text Generator:** sample from a trained checkpoint.
- **Model Comparison:** bigram vs. GPT side by side.
- **Architecture Explorer:** each component explained interactively.
"""
    )

    st.subheader("Project status")
    try:
        ds = dataset_path()
        st.write(f"Dataset: `{ds.relative_to(PROJECT_ROOT)}` "
                 f"{'found' if ds.is_file() else 'not downloaded yet (run `python -m utils.data`)'}")
    except FileNotFoundError as exc:
        st.error(str(exc))

    checkpoints = discover_checkpoints()
    if checkpoints:
        st.dataframe(
            pd.DataFrame(
                {
                    "run": [c.run for c in checkpoints],
                    "checkpoint": [c.kind for c in checkpoints],
                    "path": [str(c.path.relative_to(PROJECT_ROOT)) for c in checkpoints],
                }
            ),
            hide_index=True,
        )
    else:
        st.info("No trained checkpoints yet. Train one with `python train.py --name gpt_small`.")

    with st.expander("Default configuration (config/config.yaml)"):
        try:
            st.json(asdict(load_config(CONFIG_PATH)))
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))