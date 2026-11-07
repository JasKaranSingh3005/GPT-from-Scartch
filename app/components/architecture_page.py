from __future__ import annotations

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from models.attention import causal_average_demo
from models.tokenizer import CharTokenizer
from utils.estimate import gpt_parameter_breakdown, valid_head_counts


def scaling_demo(d_k: int, n_pairs: int = 4000, n_keys: int = 8, seed: int = 0) -> dict[str, float]:
    """Why attention divides by sqrt(d_k), measured on random vectors.

    For q, k with unit-variance entries, the dot product has variance d_k. Without
    scaling, scores get large, softmax saturates (one key gets ~all the weight) and
    gradients vanish. Dividing by sqrt(d_k) restores variance ~1.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n_pairs, d_k, generator=g)
    k = torch.randn(n_pairs, d_k, generator=g)
    raw = (q * k).sum(-1)
    scaled = raw / d_k**0.5

    q1 = torch.randn(1000, 1, d_k, generator=g)
    keys = torch.randn(1000, n_keys, d_k, generator=g)
    scores = (q1 * keys).sum(-1)  # (1000, n_keys)
    return {
        "std_raw": raw.std().item(),
        "std_scaled": scaled.std().item(),
        "max_weight_raw": F.softmax(scores, dim=-1).max(-1).values.mean().item(),
        "max_weight_scaled": F.softmax(scores / d_k**0.5, dim=-1).max(-1).values.mean().item(),
    }


def layernorm_demo(shift: float, scale: float, n: int = 8, seed: int = 0):
    """A feature vector with chosen mean/spread, before and after LayerNorm."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g) * scale + shift
    return x, F.layer_norm(x, (n,))  # gamma=1, beta=0, eps=1e-5


def render_architecture_explorer() -> None:
    st.title("Architecture Explorer")
    st.markdown("One tab per component. The demos run real PyTorch code, not canned pictures.")
    tabs = st.tabs(["Embeddings", "Self-attention", "Multi-head", "Feed-forward",
                    "Residuals & LayerNorm", "Whole model"])

    with tabs[0]:
        st.markdown(
            "Token IDs are integers, so the model first looks each one up in a learned table. "
            "A second learned table adds **position** information, because attention by "
            "itself has no notion of order."
        )
        st.latex(r"x_t = E_{\text{token}}[\,\text{id}_t\,] + E_{\text{pos}}[\,t\,]"
                 r"\qquad x_t \in \mathbb{R}^{n_{\text{embd}}}")
        word = st.text_input("Try a word", "hello", max_chars=20, key="arch_word")
        if word:
            tok = CharTokenizer.from_text(word)  # tiny vocabulary just for this demo
            st.dataframe(pd.DataFrame({
                "position t": range(len(word)),
                "character": list(word),
                "token id (this demo's vocabulary)": tok.encode(word),
            }), hide_index=True)
            st.caption("The trained models use the full Tiny Shakespeare vocabulary, so "
                       "their IDs differ from this demo's.")

    with tabs[1]:
        st.markdown(
            "**Step 1: averaging the past.** The simplest way to let a position use earlier "
            "positions is a running mean. A lower-triangular matrix with normalized rows does "
            "exactly that. Attention keeps this structure but *learns* the weights."
        )
        T = st.slider("Sequence length", 2, 8, 5, key="arch_T")
        tril = torch.tril(torch.ones(T, T))
        weights = (tril / tril.sum(1, keepdim=True)).numpy().round(2)
        st.dataframe(pd.DataFrame(weights), hide_index=False)
        probe = torch.arange(1, T + 1, dtype=torch.float).reshape(1, T, 1)
        st.caption(f"Applying it to the values 1..{T} gives running means "
                   f"{causal_average_demo(probe).flatten().tolist()}.")

        st.markdown("**Step 2: learned weights.** Each position makes a query, a key and a "
                    "value. The mask forbids looking at the future.")
        st.latex(r"\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V")
        d_k = st.select_slider("Head size d_k", [4, 16, 64, 256, 512], value=64, key="arch_dk")
        r = scaling_demo(d_k)
        a, b = st.columns(2)
        a.metric("Score std, no scaling", f"{r['std_raw']:.2f}", help="Grows like sqrt(d_k)")
        b.metric("Score std, divided by sqrt(d_k)", f"{r['std_scaled']:.2f}")
        a.metric("Avg. largest weight, no scaling", f"{r['max_weight_raw']:.2f}")
        b.metric("Avg. largest weight, scaled", f"{r['max_weight_scaled']:.2f}")
        st.caption("Measured on random vectors with 8 keys (uniform attention would give "
                   "0.125). Without scaling, a larger d_k pushes the largest weight toward 1, "
                   "so softmax saturates. Scaling keeps it from doing that.")

    with tabs[2]:
        st.markdown(
            "Instead of one attention pattern, split the embedding into `n_head` smaller "
            "heads. Each head has its own Q, K, V projections and can attend differently. "
            "Their outputs are concatenated and mixed by one more linear layer."
        )
        n_embd = st.selectbox("n_embd", [32, 64, 128, 384], index=1, key="arch_mh_embd")
        heads = valid_head_counts(n_embd, [1, 2, 3, 4, 6, 8, 12, 16])
        n_head = st.selectbox("n_head", heads, index=min(3, len(heads) - 1), key="arch_mh_head")
        st.markdown(f"`head_size = {n_embd} / {n_head} = {n_embd // n_head}`, and "
                    f"concatenating {n_head} heads gives back {n_embd} features.")
        st.caption("The parameter count does not depend on n_head: more heads means smaller "
                   "heads, not more weights.")

    with tabs[3]:
        st.markdown(
            "Attention moves information **between** positions. The feed-forward network then "
            "processes each position **on its own**, with a hidden layer 4x wider than the "
            "embedding."
        )
        st.latex(r"\mathrm{FFN}(x)=W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2")
        c = st.slider("n_embd", 16, 512, 64, step=16, key="arch_ffn_c")
        st.markdown(f"Shapes: `{c} -> {4 * c} -> {c}`, with {8 * c * c + 5 * c:,} parameters "
                    "per layer.")

    with tabs[4]:
        st.markdown(
            "**Residual connections** add each sublayer's output to its input "
            "(`x = x + sublayer(x)`), giving gradients a direct path through deep stacks. "
            "**LayerNorm** rescales each position's feature vector to zero mean and unit "
            "variance before each sublayer (pre-norm)."
        )
        st.latex(r"\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta")
        shift = st.slider("Mean of the input", -10.0, 10.0, 3.0, key="arch_ln_shift")
        scale = st.slider("Spread of the input", 0.1, 10.0, 4.0, key="arch_ln_scale")
        before, after = layernorm_demo(shift, scale)
        st.dataframe(pd.DataFrame({"before": before.numpy().round(3),
                                   "after LayerNorm": after.numpy().round(3)}).T)
        st.caption(f"Before: mean {before.mean():.2f}, std {before.std(unbiased=False):.2f}. "
                   f"After: mean {after.mean():.2f}, std {after.std(unbiased=False):.2f} "
                   "(with gamma=1, beta=0; trained models learn both).")

    with tabs[5]:
        st.markdown("Stack N blocks between the embeddings and a final LayerNorm plus linear "
                    "head, and you have a GPT. Where do the parameters live?")
        a, b, c2, d = st.columns(4)
        vocab = int(a.number_input("vocab", 2, 5000, 65, key="arch_vocab"))
        embd = b.selectbox("n_embd", [32, 64, 128, 256, 384], index=1, key="arch_embd")
        layers = c2.slider("layers", 1, 12, 4, key="arch_layers")
        block = d.selectbox("block_size", [32, 64, 128, 256], index=1, key="arch_block")
        parts = gpt_parameter_breakdown(vocab, embd, layers, block)
        st.bar_chart(pd.Series(parts, name="parameters"))
        st.metric("Total parameters", f"{sum(parts.values()):,}")