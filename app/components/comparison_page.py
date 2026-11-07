from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.common import load_checkpoint_cached, model_device, pick_checkpoint
from app.components.runs import read_summary
from models.sampling import generate_text


def render_model_comparison() -> None:
    st.title("Model Comparison")
    st.markdown(
        "The bigram model predicts each character from **only the previous character**. The "
        "GPT sees up to `block_size` previous characters through attention. Both models below "
        "get the same prompt, settings and random seed."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        bigram_info = pick_checkpoint("Bigram checkpoint", "cmp_bigram", model_type="bigram",
                                      prefer="bigram_baseline (best)")
    with col_b:
        gpt_info = pick_checkpoint("GPT checkpoint", "cmp_gpt", model_type="gpt")
    if bigram_info is None or gpt_info is None:
        return

    try:
        bigram, tok_b, _ = load_checkpoint_cached(bigram_info.path)
        gpt, tok_g, _ = load_checkpoint_cached(gpt_info.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Could not load checkpoint: {exc}")
        return
    if tok_b.chars != tok_g.chars:
        st.error("These checkpoints were trained with different vocabularies, so the same "
                 "prompt cannot be given to both.")
        return

    prompt = st.text_area("Prompt", value="ROMEO:", height=70, key="cmp_prompt")
    c1, c2, c3, c4 = st.columns(4)
    max_new = c1.slider("New characters", 50, 800, 300, step=50, key="cmp_len")
    temperature = c2.slider("Temperature", 0.1, 2.0, 0.8, step=0.05, key="cmp_temp")
    top_k = c3.slider("top-k", 1, tok_g.vocab_size, min(40, tok_g.vocab_size), key="cmp_topk")
    seed = int(c4.number_input("Seed", 0, 2**31 - 1, 42, key="cmp_seed"))

    if st.button("Generate with both", type="primary", key="cmp_button"):
        try:
            with st.spinner("Generating..."):
                st.session_state["cmp_result"] = {
                    "bigram": generate_text(bigram, tok_b, prompt, max_new, temperature,
                                            top_k, model_device(bigram), seed),
                    "gpt": generate_text(gpt, tok_g, prompt, max_new, temperature,
                                         top_k, model_device(gpt), seed),
                }
        except (ValueError, RuntimeError) as exc:
            st.error(f"Generation failed: {exc}. The prompt can only use characters "
                     "from the training vocabulary.")

    result = st.session_state.get("cmp_result")
    if result:
        left, right = st.columns(2)
        with left:
            st.subheader(f"Bigram: {bigram_info.label}")
            st.code(result["bigram"], language=None)
        with right:
            st.subheader(f"GPT: {gpt_info.label}")
            st.code(result["gpt"], language=None)
        st.caption("Output is from the last time you pressed Generate. Changing settings "
                   "above does not regenerate it.")

    rows = []
    for label, info, model in (("Bigram", bigram_info, bigram), ("GPT", gpt_info, gpt)):
        summary = read_summary(info.path.parent) or {}
        rows.append({
            "model": label,
            "run": info.label,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best val loss": summary.get("best_val_loss"),
            "final val loss": summary.get("final_val_loss"),
        })
    st.subheader("Recorded training results")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Lower validation loss means better next-character predictions on held-out "
               "text. Values come from each run's summary.json (blank if it is missing).")