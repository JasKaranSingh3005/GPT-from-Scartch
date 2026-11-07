from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from app.components.common import (  # noqa: E402
    load_checkpoint_cached,
    model_device,
    pick_checkpoint,
)
from app.components.tokenizer_page import visible_char  # noqa: E402
from models.sampling import encode_prompt, generate_text, next_token_probs  # noqa: E402
from models.tokenizer import CharTokenizer  # noqa: E402

TOP_SHOWN = 12


def next_char_distribution(
    model: torch.nn.Module, tokenizer: CharTokenizer, prompt: str,
    temperature: float, top_k: int | None,
) -> torch.Tensor:
    """The exact distribution the sampler draws the NEXT character from, shape (V,)."""
    ids = encode_prompt(tokenizer, prompt)  # ValueError on unknown characters
    block_size = getattr(model, "block_size", None)  # the bigram has no context limit
    if block_size:
        ids = ids[-block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=model_device(model))
    with torch.no_grad():
        logits, _ = model(idx)
    return next_token_probs(logits[:, -1, :], temperature, top_k)[0].cpu()


def draw_distribution(tokenizer: CharTokenizer, probs: torch.Tensor):
    top = torch.topk(probs, min(TOP_SHOWN, probs.numel()))
    labels = [visible_char(tokenizer.itos[int(i)]) for i in top.indices]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(labels)), top.values.numpy(), color="#e08a00")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("probability")
    ax.set_title(f"Top {len(labels)} candidates for the next character")
    fig.tight_layout()
    return fig


def render_text_generator() -> None:
    st.title("Text Generator")
    st.markdown(
        "The model predicts a probability for every possible next character, one is "
        "sampled, appended, and the process repeats. **Temperature** and **top-k** change "
        "how that sampling behaves."
    )

    info = pick_checkpoint("Checkpoint", "gen_ckpt")
    if info is None:
        return
    try:
        model, tokenizer, ckpt = load_checkpoint_cached(info.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Could not load checkpoint: {exc}")
        return
    n_params = sum(p.numel() for p in model.parameters())
    st.caption(f"{ckpt['model_type']} | {n_params:,} parameters | "
               f"iteration {ckpt['iteration']} | vocabulary {tokenizer.vocab_size}")

    prompt = st.text_area("Prompt (leave empty to start from a newline)", value="ROMEO:",
                          height=80, key="gen_prompt")

    col1, col2 = st.columns(2)
    with col1:
        max_new = st.slider("New characters", 50, 1000, 300, step=50, key="gen_len")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, step=0.05, key="gen_temp")
    with col2:
        use_top_k = st.checkbox("Use top-k sampling", value=True, key="gen_use_topk")
        top_k_value = st.slider(
            "top-k", 1, tokenizer.vocab_size, min(40, tokenizer.vocab_size),
            key="gen_topk", disabled=not use_top_k,
        )
        fixed_seed = st.checkbox("Fixed seed (reproducible output)", value=False, key="gen_fixed")
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                               key="gen_seed", disabled=not fixed_seed)
    top_k = top_k_value if use_top_k else None

    try:
        probs = next_char_distribution(model, tokenizer, prompt, temperature, top_k)
    except ValueError as exc:
        st.error(f"{exc}. The prompt can only use characters from the training vocabulary.")
        return

    st.subheader("Next-character distribution for this prompt")
    fig = draw_distribution(tokenizer, probs)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Lower temperature concentrates probability on the top candidates; higher "
               "temperature flattens it. Top-k sets everything outside the k best to exactly 0.")

    if st.button("Generate", type="primary", key="gen_button"):
        try:
            with st.spinner("Generating..."):
                text = generate_text(
                    model, tokenizer, prompt, max_new, temperature=temperature, top_k=top_k,
                    device=model_device(model), seed=int(seed) if fixed_seed else None,
                )
        except (ValueError, RuntimeError) as exc:
            st.error(f"Generation failed: {exc}")
        else:
            st.session_state["gen_result"] = {
                "text": text, "source": info.label, "temperature": temperature, "top_k": top_k,
            }

    result = st.session_state.get("gen_result")
    if result:
        st.subheader("Output")
        st.caption(f"{result['source']} | temperature {result['temperature']} | "
                   f"top-k {result['top_k'] or 'off'}")
        st.code(result["text"], language=None)