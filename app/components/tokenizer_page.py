from __future__ import annotations

import html
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe inside Streamlit
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.common import get_shakespeare_tokenizer, load_dataset_text  # noqa: E402
from models.tokenizer import CharTokenizer  # noqa: E402

SRC_SHAKESPEARE = "Tiny Shakespeare vocabulary"
SRC_CUSTOM = "Build from my own text"
DEFAULT_TEXT = "To be, or not to be"
MAX_CHIPS = 400  # keep the page light for very long inputs
VISIBLE = {" ": "␣", "\n": "↵", "\t": "⇥"}  # make whitespace tokens visible


def visible_char(ch: str) -> str:
    return VISIBLE.get(ch, ch)


def chip_html(ch: str, idx: int) -> str:
    """One colored 'token chip': the character with its ID as a subscript."""
    hue = (idx * 37) % 360  # spread IDs around the color wheel
    return (
        f'<span title="token id {idx}" style="display:inline-block;margin:2px;'
        f"padding:2px 6px;border-radius:6px;background:hsl({hue},70%,85%);"
        f'color:#111;font-family:monospace;">{html.escape(visible_char(ch))}'
        f'<sub style="font-size:0.65em;"> {idx}</sub></span>'
    )


@st.cache_data
def char_frequencies(top: int = 25) -> list[tuple[str, int]]:
    return Counter(load_dataset_text()).most_common(top)


def _frequency_chart() -> None:
    freqs = char_frequencies()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=[count for _, count in freqs],
        y=[visible_char(ch) for ch, _ in freqs],
        orient="h",
        color="#4f6bed",
        ax=ax,
    )
    ax.set_xlabel("occurrences in Tiny Shakespeare")
    ax.set_ylabel("character")
    ax.set_title(f"{len(freqs)} most frequent characters")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_tokenizer_explorer() -> None:
    st.title("Tokenizer Explorer")
    st.markdown(
        "A neural network only understands numbers. This project uses the simplest "
        "possible tokenizer: **every distinct character is one token**, and its ID is "
        "its position in the sorted vocabulary."
    )

    source = st.radio(
        "Vocabulary", [SRC_SHAKESPEARE, SRC_CUSTOM], horizontal=True, key="tok_source"
    )
    text = st.text_area("Text to tokenize", value=DEFAULT_TEXT, height=100, key="tok_text")

    if source == SRC_SHAKESPEARE:
        try:
            tokenizer = get_shakespeare_tokenizer()
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            st.error(f"Could not load the dataset: {exc}")
            return
    else:
        if not text:
            st.warning("Type some text to build a vocabulary from it.")
            return
        tokenizer = CharTokenizer.from_text(text)

    if not text:
        st.info("Type some text above.")
        ids: list[int] = []
    else:
        try:
            ids = tokenizer.encode(text)
        except ValueError as exc:
            st.error(
                f"{exc}. The tokenizer raises instead of silently dropping characters, "
                "because a model can only handle characters it saw when the vocabulary "
                "was built."
            )
            return

    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", len(text))
    col2.metric("Tokens", len(ids))
    col3.metric("Vocabulary size", tokenizer.vocab_size)

    if ids:
        st.subheader("Tokens")
        chips = "".join(chip_html(ch, i) for ch, i in zip(text[:MAX_CHIPS], ids))
        st.markdown(chips, unsafe_allow_html=True)
        st.caption("Whitespace is shown as ␣ (space) and ↵ (newline). Hover for the ID.")
        if len(text) > MAX_CHIPS:
            st.caption(f"Showing the first {MAX_CHIPS} of {len(text)} characters.")

        st.subheader("Token IDs")
        st.code(str(ids[:MAX_CHIPS]), language="python")

        if tokenizer.decode(ids) == text:
            st.success("Round trip OK: decode(encode(text)) gives back the original text.")
        else:  # should be impossible; the tests guard against it
            st.error("Round trip failed.")

    st.subheader("Decode IDs back to text")
    raw = st.text_input("Token IDs (comma or space separated)", value="", key="tok_decode")
    if raw.strip():
        try:
            parsed = [int(p) for p in raw.replace(",", " ").split()]
            st.code(repr(tokenizer.decode(parsed)), language="python")
        except ValueError as exc:
            st.error(f"Could not decode: {exc}")

    with st.expander(f"Full vocabulary ({tokenizer.vocab_size} tokens)"):
        st.dataframe(
            pd.DataFrame(
                {
                    "id": list(range(tokenizer.vocab_size)),
                    "character": [visible_char(c) for c in tokenizer.chars],
                    "unicode": [f"U+{ord(c):04X}" for c in tokenizer.chars],
                }
            ),
            hide_index=True,
        )

    if source == SRC_SHAKESPEARE:
        st.subheader("Which characters does the model see most?")
        try:
            _frequency_chart()
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            st.error(f"Could not compute frequencies: {exc}")
        st.caption(
            "Even before learning any context, a model can improve by predicting common "
            "characters more often. That skewed distribution is the easiest pattern to learn."
        )