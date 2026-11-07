from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Make the project root importable when launched via `streamlit run app/streamlit_app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="GPT from Scratch", page_icon="🧠", layout="wide")

from app.components.architecture_page import render_architecture_explorer  # noqa: E402
from app.components.attention_page import render_attention_visualizer  # noqa: E402
from app.components.common import get_device  # noqa: E402
from app.components.comparison_page import render_model_comparison  # noqa: E402
from app.components.config_page import render_model_configuration  # noqa: E402
from app.components.dashboard_page import render_training_dashboard  # noqa: E402
from app.components.generator_page import render_text_generator  # noqa: E402
from app.components.home import render_home  # noqa: E402
from app.components.tokenizer_page import render_tokenizer_explorer  # noqa: E402

PAGES: dict[str, Callable[[], None]] = {
    "Home": render_home,
    "Tokenizer Explorer": render_tokenizer_explorer,
    "Attention Visualizer": render_attention_visualizer,
    "Model Configuration": render_model_configuration,
    "Training Dashboard": render_training_dashboard,
    "Text Generator": render_text_generator,
    "Model Comparison": render_model_comparison,
    "Architecture Explorer": render_architecture_explorer,
}


def main() -> None:
    st.sidebar.title("GPT from Scratch")
    page = st.sidebar.radio("Navigate", list(PAGES), key="nav_page")
    st.sidebar.caption(f"Device: {get_device()}")
    PAGES[page]()


main()