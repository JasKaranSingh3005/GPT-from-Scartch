from __future__ import annotations
import sys
from dataclasses import dataclass, replace
from pathlib import Path
import streamlit as st
import torch

# `streamlit run app/streamlit_app.py` puts app/ (not the project root) on sys.path,
# so make the root importable before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tokenizer import CharTokenizer  # noqa: E402
from training.checkpoint import load_model  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.data import download_dataset, load_text  # noqa: E402
from utils.device import get_device  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


@dataclass(frozen=True)
class CheckpointInfo:
    run: str  # run folder name, e.g. "gpt_small"
    kind: str  # "best" or "last"
    path: Path

    @property
    def label(self) -> str:
        return f"{self.run} ({self.kind})"


def discover_checkpoints(runs_dir: Path = RUNS_DIR) -> list[CheckpointInfo]:
    """Find best.pt / last.pt inside each run folder, sorted by run name."""
    if not runs_dir.is_dir():
        return []
    found: list[CheckpointInfo] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for kind in ("best", "last"):
            path = run_dir / f"{kind}.pt"
            if path.is_file():
                found.append(CheckpointInfo(run_dir.name, kind, path))
    return found


def dataset_path() -> Path:
    return PROJECT_ROOT / load_config(CONFIG_PATH).data.path


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset_text() -> str:
    """Tiny Shakespeare text (downloads it first if missing). Errors are not cached."""
    cfg = load_config(CONFIG_PATH)
    return load_text(download_dataset(cfg.data.url, PROJECT_ROOT / cfg.data.path))


@st.cache_resource
def get_shakespeare_tokenizer() -> CharTokenizer:
    """Same vocabulary the models were trained with (built from the full text)."""
    return CharTokenizer.from_text(load_dataset_text())


@st.cache_resource(show_spinner="Loading model...")
def _load_cached(path: str, mtime_ns: int):
    # mtime_ns is part of the cache key, so retraining a run refreshes the cache.
    model, tokenizer, ckpt = load_model(path, get_device())
    ckpt = {k: v for k, v in ckpt.items() if k != "state_dict"}  # don't keep weights twice
    return model, tokenizer, ckpt


def load_checkpoint_cached(path: Path):
    """Returns (model in eval mode, tokenizer, checkpoint metadata)."""
    return _load_cached(str(path), path.stat().st_mtime_ns)

def model_device(model: torch.nn.Module) -> torch.device:
    """The device a loaded model lives on."""
    return next(model.parameters()).device


def pick_checkpoint(
    label: str, key: str, model_type: str | None = None, prefer: str = "gpt_small (best)"
) -> CheckpointInfo | None:
    """Sidebar-style checkpoint selector. Returns None (after showing a hint)
    when no matching checkpoint exists. `model_type` filters to "gpt" or "bigram"."""
    infos = discover_checkpoints()
    if model_type is not None:
        kept = []
        for info in infos:
            try:
                _, _, ckpt = load_checkpoint_cached(info.path)
            except (RuntimeError, ValueError, OSError):
                continue  # unreadable checkpoint: just don't offer it
            if ckpt["model_type"] == model_type:
                kept.append(info)
        infos = kept
    if not infos:
        what = f"{model_type} " if model_type else ""
        st.info(f"No trained {what}checkpoints found. Train one with "
                "`python train.py --name gpt_small`.")
        return None
    labels = [i.label for i in infos]
    index = labels.index(prefer) if prefer in labels else 0
    choice = st.selectbox(label, labels, index=index, key=key)
    return infos[labels.index(choice)]

def current_config() -> Config:
    """config.yaml plus any overrides set on the Model Configuration page."""
    cfg = load_config(CONFIG_PATH)
    over = st.session_state.get("cfg_overrides", {})
    return replace(
        cfg,
        model=replace(cfg.model, **over.get("model", {})),
        training=replace(cfg.training, **over.get("training", {})),
    )