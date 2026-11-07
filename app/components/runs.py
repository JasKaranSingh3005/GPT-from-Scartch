from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.components.common import RUNS_DIR

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def list_runs(runs_dir: Path = RUNS_DIR) -> list[str]:
    """Run folders that contain a history.csv."""
    if not runs_dir.is_dir():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if (p / "history.csv").is_file())


def load_history(run: str, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    df = pd.read_csv(runs_dir / run / "history.csv")
    missing = {"iter", "train_loss", "val_loss"} - set(df.columns)
    if missing:
        raise ValueError(f"history.csv of run {run!r} is missing columns: {sorted(missing)}")
    return df


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    """summary.json of a run folder, or None if absent or unreadable."""
    try:
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def summary_row(run: str, history: pd.DataFrame, summary: dict[str, Any] | None) -> dict[str, Any]:
    """One table row per run. Uses summary.json when present, else the history alone."""
    final = history.iloc[-1]
    best = history.loc[history["val_loss"].idxmin()]
    s = summary or {}
    return {
        "run": run,
        "model": s.get("model_type", "?"),
        "parameters": s.get("parameters"),
        "train time (s)": s.get("training_time_s"),
        "final train": float(final["train_loss"]),
        "final val": float(final["val_loss"]),
        "best val": float(best["val_loss"]),
        "best iter": int(best["iter"]),
        "val - train": float(final["val_loss"] - final["train_loss"]),
    }


def validate_run_name(name: str, runs_dir: Path = RUNS_DIR) -> str | None:
    """Error message if the name is unusable, else None."""
    if not RUN_NAME_RE.match(name):
        return "Use 1-64 letters, digits, '-' or '_' (starting with a letter or digit)."
    if (runs_dir / name).exists():
        return f"A run named {name!r} already exists. Choose another name."
    return None