from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.definitions import BASE_MODEL, EXPERIMENTS, VARIANTS
from experiments.plot_results import collect_table

BATCH_RE = re.compile(r"it(\d+)_s(\d+)")


class ExperimentNotFoundError(Exception):
    """Unknown or malformed experiment batch id (HTTP 404)."""


def definitions() -> dict[str, Any]:
    """The experiments and variants exactly as defined in experiments/definitions.py."""
    def variant(name: str) -> dict[str, Any]:
        v = VARIANTS[name]
        model = {**BASE_MODEL, **v.overrides} if v.model_type == "gpt" else {}
        return {"name": name, "label": v.label, "model_type": v.model_type,
                "overrides": dict(v.overrides), "model": model}

    return {
        "base_model": dict(BASE_MODEL),
        "experiments": [
            {"key": e.key, "title": e.title, "variants": [variant(n) for n in e.variants]}
            for e in EXPERIMENTS.values()
        ],
    }


def batch_dir(root: Path, batch_id: str) -> Path:
    if not BATCH_RE.fullmatch(batch_id):
        raise ExperimentNotFoundError(f"Unknown experiment batch {batch_id!r}")
    path = Path(root) / batch_id
    if not path.is_dir():
        raise ExperimentNotFoundError(f"Unknown experiment batch {batch_id!r}")
    return path


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))  # native types, NaN -> null


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def list_batches(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and BATCH_RE.fullmatch(p.name)):
        m = BATCH_RE.fullmatch(d.name)
        assert m is not None
        finished = sum(1 for n in VARIANTS if (d / n / "summary.json").is_file())
        out.append({
            "batch_id": d.name, "max_iters": int(m.group(1)), "seed": int(m.group(2)),
            "variants_finished": finished, "variants_total": len(VARIANTS),
            "modified": datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc),
        })
    return sorted(out, key=lambda b: b["modified"], reverse=True)


def batch_detail(root: Path, batch_id: str) -> dict[str, Any]:
    d = batch_dir(root, batch_id)
    m = BATCH_RE.fullmatch(batch_id)
    assert m is not None
    rows = _records(collect_table(d, list(VARIANTS)))  # numbers come from summary.json
    curves: dict[str, list[dict[str, float]]] = {}
    for r in rows:
        try:
            h = pd.read_csv(d / r["run"] / "history.csv")
            curves[r["run"]] = _records(h[["iter", "train_loss", "val_loss"]])
        except (OSError, KeyError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue  # a missing curve must not hide the table
    return {"batch_id": batch_id, "max_iters": int(m.group(1)), "seed": int(m.group(2)),
            "rows": rows, "curves": curves, "batch": _read_json(d / "batch.json")}


def compare(root: Path, batch_ids: list[str]) -> dict[str, Any]:
    """Per-variant results of several batches (for example, the same setup with two seeds)."""
    tables = {}
    for bid in batch_ids:
        rows = _records(collect_table(batch_dir(root, bid), list(VARIANTS)))
        tables[bid] = {r["run"]: r for r in rows}
    variants = []
    for name in VARIANTS:
        if not any(name in t for t in tables.values()):
            continue
        variants.append({
            "variant": name, "label": VARIANTS[name].label,
            "best_val_loss": {b: (t[name]["best_val_loss"] if name in t else None) for b, t in tables.items()},
            "parameters": {b: (t[name]["parameters"] if name in t else None) for b, t in tables.items()},
        })
    return {"run_ids": batch_ids, "variants": variants}


def export_csv(root: Path, batch_id: str) -> str:
    d = batch_dir(root, batch_id)
    return collect_table(d, list(VARIANTS)).to_csv(index=False)