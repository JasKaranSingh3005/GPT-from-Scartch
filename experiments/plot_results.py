from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.definitions import EXPERIMENTS, VARIANTS  # noqa: E402


def _load_run(out_dir: Path, name: str) -> tuple[dict, pd.DataFrame] | None:
    """(summary, history) of a finished run, or None if missing or unreadable."""
    run = out_dir / name
    try:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        history = pd.read_csv(run / "history.csv")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return summary, history


def collect_table(out_dir: Path, names: list[str]) -> pd.DataFrame:
    """One row per finished run. Every number comes from summary.json (measured, not typed in)."""
    rows = []
    for name in names:
        loaded = _load_run(out_dir, name)
        if loaded is None:
            continue
        s, _ = loaded
        rows.append({
            "run": name,
            "label": VARIANTS[name].label if name in VARIANTS else name,
            "parameters": s["parameters"],
            "training_time_s": round(s["training_time_s"], 1),
            "final_train_loss": s["final_train_loss"],
            "final_val_loss": s["final_val_loss"],
            "best_val_loss": s["best_val_loss"],
            "best_iter": s["best_iter"],
            "val_minus_train": s["final_val_loss"] - s["final_train_loss"],
        })
    return pd.DataFrame(rows)


def plot_experiment(key: str, out_dir: Path, plots_dir: Path) -> Path | None:
    """Loss curves (left) and best validation loss per variant (right). None if no runs exist."""
    exp = EXPERIMENTS[key]
    runs = []
    for name in exp.variants:
        loaded = _load_run(out_dir, name)
        if loaded is not None:
            runs.append((name, *loaded))
    if not runs:
        return None

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, (name, summary, history) in enumerate(runs):
        label = VARIANTS[name].label.split(" (")[0]  # drop the long parenthesis of 'base'
        color = f"C{i % 10}"
        ax_curve.plot(history["iter"], history["train_loss"], color=color, label=f"{label} train")
        ax_curve.plot(history["iter"], history["val_loss"], color=color, linestyle="--",
                      label=f"{label} val")
    ax_curve.set_xlabel("iteration")
    ax_curve.set_ylabel("cross-entropy loss")
    ax_curve.set_title(f"{exp.title}: training (solid) and validation (dashed)")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(fontsize=7)

    labels = [VARIANTS[n].label.split(" (")[0] for n, _, _ in runs]
    best = [s["best_val_loss"] for _, s, _ in runs]
    bars = ax_bar.bar(range(len(runs)), best, color=[f"C{i % 10}" for i in range(len(runs))])
    ax_bar.bar_label(bars, fmt="%.3f", fontsize=8)
    ax_bar.set_xticks(range(len(runs)))
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylabel("best validation loss (lower is better)")
    ax_bar.set_title("Best validation loss")
    fig.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{key}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot finished experiment runs")
    parser.add_argument("--dir", default="outputs/experiments/it1000_s1337")
    args = parser.parse_args()
    out_dir = Path(args.dir)
    if not out_dir.is_dir():
        raise SystemExit(f"Error: {out_dir} not found. Run experiments.run_experiments first.")

    table = collect_table(out_dir, list(VARIANTS))
    if table.empty:
        raise SystemExit(f"Error: no finished runs in {out_dir}")
    table.to_csv(out_dir / "results.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns=["label"]).round(4).to_string(index=False))
    for key in EXPERIMENTS:
        path = plot_experiment(key, out_dir, out_dir / "plots")
        print(f"{key}: {path if path else 'no finished runs, skipped'}")
    print(f"\nTable: {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()