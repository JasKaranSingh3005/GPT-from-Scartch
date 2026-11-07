from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.common import PROJECT_ROOT, RUNS_DIR, current_config  # noqa: E402
from app.components.runs import (  # noqa: E402
    list_runs,
    load_history,
    read_summary,
    summary_row,
    validate_run_name,
)
from training.train import run_experiment  # noqa: E402
from utils.config import Config  # noqa: E402


def draw_loss_curves(histories: dict[str, pd.DataFrame]):
    """Solid line = training loss, dashed = validation loss, one color per run."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (run, df) in enumerate(histories.items()):
        color = f"C{i % 10}"
        ax.plot(df["iter"], df["train_loss"], color=color, label=f"{run} train")
        ax.plot(df["iter"], df["val_loss"], color=color, linestyle="--", label=f"{run} val")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training (solid) and validation (dashed) loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _train_in_app(cfg: Config, model_type: str, name: str, iters: int) -> None:
    training = replace(cfg.training, max_iters=iters, eval_interval=max(1, iters // 10),
                       eval_iters=20)
    if model_type == "bigram":
        training = replace(training, learning_rate=1e-2)  # same choice as train.py
    cfg = replace(
        cfg,
        data=replace(cfg.data, path=str(PROJECT_ROOT / cfg.data.path)),
        training=training,
    )
    rows: list[dict[str, float]] = []
    chart = st.empty()
    bar = st.progress(0.0, text="Starting...")

    def on_eval(row: dict[str, float]) -> None:
        rows.append(row)
        chart.line_chart(pd.DataFrame(rows).set_index("iter")[["train_loss", "val_loss"]])
        bar.progress(min(1.0, row["iter"] / iters), text=f"iteration {int(row['iter'])} / {iters}")

    try:
        summary = run_experiment(cfg, model_type, name, out_root=RUNS_DIR,
                                 use_tensorboard=False, on_eval=on_eval)
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"Training failed: {exc}. Delete outputs/runs/{name} before retrying "
                 "with the same name.")
        return
    st.session_state["dash_flash"] = (
        f"Run '{name}' finished in {summary['training_time_s']:.0f}s | "
        f"final val loss {summary['final_val_loss']:.4f}"
    )
    st.rerun()


def render_training_dashboard() -> None:
    st.title("Training Dashboard")
    flash = st.session_state.pop("dash_flash", None)
    if flash:
        st.success(flash)
    st.markdown(
        "Loss curves from your runs (`outputs/runs/*/history.csv`). **Training loss** is "
        "measured on text the model learns from, **validation loss** on held-out text. "
        "A widening gap between them means overfitting."
    )

    runs = list_runs()
    if not runs:
        st.info("No finished runs yet. Train one with `python train.py --name gpt_small`, "
                "or use the section at the bottom of this page.")
    else:
        default = [r for r in ("bigram_baseline", "gpt_small") if r in runs] or runs[:2]
        selected = st.multiselect("Runs to plot", runs, default=default, key="dash_runs")
        histories: dict[str, pd.DataFrame] = {}
        for run in selected:
            try:
                histories[run] = load_history(run)
            except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                st.error(f"Could not read run {run!r}: {exc}")
        if histories:
            fig = draw_loss_curves(histories)
            st.pyplot(fig)
            plt.close(fig)
            table = pd.DataFrame(
                [summary_row(r, df, read_summary(RUNS_DIR / r)) for r, df in histories.items()]
            )
            st.dataframe(table.round(4), hide_index=True)
            st.caption("'best iter' is the evaluation with the lowest validation loss. If it is "
                       "well before the last iteration, the model started overfitting. "
                       "Live curves are also available with `tensorboard --logdir outputs/runs`.")

    with st.expander("Train a short run here (the page is busy until it finishes)"):
        try:
            cfg = current_config()
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Could not load config: {exc}")
            return
        m = cfg.model
        st.caption(f"Uses the Model Configuration page's values: n_embd {m.n_embd}, "
                   f"{m.n_head} heads, {m.n_layer} layers, block_size {m.block_size}, "
                   f"dropout {m.dropout}, batch {cfg.training.batch_size}, "
                   f"lr {cfg.training.learning_rate:g}. Use the Stop button at the top right "
                   "of the page to interrupt. Long runs are better done from the terminal.")
        model_type = st.radio("Model", ["gpt", "bigram"], horizontal=True, key="dash_model")
        name = st.text_input("Run name", value="app_run", key="dash_name")
        iters = int(st.number_input("Iterations", 50, 3000, 300, step=50, key="dash_iters"))
        if st.button("Start training", type="primary", key="dash_start"):
            error = validate_run_name(name)
            if error:
                st.error(error)
            else:
                _train_in_app(cfg, model_type, name, iters)