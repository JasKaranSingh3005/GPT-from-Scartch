from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from models.tokenizer import CharTokenizer
from training.checkpoint import (
    MODEL_TYPES,
    build_training_state,
    create_model,
    load_checkpoint,
    model_kwargs_from_config,
    restore_rng_state,
    save_checkpoint,
)
from training.evaluate import estimate_loss
from utils.config import Config
from utils.data import download_dataset, get_batch, load_text, split_text
from utils.device import get_device, set_seed

CSV_FIELDS = ["iter", "train_loss", "val_loss", "elapsed_s"]
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


def prepare_data(cfg: Config) -> tuple[CharTokenizer, dict[str, torch.Tensor]]:
    """Download (if needed), tokenize and split the dataset."""
    text = load_text(download_dataset(cfg.data.url, cfg.data.path))
    tokenizer = CharTokenizer.from_text(text)  # vocabulary from the full text
    train_text, val_text = split_text(text, cfg.data.train_fraction)
    splits = {
        "train": torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
        "val": torch.tensor(tokenizer.encode(val_text), dtype=torch.long),
    }
    return tokenizer, splits


def train_model(
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    *,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    device: torch.device | str,
    weight_decay: float = WEIGHT_DECAY,
    grad_clip: float = GRAD_CLIP,
    writer: Any | None = None,
    on_best: Callable[[int], None] | None = None,
    on_eval: Callable[[dict[str, float]], None] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_iter: int = 0,
    history: list[dict[str, float]] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_stop: Callable[[int], None] | None = None,
) -> list[dict[str, float]]:
    """Generic training loop (works for the bigram model and the GPT).

    Each step: sample a batch, compute cross-entropy, backpropagate, clip the
    gradient norm, and take an AdamW step. Every `eval_interval` steps the
    train/val losses are estimated and recorded.

    Stopping and resuming (all optional, defaults give the original behaviour):
      should_stop  polled once per iteration; when it returns True the loop calls
                   `on_stop(iterations_completed)` and returns early.
      optimizer    pass an existing optimizer (e.g. with restored state) instead of a new one.
      start_iter   iterations already completed; `history` holds their evaluation rows.
    An iteration that already has a row in `history` is not evaluated again.

    Returns the full history (previous rows first): one dict per evaluation with keys
    iter, train_loss, val_loss, elapsed_s. `on_best(iter)` fires on a new validation
    minimum; `on_eval(row)` fires after every new evaluation.
    """
    if max_iters < 1 or eval_interval < 1 or eval_iters < 1:
        raise ValueError("max_iters, eval_interval and eval_iters must all be >= 1")
    if not 0 <= start_iter <= max_iters:
        raise ValueError(f"start_iter must be between 0 and max_iters ({max_iters})")

    model.to(device)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    history = [dict(r) for r in history] if history else []
    evaluated = {int(r["iter"]) for r in history}
    best_val = min((r["val_loss"] for r in history), default=float("inf"))
    start = time.perf_counter() - (history[-1]["elapsed_s"] if history else 0.0)

    for it in range(start_iter, max_iters + 1):
        if (it % eval_interval == 0 or it == max_iters) and it not in evaluated:
            losses = estimate_loss(model, splits, block_size, batch_size, eval_iters, device)
            row = {
                "iter": it,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed_s": time.perf_counter() - start,
            }
            history.append(row)
            evaluated.add(it)
            print(
                f"iter {it:5d} | train {row['train_loss']:.4f} | "
                f"val {row['val_loss']:.4f} | {row['elapsed_s']:.1f}s"
            )
            if writer is not None:
                writer.add_scalar("loss/train", row["train_loss"], it)
                writer.add_scalar("loss/val", row["val_loss"], it)
            if row["val_loss"] < best_val:
                best_val = row["val_loss"]
                if on_best is not None:
                    on_best(it)
            if on_eval is not None:
                on_eval(row)
        if it == max_iters:
            break
        if should_stop is not None and should_stop():
            if on_stop is not None:
                on_stop(it)  # `it` iterations have been completed
            break

        x, y = get_batch(splits["train"], block_size, batch_size, device)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss became {loss.item()} at iter {it}. Try a lower learning rate."
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clipping the gradient norm guards against occasional huge updates.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return history


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _make_writer(log_dir: Path) -> Any | None:
    """TensorBoard writer, or None (with a warning) if TensorBoard is unusable."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # missing package or incompatible install
        print(f"Warning: TensorBoard logging disabled ({exc})")
        return None


def run_experiment(
    cfg: Config,
    model_type: str,
    run_name: str,
    out_root: str | Path = "outputs/runs",
    use_tensorboard: bool = True,
    device: torch.device | str | None = None,
    on_eval: Callable[[dict[str, float]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    resumable: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one model end to end and save everything under out_root/run_name/:

        history.csv   loss curve data (used later for plots)
        summary.json  parameters, training time, final/best losses, config (completed runs only)
        best.pt       checkpoint with the lowest validation loss
        last.pt       latest checkpoint (with full history)
        events.*      TensorBoard logs

    resumable=True stores optimizer and RNG state in last.pt at every evaluation and at
    the end, so the run can be continued. resume=True continues from last.pt: model
    settings and hyperparameters come from the checkpoint, while max_iters, eval_interval
    and eval_iters come from cfg (so a run can be extended).

    If `should_stop` returns True the run stops, saves last.pt and history.csv, writes NO
    summary.json (so it is never mistaken for a finished run) and returns a dict with
    status "stopped".
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")

    tc = cfg.training
    device = device or get_device()
    run_dir = Path(out_root) / run_name
    print(f"Run: {run_name} | model: {model_type} | device: {device}"
          f"{' | resuming' if resume else ''}")
    if not resume:
        set_seed(tc.seed)

    tokenizer, splits = prepare_data(cfg)

    state: dict[str, Any] | None = None
    if resume:
        ckpt = load_checkpoint(run_dir / "last.pt", device)
        state = ckpt.get("training_state")
        if state is None:
            raise ValueError(
                f"{run_dir / 'last.pt'} has no training state; only runs started with "
                "resumable=True (--resumable) can be resumed"
            )
        if ckpt["model_type"] != model_type:
            raise ValueError(f"Run is a {ckpt['model_type']!r} model, not {model_type!r}")
        if list(ckpt["chars"]) != tokenizer.chars:
            raise ValueError("The dataset's vocabulary differs from the one this run used")
        start_iter = int(state["iteration"])
        if start_iter >= tc.max_iters:
            raise ValueError(
                f"Run is already at iteration {start_iter}; set max_iters above that to continue"
            )
        hp = dict(state["hyperparameters"])
        model_kwargs = dict(ckpt["model_kwargs"])
        history = [dict(r) for r in ckpt["history"]]
        model = create_model(model_type, model_kwargs)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        resumable = True
    else:
        start_iter, history = 0, []
        hp = {
            "learning_rate": tc.learning_rate, "batch_size": tc.batch_size,
            "block_size": cfg.model.block_size, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "seed": tc.seed,
        }
        model_kwargs = model_kwargs_from_config(model_type, cfg.model, tokenizer.vocab_size)
        model = create_model(model_type, model_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"]
    )
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state["rng"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab size: {tokenizer.vocab_size} | Parameters: {n_params:,}")

    run_dir.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(run_dir) if use_tensorboard else None
    if writer is not None:
        writer.add_text("config", json.dumps({"model": model_kwargs, "lr": hp["learning_rate"]}))

    rows = [dict(r) for r in history]  # every evaluation so far, for checkpoints

    def save_last(iteration: int) -> None:
        save_checkpoint(
            run_dir / "last.pt", model, model_type=model_type, model_kwargs=model_kwargs,
            chars=tokenizer.chars, iteration=iteration, history=rows,
            training_state=build_training_state(optimizer, iteration, hp) if resumable else None,
        )

    def save_best(iteration: int) -> None:
        save_checkpoint(
            run_dir / "best.pt", model, model_type=model_type, model_kwargs=model_kwargs,
            chars=tokenizer.chars, iteration=iteration,
        )

    def handle_eval(row: dict[str, float]) -> None:
        rows.append(row)
        if resumable:
            save_last(int(row["iter"]))  # a crash loses at most one evaluation interval
        if on_eval is not None:
            on_eval(row)

    stopped_at: list[int] = []
    prior_elapsed = history[-1]["elapsed_s"] if history else 0.0
    start = time.perf_counter()
    try:
        history = train_model(
            model, splits,
            block_size=hp["block_size"], batch_size=hp["batch_size"],
            learning_rate=hp["learning_rate"], max_iters=tc.max_iters,
            eval_interval=tc.eval_interval, eval_iters=tc.eval_iters,
            device=device, weight_decay=hp["weight_decay"], grad_clip=hp["grad_clip"],
            writer=writer, on_best=save_best, on_eval=handle_eval,
            optimizer=optimizer, start_iter=start_iter, history=history,
            should_stop=should_stop, on_stop=stopped_at.append,
        )
    finally:
        if writer is not None:
            writer.close()
    training_time = prior_elapsed + (time.perf_counter() - start)

    save_last(stopped_at[0] if stopped_at else tc.max_iters)
    write_history_csv(history, run_dir / "history.csv")

    if stopped_at:
        print(f"Stopped at iteration {stopped_at[0]} | saved {run_dir / 'last.pt'}")
        return {
            "run_name": run_name, "status": "stopped", "model_type": model_type,
            "iteration": stopped_at[0], "parameters": n_params,
            "training_time_s": training_time, "run_dir": str(run_dir),
            "resumable": resumable,
        }

    best = min(history, key=lambda r: r["val_loss"])
    summary = {
        "run_name": run_name,
        "status": "completed",
        "model_type": model_type,
        "device": str(device),
        "parameters": n_params,
        "vocab_size": tokenizer.vocab_size,
        "model_kwargs": model_kwargs,
        "training": {
            "batch_size": hp["batch_size"], "learning_rate": hp["learning_rate"],
            "max_iters": tc.max_iters, "seed": hp["seed"], "block_size": hp["block_size"],
        },
        "resumed_from_iteration": start_iter if resume else None,
        "training_time_s": training_time,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_loss": best["val_loss"],
        "best_iter": int(best["iter"]),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Done in {training_time:.1f}s | final val {summary['final_val_loss']:.4f} | "
        f"best val {summary['best_val_loss']:.4f} (iter {summary['best_iter']})"
    )
    print(f"Saved to {run_dir}")
    return summary