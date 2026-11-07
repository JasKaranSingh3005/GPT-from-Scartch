from __future__ import annotations
import argparse
import time
from pathlib import Path
import torch
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
    create_model,
    model_kwargs_from_config,
    save_checkpoint,
)
from training.evaluate import estimate_loss
from utils.config import Config
from utils.data import download_dataset, get_batch, load_text, split_text
from utils.device import get_device, set_seed

CSV_FIELDS = ["iter", "train_loss", "val_loss", "elapsed_s"]


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
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    writer: Any | None = None,
    on_best: Callable[[int], None] | None = None,
    on_eval: Callable[[dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    """Generic training loop (works for the bigram model and the GPT).

    Each step: sample a batch, compute cross-entropy, backpropagate, clip the
    gradient norm, and take an AdamW step. Every `eval_interval` steps the
    train/val losses are estimated and recorded.

    Returns the history: one dict per evaluation with keys
    iter, train_loss, val_loss, elapsed_s.
    `on_best(iter)` is called whenever validation loss hits a new minimum;
    `on_eval(row)` is called after every evaluation (e.g. for a live UI).
    """
    if max_iters < 1 or eval_interval < 1 or eval_iters < 1:
        raise ValueError("max_iters, eval_interval and eval_iters must all be >= 1")

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    start = time.perf_counter()

    for it in range(max_iters + 1):
        if it % eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, splits, block_size, batch_size, eval_iters, device)
            row = {
                "iter": it,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed_s": time.perf_counter() - start,
            }
            history.append(row)
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
) -> dict[str, Any]:
    """Train one model end to end and save everything under out_root/run_name/:

        history.csv   loss curve data (used later for plots)
        summary.json  parameters, training time, final/best losses, config
        best.pt       checkpoint with the lowest validation loss
        last.pt       checkpoint at the end of training (with full history)
        events.*      TensorBoard logs
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")

    tc = cfg.training
    set_seed(tc.seed)
    device = device or get_device()
    print(f"Run: {run_name} | model: {model_type} | device: {device}")

    tokenizer, splits = prepare_data(cfg)
    model_kwargs = model_kwargs_from_config(model_type, cfg.model, tokenizer.vocab_size)
    model = create_model(model_type, model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab size: {tokenizer.vocab_size} | Parameters: {n_params:,}")

    run_dir = Path(out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(run_dir) if use_tensorboard else None
    if writer is not None:
        writer.add_text("config", json.dumps({"model": model_kwargs, "lr": tc.learning_rate}))

    def save_best(iteration: int) -> None:
        save_checkpoint(
            run_dir / "best.pt", model, model_type=model_type, model_kwargs=model_kwargs,
            chars=tokenizer.chars, iteration=iteration,
        )

    start = time.perf_counter()
    try:
        history = train_model(
            model, splits,
            block_size=cfg.model.block_size, batch_size=tc.batch_size,
            learning_rate=tc.learning_rate, max_iters=tc.max_iters,
            eval_interval=tc.eval_interval, eval_iters=tc.eval_iters,
            device=device, writer=writer, on_best=save_best, on_eval=on_eval,
        )
    finally:
        if writer is not None:
            writer.close()
    training_time = time.perf_counter() - start

    save_checkpoint(
        run_dir / "last.pt", model, model_type=model_type, model_kwargs=model_kwargs,
        chars=tokenizer.chars, iteration=tc.max_iters, history=history,
    )
    write_history_csv(history, run_dir / "history.csv")

    best = min(history, key=lambda r: r["val_loss"])
    summary = {
        "run_name": run_name,
        "model_type": model_type,
        "device": str(device),
        "parameters": n_params,
        "vocab_size": tokenizer.vocab_size,
        "model_kwargs": model_kwargs,
        "training": {
            "batch_size": tc.batch_size, "learning_rate": tc.learning_rate,
            "max_iters": tc.max_iters, "seed": tc.seed,
            "block_size": cfg.model.block_size,
        },
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
from models.bigram import BigramLanguageModel
from models.tokenizer import CharTokenizer
from utils.config import load_config
from utils.data import download_dataset, get_batch, load_text, split_text
from utils.device import get_device, set_seed


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device) -> dict[str, float]:
    """Average loss over several random batches for each split."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bigram baseline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)  # bigram tolerates a high LR
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc = cfg.training
    max_iters = args.max_iters or tc.max_iters
    set_seed(tc.seed)
    device = get_device()
    print(f"Device: {device}")

    text = load_text(download_dataset(cfg.data.url, cfg.data.path))
    tokenizer = CharTokenizer.from_text(text)  # vocab from the full text
    train_text, val_text = split_text(text, cfg.data.train_fraction)
    splits = {
        "train": torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
        "val": torch.tensor(tokenizer.encode(val_text), dtype=torch.long),
    }

    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab size: {tokenizer.vocab_size} | Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    block_size = cfg.model.block_size
    start = time.time()

    for it in range(max_iters + 1):
        if it % tc.eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, splits, block_size, tc.batch_size, tc.eval_iters, device)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                  f" | {time.time() - start:.1f}s")
        if it == max_iters:
            break
        x, y = get_batch(splits["train"], block_size, tc.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    out_path = Path("outputs/bigram.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "chars": tokenizer.chars}, out_path)
    print(f"Saved {out_path}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=300)[0].tolist()
    print("\n--- Bigram sample ---\n" + tokenizer.decode(sample))


if __name__ == "__main__":
    main()