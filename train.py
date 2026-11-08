from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime

from training.checkpoint import MODEL_TYPES
from training.train import run_experiment
from utils.config import load_config

import signal
import threading

MODEL_FIELDS = {"n_embd": int, "n_head": int, "n_layer": int, "block_size": int, "dropout": float}
TRAIN_FIELDS = {
    "batch_size": int, "max_iters": int, "eval_interval": int, "eval_iters": int, "seed": int,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bigram or GPT language model")
    parser.add_argument("--model", choices=MODEL_TYPES, default="gpt")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--name", default=None, help="run name (default: model + timestamp)")
    parser.add_argument("--out-dir", default="outputs/runs")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--resumable", action="store_true",
                        help="save optimizer and RNG state so the run can be resumed")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run named by --name from its last.pt")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=None)
    # Optional overrides of values in the config file
    for name, typ in {**MODEL_FIELDS, **TRAIN_FIELDS}.items():
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_over = {k: getattr(args, k) for k in MODEL_FIELDS if getattr(args, k) is not None}
    train_over = {k: getattr(args, k) for k in TRAIN_FIELDS if getattr(args, k) is not None}
    if args.learning_rate is not None:
        train_over["learning_rate"] = args.learning_rate
    elif args.model == "bigram":
        train_over["learning_rate"] = 1e-2  # the bigram is one lookup table; it tolerates a high LR

    # dataclasses.replace re-runs __post_init__, so bad head counts are caught here
    cfg = replace(
        cfg,
        model=replace(cfg.model, **model_over),
        training=replace(cfg.training, **train_over),
    )

    if args.resume and not args.name:
        parser.error("--resume needs --name")

    stop = threading.Event()

    def on_sigint(signum, frame):  # first Ctrl+C: stop cleanly; second: abort
        if stop.is_set():
            raise KeyboardInterrupt
        print("\nStopping after the current step (press Ctrl+C again to abort)...")
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)

    name = args.name or f"{args.model}-{datetime.now():%Y%m%d-%H%M%S}"
    run_experiment(
        cfg, args.model, name, out_root=args.out_dir, use_tensorboard=not args.no_tensorboard,
        should_stop=stop.is_set, resumable=args.resumable, resume=args.resume,
    )


if __name__ == "__main__":
    main()