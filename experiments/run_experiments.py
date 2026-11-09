from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from experiments.definitions import EXPERIMENTS, VARIANTS, build_config, selected_variants
from training.train import run_experiment
from utils.config import Config, load_config


def run_all(
    base_cfg: Config,
    variant_names: list[str],
    out_dir: Path,
    max_iters: int,
    *,
    force: bool = False,
    use_tensorboard: bool = False,
    run_fn: Callable[..., dict[str, Any]] = run_experiment,
    should_stop: Callable[[], bool] | None = None,
    on_variant: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Train each variant. Returns (summaries of runs done now, {name: error message}).

    A variant whose summary.json already exists is skipped, so an interrupted batch
    can be resumed by running the same command again. One failing run is reported
    and does not stop the others.

    Optional hooks: `should_stop` is checked before each variant and passed to the runs
    (a stopped run ends the batch); `on_variant(name, index, total)` is called before each
    variant, including skipped ones.
    """
    summaries: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    total = len(variant_names)
    extra: dict[str, Any] = {"should_stop": should_stop} if should_stop is not None else {}
    for i, name in enumerate(variant_names, 1):
        if should_stop is not None and should_stop():
            print("\nStop requested; not starting further runs.")
            break
        if on_variant is not None:
            on_variant(name, i, total)
        if (out_dir / name / "summary.json").is_file() and not force:
            print(f"\n[{i}/{total}] {name}: already done, skipping (use --force to rerun)")
            continue
        print(f"\n[{i}/{total}] {name}: {VARIANTS[name].label}")
        cfg = build_config(base_cfg, VARIANTS[name], max_iters)
        try:
            summaries[name] = run_fn(
                cfg, VARIANTS[name].model_type, name,
                out_root=out_dir, use_tensorboard=use_tensorboard, **extra,
            )
        except Exception as exc:  # keep going; KeyboardInterrupt is not caught
            failures[name] = str(exc)
            print(f"  FAILED: {exc}")
            continue
        if summaries[name].get("status") == "stopped":
            break
    return summaries, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model-comparison experiments")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--only", nargs="+", metavar="EXPERIMENT",
                        help=f"subset of: {', '.join(EXPERIMENTS)} (default: all)")
    parser.add_argument("--max-iters", type=int, default=1000,
                        help="training iterations per run (same for every run)")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--force", action="store_true", help="rerun finished runs")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    args = parser.parse_args()

    if args.list:
        for exp in EXPERIMENTS.values():
            print(f"{exp.key:14s} {exp.title}: {', '.join(exp.variants)}")
        return

    base = load_config(args.config)
    if args.seed is not None:
        base = replace(base, training=replace(base.training, seed=args.seed))
    try:
        names = selected_variants(args.only)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}")

    out_dir = Path("outputs/experiments") / f"it{args.max_iters}_s{base.training.seed}"
    print(f"Running {len(names)} runs ({args.max_iters} iterations each) into {out_dir}")
    _, failures = run_all(base, names, out_dir, args.max_iters,
                          force=args.force, use_tensorboard=args.tensorboard)
    if failures:
        print("\nFailed runs:")
        for name, msg in failures.items():
            print(f"  {name}: {msg}")
    print(f"\nNext: python -m experiments.plot_results --dir {out_dir}")


if __name__ == "__main__":
    main()