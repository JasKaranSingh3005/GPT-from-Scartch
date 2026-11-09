"""Training worker: `python -m training.worker <job_dir>`.

Reads <job_dir>/spec.json, runs the existing run_experiment, and reports through files:
  metrics.jsonl  one JSON line per evaluation (appended as training proceeds)
  result.json    the run summary (status "completed" or "stopped")
  error.json     {"error": ...} if training failed
Creating <job_dir>/stop.flag asks the worker to stop cleanly after the current step.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import functools

import torch

from training.checkpoint import load_checkpoint
from training.train import run_experiment
from utils.config import Config, DataConfig, ModelConfig, TrainingConfig
from utils.device import get_device

from datetime import datetime, timezone

def config_from_dict(d: dict[str, Any]) -> Config:
    return Config(
        data=DataConfig(**d["data"]),
        model=ModelConfig(**d["model"]),
        training=TrainingConfig(**d["training"]),
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to this PyTorch build")
    return torch.device(name)


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def append_metric(path: Path, job_id: str, row: dict[str, float]) -> None:
    line = json.dumps({"job_id": job_id, **{k: row[k] for k in
                       ("iter", "train_loss", "val_loss", "elapsed_s")}})
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

def run_experiment_batch(job_dir: Path, spec: dict[str, Any], stop_file: Path) -> dict[str, Any]:
    """Run an experiment batch with the existing run_all. Raises if any variant failed."""
    from experiments.definitions import selected_variants
    from experiments.run_experiments import run_all
    from training.train import run_experiment

    base = config_from_dict(spec["config"])
    names = selected_variants(spec["experiments"] or None)
    out_dir = Path(spec["out_root"]) / spec["batch_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "batch.json", {
        "batch_id": spec["batch_id"], "max_iters": spec["max_iters"], "seed": spec["seed"],
        "experiments": spec["experiments"], "variants": names, "base_config": spec["config"],
        "device_setting": spec["device"], "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    def on_variant(name: str, i: int, total: int) -> None:
        write_json_atomic(job_dir / "progress.json", {"current": name, "done": i - 1, "total": total})

    run_fn = functools.partial(run_experiment, device=resolve_device(spec["device"]))
    summaries, failures = run_all(
        base, names, out_dir, spec["max_iters"], force=spec["force"], use_tensorboard=False,
        run_fn=run_fn, should_stop=stop_file.exists, on_variant=on_variant,
    )
    if failures:
        raise RuntimeError("Variants failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))
    ran = [n for n, s in summaries.items() if s.get("status") != "stopped"]
    stopped = stop_file.exists()
    return {
        "status": "stopped" if stopped else "completed", "batch_id": spec["batch_id"],
        "variants_run": ran, "variants_total": len(names),
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m training.worker <job_dir>", file=sys.stderr)
        return 2
    job_dir = Path(args[0])
    metrics_path = job_dir / "metrics.jsonl"
    stop_file = job_dir / "stop.flag"
    try:
        spec = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
        job_id = spec["job_id"]
        cfg = config_from_dict(spec["config"])
        run_dir = Path(spec["out_root"]) / spec["run_name"]

        if spec.get("kind") == "experiment":
            summary = run_experiment_batch(job_dir, spec, stop_file)
        else:
            if spec["resume"]:  # seed the curve with the evaluations already done
                prior = load_checkpoint(run_dir / "last.pt", "cpu")["history"]
                for row in prior:
                    append_metric(metrics_path, job_id, row)
            summary = run_experiment(
                cfg, spec["model_type"], spec["run_name"], out_root=spec["out_root"],
                use_tensorboard=False, device=resolve_device(spec["device"]),
                on_eval=lambda row: append_metric(metrics_path, job_id, row),
                should_stop=stop_file.exists, resumable=True, resume=spec["resume"],
            )
        write_json_atomic(job_dir / "result.json", summary)
        return 0
    
    except Exception as exc:
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            message += " (reduce batch_size, block_size or n_embd)"
        try:
            write_json_atomic(job_dir / "error.json", {"error": message})
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())