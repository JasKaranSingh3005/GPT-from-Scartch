from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.tokenizer import CharTokenizer
from training.checkpoint import load_checkpoint, load_model
from utils.estimate import gpt_parameter_breakdown

RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
KINDS = ("best", "last")


class ModelNotFoundError(Exception):
    """Unknown model id or missing checkpoint file (HTTP 404)."""


class CheckpointError(Exception):
    """Checkpoint exists but cannot be read or built into a model (HTTP 422)."""


class ModelLimitError(Exception):
    """Too many models already loaded (HTTP 409)."""


def parse_model_id(model_id: str) -> tuple[str, str]:
    """'gpt_small:best' -> ('gpt_small', 'best'). Anything else is 'not found',
    which also rules out path traversal (names may not contain '/', '\\' or '.')."""
    run, sep, kind = model_id.partition(":")
    if not sep or not RUN_NAME_RE.fullmatch(run) or kind not in KINDS:
        raise ModelNotFoundError(f"Unknown model id {model_id!r}; expected '<run>:<best|last>'")
    return run, kind


@dataclass(frozen=True)
class CheckpointFile:
    model_id: str
    run: str
    kind: str
    path: Path
    size_bytes: int
    mtime_ns: int

    @property
    def modified(self) -> datetime:
        return datetime.fromtimestamp(self.mtime_ns / 1e9, tz=timezone.utc)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    numel: int


@dataclass(frozen=True)
class CheckpointMeta:
    """Facts read from the checkpoint file itself (never from constants)."""

    file: CheckpointFile
    model_type: str
    model_kwargs: dict[str, Any]
    iteration: int
    parameters: int
    vocab_size: int
    tensors: tuple[TensorInfo, ...]


@dataclass(frozen=True)
class RunSummary:
    """Run-level results from summary.json (not specific to best vs last)."""

    best_val_loss: float | None = None
    final_val_loss: float | None = None
    training_time_s: float | None = None


@dataclass
class LoadedModel:
    model_id: str
    model: nn.Module
    tokenizer: CharTokenizer
    meta: CheckpointMeta
    device: torch.device
    loaded_at: datetime


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def gpt_breakdown(meta: CheckpointMeta) -> dict[str, int] | None:
    """Hand-derived per-component parameter counts for a GPT, else None."""
    if meta.model_type != "gpt":
        return None
    kw = meta.model_kwargs
    try:
        return gpt_parameter_breakdown(
            kw["vocab_size"], kw["n_embd"], kw["n_layer"], kw["block_size"]
        )
    except KeyError:
        return None


class ModelRegistry:
    """Discovers checkpoints under runs_dir and manages which models are in memory."""

    def __init__(self, runs_dir: Path, device: torch.device, max_loaded: int = 3) -> None:
        self._runs_dir = Path(runs_dir)
        self._device = device
        self._max_loaded = max_loaded
        self._lock = threading.RLock()  # guards _loaded and _meta_cache
        self._loaded: dict[str, LoadedModel] = {}
        self._meta_cache: dict[tuple[str, int], CheckpointMeta] = {}
        # Serializes model execution so requests don't run on the device concurrently.
        self.inference_lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        return self._device

    # ---- discovery (file system only, nothing is loaded) -------------------

    def discover(self) -> list[CheckpointFile]:
        if not self._runs_dir.is_dir():
            return []
        found: list[CheckpointFile] = []
        run_dirs = sorted(
            p for p in self._runs_dir.iterdir()
            if p.is_dir() and RUN_NAME_RE.fullmatch(p.name)
        )
        for run_dir in run_dirs:
            for kind in KINDS:
                path = run_dir / f"{kind}.pt"
                if path.is_file():
                    st = path.stat()
                    found.append(CheckpointFile(
                        f"{run_dir.name}:{kind}", run_dir.name, kind, path,
                        st.st_size, st.st_mtime_ns,
                    ))
        return found

    def resolve(self, model_id: str) -> CheckpointFile:
        run, kind = parse_model_id(model_id)
        path = self._runs_dir / run / f"{kind}.pt"
        if not path.is_file():
            raise ModelNotFoundError(f"No checkpoint for model id {model_id!r}")
        st = path.stat()
        return CheckpointFile(model_id, run, kind, path, st.st_size, st.st_mtime_ns)

    # ---- metadata (reads the file; cached until the file changes) ----------

    def metadata(self, model_id: str) -> CheckpointMeta:
        file = self.resolve(model_id)
        key = (model_id, file.mtime_ns)
        with self._lock:
            cached = self._meta_cache.get(key)
        if cached is not None:
            return cached
        meta = self._read_meta(file)
        with self._lock:
            for stale in [k for k in self._meta_cache if k[0] == model_id]:
                del self._meta_cache[stale]
            self._meta_cache[key] = meta
        return meta

    @staticmethod
    def _read_meta(file: CheckpointFile) -> CheckpointMeta:
        try:
            ckpt = load_checkpoint(file.path, "cpu")
            tensors = tuple(
                TensorInfo(name, tuple(int(d) for d in t.shape), int(t.numel()))
                for name, t in ckpt["state_dict"].items()
            )
            return CheckpointMeta(
                file=file,
                model_type=str(ckpt["model_type"]),
                model_kwargs=dict(ckpt["model_kwargs"]),
                iteration=int(ckpt["iteration"]),
                # Valid because these models register no buffers: state_dict == parameters.
                parameters=sum(t.numel for t in tensors),
                vocab_size=len(ckpt["chars"]),
                tensors=tensors,
            )
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
            raise CheckpointError(f"Checkpoint {file.model_id!r} is unusable: {exc}") from exc

    def run_summary(self, model_id: str) -> RunSummary:
        """Read fresh each call: summary.json is written after best.pt during training."""
        file = self.resolve(model_id)
        try:
            data = json.loads((file.path.parent / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return RunSummary()
        if not isinstance(data, dict):
            return RunSummary()
        return RunSummary(
            _opt_float(data.get("best_val_loss")),
            _opt_float(data.get("final_val_loss")),
            _opt_float(data.get("training_time_s")),
        )

    # ---- lifecycle ----------------------------------------------------------

    def load(self, model_id: str) -> LoadedModel:
        """Load into memory. Idempotent; reloads if the file changed on disk."""
        file = self.resolve(model_id)
        with self._lock:
            current = self._loaded.get(model_id)
            if current is not None and current.meta.file.mtime_ns == file.mtime_ns:
                return current
            if current is None and len(self._loaded) >= self._max_loaded:
                raise ModelLimitError(
                    f"{len(self._loaded)} models already loaded (limit {self._max_loaded}); "
                    "unload one first"
                )
            meta = self.metadata(model_id)
            try:
                model, tokenizer, _ = load_model(file.path, self._device)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                raise CheckpointError(f"Could not load {model_id!r}: {exc}") from exc
            loaded = LoadedModel(
                model_id, model, tokenizer, meta, self._device, datetime.now(timezone.utc)
            )
            self._loaded[model_id] = loaded
            del current
        self._free_cache()
        return loaded

    def unload(self, model_id: str) -> bool:
        """Drop a model from memory. Returns False if it was not loaded."""
        with self._lock:
            removed = self._loaded.pop(model_id, None)
        was_loaded = removed is not None
        del removed
        self._free_cache()
        return was_loaded

    def unload_all(self) -> None:
        with self._lock:
            self._loaded.clear()
        self._free_cache()

    def get_loaded(self, model_id: str) -> LoadedModel | None:
        with self._lock:
            return self._loaded.get(model_id)

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._loaded

    def loaded_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._loaded)

    def _free_cache(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _was_removed(self, model_id: str) -> bool:  # placeholder replaced below
        raise NotImplementedError