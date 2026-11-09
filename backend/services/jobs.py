from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from backend.core.config import PROJECT_ROOT, Settings
from training.checkpoint import load_checkpoint
from training.worker import config_from_dict
from utils.config import Config, ModelConfig, TrainingConfig, load_config

from experiments.definitions import selected_variants

log = logging.getLogger("gptlab.jobs")

QUEUED, RUNNING, COMPLETED, FAILED, STOPPED = "QUEUED", "RUNNING", "COMPLETED", "FAILED", "STOPPED"
JOB_ID_RE = re.compile(r"[0-9a-f]{12}")
METRIC_KEYS = ("iter", "train_loss", "val_loss", "elapsed_s")
_KILL_AFTER_TERMINATE_S = 10.0


class JobNotFoundError(Exception):
    """Unknown job id (HTTP 404)."""


class JobConflictError(Exception):
    """Request conflicts with the job's or run's current state (HTTP 409)."""


class JobRejectedError(Exception):
    """Request is well-formed but not acceptable (HTTP 422)."""


class QueueFullError(Exception):
    """Too many queued jobs (HTTP 429)."""


class TrainingDisabledError(Exception):
    """Training is turned off on this server (HTTP 403)."""


# ---- storage --------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, run_name TEXT NOT NULL, model_type TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
    spec TEXT NOT NULL, result TEXT, error TEXT, resumed_from TEXT, pid INTEGER,
    stop_requested INTEGER NOT NULL DEFAULT 0
)"""


class SqliteJobStore:
    """Job metadata only. Metrics, logs and checkpoints stay on the file system."""

    _UPDATABLE = {"status", "started_at", "finished_at", "result", "error", "pid",
                  "stop_requested"}

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit or roll back
                yield conn
        finally:
            conn.close()

    def insert(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._conn() as c:
            c.execute(f"INSERT INTO jobs ({cols}) VALUES ({marks})", list(row.values()))

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(r) if r else None

    def update(self, job_id: str, **fields: Any) -> None:
        bad = set(fields) - self._UPDATABLE
        if bad:
            raise ValueError(f"cannot update fields {sorted(bad)}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*fields.values(), job_id])

    def list(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM jobs", []
        if status:
            sql, args = sql + " WHERE status = ?", [status]
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, [*args, limit]).fetchall()]

    def count(self, status: str) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)).fetchone()[0]

    def oldest_queued(self) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE status = ? "
                          "ORDER BY created_at ASC, rowid ASC LIMIT 1", (QUEUED,)).fetchone()
        return dict(r) if r else None

    def active_for_run(self, run_name: str) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT 1 FROM jobs WHERE run_name = ? AND status IN (?, ?) LIMIT 1",
                          (run_name, QUEUED, RUNNING)).fetchone()
        return r is not None


# ---- file helpers -----------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Valid metric rows only; a torn last line (worker mid-write) is skipped."""
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and all(isinstance(row.get(k), (int, float)) for k in METRIC_KEYS):
            rows.append(row)
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def tail_lines(path: Path, n: int, max_bytes: int = 256 * 1024) -> list[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- process launching --------------------------------------------------------

def default_launcher(job_dir: Path, log_file: Any) -> subprocess.Popen:
    """Start the worker in its own process group, so Ctrl+C in the server's terminal
    does not hit it directly; the manager stops it through stop.flag instead."""
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [sys.executable, "-m", "training.worker", str(job_dir)],
        cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}, **kwargs,
    )


@dataclass
class _Running:
    job_id: str
    process: Any
    log_file: Any
    stop_deadline: float | None = None
    terminated: bool = False
    stop_requested: bool = False


class JobManager:
    """Queue, supervise and report on training jobs. One job runs at a time."""

    def __init__(
        self,
        settings: Settings,
        store: SqliteJobStore | None = None,
        launcher: Callable[[Path, Any], Any] = default_launcher,
        grace_seconds: float = 60.0,
        poll_seconds: float = 0.25,
        shutdown_wait_seconds: float = 20.0,
    ) -> None:
        self._s = settings
        self._store = store  # created in start(), so importing the app has no side effects
        self._launcher = launcher
        self._grace = grace_seconds
        self._poll = poll_seconds
        self._shutdown_wait = shutdown_wait_seconds
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: _Running | None = None

    # ---- lifecycle ----------------------------------------------------------

    @property
    def store(self) -> SqliteJobStore:
        if self._store is None:
            raise RuntimeError("JobManager has not been started")
        return self._store

    def start(self) -> None:
        if self._thread is not None:
            return
        self._s.jobs_dir.mkdir(parents=True, exist_ok=True)
        if self._store is None:
            self._store = SqliteJobStore(self._s.db_path)
        self._recover()
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._loop, name="gptlab-jobs", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the dispatcher, then ask a running worker to stop and wait for it."""
        self._shutdown.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self._lock:
            cur = self._current
            if cur is None:
                return
            self._request_stop(cur)
        try:
            cur.process.wait(timeout=self._shutdown_wait)
        except subprocess.TimeoutExpired:
            cur.terminated = True
            cur.process.terminate()
            try:
                cur.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cur.process.kill()
        with self._lock:
            self._finish(cur)

    def _recover(self) -> None:
        for row in self.store.list(RUNNING, limit=1000):
            self.store.update(
                row["id"], status=FAILED, finished_at=_now(), pid=None,
                error="The server restarted while this job was running. Resume it if a "
                      "checkpoint exists.",
            )

    def _loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                self._tick()
            except Exception:  # keep supervising no matter what
                log.exception("job dispatcher error")
            self._wake.wait(self._poll)
            self._wake.clear()

    def _tick(self) -> None:
        with self._lock:
            cur = self._current
            if cur is None:
                row = self.store.oldest_queued()
                if row is not None:
                    self._launch(row)
                return
            if cur.process.poll() is not None:
                self._finish(cur)
                return
            if cur.stop_deadline is not None and time.monotonic() > cur.stop_deadline:
                if not cur.terminated:
                    cur.terminated = True
                    cur.process.terminate()
                    cur.stop_deadline = time.monotonic() + _KILL_AFTER_TERMINATE_S
                else:
                    cur.process.kill()
                    cur.stop_deadline = None

    def _launch(self, row: dict[str, Any]) -> None:
        job_id = row["id"]
        job_dir = self.job_dir(job_id)
        log_file = open(job_dir / "log.txt", "ab")
        try:
            proc = self._launcher(job_dir, log_file)
        except Exception as exc:
            log_file.close()
            self.store.update(job_id, status=FAILED, finished_at=_now(),
                              error=f"Could not start the worker process: {exc}")
            return
        self.store.update(job_id, status=RUNNING, started_at=_now(),
                          pid=getattr(proc, "pid", None))
        self._current = _Running(job_id, proc, log_file)

    def _request_stop(self, cur: _Running) -> None:
        (self.job_dir(cur.job_id) / "stop.flag").touch()
        cur.stop_requested = True
        cur.stop_deadline = time.monotonic() + self._grace
        self.store.update(cur.job_id, stop_requested=1)

    def _finish(self, cur: _Running) -> None:
        rc = cur.process.poll()
        cur.log_file.close()
        self._current = None
        job_dir = self.job_dir(cur.job_id)
        result = read_json(job_dir / "result.json")
        error: str | None = None
        if result is not None and result.get("status") in ("completed", "stopped"):
            status = COMPLETED if result["status"] == "completed" else STOPPED
        elif cur.terminated and cur.stop_requested:
            status = STOPPED
            error = ("The worker did not stop in time and was terminated; the latest "
                     "checkpoint may be older than the stop point.")
        else:
            status = FAILED
            err = read_json(job_dir / "error.json")
            if err and isinstance(err.get("error"), str):
                error = err["error"]
            else:
                tail = " | ".join(tail_lines(job_dir / "log.txt", 3))
                error = f"Worker exited with code {rc}" + (f". Last log lines: {tail}" if tail else "")
        self.store.update(
            cur.job_id, status=status, finished_at=_now(), error=error, pid=None,
            result=json.dumps(result) if result is not None else None,
        )

    # ---- requests -----------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        return self._s.jobs_dir / job_id

    def _abs(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def _require_training(self) -> None:
        if not self._s.training_enabled:
            raise TrainingDisabledError("Training is disabled on this server")

    def _build_config(self, params: dict[str, Any]) -> Config:
        base = load_config(self._s.config_path)
        model_type = params["model_type"]
        # The bigram has no embedding width or heads; use valid placeholders.
        n_embd, n_head = (params["n_embd"], params["n_head"]) if model_type == "gpt" else (64, 4)
        try:
            model = ModelConfig(n_embd=n_embd, n_head=n_head, n_layer=params["n_layer"],
                                block_size=params["block_size"], dropout=params["dropout"])
        except ValueError as exc:
            raise JobRejectedError(str(exc)) from exc
        training = TrainingConfig(
            batch_size=params["batch_size"], learning_rate=params["learning_rate"],
            max_iters=params["max_iters"], eval_interval=params["eval_interval"],
            eval_iters=params["eval_iters"], seed=params["seed"],
        )
        data = replace(base.data, path=str(self._abs(base.data.path)))
        return Config(data=data, model=model, training=training)

    def _check_iters(self, max_iters: int) -> None:
        if max_iters > self._s.max_train_iters:
            raise JobRejectedError(
                f"max_iters {max_iters} exceeds the server limit of {self._s.max_train_iters}")

    def _check_queue(self) -> None:
        if self.store.count(QUEUED) >= self._s.max_queued_jobs:
            raise QueueFullError(
                f"{self._s.max_queued_jobs} jobs are already queued; try again later")

    def submit(self, params: dict[str, Any]) -> str:
        self._require_training()
        name = params["name"]
        self._check_iters(params["max_iters"])
        with self._lock:
            if (self._s.runs_dir / name).exists():
                raise JobConflictError(f"A run named {name!r} already exists")
            if self.store.active_for_run(name):
                raise JobConflictError(f"A job for run {name!r} is already queued or running")
            self._check_queue()
            cfg = self._build_config(params)
            return self._create(cfg, params["model_type"], name, resume=False, resumed_from=None)

    def submit_experiment(self, params: dict[str, Any]) -> str:
        """Queue an experiment batch: the existing run_all over the chosen experiments."""
        self._require_training()
        max_iters, seed = params["max_iters"], params["seed"]
        self._check_iters(max_iters)
        try:
            selected_variants(params["experiments"])
        except ValueError as exc:
            raise JobRejectedError(str(exc)) from exc
        batch_id = f"it{max_iters}_s{seed}"
        run_name = f"exp-{batch_id}"
        with self._lock:
            if self.store.active_for_run(run_name):
                raise JobConflictError(f"A batch {batch_id!r} is already queued or running")
            self._check_queue()
            base = load_config(self._s.config_path)
            base = replace(
                base, data=replace(base.data, path=str(self._abs(base.data.path))),
                training=replace(base.training, seed=seed, max_iters=max_iters),
            )
            extra = {
                "kind": "experiment", "batch_id": batch_id, "experiments": params["experiments"],
                "max_iters": max_iters, "seed": seed, "force": bool(params.get("force", False)),
                "out_root": str(self._s.experiments_dir),
            }
            return self._create(base, "experiment", run_name, resume=False, resumed_from=None,
                                spec_extra=extra)
    
    def resume(self, job_id: str, max_iters: int | None = None) -> str:
        """New job continuing the same run from its last.pt (optionally with a higher max_iters)."""
        self._require_training()
        with self._lock:
            row = self._row(job_id)
            if row["status"] in (QUEUED, RUNNING):
                raise JobConflictError(f"Job is still {row['status']}; stop it first")
            spec = json.loads(row["spec"])
            if spec.get("kind") == "experiment":
                raise JobRejectedError(
                    "Experiment batches continue by submitting the same experiment again; "
                    "finished variants are skipped")
            run_name = spec["run_name"]
            if self.store.active_for_run(run_name):
                raise JobConflictError(f"A job for run {run_name!r} is already queued or running")
            self._check_queue()
            path = self._s.runs_dir / run_name / "last.pt"
            if not path.is_file():
                raise JobConflictError("There is no checkpoint to resume from")
            try:
                state = load_checkpoint(path, "cpu").get("training_state")
            except (RuntimeError, ValueError, OSError) as exc:
                raise JobConflictError(f"The checkpoint cannot be read: {exc}") from exc
            if state is None:
                raise JobRejectedError("The checkpoint has no training state to resume from")
            cfg = config_from_dict(spec["config"])
            new_max = max_iters if max_iters is not None else cfg.training.max_iters
            self._check_iters(new_max)
            done = int(state["iteration"])
            if done >= new_max:
                raise JobRejectedError(
                    f"The run is already at iteration {done}; pass max_iters above that")
            cfg = replace(cfg, training=replace(cfg.training, max_iters=new_max))
            return self._create(cfg, spec["model_type"], run_name, resume=True, resumed_from=job_id)

    def _create(self, cfg: Config, model_type: str, run_name: str, *, resume: bool,
            resumed_from: str | None, spec_extra: dict[str, Any] | None = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True)
        spec = {
            "job_id": job_id, "config": asdict(cfg), "model_type": model_type,
            "run_name": run_name, "out_root": str(self._s.runs_dir),
            "device": self._s.device, "resume": resume,
        }
        if spec_extra:
            spec.update(spec_extra)
        (job_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        self.store.insert({
            "id": job_id, "run_name": run_name, "model_type": model_type, "status": QUEUED,
            "created_at": _now(), "spec": json.dumps(spec), "resumed_from": resumed_from,
        })
        self._wake.set()
        return job_id

    def stop(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._row(job_id)
            if row["status"] == QUEUED:
                self.store.update(job_id, status=STOPPED, finished_at=_now(), stop_requested=1)
            elif row["status"] == RUNNING:
                cur = self._current
                if cur is None or cur.job_id != job_id:
                    raise JobConflictError("Job is not attached to a worker process")
                self._request_stop(cur)
            else:
                raise JobConflictError(f"Job is already {row['status']}")
            return self.view(job_id)

    # ---- queries --------------------------------------------------------------

    def _row(self, job_id: str) -> dict[str, Any]:
        if not JOB_ID_RE.fullmatch(job_id):
            raise JobNotFoundError(f"Unknown job {job_id!r}")
        row = self.store.get(job_id)
        if row is None:
            raise JobNotFoundError(f"Unknown job {job_id!r}")
        return row

    def view(self, job_id: str) -> dict[str, Any]:
        return self._view(self._row(job_id))

    def list_jobs(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [self._view(r) for r in self.store.list(status, limit)]

    def _view(self, row: dict[str, Any]) -> dict[str, Any]:
        spec = json.loads(row["spec"])
        cfg = spec["config"]
        kind = spec.get("kind", "training")
        metrics = read_jsonl(self.job_dir(row["id"]) / "metrics.jsonl")
        latest = {k: metrics[-1][k] for k in METRIC_KEYS} if metrics else None
        status = row["status"]
        detail: dict[str, Any] | None = None
        if status == COMPLETED:
            progress: float | None = 1.0
        elif latest is not None:
            progress = min(1.0, latest["iter"] / cfg["training"]["max_iters"])
        else:
            progress = 0.0 if status == QUEUED else None
        if kind == "experiment" and status != COMPLETED:
            prog = read_json(self.job_dir(row["id"]) / "progress.json")
            if prog and isinstance(prog.get("total"), int) and prog["total"] > 0:
                detail = prog
                progress = min(1.0, float(prog["done"]) / prog["total"])
        result = json.loads(row["result"]) if row["result"] else None
        if result is not None:
            result.pop("run_dir", None)  # don't expose server paths
        model = cfg["model"] if spec["model_type"] == "gpt" else {"block_size": cfg["model"]["block_size"]}
        return {
            "job_id": row["id"], "run_name": row["run_name"], "model_type": row["model_type"],
            "model_id": f"{row['run_name']}:best", "status": status, "kind": kind,
            "created_at": row["created_at"], "started_at": row["started_at"],
            "finished_at": row["finished_at"], "config": {"model": model, "training": cfg["training"]},
            "error": row["error"], "resumed_from": row["resumed_from"],
            "stop_requested": bool(row["stop_requested"]), "latest": latest,
            "progress": progress, "result": result, "device": spec["device"], "detail": detail,
        }

    def metrics(self, job_id: str, after_iter: int | None = None) -> list[dict[str, Any]]:
        self._row(job_id)
        rows = read_jsonl(self.job_dir(job_id) / "metrics.jsonl")
        rows = [{k: r[k] for k in METRIC_KEYS} for r in rows]
        return [r for r in rows if after_iter is None or r["iter"] > after_iter]

    def logs(self, job_id: str, tail: int) -> list[str]:
        self._row(job_id)
        return tail_lines(self.job_dir(job_id) / "log.txt", tail)