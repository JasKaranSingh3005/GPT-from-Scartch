from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.jobs import (
    COMPLETED, FAILED, STOPPED, JobManager, JobNotFoundError,
)

router = APIRouter()

POLL_SECONDS = 0.5
HEARTBEAT_SECONDS = 15.0
TERMINAL = {COMPLETED: "training.completed", FAILED: "training.failed", STOPPED: "training.stopped"}


def event(name: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": name, "data": data}


class TrainingStream:
    """Turns the job's files into a sequence of events. Pure logic, no socket."""

    def __init__(self, jobs: JobManager, job_id: str) -> None:
        self._jobs = jobs
        self._job_id = job_id
        self._last_iter: int | None = None
        self._best = float("inf")
        self._started = False

    def poll(self) -> tuple[list[dict[str, Any]], bool]:
        """New events since the last call, and whether the stream is finished."""
        job = self._jobs.view(self._job_id)  # raises JobNotFoundError
        status = job["status"]
        out: list[dict[str, Any]] = []
        if not self._started and status != "QUEUED":
            self._started = True
            out.append(event("training.started", {
                "job_id": self._job_id, "status": status, "run_name": job["run_name"],
                "config": job["config"],
            }))
        for row in self._jobs.metrics(self._job_id, self._last_iter):
            self._last_iter = int(row["iter"])
            out.append(event("training.metrics", {**row, "job_id": self._job_id, "status": status}))
            if row["val_loss"] < self._best:
                self._best = row["val_loss"]
                out.append(event("training.checkpoint_saved", {
                    "job_id": self._job_id, "kind": "best", "iter": row["iter"],
                    "val_loss": row["val_loss"],
                }))
        if status in TERMINAL:
            # Re-read once so rows written just before the worker exited are not lost.
            for row in self._jobs.metrics(self._job_id, self._last_iter):
                self._last_iter = int(row["iter"])
                out.append(event("training.metrics", {**row, "job_id": self._job_id, "status": status}))
            out.append(event(TERMINAL[status], {
                "job_id": self._job_id, "status": status, "error": job["error"],
                "result": job["result"],
            }))
            return out, True
        return out, False


def origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    allowed = websocket.app.state.settings.cors_origins
    return origin is None or origin in allowed  # non-browser clients send no Origin


@router.websocket("/ws/training/{job_id}")
async def training_socket(websocket: WebSocket, job_id: str) -> None:
    if not origin_allowed(websocket):
        await websocket.close(code=4403)
        return
    jobs: JobManager = websocket.app.state.jobs
    await websocket.accept()
    stream = TrainingStream(jobs, job_id)
    last_sent = time.monotonic()
    try:
        while True:
            try:
                events, done = await asyncio.to_thread(stream.poll)
            except JobNotFoundError as exc:
                await websocket.send_json(event("error", {"detail": str(exc)}))
                await websocket.close(code=4404)
                return
            for ev in events:
                await websocket.send_json(ev)
                last_sent = time.monotonic()
            if done:
                await websocket.close()
                return
            if time.monotonic() - last_sent > HEARTBEAT_SECONDS:
                await websocket.send_json(event("ping", {}))
                last_sent = time.monotonic()
            await asyncio.sleep(POLL_SECONDS)
    except (WebSocketDisconnect, RuntimeError):
        return  # client went away