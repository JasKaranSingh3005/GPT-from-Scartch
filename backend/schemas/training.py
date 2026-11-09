from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from backend.schemas.common import ApiModel

JobStatus = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "STOPPED"]


class TrainingJobRequest(ApiModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    model_type: Literal["gpt", "bigram"] = "gpt"
    n_embd: int = Field(default=64, ge=8, le=512)
    n_head: int = Field(default=4, ge=1, le=16)
    n_layer: int = Field(default=4, ge=1, le=12)
    block_size: int = Field(default=64, ge=2, le=256)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    batch_size: int = Field(default=32, ge=1, le=128)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    max_iters: int = Field(default=1000, ge=1)
    eval_interval: int = Field(default=100, ge=1)
    eval_iters: int = Field(default=50, ge=1, le=500)
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def heads_divide_width(self) -> "TrainingJobRequest":
        if self.model_type == "gpt" and self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        return self


class ResumeRequest(ApiModel):
    max_iters: int | None = Field(default=None, ge=1)


class MetricRow(ApiModel):
    iter: int
    train_loss: float
    val_loss: float
    elapsed_s: float


class TrainingJob(ApiModel):
    job_id: str
    run_name: str
    model_type: str
    model_id: str  # "<run>:best", usable in the model endpoints once a checkpoint exists
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    device: str  # the server's device setting; the worker resolves "auto" itself
    config: dict[str, Any]
    error: str | None
    resumed_from: str | None
    stop_requested: bool
    latest: MetricRow | None
    progress: float | None
    result: dict[str, Any] | None
    kind: str = "training"
    detail: dict[str, Any] | None = None


class TrainingJobList(ApiModel):
    jobs: list[TrainingJob]


class MetricsResponse(ApiModel):
    job_id: str
    status: JobStatus
    rows: list[MetricRow]


class LogsResponse(ApiModel):
    job_id: str
    status: JobStatus
    lines: list[str]