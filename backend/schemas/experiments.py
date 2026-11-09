from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from backend.schemas.common import ApiModel


class ExperimentVariantInfo(ApiModel):
    name: str
    label: str
    model_type: str
    overrides: dict[str, Any]
    model: dict[str, Any]


class ExperimentInfo(ApiModel):
    key: str
    title: str
    variants: list[ExperimentVariantInfo]


class ExperimentList(ApiModel):
    base_model: dict[str, Any]
    experiments: list[ExperimentInfo]


class ExperimentRunRequest(ApiModel):
    experiments: list[str] | None = None  # None = all experiments
    max_iters: int = Field(default=1000, ge=1)
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)
    force: bool = False  # rerun variants that already have results


class BatchSummary(ApiModel):
    batch_id: str
    max_iters: int
    seed: int
    variants_finished: int
    variants_total: int
    modified: datetime


class BatchList(ApiModel):
    runs: list[BatchSummary]


class BatchDetail(ApiModel):
    batch_id: str
    max_iters: int
    seed: int
    rows: list[dict[str, Any]]
    curves: dict[str, list[dict[str, float]]]
    batch: dict[str, Any] | None


class CompareVariant(ApiModel):
    variant: str
    label: str
    best_val_loss: dict[str, float | None]
    parameters: dict[str, int | None]


class CompareResponse(ApiModel):
    run_ids: list[str]
    variants: list[CompareVariant]