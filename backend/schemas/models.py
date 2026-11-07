from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from backend.schemas.common import ApiModel


class ModelSummary(ApiModel):
    model_id: str
    run: str
    kind: Literal["best", "last"]
    size_bytes: int
    modified: datetime
    loaded: bool


class ModelListResponse(ApiModel):
    models: list[ModelSummary]


class ModelDetail(ModelSummary):
    model_type: str
    iteration: int
    parameters: int
    vocab_size: int
    run_best_val_loss: float | None
    run_final_val_loss: float | None
    run_training_time_s: float | None


class ModelConfigResponse(ApiModel):
    model_id: str
    model_type: str
    model_kwargs: dict[str, Any]
    head_size: int | None
    context_length: int | None


class TensorShape(ApiModel):
    name: str
    shape: list[int]
    numel: int


class ParametersResponse(ApiModel):
    model_id: str
    total: int
    breakdown: dict[str, int] | None
    breakdown_matches_actual: bool | None
    tensors: list[TensorShape]


class LoadRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)


class LoadResponse(ApiModel):
    model_id: str
    loaded: bool
    device: str
    loaded_at: datetime
    parameters: int


class UnloadResponse(ApiModel):
    model_id: str
    unloaded: bool