from __future__ import annotations

from pydantic import Field

from backend.schemas.common import ApiModel
from backend.schemas.tokenizer import EncodedToken


class AttentionRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)
    prompt: str = Field(max_length=4000)
    layer: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=64, ge=1, le=128)


class AttentionResponse(ApiModel):
    model_id: str
    tokens: list[EncodedToken]
    layer: int
    n_layers: int
    n_heads: int
    block_size: int
    cropped: bool
    # weights[head][query][key]; each row sums to ~1; keys after the query are exactly 0
    weights: list[list[list[float]]]


class ForwardPassRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)
    prompt: str = Field(max_length=4000)
    max_tokens: int = Field(default=64, ge=1, le=128)


class TensorStat(ApiModel):
    name: str
    shape: list[int]
    mean: float
    std: float
    min: float
    max: float


class ForwardPassResponse(ApiModel):
    model_id: str
    tokens: list[EncodedToken]
    cropped: bool
    stages: list[TensorStat]


class ArchitectureStage(ApiModel):
    key: str
    label: str
    input_shape: str
    output_shape: str
    parameters: int
    details: dict[str, int | str]


class ArchitectureResponse(ApiModel):
    model_id: str
    config: dict[str, int | float]
    stages: list[ArchitectureStage]
    parameters_total: int
    stage_sum_matches_total: bool