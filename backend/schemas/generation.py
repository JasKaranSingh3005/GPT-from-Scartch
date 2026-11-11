from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from backend.schemas.common import ApiModel


class GenerateRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)
    prompt: str = Field(default="", max_length=4000)
    max_new_tokens: int = Field(default=200, ge=1)
    temperature: float = Field(default=0.8, gt=0, le=5.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class GenerateResponse(ApiModel):
    model_id: str
    prompt: str
    text: str  # prompt + continuation
    completion: str
    tokens_generated: int
    prompt_tokens: int
    seconds: float
    tokens_per_second: float
    device: str
    seed: int | None
    # False means a CUDA seed reproduces only on CUDA (and CPU only on CPU).
    seed_note: str | None


class CompareVariant(ApiModel):
    label: str = Field(min_length=1, max_length=60)
    model_id: str | None = Field(default=None, min_length=1, max_length=130)
    temperature: float | None = Field(default=None, gt=0, le=5.0)
    top_k: int | None = Field(default=None, ge=1)


class CompareRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)  # default for variants
    prompt: str = Field(default="", max_length=4000)
    max_new_tokens: int = Field(default=200, ge=1)
    temperature: float = Field(default=0.8, gt=0, le=5.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)  # same seed for every variant
    variants: list[CompareVariant] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def labels_unique(self) -> "CompareRequest":
        labels = [v.label for v in self.variants]
        if len(set(labels)) != len(labels):
            raise ValueError("variant labels must be unique")
        return self


class CompareResult(ApiModel):
    label: str
    result: GenerateResponse


class CompareResponse(ApiModel):
    seed: int
    results: list[CompareResult]


StreamEventType = Literal["start", "token", "done", "error"]

class DistributionRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)
    prompt: str = Field(default="", max_length=4000)
    temperature: float = Field(default=0.8, gt=0, le=5.0)
    top_k: int | None = Field(default=None, ge=1)
    top_n: int = Field(default=12, ge=1, le=100)  # how many candidates to return


class DistributionEntry(ApiModel):
    id: int
    char: str
    probability: float


class DistributionResponse(ApiModel):
    model_id: str
    prompt_tokens: int  # tokens the model actually saw (after cropping)
    cropped: bool
    temperature: float
    top_k: int | None
    shown_mass: float  # total probability of the returned entries
    entries: list[DistributionEntry]  # most likely first