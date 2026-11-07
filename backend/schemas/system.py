from __future__ import annotations

from backend.schemas.common import ApiModel


class HealthResponse(ApiModel):
    status: str
    version: str
    uptime_seconds: float


class GPUInfo(ApiModel):
    index: int
    name: str
    total_memory_bytes: int
    compute_capability: str


class Limits(ApiModel):
    max_new_tokens: int
    max_loaded_models: int


class SystemInfo(ApiModel):
    app_version: str
    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device: str
    gpus: list[GPUInfo]
    cpu_count: int | None
    runs_dir_exists: bool
    checkpoints_found: int
    models_loaded: int
    limits: Limits