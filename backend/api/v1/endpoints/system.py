from __future__ import annotations

import os
import platform
import time

import torch
from fastapi import APIRouter, Request

from backend import __version__
from backend.dependencies import RegistryDep, SettingsDep
from backend.schemas.system import GPUInfo, HealthResponse, Limits, SystemInfo

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=time.monotonic() - request.app.state.started_at,
    )


@router.get("/system/info", response_model=SystemInfo)
def system_info(settings: SettingsDep, registry: RegistryDep) -> SystemInfo:
    """Everything here is read from the running process, not from constants."""
    cuda = torch.cuda.is_available()
    gpus: list[GPUInfo] = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory_bytes=int(props.total_memory),
                compute_capability=f"{props.major}.{props.minor}",
            ))
    return SystemInfo(
        app_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=cuda,
        cuda_version=torch.version.cuda,
        device=str(registry.device),
        gpus=gpus,
        cpu_count=os.cpu_count(),
        runs_dir_exists=settings.runs_dir.is_dir(),
        checkpoints_found=len(registry.discover()),
        models_loaded=len(registry.loaded_ids()),
        limits=Limits(
            max_new_tokens=settings.max_new_tokens,
            max_loaded_models=settings.max_loaded_models,
        ),
    )