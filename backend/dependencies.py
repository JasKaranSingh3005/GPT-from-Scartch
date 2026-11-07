from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.core.config import Settings
from backend.services.model_registry import ModelRegistry


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]