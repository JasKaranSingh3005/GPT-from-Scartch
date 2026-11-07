from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.api.v1.router import api_router
from backend.core.config import Settings, resolve_device
from backend.core.errors import install_error_handlers
from backend.services.model_registry import ModelRegistry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    device = resolve_device(settings.device)  # fails fast if cuda was requested but unusable
    registry = ModelRegistry(settings.runs_dir, device, settings.max_loaded_models)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        registry.unload_all()

    app = FastAPI(
        title="GPT Lab API",
        version=__version__,
        description="HTTP API over the from-scratch PyTorch GPT in this repository.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.started_at = time.monotonic()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()