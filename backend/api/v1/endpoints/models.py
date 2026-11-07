from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.dependencies import RegistryDep
from backend.schemas.models import (
    LoadRequest,
    LoadResponse,
    ModelConfigResponse,
    ModelDetail,
    ModelListResponse,
    ModelSummary,
    ParametersResponse,
    TensorShape,
    UnloadResponse,
)
from backend.services.model_registry import CheckpointFile, ModelRegistry, gpt_breakdown

router = APIRouter(prefix="/models", tags=["models"])


def _base(file: CheckpointFile, registry: ModelRegistry) -> dict[str, Any]:
    return dict(
        model_id=file.model_id, run=file.run, kind=file.kind, size_bytes=file.size_bytes,
        modified=file.modified, loaded=registry.is_loaded(file.model_id),
    )


@router.get("", response_model=ModelListResponse)
def list_models(registry: RegistryDep) -> ModelListResponse:
    """Lists checkpoint files only. Nothing is read or loaded, so a corrupt file still
    appears here and fails (422) when opened."""
    return ModelListResponse(
        models=[ModelSummary(**_base(f, registry)) for f in registry.discover()]
    )


@router.post("/load", response_model=LoadResponse)
def load_model(body: LoadRequest, registry: RegistryDep) -> LoadResponse:
    loaded = registry.load(body.model_id)
    return LoadResponse(
        model_id=loaded.model_id, loaded=True, device=str(loaded.device),
        loaded_at=loaded.loaded_at, parameters=loaded.meta.parameters,
    )


@router.post("/unload", response_model=UnloadResponse)
def unload_model(body: LoadRequest, registry: RegistryDep) -> UnloadResponse:
    return UnloadResponse(model_id=body.model_id, unloaded=registry.unload(body.model_id))


@router.get("/{model_id}", response_model=ModelDetail)
def model_detail(model_id: str, registry: RegistryDep) -> ModelDetail:
    meta = registry.metadata(model_id)
    summary = registry.run_summary(model_id)
    return ModelDetail(
        **_base(meta.file, registry),
        model_type=meta.model_type, iteration=meta.iteration,
        parameters=meta.parameters, vocab_size=meta.vocab_size,
        run_best_val_loss=summary.best_val_loss,
        run_final_val_loss=summary.final_val_loss,
        run_training_time_s=summary.training_time_s,
    )


@router.get("/{model_id}/config", response_model=ModelConfigResponse)
def model_config(model_id: str, registry: RegistryDep) -> ModelConfigResponse:
    meta = registry.metadata(model_id)
    kw = meta.model_kwargs
    n_embd, n_head, block = kw.get("n_embd"), kw.get("n_head"), kw.get("block_size")
    head_size = n_embd // n_head if isinstance(n_embd, int) and isinstance(n_head, int) and n_head else None
    return ModelConfigResponse(
        model_id=model_id, model_type=meta.model_type, model_kwargs=kw,
        head_size=head_size, context_length=block if isinstance(block, int) else None,
    )


@router.get("/{model_id}/parameters", response_model=ParametersResponse)
def model_parameters(model_id: str, registry: RegistryDep) -> ParametersResponse:
    meta = registry.metadata(model_id)
    breakdown = gpt_breakdown(meta)
    return ParametersResponse(
        model_id=model_id,
        total=meta.parameters,
        breakdown=breakdown,
        breakdown_matches_actual=None if breakdown is None else sum(breakdown.values()) == meta.parameters,
        tensors=[TensorShape(name=t.name, shape=list(t.shape), numel=t.numel) for t in meta.tensors],
    )