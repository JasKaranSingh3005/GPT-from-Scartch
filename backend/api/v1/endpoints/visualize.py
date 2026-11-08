from __future__ import annotations

from fastapi import APIRouter

from backend.dependencies import RegistryDep
from backend.schemas.visualization import (
    ArchitectureResponse,
    AttentionRequest,
    AttentionResponse,
    ForwardPassRequest,
    ForwardPassResponse,
)
from backend.services import visualization as viz

router = APIRouter(prefix="/visualize", tags=["visualization"])


@router.get("/architecture", response_model=ArchitectureResponse)
def architecture(model_id: str, registry: RegistryDep) -> ArchitectureResponse:
    meta = registry.metadata(model_id)  # reads the checkpoint, no model load needed
    return ArchitectureResponse(model_id=model_id, **viz.architecture(meta))


@router.post("/attention", response_model=AttentionResponse)
def attention(body: AttentionRequest, registry: RegistryDep) -> AttentionResponse:
    loaded = registry.load(body.model_id)
    return AttentionResponse(
        **viz.attention_for_prompt(registry, loaded, body.prompt, body.layer, body.max_tokens)
    )


@router.post("/forward-pass", response_model=ForwardPassResponse)
def forward_pass(body: ForwardPassRequest, registry: RegistryDep) -> ForwardPassResponse:
    loaded = registry.load(body.model_id)
    return ForwardPassResponse(
        **viz.forward_pass(registry, loaded, body.prompt, body.max_tokens)
    )