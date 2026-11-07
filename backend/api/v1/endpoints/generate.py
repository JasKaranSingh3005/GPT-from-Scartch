from __future__ import annotations
from backend.services.inference import make_generator, prepare  # noqa: F401  (prefetch)
import json
from typing import Iterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.dependencies import RegistryDep, SettingsDep
from backend.schemas.generation import (
    CompareRequest,
    CompareResponse,
    CompareResult,
    GenerateRequest,
    GenerateResponse,
)
from backend.services.inference import (
    GenerationParams,
    GenerationResult,
    check_limits,
    run_generation,
    stream_tokens,
)
from backend.services.model_registry import LoadedModel, ModelRegistry
import torch
import time 

router = APIRouter(prefix="/generate", tags=["generation"])

SEED_NOTE = "Seeded output is reproducible on the same device type only (CPU vs CUDA differ)."


def _params(body: GenerateRequest | CompareRequest, **over) -> GenerationParams:
    values = dict(
        prompt=body.prompt, max_new_tokens=body.max_new_tokens,
        temperature=body.temperature, top_k=body.top_k, seed=getattr(body, "seed", None),
    )
    values.update({k: v for k, v in over.items() if v is not None})
    return GenerationParams(**values)


def _response(result: GenerationResult) -> GenerateResponse:
    return GenerateResponse(
        model_id=result.model_id, prompt=result.prompt,
        text=result.prompt + result.completion, completion=result.completion,
        tokens_generated=result.tokens_generated, prompt_tokens=result.prompt_tokens,
        seconds=result.seconds, tokens_per_second=result.tokens_per_second,
        device=result.device, seed=result.seed,
        seed_note=SEED_NOTE if result.seed is not None else None,
    )


def _loaded(registry: ModelRegistry, model_id: str) -> LoadedModel:
    """Loads on demand (idempotent), so the Playground never needs a separate load step."""
    return registry.load(model_id)


@router.post("", response_model=GenerateResponse)
def generate(body: GenerateRequest, registry: RegistryDep, settings: SettingsDep) -> GenerateResponse:
    params = _params(body)
    check_limits(params, settings.max_new_tokens)
    loaded = _loaded(registry, body.model_id)
    return _response(run_generation(registry, loaded, params))


@router.post("/stream")
def generate_stream_endpoint(
    body: GenerateRequest, registry: RegistryDep, settings: SettingsDep
) -> StreamingResponse:
    """Server-Sent Events: `start`, one `token` per sampled token, then `done` (or `error`).

    Validation errors raise before streaming starts, so they come back as normal 4xx.
    """
    params = _params(body)
    check_limits(params, settings.max_new_tokens)
    loaded = _loaded(registry, body.model_id)
    prepare(loaded, params.prompt)  # raises InvalidRequestError for unknown characters

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def events() -> Iterator[str]:
        yield sse("start", {"model_id": loaded.model_id, "device": str(loaded.device),
                            "seed": params.seed, "max_new_tokens": params.max_new_tokens})
        start = time.perf_counter()
        count = 0
        try:
            for token_id in stream_tokens(registry, loaded, params):
                count += 1
                yield sse("token", {"id": token_id,
                                    "text": loaded.tokenizer.itos[token_id], "index": count})
        except Exception as exc:  # report mid-stream failures as an event, not a hung socket
            yield sse("error", {"detail": str(exc)})
            return
        if loaded.device.type == "cuda":
            torch.cuda.synchronize(loaded.device)
        seconds = time.perf_counter() - start
        yield sse("done", {"tokens_generated": count, "seconds": seconds,
                           "tokens_per_second": count / seconds if seconds > 0 else 0.0})

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/compare", response_model=CompareResponse)
def compare(body: CompareRequest, registry: RegistryDep, settings: SettingsDep) -> CompareResponse:
    """Run each variant with the SAME seed. Variants can override model, temperature, top-k."""
    results: list[CompareResult] = []
    for variant in body.variants:
        params = _params(body, temperature=variant.temperature)
        if variant.top_k is not None:
            params = GenerationParams(params.prompt, params.max_new_tokens,
                                      params.temperature, variant.top_k, params.seed)
        check_limits(params, settings.max_new_tokens)
        loaded = _loaded(registry, variant.model_id or body.model_id)
        results.append(CompareResult(label=variant.label,
                                     result=_response(run_generation(registry, loaded, params))))
    return CompareResponse(seed=body.seed, results=results)