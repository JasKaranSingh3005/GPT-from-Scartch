from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import torch

from backend.services.model_registry import LoadedModel, ModelRegistry
from models.sampling import encode_prompt, generate_stream


class InvalidRequestError(Exception):
    """Request is well-formed but not acceptable, e.g. unknown characters (HTTP 422)."""


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int | None
    seed: int | None


@dataclass(frozen=True)
class GenerationResult:
    model_id: str
    prompt: str
    completion: str
    tokens_generated: int
    prompt_tokens: int
    seconds: float
    device: str
    seed: int | None

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_generated / self.seconds if self.seconds > 0 else 0.0


def check_limits(params: GenerationParams, max_new_tokens_limit: int) -> None:
    if params.max_new_tokens > max_new_tokens_limit:
        raise InvalidRequestError(
            f"max_new_tokens {params.max_new_tokens} exceeds the server limit "
            f"of {max_new_tokens_limit}"
        )


def prepare(loaded: LoadedModel, prompt: str) -> torch.Tensor:
    """Encode the prompt onto the model's device. Names any unknown characters."""
    try:
        ids = encode_prompt(loaded.tokenizer, prompt)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    return torch.tensor([ids], dtype=torch.long, device=loaded.device)


def make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    """Private RNG stream for one request. None means 'use the global RNG' (unseeded)."""
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def stream_tokens(
    registry: ModelRegistry, loaded: LoadedModel, params: GenerationParams
) -> Iterator[int]:
    """Yield generated token IDs.

    The inference lock is held only while one token is computed, never across a yield,
    so an abandoned or disconnected stream cannot keep other requests waiting.
    """
    idx = prepare(loaded, params.prompt)
    gen = make_generator(loaded.device, params.seed)
    stream = generate_stream(
        loaded.model, idx, params.max_new_tokens, params.temperature, params.top_k, gen
    )
    try:
        while True:
            with registry.inference_lock:
                try:
                    token = next(stream)
                except StopIteration:
                    return
            yield token
    finally:
        stream.close()


def run_generation(
    registry: ModelRegistry, loaded: LoadedModel, params: GenerationParams
) -> GenerationResult:
    prompt_ids = prepare(loaded, params.prompt).shape[1]
    start = time.perf_counter()
    ids = list(stream_tokens(registry, loaded, params))
    if loaded.device.type == "cuda":
        torch.cuda.synchronize(loaded.device)  # make the timing include queued GPU work
    seconds = time.perf_counter() - start
    return GenerationResult(
        model_id=loaded.model_id,
        prompt=params.prompt,
        completion=loaded.tokenizer.decode(ids),
        tokens_generated=len(ids),
        prompt_tokens=prompt_ids,
        seconds=seconds,
        device=str(loaded.device),
        seed=params.seed,
    )