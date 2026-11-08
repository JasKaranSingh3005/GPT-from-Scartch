from __future__ import annotations

from typing import Any

import torch

from backend.services.inference import InvalidRequestError
from backend.services.model_registry import CheckpointMeta, LoadedModel, ModelRegistry


def require_gpt(model_type: str) -> None:
    if model_type != "gpt":
        raise InvalidRequestError(
            f"This view needs a GPT checkpoint; this one is a {model_type!r} model "
            "(it has no attention layers or Transformer blocks)."
        )


def _prepare(loaded: LoadedModel, prompt: str, max_tokens: int):
    """Crop to the last min(block_size, max_tokens) characters and encode them."""
    require_gpt(loaded.meta.model_type)
    if not prompt:
        raise InvalidRequestError("Enter a prompt")
    limit = min(loaded.model.block_size, max_tokens)
    used = prompt[-limit:]
    try:
        ids = loaded.tokenizer.encode(used)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    idx = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    tokens = [{"char": c, "id": i} for c, i in zip(used, ids)]
    return idx, tokens, len(prompt) > limit


def attention_for_prompt(
    registry: ModelRegistry, loaded: LoadedModel, prompt: str, layer: int, max_tokens: int
) -> dict[str, Any]:
    idx, tokens, cropped = _prepare(loaded, prompt, max_tokens)
    with registry.inference_lock, torch.inference_mode():
        _, _, weights = loaded.model(idx, return_weights=True)  # per layer: (1, n_head, T, T)
    if layer >= len(weights):
        raise InvalidRequestError(f"layer {layer} out of range; the model has {len(weights)} layers")
    w = weights[layer][0].float().cpu()  # (n_head, T, T)
    w = torch.round(w * 1e5) / 1e5
    return {
        "model_id": loaded.model_id, "tokens": tokens, "layer": layer,
        "n_layers": len(weights), "n_heads": int(w.shape[0]),
        "block_size": loaded.model.block_size, "cropped": cropped, "weights": w.tolist(),
    }


def forward_pass(
    registry: ModelRegistry, loaded: LoadedModel, prompt: str, max_tokens: int
) -> dict[str, Any]:
    """Run one real forward pass and record each module's output shape and statistics."""
    idx, tokens, cropped = _prepare(loaded, prompt, max_tokens)
    model = loaded.model
    names = ["token_embedding", "position_embedding"]
    names += [f"blocks.{i}" for i in range(len(model.blocks))] + ["ln_f", "lm_head"]
    modules = dict(model.named_modules())
    stages: list[dict[str, Any]] = []
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            t = output.detach().float()
            stages.append({
                "name": name, "shape": list(t.shape), "mean": t.mean().item(),
                "std": t.std().item(), "min": t.min().item(), "max": t.max().item(),
            })
        return hook

    try:
        for name in names:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        with registry.inference_lock, torch.inference_mode():
            model(idx)
    finally:
        for h in handles:
            h.remove()
    return {"model_id": loaded.model_id, "tokens": tokens, "cropped": cropped, "stages": stages}


def _params(meta: CheckpointMeta, prefix: str) -> int:
    return sum(t.numel for t in meta.tensors if t.name.startswith(prefix))


def architecture(meta: CheckpointMeta) -> dict[str, Any]:
    require_gpt(meta.model_type)
    kw = meta.model_kwargs
    try:
        C, H, L = int(kw["n_embd"]), int(kw["n_head"]), int(kw["n_layer"])
        S, V = int(kw["block_size"]), int(kw["vocab_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidRequestError(f"Checkpoint config is incomplete: {exc}") from exc
    dropout = float(kw.get("dropout", 0.0))

    def stage(key, label, i, o, params, **details):
        return {"key": key, "label": label, "input_shape": i, "output_shape": o,
                "parameters": params, "details": details}

    stages = [
        stage("tokenizer", "Tokenizer (character level)", "text", "(B, T) token IDs", 0,
              vocab_size=V, max_length=S),
        stage("token_embedding", "Token embedding", "(B, T)", f"(B, T, {C})",
              _params(meta, "token_embedding."), table=f"{V} x {C}"),
        stage("position_embedding", "Positional embedding (added to token embedding)", "(T,)",
              f"(T, {C})", _params(meta, "position_embedding."), table=f"{S} x {C}"),
        stage("blocks", f"Transformer blocks x {L}", f"(B, T, {C})", f"(B, T, {C})",
              _params(meta, "blocks."), n_layer=L, n_head=H, head_size=C // H,
              ffn_hidden=4 * C, per_block=_params(meta, "blocks.0.")),
        stage("ln_f", "Final LayerNorm", f"(B, T, {C})", f"(B, T, {C})", _params(meta, "ln_f.")),
        stage("lm_head", "Linear language-modelling head", f"(B, T, {C})", f"(B, T, {V})",
              _params(meta, "lm_head."), logits=V),
        stage("softmax", "Softmax and sampling (last position)", f"(B, {V}) logits", "(B, 1) token", 0,
              temperature="applied before softmax", top_k="optional"),
    ]
    total = sum(s["parameters"] for s in stages)
    return {
        "config": {"n_embd": C, "n_head": H, "n_layer": L, "block_size": S,
                   "vocab_size": V, "dropout": dropout},
        "stages": stages, "parameters_total": meta.parameters,
        "stage_sum_matches_total": total == meta.parameters,
    }