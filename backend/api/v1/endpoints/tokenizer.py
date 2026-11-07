from __future__ import annotations

from fastapi import APIRouter

from backend.dependencies import RegistryDep
from backend.schemas.tokenizer import (
    DecodeRequest,
    DecodeResponse,
    EncodedToken,
    EncodeRequest,
    EncodeResponse,
    VocabEntry,
    VocabularyResponse,
)
from backend.services.inference import InvalidRequestError

router = APIRouter(prefix="/tokenizer", tags=["tokenizer"])


@router.post("/encode", response_model=EncodeResponse)
def encode(body: EncodeRequest, registry: RegistryDep) -> EncodeResponse:
    meta = registry.metadata(body.model_id)  # reads the checkpoint, no model load needed

    tokenizer = _tokenizer(registry, body.model_id)
    try:
        ids = tokenizer.encode(body.text)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    block = meta.model_kwargs.get("block_size")
    return EncodeResponse(
        model_id=body.model_id, ids=ids,
        tokens=[EncodedToken(char=c, id=i) for c, i in zip(body.text, ids)],
        length=len(ids), vocab_size=tokenizer.vocab_size,
        block_size=block if isinstance(block, int) else None,
    )


@router.post("/decode", response_model=DecodeResponse)
def decode(body: DecodeRequest, registry: RegistryDep) -> DecodeResponse:
    tokenizer = _tokenizer(registry, body.model_id)
    try:
        return DecodeResponse(model_id=body.model_id, text=tokenizer.decode(body.ids))
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


@router.get("/vocabulary", response_model=VocabularyResponse)
def vocabulary(model_id: str, registry: RegistryDep) -> VocabularyResponse:
    tokenizer = _tokenizer(registry, model_id)
    return VocabularyResponse(
        model_id=model_id, vocab_size=tokenizer.vocab_size,
        entries=[VocabEntry(id=i, char=c, codepoint=f"U+{ord(c):04X}")
                 for i, c in enumerate(tokenizer.chars)],
    )


def _tokenizer(registry, model_id: str):
    """The tokenizer stored inside the checkpoint (the one the model was trained with)."""
    from models.tokenizer import CharTokenizer
    from training.checkpoint import load_checkpoint

    file = registry.resolve(model_id)
    loaded = registry.get_loaded(model_id)
    if loaded is not None:
        return loaded.tokenizer
    try:
        ckpt = load_checkpoint(file.path, "cpu")
        return CharTokenizer(ckpt["chars"])
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        from backend.services.model_registry import CheckpointError

        raise CheckpointError(f"Checkpoint {model_id!r} is unusable: {exc}") from exc