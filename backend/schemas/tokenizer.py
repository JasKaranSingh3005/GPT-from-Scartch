from __future__ import annotations

from pydantic import Field

from backend.schemas.common import ApiModel


class EncodeRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)
    text: str = Field(max_length=20000)


class EncodedToken(ApiModel):
    char: str
    id: int


class EncodeResponse(ApiModel):
    model_id: str
    ids: list[int]
    tokens: list[EncodedToken]
    length: int
    vocab_size: int
    block_size: int | None  # None for models without a context limit (bigram)


class DecodeRequest(ApiModel):
    model_id: str = Field(min_length=1, max_length=130)
    ids: list[int] = Field(max_length=20000)


class DecodeResponse(ApiModel):
    model_id: str
    text: str


class VocabEntry(ApiModel):
    id: int
    char: str
    codepoint: str


class VocabularyResponse(ApiModel):
    model_id: str
    vocab_size: int
    entries: list[VocabEntry]