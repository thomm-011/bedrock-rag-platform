"""Shared data models used across interfaces and implementations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """A single loaded document before chunking."""

    source: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of a document, ready to embed and store."""

    chunk_id: str
    source: str
    heading: str | None = None
    text: str
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    """A retrieved chunk with its similarity score."""

    chunk_id: str
    source: str
    heading: str | None
    text: str
    score: float


class LLMResult(BaseModel):
    """Output of an LLMProvider.generate call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
