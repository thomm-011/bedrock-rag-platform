"""Protocol (structural) interfaces for every pluggable backend.

Defining these first lets local (Ollama/pgvector) and future AWS (Bedrock/
OpenSearch Serverless, M4) implementations satisfy the exact same contract,
so the API, CLI, and tests never need to know which backend is active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from bedrock_rag_platform.interfaces.models import Chunk, LLMResult, RawDocument, SearchResult


@runtime_checkable
class LLMProvider(Protocol):
    """Generates grounded answers from a prompt."""

    async def generate(self, prompt: str, *, max_tokens: int = 256) -> LLMResult: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists and searches chunk embeddings."""

    async def upsert(self, chunks: list[Chunk]) -> None: ...

    async def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]: ...

    async def count(self) -> int: ...


@runtime_checkable
class DocumentLoader(Protocol):
    """Loads raw documents from a path (file or directory)."""

    def load(self, path: Path) -> list[RawDocument]: ...

    def can_load(self, path: Path) -> bool: ...
