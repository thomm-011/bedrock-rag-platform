"""Deterministic fakes used by unit tests (no network, no Postgres, no Ollama)."""

from __future__ import annotations

import hashlib

from bedrock_rag_platform.interfaces.models import Chunk, LLMResult, SearchResult


class FakeEmbedder:
    """Deterministic hashing embedder: same text -> same vector, every run."""

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._hash_vector(text) for text in texts]

    def _hash_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        values = [digest[i % len(digest)] for i in range(self.dimensions)]
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]


class FakeLLM:
    """Returns a canned/echoed answer instead of calling a real model."""

    def __init__(self, fixed_answer: str | None = None) -> None:
        self.fixed_answer = fixed_answer
        self.prompts: list[str] = []

    async def generate(self, prompt: str, *, max_tokens: int = 256) -> LLMResult:
        self.prompts.append(prompt)
        text = self.fixed_answer if self.fixed_answer is not None else f"echo:{len(prompt)}"
        return LLMResult(
            text=text, prompt_tokens=len(prompt.split()), completion_tokens=5, latency_ms=1.0
        )


class InMemoryVectorStore:
    """Cosine-similarity search over an in-process list; no DB needed."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    async def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        scored = []
        for chunk in self._chunks.values():
            assert chunk.embedding is not None
            score = self._cosine(query_embedding, chunk.embedding)
            scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                heading=chunk.heading,
                text=chunk.text,
                score=score,
            )
            for score, chunk in scored[:top_k]
        ]

    async def count(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sum(x * x for x in a) ** 0.5 or 1.0
        norm_b = sum(y * y for y in b) ** 0.5 or 1.0
        return dot / (norm_a * norm_b)
