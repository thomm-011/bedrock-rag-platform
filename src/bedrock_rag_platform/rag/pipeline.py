"""Ties loaders, chunker, embedder, vector store, and LLM into ingest/ask."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from bedrock_rag_platform.chunking.chunker import HeadingAwareChunker
from bedrock_rag_platform.interfaces.models import SearchResult
from bedrock_rag_platform.interfaces.protocols import (
    DocumentLoader,
    Embedder,
    LLMProvider,
    VectorStore,
)
from bedrock_rag_platform.rag.prompts import build_grounded_prompt, build_no_rag_prompt


class IngestResult(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_stored: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class Citation(BaseModel):
    index: int
    source: str
    heading: str | None
    snippet: str
    score: float


class AskResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    usage: Usage


class RagPipeline:
    """Orchestrates ingestion and question answering over any backend set."""

    def __init__(
        self,
        llm: LLMProvider,
        embedder: Embedder,
        vector_store: VectorStore,
        loader: DocumentLoader,
        chunker: HeadingAwareChunker | None = None,
        max_context_chars: int = 6000,
        num_predict: int = 256,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.loader = loader
        self.chunker = chunker or HeadingAwareChunker()
        self.max_context_chars = max_context_chars
        self.num_predict = num_predict

    async def ingest_path(self, path: Path) -> IngestResult:
        documents = self.loader.load(path)
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunker.chunk(doc))

        if not all_chunks:
            return IngestResult(documents_loaded=len(documents), chunks_created=0, chunks_stored=0)

        texts = [c.text for c in all_chunks]
        embeddings = await self.embedder.embed(texts)
        for chunk, embedding in zip(all_chunks, embeddings, strict=True):
            chunk.embedding = embedding

        await self.vector_store.upsert(all_chunks)
        return IngestResult(
            documents_loaded=len(documents),
            chunks_created=len(all_chunks),
            chunks_stored=len(all_chunks),
        )

    async def retrieve(self, question: str, top_k: int) -> list[SearchResult]:
        [query_embedding] = await self.embedder.embed([question])
        return await self.vector_store.search(query_embedding, top_k)

    async def ask(self, question: str, top_k: int) -> AskResult:
        started = time.perf_counter()
        results = await self.retrieve(question, top_k)

        snippets: list[str] = []
        budget = self.max_context_chars
        for result in results:
            snippet = result.text[: max(budget, 0)]
            if not snippet:
                break
            snippets.append(snippet)
            budget -= len(snippet)

        prompt = build_grounded_prompt(question, snippets)
        llm_result = await self.llm.generate(prompt, max_tokens=self.num_predict)
        total_latency_ms = (time.perf_counter() - started) * 1000

        citations = [
            Citation(
                index=i + 1,
                source=result.source,
                heading=result.heading,
                snippet=snippet,
                score=result.score,
            )
            for i, (result, snippet) in enumerate(
                zip(results[: len(snippets)], snippets, strict=True)
            )
        ]

        return AskResult(
            question=question,
            answer=llm_result.text,
            citations=citations,
            usage=Usage(
                prompt_tokens=llm_result.prompt_tokens,
                completion_tokens=llm_result.completion_tokens,
                latency_ms=total_latency_ms,
            ),
        )

    async def ask_no_rag(self, question: str) -> AskResult:
        """Baseline: ask the LLM directly, no retrieval. Used by the eval's before/after row."""
        prompt = build_no_rag_prompt(question)
        llm_result = await self.llm.generate(prompt, max_tokens=self.num_predict)
        return AskResult(
            question=question,
            answer=llm_result.text,
            citations=[],
            usage=Usage(
                prompt_tokens=llm_result.prompt_tokens,
                completion_tokens=llm_result.completion_tokens,
                latency_ms=llm_result.latency_ms,
            ),
        )
