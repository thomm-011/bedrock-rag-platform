"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from bedrock_rag_platform.chunking.chunker import HeadingAwareChunker
from bedrock_rag_platform.loaders.composite_loader import CompositeLoader
from bedrock_rag_platform.rag.pipeline import RagPipeline
from tests.fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=16)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def pipeline(
    fake_llm: FakeLLM, fake_embedder: FakeEmbedder, in_memory_store: InMemoryVectorStore
) -> RagPipeline:
    return RagPipeline(
        llm=fake_llm,
        embedder=fake_embedder,
        vector_store=in_memory_store,
        loader=CompositeLoader(),
        chunker=HeadingAwareChunker(chunk_size_chars=200, chunk_overlap_chars=30),
    )
