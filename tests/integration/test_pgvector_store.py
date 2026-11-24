"""Integration test against a real Postgres+pgvector instance.

Run via `make test-integration` (starts the compose `db` service) or against
any reachable Postgres with the pgvector extension available. Skipped by
default; CI runs it with a `pgvector/pgvector:pg16` service container and a
FakeEmbedder-sized dimension (see .github/workflows/ci.yml).
"""

from __future__ import annotations

import os

import pytest

from bedrock_rag_platform.interfaces.models import Chunk
from bedrock_rag_platform.providers.pgvector_store import PgVectorStore

pytestmark = pytest.mark.integration

DSN = os.environ.get("RAG_TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
DIM = 8


@pytest.fixture
def store() -> PgVectorStore:
    store = PgVectorStore(dsn=DSN, dimensions=DIM, hnsw_m=16, hnsw_ef_construction=64)
    store.ensure_schema()
    return store


async def test_upsert_and_search_roundtrip(store: PgVectorStore) -> None:
    chunks = [
        Chunk(
            chunk_id="it-a",
            source="itest.md",
            heading="A",
            text="Karpenter provisions nodes.",
            embedding=[1.0] + [0.0] * (DIM - 1),
        ),
        Chunk(
            chunk_id="it-b",
            source="itest.md",
            heading="B",
            text="KEDA scales on event sources.",
            embedding=[0.0, 1.0] + [0.0] * (DIM - 2),
        ),
    ]
    await store.upsert(chunks)

    results = await store.search([1.0] + [0.0] * (DIM - 1), top_k=1)
    assert results[0].chunk_id == "it-a"
    assert await store.count() >= 2


async def test_upsert_without_embedding_raises(store: PgVectorStore) -> None:
    chunk = Chunk(chunk_id="it-c", source="itest.md", heading=None, text="no vector")
    with pytest.raises(ValueError, match="no embedding"):
        await store.upsert([chunk])
