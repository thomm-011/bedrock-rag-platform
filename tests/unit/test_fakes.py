from __future__ import annotations

from bedrock_rag_platform.interfaces.models import Chunk
from tests.fakes import FakeEmbedder, InMemoryVectorStore


async def test_fake_embedder_is_deterministic() -> None:
    embedder = FakeEmbedder(dimensions=8)
    v1 = await embedder.embed(["hello"])
    v2 = await embedder.embed(["hello"])
    assert v1 == v2
    assert len(v1[0]) == 8


async def test_in_memory_store_ranks_by_similarity() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            Chunk(chunk_id="a", source="s", heading=None, text="t", embedding=[1.0, 0.0]),
            Chunk(chunk_id="b", source="s", heading=None, text="t", embedding=[0.0, 1.0]),
        ]
    )
    results = await store.search([1.0, 0.0], top_k=1)
    assert results[0].chunk_id == "a"
    assert await store.count() == 2
