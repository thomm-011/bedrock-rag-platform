from __future__ import annotations

import pytest

from bedrock_rag_platform.providers.bedrock_stubs import (
    BedrockEmbedder,
    BedrockLLM,
    OpenSearchServerlessStore,
)


async def test_bedrock_llm_not_implemented() -> None:
    llm = BedrockLLM(model_id="anthropic.claude-3-haiku", region="us-east-1")
    with pytest.raises(NotImplementedError):
        await llm.generate("hi")


async def test_bedrock_embedder_not_implemented() -> None:
    embedder = BedrockEmbedder(model_id="amazon.titan-embed-text-v2:0", region="us-east-1")
    with pytest.raises(NotImplementedError):
        await embedder.embed(["hi"])


async def test_opensearch_store_not_implemented() -> None:
    store = OpenSearchServerlessStore(collection_endpoint="https://example", region="us-east-1")
    with pytest.raises(NotImplementedError):
        await store.upsert([])
    with pytest.raises(NotImplementedError):
        await store.search([], 1)
    with pytest.raises(NotImplementedError):
        await store.count()
