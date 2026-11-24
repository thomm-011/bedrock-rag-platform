"""Placeholders for the AWS backends planned in M4 (Terraform + Bedrock).

These classes exist only to document the shape of the future implementation
and to make the interface contract concrete in code review. They are not
wired into any route, CLI command, or test, and calling them always raises
NotImplementedError. Do not use these in M1/M2 — implement in M4.
"""

from __future__ import annotations

from bedrock_rag_platform.interfaces.models import Chunk, LLMResult, SearchResult


class BedrockLLM:
    """Future LLMProvider over Amazon Bedrock (Claude Haiku/Sonnet)."""

    def __init__(self, model_id: str, region: str) -> None:
        self.model_id = model_id
        self.region = region

    async def generate(self, prompt: str, *, max_tokens: int = 256) -> LLMResult:
        raise NotImplementedError("Bedrock backend ships in M4 (Terraform AWS milestone)")


class BedrockEmbedder:
    """Future Embedder over Amazon Bedrock (Titan Embeddings v2)."""

    def __init__(self, model_id: str, region: str, dimensions: int = 1024) -> None:
        self.model_id = model_id
        self.region = region
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Bedrock backend ships in M4 (Terraform AWS milestone)")


class OpenSearchServerlessStore:
    """Future VectorStore over Amazon OpenSearch Serverless."""

    def __init__(self, collection_endpoint: str, region: str) -> None:
        self.collection_endpoint = collection_endpoint
        self.region = region

    async def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("OpenSearch Serverless backend ships in M4")

    async def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        raise NotImplementedError("OpenSearch Serverless backend ships in M4")

    async def count(self) -> int:
        raise NotImplementedError("OpenSearch Serverless backend ships in M4")
