"""Protocol interfaces and shared models for pluggable RAG backends."""

from bedrock_rag_platform.interfaces.models import Chunk, LLMResult, RawDocument, SearchResult
from bedrock_rag_platform.interfaces.protocols import (
    DocumentLoader,
    Embedder,
    LLMProvider,
    VectorStore,
)

__all__ = [
    "Chunk",
    "DocumentLoader",
    "Embedder",
    "LLMProvider",
    "LLMResult",
    "RawDocument",
    "SearchResult",
    "VectorStore",
]
