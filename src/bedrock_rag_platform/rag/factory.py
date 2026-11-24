"""Builds a RagPipeline from Settings, selecting local backends.

Only "ollama"/"pgvector" are implemented in M1/M2; any other backend name
raises, pointing at the M4 milestone. This keeps backend selection explicit
and fails loudly instead of silently falling back.
"""

from __future__ import annotations

from bedrock_rag_platform.chunking.chunker import HeadingAwareChunker
from bedrock_rag_platform.config import Settings
from bedrock_rag_platform.loaders.composite_loader import CompositeLoader
from bedrock_rag_platform.providers.ollama_embedder import OllamaEmbedder
from bedrock_rag_platform.providers.ollama_llm import OllamaLLM
from bedrock_rag_platform.providers.pgvector_store import PgVectorStore
from bedrock_rag_platform.rag.pipeline import RagPipeline


def build_pipeline(settings: Settings) -> RagPipeline:
    if settings.llm_backend != "ollama":
        raise ValueError(f"Unsupported llm_backend '{settings.llm_backend}' (Bedrock ships in M4)")
    if settings.embedder_backend != "ollama":
        raise ValueError(
            f"Unsupported embedder_backend '{settings.embedder_backend}' (Bedrock ships in M4)"
        )
    if settings.vector_store_backend != "pgvector":
        raise ValueError(
            f"Unsupported vector_store_backend '{settings.vector_store_backend}' "
            "(OpenSearch Serverless ships in M4)"
        )

    llm = OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
        timeout_s=settings.ollama_timeout_s,
    )
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
        dimensions=settings.vector_dim,
    )
    vector_store = PgVectorStore(
        dsn=settings.database_url,
        dimensions=settings.vector_dim,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
    )
    loader = CompositeLoader()
    chunker = HeadingAwareChunker(
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
    )

    return RagPipeline(
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        loader=loader,
        chunker=chunker,
        max_context_chars=settings.max_context_chars,
        num_predict=settings.ollama_num_predict,
    )
