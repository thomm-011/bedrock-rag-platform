from __future__ import annotations

import pytest

from bedrock_rag_platform.config import Settings
from bedrock_rag_platform.providers.ollama_embedder import OllamaEmbedder
from bedrock_rag_platform.providers.ollama_llm import OllamaLLM
from bedrock_rag_platform.providers.pgvector_store import PgVectorStore
from bedrock_rag_platform.rag.factory import build_pipeline


def test_build_pipeline_wires_local_backends() -> None:
    settings = Settings()
    pipeline = build_pipeline(settings)
    assert isinstance(pipeline.llm, OllamaLLM)
    assert isinstance(pipeline.embedder, OllamaEmbedder)
    assert isinstance(pipeline.vector_store, PgVectorStore)


@pytest.mark.parametrize("field", ["llm_backend", "embedder_backend", "vector_store_backend"])
def test_build_pipeline_rejects_unimplemented_backend(field: str) -> None:
    settings = Settings.model_validate({field: "bedrock"})
    with pytest.raises(ValueError, match="M4"):
        build_pipeline(settings)
