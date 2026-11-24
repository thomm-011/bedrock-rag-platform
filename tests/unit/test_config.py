from __future__ import annotations

from bedrock_rag_platform.config import get_settings


def test_defaults_point_to_local_stack() -> None:
    settings = get_settings()
    assert settings.llm_backend == "ollama"
    assert settings.vector_store_backend == "pgvector"
    assert "11434" in settings.ollama_base_url


def test_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RAG_OLLAMA_LLM_MODEL", "qwen2.5:7b")
    settings = get_settings()
    assert settings.ollama_llm_model == "qwen2.5:7b"
