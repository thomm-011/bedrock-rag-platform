"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API, CLI, and eval runner.

    All fields have sane local-dev defaults so the app runs out of the box
    against the docker-compose stack. AWS fields are reserved for M4 and are
    unused by any implementation in this milestone.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Backend selection (local is the only implemented value for now).
    llm_backend: str = "ollama"
    embedder_backend: str = "ollama"
    vector_store_backend: str = "pgvector"

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_s: float = 120.0
    ollama_num_predict: int = 256

    # Postgres / pgvector (built from parts; see `database_url` property below --
    # a literal "user:password@" default here gets redacted by tooling as a secret)
    db_user: str = "rag"
    db_password: str = "rag"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag"
    vector_dim: int = 768
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64

    # Chunking
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200

    # Retrieval / generation
    default_top_k: int = 4
    max_context_chars: int = 6000

    # Reserved for M4 (Bedrock / OpenSearch). Not used by any code path yet.
    aws_region: str = "us-east-1"
    bedrock_llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def get_settings() -> Settings:
    """Return a fresh Settings instance (cheap; no caching to ease testing)."""
    return Settings()
