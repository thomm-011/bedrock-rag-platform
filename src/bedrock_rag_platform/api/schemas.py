"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bedrock_rag_platform.rag.pipeline import Citation, Usage


class IngestPathRequest(BaseModel):
    path: str = Field(description="Server-local path to a file or directory to ingest")


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_stored: int


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    usage: Usage


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int | None = None
