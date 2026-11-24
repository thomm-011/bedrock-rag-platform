"""FastAPI application: /health, /ingest, /ask."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from bedrock_rag_platform.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestPathRequest,
    IngestResponse,
)
from bedrock_rag_platform.config import get_settings
from bedrock_rag_platform.rag.factory import build_pipeline
from bedrock_rag_platform.rag.pipeline import RagPipeline

app = FastAPI(
    title="Bedrock RAG Platform",
    description="Local-first RAG API (Ollama + pgvector today, Bedrock-ready interfaces).",
    version="0.1.0",
)

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline(get_settings())
    return _pipeline


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        pipeline = get_pipeline()
        count = await pipeline.vector_store.count()
        return HealthResponse(status="ok", vector_store_count=count)
    except Exception as exc:
        return HealthResponse(status=f"degraded: {exc}", vector_store_count=None)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestPathRequest) -> IngestResponse:
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    pipeline = get_pipeline()
    result = await pipeline.ingest_path(path)
    return IngestResponse(**result.model_dump())


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: Annotated[UploadFile, File()]) -> IngestResponse:
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        pipeline = get_pipeline()
        result = await pipeline.ingest_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return IngestResponse(**result.model_dump())


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    result = await pipeline.ask(request.question, request.top_k)
    return AskResponse(answer=result.answer, citations=result.citations, usage=result.usage)
