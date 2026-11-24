from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bedrock_rag_platform.api import app as app_module
from bedrock_rag_platform.chunking.chunker import HeadingAwareChunker
from bedrock_rag_platform.loaders.composite_loader import CompositeLoader
from bedrock_rag_platform.rag.pipeline import RagPipeline
from tests.fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore


def _make_test_pipeline() -> RagPipeline:
    return RagPipeline(
        llm=FakeLLM(fixed_answer="Karpenter provisions nodes. [1]"),
        embedder=FakeEmbedder(dimensions=8),
        vector_store=InMemoryVectorStore(),
        loader=CompositeLoader(),
        chunker=HeadingAwareChunker(chunk_size_chars=200, chunk_overlap_chars=20),
    )


def test_health_ok(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    app_module._pipeline = _make_test_pipeline()
    client = TestClient(app_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_and_ask_roundtrip(fixtures_dir: Path) -> None:
    app_module._pipeline = _make_test_pipeline()
    client = TestClient(app_module.app)

    ingest_response = client.post("/ingest", json={"path": str(fixtures_dir / "sample.md")})
    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunks_stored"] > 0

    ask_response = client.post("/ask", json={"question": "What is Karpenter?", "top_k": 2})
    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["answer"] == "Karpenter provisions nodes. [1]"
    assert len(body["citations"]) >= 1
    assert "latency_ms" in body["usage"]


def test_ingest_missing_path_returns_404() -> None:
    app_module._pipeline = _make_test_pipeline()
    client = TestClient(app_module.app)
    response = client.post("/ingest", json={"path": "/no/such/path"})
    assert response.status_code == 404


def test_ingest_upload(fixtures_dir: Path) -> None:
    app_module._pipeline = _make_test_pipeline()
    client = TestClient(app_module.app)
    with (fixtures_dir / "sample.md").open("rb") as f:
        response = client.post("/ingest/upload", files={"file": ("sample.md", f, "text/markdown")})
    assert response.status_code == 200
    assert response.json()["chunks_stored"] > 0
