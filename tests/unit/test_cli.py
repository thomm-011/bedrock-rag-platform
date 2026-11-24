from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bedrock_rag_platform.chunking.chunker import HeadingAwareChunker
from bedrock_rag_platform.cli import main as cli_main
from bedrock_rag_platform.loaders.composite_loader import CompositeLoader
from bedrock_rag_platform.rag.pipeline import RagPipeline
from tests.fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore

runner = CliRunner()


def _fake_pipeline() -> RagPipeline:
    return RagPipeline(
        llm=FakeLLM(fixed_answer="Karpenter provisions nodes. [1]"),
        embedder=FakeEmbedder(dimensions=8),
        vector_store=InMemoryVectorStore(),
        loader=CompositeLoader(),
        chunker=HeadingAwareChunker(chunk_size_chars=200, chunk_overlap_chars=20),
    )


def test_cli_ingest_and_ask(monkeypatch, fixtures_dir: Path) -> None:  # type: ignore[no-untyped-def]
    shared_pipeline = _fake_pipeline()
    monkeypatch.setattr(cli_main, "build_pipeline", lambda settings: shared_pipeline)

    ingest_result = runner.invoke(cli_main.app, ["ingest", str(fixtures_dir / "sample.md")])
    assert ingest_result.exit_code == 0
    assert "stored" in ingest_result.stdout

    ask_result = runner.invoke(cli_main.app, ["ask", "What is Karpenter?"])
    assert ask_result.exit_code == 0
    assert "Karpenter provisions nodes" in ask_result.stdout


def test_cli_ask_json_output(monkeypatch, fixtures_dir: Path) -> None:  # type: ignore[no-untyped-def]
    shared_pipeline = _fake_pipeline()
    monkeypatch.setattr(cli_main, "build_pipeline", lambda settings: shared_pipeline)
    runner.invoke(cli_main.app, ["ingest", str(fixtures_dir / "sample.md")])

    result = runner.invoke(cli_main.app, ["ask", "What is Karpenter?", "--json"])
    assert result.exit_code == 0
    assert '"answer"' in result.stdout
