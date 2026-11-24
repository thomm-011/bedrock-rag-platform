from __future__ import annotations

from pathlib import Path

from bedrock_rag_platform.rag.pipeline import RagPipeline
from tests.fakes import FakeLLM


async def test_ingest_path_stores_chunks(pipeline: RagPipeline, fixtures_dir: Path) -> None:
    result = await pipeline.ingest_path(fixtures_dir / "sample.md")
    assert result.documents_loaded == 1
    assert result.chunks_created > 0
    assert await pipeline.vector_store.count() == result.chunks_created


async def test_ingest_empty_dir_is_noop(pipeline: RagPipeline, tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = await pipeline.ingest_path(empty_dir)
    assert result.documents_loaded == 0
    assert result.chunks_created == 0


async def test_ask_returns_citations_grounded_in_retrieved_chunks(
    pipeline: RagPipeline, fixtures_dir: Path
) -> None:
    await pipeline.ingest_path(fixtures_dir / "sample.md")
    pipeline.llm = FakeLLM(fixed_answer="Karpenter provisions nodes. [1]")

    result = await pipeline.ask("What is Karpenter?", top_k=2)

    assert result.answer == "Karpenter provisions nodes. [1]"
    assert len(result.citations) >= 1
    assert result.citations[0].source.endswith("sample.md")
    assert result.usage.latency_ms >= 0


async def test_ask_builds_prompt_with_context(pipeline: RagPipeline, fixtures_dir: Path) -> None:
    await pipeline.ingest_path(fixtures_dir / "sample.md")
    llm = FakeLLM(fixed_answer="irrelevant")
    pipeline.llm = llm

    await pipeline.ask("What is Karpenter?", top_k=1)

    assert len(llm.prompts) == 1
    assert "CONTEXT" in llm.prompts[0]
    assert "What is Karpenter?" in llm.prompts[0]


async def test_ask_no_rag_skips_retrieval(pipeline: RagPipeline) -> None:
    llm = FakeLLM(fixed_answer="furniture brand")
    pipeline.llm = llm

    result = await pipeline.ask_no_rag("What is Karpenter?")

    assert result.answer == "furniture brand"
    assert result.citations == []
    assert "CONTEXT" not in llm.prompts[0]


async def test_ask_truncates_context_to_budget(pipeline: RagPipeline, fixtures_dir: Path) -> None:
    await pipeline.ingest_path(fixtures_dir / "sample.md")
    pipeline.max_context_chars = 10
    result = await pipeline.ask("What is Karpenter?", top_k=5)
    assert sum(len(c.snippet) for c in result.citations) <= 10
