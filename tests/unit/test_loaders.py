from __future__ import annotations

from pathlib import Path

from bedrock_rag_platform.loaders.composite_loader import CompositeLoader
from bedrock_rag_platform.loaders.markdown_loader import MarkdownLoader


def test_markdown_loader_reads_file(fixtures_dir: Path) -> None:
    loader = MarkdownLoader()
    docs = loader.load(fixtures_dir / "sample.md")
    assert len(docs) == 1
    assert "Karpenter" in docs[0].text
    assert docs[0].metadata["type"] == "markdown"


def test_markdown_loader_can_load() -> None:
    loader = MarkdownLoader()
    assert loader.can_load(Path("a.md"))
    assert not loader.can_load(Path("a.pdf"))


def test_composite_loader_walks_directory(fixtures_dir: Path) -> None:
    loader = CompositeLoader()
    docs = loader.load(fixtures_dir)
    assert any("Karpenter" in d.text for d in docs)


def test_composite_loader_raises_on_unknown_type(tmp_path: Path) -> None:
    unknown = tmp_path / "file.xyz"
    unknown.write_text("data")
    loader = CompositeLoader()
    try:
        loader.load(unknown)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
