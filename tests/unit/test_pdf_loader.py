from __future__ import annotations

from pathlib import Path

from bedrock_rag_platform.loaders.pdf_loader import PdfLoader


def test_pdf_loader_can_load() -> None:
    loader = PdfLoader()
    assert loader.can_load(Path("a.pdf"))
    assert not loader.can_load(Path("a.md"))


def test_pdf_loader_extracts_document(fixtures_dir: Path) -> None:
    loader = PdfLoader()
    docs = loader.load(fixtures_dir / "sample.pdf")
    assert len(docs) == 1
    assert docs[0].metadata["type"] == "pdf"
    assert docs[0].metadata["pages"] == "1"
