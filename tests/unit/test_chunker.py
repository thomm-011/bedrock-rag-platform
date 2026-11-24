from __future__ import annotations

import pytest

from bedrock_rag_platform.chunking.chunker import HeadingAwareChunker
from bedrock_rag_platform.interfaces.models import RawDocument


def test_chunk_respects_headings() -> None:
    doc = RawDocument(
        source="doc.md",
        text="# Title\n\nIntro text.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.",
    )
    chunker = HeadingAwareChunker(chunk_size_chars=500, chunk_overlap_chars=10)
    chunks = chunker.chunk(doc)

    headings = {c.heading for c in chunks}
    assert "Section A" in headings
    assert "Section B" in headings
    assert all(c.source == "doc.md" for c in chunks)


def test_chunk_ids_are_unique() -> None:
    doc = RawDocument(source="doc.md", text="Para one.\n\nPara two.\n\nPara three." * 20)
    chunker = HeadingAwareChunker(chunk_size_chars=100, chunk_overlap_chars=20)
    chunks = chunker.chunk(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert len(chunks) > 1


def test_chunk_size_is_respected_on_long_text() -> None:
    long_text = "word " * 2000
    doc = RawDocument(source="doc.md", text=long_text)
    chunker = HeadingAwareChunker(chunk_size_chars=300, chunk_overlap_chars=50)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    # allow the configured overlap on top of the target size
    assert all(len(c.text) <= 300 + 60 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap_chars"):
        HeadingAwareChunker(chunk_size_chars=100, chunk_overlap_chars=100)


def test_empty_document_produces_no_chunks() -> None:
    doc = RawDocument(source="doc.md", text="   \n\n  ")
    chunker = HeadingAwareChunker()
    assert chunker.chunk(doc) == []
