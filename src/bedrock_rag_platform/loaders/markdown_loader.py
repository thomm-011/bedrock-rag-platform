"""Loads markdown files as raw documents."""

from __future__ import annotations

from pathlib import Path

from bedrock_rag_platform.interfaces.models import RawDocument

MARKDOWN_SUFFIXES = {".md", ".markdown"}


class MarkdownLoader:
    """Reads .md/.markdown files as UTF-8 text."""

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in MARKDOWN_SUFFIXES

    def load(self, path: Path) -> list[RawDocument]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [RawDocument(source=str(path), text=text, metadata={"type": "markdown"})]
