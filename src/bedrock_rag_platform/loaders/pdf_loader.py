"""Loads PDF files as raw documents using pypdf."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from bedrock_rag_platform.interfaces.models import RawDocument


class PdfLoader:
    """Extracts text per page and joins with form-feed-free newlines."""

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(self, path: Path) -> list[RawDocument]:
        reader = PdfReader(str(path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages_text)
        return [
            RawDocument(
                source=str(path),
                text=text,
                metadata={"type": "pdf", "pages": str(len(reader.pages))},
            )
        ]
