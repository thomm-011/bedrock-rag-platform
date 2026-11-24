"""Dispatches loading to the right loader based on file extension, and
walks directories recursively."""

from __future__ import annotations

from pathlib import Path

from bedrock_rag_platform.interfaces.models import RawDocument
from bedrock_rag_platform.interfaces.protocols import DocumentLoader
from bedrock_rag_platform.loaders.markdown_loader import MarkdownLoader
from bedrock_rag_platform.loaders.pdf_loader import PdfLoader


class CompositeLoader:
    """Default loader used by ingestion: markdown + PDF, recursive directory walk."""

    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders: list[DocumentLoader] = loaders or [MarkdownLoader(), PdfLoader()]

    def can_load(self, path: Path) -> bool:
        return any(loader.can_load(path) for loader in self._loaders)

    def load(self, path: Path) -> list[RawDocument]:
        if path.is_dir():
            docs: list[RawDocument] = []
            for child in sorted(path.rglob("*")):
                if child.is_file() and self.can_load(child):
                    docs.extend(self.load(child))
            return docs

        for loader in self._loaders:
            if loader.can_load(path):
                return loader.load(path)
        raise ValueError(f"No loader registered for file type: {path}")
