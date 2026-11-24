"""Heading-aware recursive chunker with overlap.

Strategy:
1. Split the document on markdown headings (#, ##, ### ...), keeping the
   most recent heading as metadata for every chunk produced from that
   section (improves citation quality: "KEDA Concepts > Scaling Deployments"
   reads better than an anonymous offset).
2. Within each section, recursively split on paragraph/line/sentence
   boundaries (in that order of preference) so chunks don't cut mid-sentence
   when avoidable, up to `chunk_size_chars`.
3. Add `chunk_overlap_chars` of trailing context from the previous chunk to
   the next chunk's start, so retrieval doesn't lose context at a boundary.
"""

from __future__ import annotations

import hashlib
import re

from bedrock_rag_platform.interfaces.models import Chunk, RawDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on_headings(text: str) -> list[tuple[str | None, str]]:
    """Return [(heading_or_none, section_text), ...] preserving order."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def _recursive_split(text: str, max_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]
    if not separators:
        # Hard cut as last resort.
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    sep, rest = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return _recursive_split(text, max_size, rest)

    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = (buffer + sep + part) if buffer else part
        if len(candidate) <= max_size:
            buffer = candidate
        else:
            if buffer:
                chunks.extend(_recursive_split(buffer, max_size, rest))
            buffer = part
    if buffer:
        chunks.extend(_recursive_split(buffer, max_size, rest))
    return chunks


class HeadingAwareChunker:
    """Splits documents into overlapping, heading-tagged chunks."""

    def __init__(self, chunk_size_chars: int = 1200, chunk_overlap_chars: int = 200) -> None:
        if chunk_overlap_chars >= chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def chunk(self, document: RawDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        previous_tail = ""

        for heading, section_text in _split_on_headings(document.text):
            raw_pieces = _recursive_split(section_text, self.chunk_size_chars, _SPLIT_SEPARATORS)
            for piece in raw_pieces:
                body = (previous_tail + "\n" + piece).strip() if previous_tail else piece
                chunk_id = self._make_id(document.source, heading, len(chunks))
                chunks.append(
                    Chunk(chunk_id=chunk_id, source=document.source, heading=heading, text=body)
                )
                previous_tail = piece[-self.chunk_overlap_chars :]

        return chunks

    @staticmethod
    def _make_id(source: str, heading: str | None, index: int) -> str:
        digest = hashlib.sha1(f"{source}:{heading}:{index}".encode()).hexdigest()[:12]
        return f"{digest}-{index}"
