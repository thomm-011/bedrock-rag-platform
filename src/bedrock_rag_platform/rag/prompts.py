"""Prompt templates for the grounded RAG answer."""

from __future__ import annotations

SYSTEM_INSTRUCTIONS = (
    "You are a documentation assistant. Answer the QUESTION using ONLY the "
    "numbered CONTEXT excerpts below. Cite every claim with the excerpt "
    "number in square brackets, e.g. [1] or [1][3]. If the context does not "
    'contain enough information to answer, reply exactly: "I don\'t know." '
    "Do not use outside knowledge. Be concise."
)


def build_context_block(snippets: list[str]) -> str:
    lines = [f"[{i + 1}] {snippet}" for i, snippet in enumerate(snippets)]
    return "\n\n".join(lines)


def build_grounded_prompt(question: str, snippets: list[str]) -> str:
    context = build_context_block(snippets) if snippets else "(no context retrieved)"
    return f"{SYSTEM_INSTRUCTIONS}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"


def build_no_rag_prompt(question: str) -> str:
    return f"Answer concisely.\n\nQUESTION: {question}\n\nANSWER:"
