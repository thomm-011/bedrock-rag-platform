from __future__ import annotations

from bedrock_rag_platform.rag.prompts import (
    build_context_block,
    build_grounded_prompt,
    build_no_rag_prompt,
)


def test_context_block_numbers_snippets() -> None:
    block = build_context_block(["alpha", "beta"])
    assert "[1] alpha" in block
    assert "[2] beta" in block


def test_grounded_prompt_includes_question_and_context() -> None:
    prompt = build_grounded_prompt("What is X?", ["X is Y."])
    assert "What is X?" in prompt
    assert "[1] X is Y." in prompt
    assert "I don't know" in prompt


def test_grounded_prompt_handles_no_context() -> None:
    prompt = build_grounded_prompt("What is X?", [])
    assert "no context retrieved" in prompt


def test_no_rag_prompt_has_no_context_section() -> None:
    prompt = build_no_rag_prompt("What is X?")
    assert "CONTEXT" not in prompt
    assert "What is X?" in prompt
