from __future__ import annotations

import httpx
import respx

from bedrock_rag_platform.providers.ollama_embedder import OllamaEmbedder
from bedrock_rag_platform.providers.ollama_llm import OllamaLLM


@respx.mock
async def test_ollama_llm_generate_parses_response() -> None:
    respx.post("http://127.0.0.1:11434/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": " Karpenter provisions nodes. ",
                "prompt_eval_count": 42,
                "eval_count": 7,
            },
        )
    )
    llm = OllamaLLM(base_url="http://127.0.0.1:11434", model="llama3.2:3b")

    result = await llm.generate("What is Karpenter?", max_tokens=64)

    assert result.text == "Karpenter provisions nodes."
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 7
    assert result.latency_ms >= 0


@respx.mock
async def test_ollama_embedder_embed_batches() -> None:
    respx.post("http://127.0.0.1:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    )
    embedder = OllamaEmbedder(base_url="http://127.0.0.1:11434", model="nomic-embed-text")

    vectors = await embedder.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_ollama_embedder_empty_input_short_circuits() -> None:
    embedder = OllamaEmbedder()
    assert await embedder.embed([]) == []
