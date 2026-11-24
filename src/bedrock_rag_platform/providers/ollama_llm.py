"""LLMProvider implementation backed by a local Ollama server."""

from __future__ import annotations

import time

import httpx

from bedrock_rag_platform.interfaces.models import LLMResult


class OllamaLLM:
    """Calls Ollama's /api/generate endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.2:3b",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def generate(self, prompt: str, *, max_tokens: int = 256) -> LLMResult:
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - started) * 1000
        return LLMResult(
            text=data.get("response", "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_ms=latency_ms,
        )
