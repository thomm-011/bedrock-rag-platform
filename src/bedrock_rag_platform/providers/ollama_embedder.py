"""Embedder implementation backed by a local Ollama server."""

from __future__ import annotations

import httpx


class OllamaEmbedder:
    """Calls Ollama's /api/embed endpoint (batched)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout_s = timeout_s

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
        embeddings: list[list[float]] = data["embeddings"]
        return embeddings
