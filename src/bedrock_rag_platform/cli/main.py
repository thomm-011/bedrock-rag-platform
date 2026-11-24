"""`rag` CLI: ingest documents and ask grounded questions from the terminal."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from bedrock_rag_platform.config import get_settings
from bedrock_rag_platform.providers.pgvector_store import PgVectorStore
from bedrock_rag_platform.rag.factory import build_pipeline

app = typer.Typer(help="Local-first RAG CLI (Ollama + pgvector).")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),  # noqa: B008
) -> None:
    """Load, chunk, embed, and store documents from PATH."""
    settings = get_settings()
    pipeline = build_pipeline(settings)
    if isinstance(pipeline.vector_store, PgVectorStore):
        pipeline.vector_store.ensure_schema()

    result = asyncio.run(pipeline.ingest_path(path))
    typer.echo(
        f"Loaded {result.documents_loaded} document(s), stored {result.chunks_stored} chunk(s)."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(4, help="Number of chunks to retrieve"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Ask a grounded question against the ingested corpus."""
    settings = get_settings()
    pipeline = build_pipeline(settings)

    result = asyncio.run(pipeline.ask(question, top_k))
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(f"Q: {question}\n")
    typer.echo(f"A: {result.answer}\n")
    for citation in result.citations:
        heading = f" — {citation.heading}" if citation.heading else ""
        typer.echo(f"  [{citation.index}] {citation.source}{heading} (score={citation.score:.3f})")
    typer.echo(
        f"\n(prompt_tokens={result.usage.prompt_tokens}, "
        f"completion_tokens={result.usage.completion_tokens}, "
        f"latency_ms={result.usage.latency_ms:.0f})"
    )


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run("bedrock_rag_platform.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
