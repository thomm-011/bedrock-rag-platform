"""VectorStore implementation backed by Postgres + pgvector."""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from bedrock_rag_platform.interfaces.models import Chunk, SearchResult

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    heading TEXT,
    text TEXT NOT NULL,
    embedding vector({dim}) NOT NULL
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = {m}, ef_construction = {ef_construction})
"""

_UPSERT = """
INSERT INTO chunks (chunk_id, source, heading, text, embedding)
VALUES (%(chunk_id)s, %(source)s, %(heading)s, %(text)s, %(embedding)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    source = EXCLUDED.source,
    heading = EXCLUDED.heading,
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding
"""

_SEARCH = """
SELECT chunk_id, source, heading, text, 1 - (embedding <=> %(query)s) AS score
FROM chunks
ORDER BY embedding <=> %(query)s
LIMIT %(top_k)s
"""


def _vec_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


class PgVectorStore:
    """Synchronous psycopg3 connection wrapped behind the async VectorStore protocol.

    Each call opens a short-lived connection; for a portfolio-scale demo this
    is simpler and safer than managing a pool's lifecycle across FastAPI/CLI
    entry points, and Postgres handles brief connections cheaply.
    """

    def __init__(
        self,
        dsn: str,
        dimensions: int = 768,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ) -> None:
        self.dsn = dsn
        self.dimensions = dimensions
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction

    def ensure_schema(self) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(_CREATE_EXTENSION)
            cur.execute(_CREATE_TABLE.format(dim=self.dimensions))
            cur.execute(
                _CREATE_INDEX.format(m=self.hnsw_m, ef_construction=self.hnsw_ef_construction)
            )

    async def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with psycopg.connect(self.dsn, autocommit=True) as conn, conn.cursor() as cur:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.chunk_id} has no embedding to store")
                cur.execute(
                    _UPSERT,
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "heading": chunk.heading,
                        "text": chunk.text,
                        "embedding": _vec_literal(chunk.embedding),
                    },
                )

    async def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        with (
            psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(_SEARCH, {"query": _vec_literal(query_embedding), "top_k": top_k})
            rows = cur.fetchall()
        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                source=row["source"],
                heading=row["heading"],
                text=row["text"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    async def count(self) -> int:
        with psycopg.connect(self.dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            row = cur.fetchone()
            return int(row[0]) if row else 0
