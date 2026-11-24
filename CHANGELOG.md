# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — M2: Evaluation
- `eval/dataset.jsonl`: 35 hand-written Q&A pairs (30 answerable + 5
  unanswerable) grounded in the fetched Karpenter/KEDA corpus.
- `eval/run_eval.py`: retrieval metrics (Hit@k, Recall@k, MRR, no LLM needed),
  abstention correctness on unanswerable questions, no-RAG-vs-RAG baseline,
  optional weak LLM-as-judge faithfulness score. Writes `eval/report.md` +
  `eval/report.json`.
- `tests/integration/test_pgvector_store.py`: real-Postgres pgvector
  integration tests, marked `integration`, run via `make test-integration`.
- `.github/workflows/ci.yml`: lint/format/mypy/unit-test job (no services
  needed), a pgvector-service integration-test job, and a Docker build job.
- Asymmetric `search_query:`/`search_document:` embedding prefixes for
  `nomic-embed-text`, applied in `RagPipeline` — measurably improves
  retrieval over the un-prefixed baseline.
- Batched embedding calls during ingestion (`embed_batch_size`) so large
  corpora don't blow past the HTTP client timeout on CPU-only Ollama.
- README with before/after Karpenter demo, architecture diagram, ADR-style
  design decisions, roadmap, honest limitations; MIT LICENSE.

### Added — M1: Local core
- FastAPI app (`/health`, `/ingest`, `/ask`) and Typer CLI (`rag ingest`,
  `rag ask`), both backed by a shared `RagPipeline`.
- `interfaces/protocols.py`: `LLMProvider`, `Embedder`, `VectorStore`,
  `DocumentLoader` Protocols decoupling core logic from backends.
- Local implementations: `OllamaLLM`, `OllamaEmbedder` (async httpx),
  `PgVectorStore` (psycopg3 + pgvector, HNSW cosine index).
- `providers/bedrock_stubs.py`: documented, unimplemented stubs for the
  M4 AWS backends (Bedrock LLM/Embedder, RDS/OpenSearch vector store).
- Markdown + PDF loaders, `HeadingAwareChunker` (heading-aware, recursive,
  overlap, per-chunk source/heading/id metadata).
- Grounded prompting: answer only from retrieved context, numbered
  citations `[n]`, explicit "I don't know" instruction when unsupported.
- `docker-compose.yml` (pgvector:pg16 + api, `host.docker.internal` for
  host Ollama), multi-stage non-root `Dockerfile`, `Makefile` dev targets.
- `scripts/fetch_corpus.py`: deterministic fetcher pinning Karpenter
  (`aws/karpenter-provider-aws@v1.15.3`) and KEDA
  (`kedacore/keda-docs@943edda`, docs/2.17) docs, with `NOTICE.md`
  attribution; corpus committed (~560 KB, Apache-2.0).
- Unit test suite with deterministic fakes (`FakeEmbedder`, `FakeLLM`,
  `InMemoryVectorStore`) — ≥85% coverage, no network/DB required.
