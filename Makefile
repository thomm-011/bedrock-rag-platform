.PHONY: up down logs ingest ask demo eval test test-integration lint fmt typecheck fetch-corpus build

COMPOSE := docker compose

up:
	$(COMPOSE) up -d --build
	@echo "Waiting for Postgres to be healthy..."
	@until $(COMPOSE) ps db | grep -q healthy; do sleep 1; done
	@echo "Stack is up. API: http://localhost:8000  (health: /health)"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

fetch-corpus:
	uv run python scripts/fetch_corpus.py

ingest:
	uv run rag ingest data/corpus

ask:
	uv run rag ask "$(Q)"

demo:
	uv run rag ask "What is Karpenter?"
	uv run rag ask "What does KEDA's AWS SQS scaler use to determine how many replicas to run?"
	uv run rag ask "What is the capital of France?"

eval:
	uv run python eval/run_eval.py

test:
	uv run pytest tests/unit -m "not integration" --cov --cov-report=term-missing

test-integration:
	$(COMPOSE) up -d db
	@until $(COMPOSE) ps db | grep -q healthy; do sleep 1; done
	RAG_DB_HOST=localhost RAG_DB_PORT=5432 RAG_DB_USER=rag RAG_DB_PASSWORD=rag RAG_DB_NAME=rag \
		uv run pytest tests/integration -m integration -v

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy src

build:
	docker build -t bedrock-rag-platform:local .
