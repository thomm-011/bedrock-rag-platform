"""Evaluation runner for the RAG pipeline (M2).

Computes, without any LLM:
  - hit@k / recall@k against expected source files
  - MRR (mean reciprocal rank) of the first correct source

Computes, with the local LLM (bounded, see GENERATION_SAMPLE_SIZE below):
  - abstention correctness on the unanswerable questions
  - a no-RAG baseline (LLM alone, no retrieval) on a small question sample
  - an optional, explicitly-labeled-weak LLM-as-judge faithfulness score

CPU notes: generation with llama3.2:3b on CPU is slow (~5-15s per call with
num_predict=256). Running generation for all 35 dataset rows could take many
minutes, so by default this script only generates for the unanswerable rows
(abstention is the metric that needs it) plus a small sample of answerable
rows (GENERATION_SAMPLE_SIZE). Retrieval metrics still cover the FULL dataset
since they need no LLM call. Pass --full-generation to run every row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from bedrock_rag_platform.config import get_settings
from bedrock_rag_platform.rag.factory import build_pipeline
from bedrock_rag_platform.rag.pipeline import AskResult, RagPipeline

GENERATION_SAMPLE_SIZE = 10  # answerable rows that get a real generation call
JUDGE_SAMPLE_SIZE = 5  # how many of the generated answers also get LLM-judged
BASELINE_QUESTIONS = [
    "What is Karpenter?",
    "What does KEDA's AWS SQS scaler use to determine how many replicas to run?",
    "What is the capital of France?",
]

JUDGE_PROMPT = """You are grading an AI-generated answer for faithfulness to the \
provided reference answer. Reply with a single integer score from 1 (completely \
wrong or unsupported) to 5 (fully faithful and correct), then a one-line reason.

REFERENCE ANSWER: {reference}
AI ANSWER: {answer}

SCORE (format "N - reason"):"""


@dataclass
class DatasetRow:
    id: str
    question: str
    reference_answer: str
    expected_sources: list[str]
    unanswerable: bool = False


@dataclass
class RetrievalMetrics:
    hit_at_k: float
    recall_at_k: float
    mrr: float
    n_questions: int


@dataclass
class GenerationMetrics:
    abstention_correct: int
    abstention_total: int
    judged: list[dict] = field(default_factory=list)
    baseline: list[dict] = field(default_factory=list)
    sampled_answers: list[dict] = field(default_factory=list)


def load_dataset(path: Path) -> list[DatasetRow]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        rows.append(
            DatasetRow(
                id=data["id"],
                question=data["question"],
                reference_answer=data["reference_answer"],
                expected_sources=data.get("expected_sources", []),
                unanswerable=data.get("unanswerable", False),
            )
        )
    return rows


def _basename(path: str) -> str:
    return Path(path).name


async def compute_retrieval_metrics(
    pipeline: RagPipeline, rows: list[DatasetRow], top_k: int
) -> tuple[RetrievalMetrics, list[dict]]:
    """Hit@k / recall@k / MRR over the answerable subset (unanswerable rows
    have no expected source, so they are excluded from retrieval scoring)."""
    answerable = [r for r in rows if not r.unanswerable]
    hits = 0
    recalls = []
    reciprocal_ranks = []
    per_question: list[dict] = []

    for row in answerable:
        results = await pipeline.retrieve(row.question, top_k)
        retrieved_sources = [_basename(r.source) for r in results]
        expected = {_basename(s) for s in row.expected_sources}

        matched_any = any(s in expected for s in retrieved_sources)
        matched_count = len({s for s in retrieved_sources if s in expected})
        recall = matched_count / len(expected) if expected else 0.0

        rank = next((i + 1 for i, s in enumerate(retrieved_sources) if s in expected), None)
        reciprocal_rank = 1.0 / rank if rank else 0.0

        hits += int(matched_any)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        per_question.append(
            {
                "id": row.id,
                "question": row.question,
                "expected_sources": sorted(expected),
                "retrieved_sources": retrieved_sources,
                "hit": matched_any,
                "recall": recall,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    n = len(answerable)
    metrics = RetrievalMetrics(
        hit_at_k=hits / n if n else 0.0,
        recall_at_k=sum(recalls) / n if n else 0.0,
        mrr=sum(reciprocal_ranks) / n if n else 0.0,
        n_questions=n,
    )
    return metrics, per_question


def _looks_like_abstention(answer: str) -> bool:
    return "don't know" in answer.lower() or "do not know" in answer.lower()


async def judge_answer(pipeline: RagPipeline, reference: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(reference=reference, answer=answer)
    result = await pipeline.llm.generate(prompt, max_tokens=60)
    return {"raw": result.text}


async def compute_generation_metrics(
    pipeline: RagPipeline,
    rows: list[DatasetRow],
    top_k: int,
    sample_size: int,
    judge_sample_size: int,
    run_judge: bool,
) -> GenerationMetrics:
    unanswerable = [r for r in rows if r.unanswerable]
    answerable_sample = [r for r in rows if not r.unanswerable][:sample_size]

    abstention_correct = 0
    sampled_answers: list[dict] = []

    for row in unanswerable:
        result = await pipeline.ask(row.question, top_k)
        correct = _looks_like_abstention(result.answer)
        abstention_correct += int(correct)
        sampled_answers.append(
            {
                "id": row.id,
                "question": row.question,
                "answer": result.answer,
                "expected_abstention": True,
                "abstained": correct,
                "latency_ms": result.usage.latency_ms,
            }
        )

    judged: list[dict] = []
    for i, row in enumerate(answerable_sample):
        result: AskResult = await pipeline.ask(row.question, top_k)
        entry = {
            "id": row.id,
            "question": row.question,
            "answer": result.answer,
            "reference_answer": row.reference_answer,
            "latency_ms": result.usage.latency_ms,
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
        }
        sampled_answers.append(entry)
        if run_judge and i < judge_sample_size:
            judged.append(
                {**entry, **await judge_answer(pipeline, row.reference_answer, result.answer)}
            )

    baseline: list[dict] = []
    for question in BASELINE_QUESTIONS:
        no_rag = await pipeline.ask_no_rag(question)
        rag = await pipeline.ask(question, top_k)
        baseline.append(
            {
                "question": question,
                "no_rag_answer": no_rag.answer,
                "rag_answer": rag.answer,
                "rag_citations": [c.source for c in rag.citations],
            }
        )

    return GenerationMetrics(
        abstention_correct=abstention_correct,
        abstention_total=len(unanswerable),
        judged=judged,
        baseline=baseline,
        sampled_answers=sampled_answers,
    )


def render_report(
    retrieval: RetrievalMetrics,
    generation: GenerationMetrics,
    per_question: list[dict],
    top_k: int,
    elapsed_s: float,
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Dataset: `eval/dataset.jsonl` ({retrieval.n_questions} answerable + "
        f"{generation.abstention_total} unanswerable questions). top_k={top_k}. "
        f"Total eval wall time: {elapsed_s:.1f}s.",
        "",
        "## Retrieval metrics (no LLM, full dataset)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hit@{top_k} | {retrieval.hit_at_k:.2%} |",
        f"| Recall@{top_k} | {retrieval.recall_at_k:.2%} |",
        f"| MRR | {retrieval.mrr:.3f} |",
        "",
        "## Abstention correctness (unanswerable questions)",
        "",
        f"{generation.abstention_correct}/{generation.abstention_total} correctly "
        'answered "I don\'t know" instead of hallucinating.',
        "",
        "## Before/after: RAG vs. no-RAG baseline",
        "",
    ]
    for item in generation.baseline:
        lines += [
            f"**Q: {item['question']}**",
            f"- No-RAG (LLM alone): {item['no_rag_answer']}",
            f"- RAG (grounded): {item['rag_answer']}",
            f"- Citations: {', '.join(item['rag_citations']) or '(none)'}",
            "",
        ]

    if generation.judged:
        lines += ["## LLM-as-judge faithfulness (WEAK signal — see caveat below)", ""]
        for item in generation.judged:
            lines.append(f"- `{item['id']}`: {item['raw'].strip()[:200]}")
        lines += [
            "",
            "> **Caveat:** this score is produced by the same small local model "
            "(llama3.2:3b) being evaluated, with no calibration or human "
            "validation. Treat it as a rough signal only, not a quality bar.",
            "",
        ]

    lines += [
        "## Per-question retrieval detail",
        "",
        "| id | hit | recall | RR | expected | retrieved |",
        "|---|---|---|---|---|---|",
    ]
    for q in per_question:
        lines.append(
            f"| {q['id']} | {'✓' if q['hit'] else '✗'} | {q['recall']:.2f} | "
            f"{q['reciprocal_rank']:.2f} | {', '.join(q['expected_sources'])} | "
            f"{', '.join(q['retrieved_sources'])} |"
        )

    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = build_pipeline(settings)

    rows = load_dataset(Path(args.dataset))
    sample_size = len(rows) if args.full_generation else args.sample_size

    started = time.perf_counter()
    retrieval, per_question = await compute_retrieval_metrics(pipeline, rows, args.top_k)
    generation = await compute_generation_metrics(
        pipeline, rows, args.top_k, sample_size, args.judge_sample_size, args.judge
    )
    elapsed_s = time.perf_counter() - started

    report_md = render_report(retrieval, generation, per_question, args.top_k, elapsed_s)
    Path(args.report_md).write_text(report_md)

    report_json = {
        "top_k": args.top_k,
        "elapsed_s": elapsed_s,
        "retrieval": {
            "hit_at_k": retrieval.hit_at_k,
            "recall_at_k": retrieval.recall_at_k,
            "mrr": retrieval.mrr,
            "n_questions": retrieval.n_questions,
        },
        "abstention": {
            "correct": generation.abstention_correct,
            "total": generation.abstention_total,
        },
        "baseline": generation.baseline,
        "sampled_answers": generation.sampled_answers,
        "judged": generation.judged,
        "per_question": per_question,
    }
    Path(args.report_json).write_text(json.dumps(report_json, indent=2))

    print(report_md)
    print(f"\nWrote {args.report_md} and {args.report_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--report-md", default="eval/report.md")
    parser.add_argument("--report-json", default="eval/report.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=GENERATION_SAMPLE_SIZE)
    parser.add_argument("--judge-sample-size", type=int, default=JUDGE_SAMPLE_SIZE)
    parser.add_argument("--judge", action="store_true", help="Run the weak LLM-as-judge score")
    parser.add_argument(
        "--full-generation", action="store_true", help="Generate for every dataset row"
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
