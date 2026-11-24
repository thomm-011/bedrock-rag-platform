# Evaluation Report

Dataset: `eval/dataset.jsonl` (30 answerable + 5 unanswerable questions). top_k=4. Total eval wall time: 267.1s.

## Retrieval metrics (no LLM, full dataset)

| Metric | Value |
|---|---|
| Hit@4 | 96.67% |
| Recall@4 | 93.06% |
| MRR | 0.894 |

## Abstention correctness (unanswerable questions)

5/5 correctly answered "I don't know" instead of hallucinating.

## Before/after: RAG vs. no-RAG baseline

**Q: What is Karpenter?**
- No-RAG (LLM alone): The Carpenter is a position in the British Armed Forces, equivalent to a Sergeant.
- RAG (grounded): Karpenter is an open-source node lifecycle management project built for Kubernetes [1]. It is designed to improve the efficiency and cost of running workloads on a Kubernetes cluster by automatically provisioning and managing nodes [1].
- Citations: data/corpus/karpenter/index.md, data/corpus/karpenter/concepts/index.md, data/corpus/karpenter/faq.md, data/corpus/karpenter/troubleshooting.md

**Q: What does KEDA's AWS SQS scaler use to determine how many replicas to run?**
- No-RAG (LLM alone): CPU utilization.
- RAG (grounded): KEDA's AWS SQS scaler uses the `queueLength` from the `metadata` section of the `scaledObject` definition to determine how many replicas to run. This value is specified in the `aws-sqs-queue-scaledobject` definition as "5" [1].
- Citations: data/corpus/keda/scalers/aws-sqs.md, data/corpus/keda/concepts/scaling-deployments.md, data/corpus/keda/scalers/aws-sqs.md, data/corpus/keda/concepts/scaling-deployments.md

**Q: What is the capital of France?**
- No-RAG (LLM alone): Paris.
- RAG (grounded): I don't know. [4]
- Citations: data/corpus/karpenter/concepts/disruption.md, data/corpus/karpenter/concepts/disruption.md, data/corpus/karpenter/concepts/nodeclasses.md, data/corpus/karpenter/concepts/disruption.md

## Per-question retrieval detail

| id | hit | recall | RR | expected | retrieved |
|---|---|---|---|---|---|
| q01 | ✓ | 1.00 | 1.00 | index.md | index.md, index.md, faq.md, troubleshooting.md |
| q02 | ✓ | 1.00 | 1.00 | index.md | index.md, nodepools.md, nodepools.md, troubleshooting.md |
| q03 | ✓ | 1.00 | 1.00 | disruption.md | disruption.md, disruption.md, managing-amis.md, disruption.md |
| q04 | ✓ | 1.00 | 0.50 | disruption.md | index.md, disruption.md, scheduling.md, index.md |
| q05 | ✓ | 1.00 | 1.00 | disruption.md | disruption.md, disruption.md, troubleshooting.md, troubleshooting.md |
| q06 | ✓ | 1.00 | 1.00 | odcrs.md | odcrs.md, odcrs.md, odcrs.md, odcrs.md |
| q07 | ✓ | 1.00 | 1.00 | odcrs.md | odcrs.md, odcrs.md, odcrs.md, odcrs.md |
| q08 | ✓ | 1.00 | 1.00 | odcrs.md | odcrs.md, odcrs.md, odcrs.md, odcrs.md |
| q09 | ✓ | 1.00 | 1.00 | aws-sqs.md | aws-sqs.md, aws-sqs.md, aws-sqs.md, aws-sqs.md |
| q10 | ✓ | 1.00 | 1.00 | aws-sqs.md | aws-sqs.md, aws-sqs.md, aws-sqs.md, aws-sqs.md |
| q11 | ✓ | 1.00 | 1.00 | aws-sqs.md | aws-sqs.md, aws-sqs.md, aws-sqs.md, aws-sqs.md |
| q12 | ✓ | 1.00 | 1.00 | aws-sqs.md | aws-sqs.md, aws-sqs.md, aws-sqs.md, aws-sqs.md |
| q13 | ✓ | 1.00 | 1.00 | aws-sqs.md | aws-sqs.md, index.md, authentication.md, index.md |
| q14 | ✓ | 1.00 | 1.00 | aws-cloudwatch.md | aws-cloudwatch.md, aws-cloudwatch.md, aws-cloudwatch.md, aws-cloudwatch.md |
| q15 | ✓ | 1.00 | 1.00 | aws-dynamodb.md | aws-dynamodb.md, aws-dynamodb.md, aws-dynamodb-streams.md, aws-dynamodb-streams.md |
| q16 | ✓ | 1.00 | 1.00 | aws-dynamodb.md | aws-dynamodb.md, scaling-deployments.md, index.md, scaling-deployments.md |
| q17 | ✓ | 1.00 | 1.00 | aws-kinesis.md | aws-kinesis.md, aws-kinesis.md, aws-kinesis.md, aws-dynamodb-streams.md |
| q18 | ✓ | 1.00 | 1.00 | scaling-deployments.md | scaling-deployments.md, scaling-jobs.md, index.md, scaling-deployments.md |
| q19 | ✓ | 1.00 | 1.00 | scaling-deployments.md | scaling-deployments.md, scaling-deployments.md, scaling-jobs.md, scaling-deployments.md |
| q20 | ✓ | 1.00 | 1.00 | scaling-deployments.md | scaling-deployments.md, index.md, scaling-deployments.md, scaling-deployments.md |
| q21 | ✓ | 1.00 | 1.00 | scaling-deployments.md | scaling-deployments.md, index.md, scaling-deployments.md, index.md |
| q22 | ✗ | 0.00 | 0.00 | index.md | nodepools.md, nodepools.md, nodepools.md, scheduling.md |
| q23 | ✓ | 1.00 | 0.33 | index.md | scheduling.md, troubleshooting.md, index.md, faq.md |
| q24 | ✓ | 1.00 | 1.00 | disruption.md | disruption.md, disruption.md, troubleshooting.md, disruption.md |
| q25 | ✓ | 1.00 | 0.50 | disruption.md | nodepools.md, disruption.md, disruption.md, nodepools.md |
| q26 | ✓ | 0.75 | 1.00 | aws-cloudwatch.md, aws-dynamodb.md, aws-kinesis.md, aws-sqs.md | aws-kinesis.md, aws-sqs.md, aws-dynamodb.md, aws-kinesis.md |
| q27 | ✓ | 0.50 | 0.50 | aws-kinesis.md, aws-sqs.md | authentication.md, aws-sqs.md, aws-dynamodb.md, aws-cloudwatch.md |
| q28 | ✓ | 1.00 | 1.00 | disruption.md | disruption.md, disruption.md, nodepools.md, disruption.md |
| q29 | ✓ | 0.67 | 1.00 | aws-cloudwatch.md, aws-dynamodb.md, aws-sqs.md | aws-dynamodb.md, aws-cloudwatch.md, aws-dynamodb-streams.md, admission-webhooks.md |
| q30 | ✓ | 1.00 | 1.00 | aws-sqs.md | aws-sqs.md, aws-sqs.md, aws-sqs.md, aws-sqs.md |
