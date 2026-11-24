"""Fetches a modest, pinned subset of Karpenter + KEDA docs for the RAG corpus.

Both projects are Apache-2.0 licensed. We pin an exact tag/commit so the
corpus (and therefore eval results) are reproducible — "latest" would drift
and silently invalidate eval/dataset.jsonl's expected answers and sources.

Usage: uv run python scripts/fetch_corpus.py [--out data/corpus]
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

KARPENTER_REPO = "aws/karpenter-provider-aws"
KARPENTER_REF = "v1.15.3"  # pinned tag, commit 8c91894e993eda682f0785091b79a93da68f9830

KEDA_REPO = "kedacore/keda-docs"
KEDA_REF = "943edda82d621bd70b851df9385f2a7f55d57abd"  # pinned commit (main, Nov 2026)
KEDA_VERSION = "2.17"


@dataclass(frozen=True)
class CorpusFile:
    repo: str
    ref: str
    remote_path: str
    local_path: str  # relative to --out


KARPENTER_BASE = "website/content/en/docs"
KEDA_BASE = f"content/docs/{KEDA_VERSION}"

FILES: list[CorpusFile] = [
    # --- Karpenter (kubernetes node autoscaler) ---
    CorpusFile(KARPENTER_REPO, KARPENTER_REF, f"{KARPENTER_BASE}/_index.md", "karpenter/index.md"),
    CorpusFile(KARPENTER_REPO, KARPENTER_REF, f"{KARPENTER_BASE}/faq.md", "karpenter/faq.md"),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/troubleshooting.md",
        "karpenter/troubleshooting.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/concepts/_index.md",
        "karpenter/concepts/index.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/concepts/disruption.md",
        "karpenter/concepts/disruption.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/concepts/capacitybuffers.md",
        "karpenter/concepts/capacitybuffers.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/concepts/nodepools.md",
        "karpenter/concepts/nodepools.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/concepts/nodeclasses.md",
        "karpenter/concepts/nodeclasses.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/concepts/scheduling.md",
        "karpenter/concepts/scheduling.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/getting-started/_index.md",
        "karpenter/getting-started/index.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/tasks/_index.md",
        "karpenter/tasks/index.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/tasks/managing-amis.md",
        "karpenter/tasks/managing-amis.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/tasks/monitoring-ec2-api-usage.md",
        "karpenter/tasks/monitoring-ec2-api-usage.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/tasks/odcrs.md",
        "karpenter/tasks/odcrs.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/upgrading/_index.md",
        "karpenter/upgrading/index.md",
    ),
    CorpusFile(
        KARPENTER_REPO,
        KARPENTER_REF,
        f"{KARPENTER_BASE}/reference/_index.md",
        "karpenter/reference/index.md",
    ),
    # --- KEDA (Kubernetes Event-Driven Autoscaling) ---
    CorpusFile(KEDA_REPO, KEDA_REF, f"{KEDA_BASE}/_index.md", "keda/index.md"),
    CorpusFile(KEDA_REPO, KEDA_REF, f"{KEDA_BASE}/concepts/_index.md", "keda/concepts/index.md"),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/concepts/scaling-deployments.md",
        "keda/concepts/scaling-deployments.md",
    ),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/concepts/scaling-jobs.md",
        "keda/concepts/scaling-jobs.md",
    ),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/concepts/authentication.md",
        "keda/concepts/authentication.md",
    ),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/concepts/external-scalers.md",
        "keda/concepts/external-scalers.md",
    ),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/concepts/admission-webhooks.md",
        "keda/concepts/admission-webhooks.md",
    ),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/concepts/troubleshooting.md",
        "keda/concepts/troubleshooting.md",
    ),
    CorpusFile(KEDA_REPO, KEDA_REF, f"{KEDA_BASE}/deploy.md", "keda/deploy.md"),
    CorpusFile(KEDA_REPO, KEDA_REF, f"{KEDA_BASE}/scalers/aws-sqs.md", "keda/scalers/aws-sqs.md"),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/scalers/aws-cloudwatch.md",
        "keda/scalers/aws-cloudwatch.md",
    ),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/scalers/aws-dynamodb.md",
        "keda/scalers/aws-dynamodb.md",
    ),
    CorpusFile(
        KEDA_REPO,
        KEDA_REF,
        f"{KEDA_BASE}/scalers/aws-dynamodb-streams.md",
        "keda/scalers/aws-dynamodb-streams.md",
    ),
    CorpusFile(
        KEDA_REPO, KEDA_REF, f"{KEDA_BASE}/scalers/aws-kinesis.md", "keda/scalers/aws-kinesis.md"
    ),
]

NOTICE_TEMPLATE = """\
# NOTICE / Attribution

This corpus is a small, pinned subset of two Apache-2.0 licensed open-source
documentation sites, downloaded for RAG demo/evaluation purposes only. No
modifications were made to the downloaded files beyond selection.

## Karpenter
- Source: https://github.com/{karpenter_repo}
- Ref (pinned): {karpenter_ref}
- License: Apache-2.0 (https://github.com/{karpenter_repo}/blob/{karpenter_ref}/LICENSE)
- Copyright: the Karpenter Authors / Amazon.com, Inc. or its affiliates.

## KEDA
- Source: https://github.com/{keda_repo}
- Ref (pinned): {keda_ref}
- Docs version: {keda_version}
- License: Apache-2.0 (https://github.com/{keda_repo}/blob/{keda_ref}/LICENSE)
- Copyright: the KEDA Authors.

## Files fetched
{file_list}
"""


def download(file: CorpusFile, out_dir: Path) -> bool:
    url = f"https://raw.githubusercontent.com/{file.repo}/{file.ref}/{file.remote_path}"
    dest = out_dir / file.local_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        print(f"  SKIP {file.local_path}: HTTP {exc.code}", file=sys.stderr)
        return False
    except urllib.error.URLError as exc:
        print(f"  SKIP {file.local_path}: {exc}", file=sys.stderr)
        return False
    dest.write_bytes(data)
    print(f"  OK   {file.local_path} ({len(data)} bytes)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/corpus", type=Path)
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[CorpusFile] = []
    print(f"Fetching {len(FILES)} candidate pages into {out_dir}/ ...")
    for file in FILES:
        if download(file, out_dir):
            fetched.append(file)
        time.sleep(0.1)  # be polite to raw.githubusercontent.com

    notice = NOTICE_TEMPLATE.format(
        karpenter_repo=KARPENTER_REPO,
        karpenter_ref=KARPENTER_REF,
        keda_repo=KEDA_REPO,
        keda_ref=KEDA_REF,
        keda_version=KEDA_VERSION,
        file_list="\n".join(f"- {f.local_path}" for f in fetched),
    )
    (out_dir / "NOTICE.md").write_text(notice)

    print(f"\nDone: {len(fetched)}/{len(FILES)} pages fetched. See {out_dir}/NOTICE.md")
    if len(fetched) < 15:
        print(
            "WARNING: fewer than 15 pages fetched — check network access or updated file paths.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
