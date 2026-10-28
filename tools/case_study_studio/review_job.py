"""Detached review worker; reads the same frozen evidence as the annotation page."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from tools.ai_review.client import ReviewError
from tools.ai_review.pipeline import run_review, usage
from tools.case_study_studio.control import Control, now

if TYPE_CHECKING:
    from tools.case_study_studio.review_queue import AIReviews


def execute(queue: AIReviews, job_id: int) -> None:
    job = queue.row(job_id)
    if job["status"] != "running" or job["pid"] != os.getpid():
        return
    output = queue.output(job_id)
    try:
        source, digest = queue.control.evidence(job["run_id"], job["benchmark"], job["task_id"])
        result = run_review(source, output, retry_failed=True)
        with queue.control.connect() as db:
            db.execute("UPDATE ai_reviews SET status='completed',finished_at=?,evidence_sha=?,usage=?,error=NULL "
                       "WHERE id=? AND status='running' AND pid=?",
                       (now(), digest, json.dumps(result["usage"]), job_id, os.getpid()))
    except Exception as exc:
        message = str(exc) if isinstance(exc, ReviewError) else f"AI Review 未完成（{type(exc).__name__}）"
        with queue.control.connect() as db:
            db.execute("UPDATE ai_reviews SET status='failed',finished_at=?,error=?,usage=? "
                       "WHERE id=? AND status='running' AND pid=?",
                       (now(), message, json.dumps(usage(output / "calls.jsonl")), job_id, os.getpid()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    control = Control(args.root, args.state)
    execute(control.ai_reviews, args.job_id)


if __name__ == "__main__":
    main()
