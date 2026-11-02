"""Detached worker for a fixed Run annotation snapshot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from tools.ai_review.client import ReviewError
from tools.ai_review.pipeline import usage
from tools.case_study_studio.control import Control, now, read_json
from tools.case_study_studio.run_summary import input_key, run_summary

if TYPE_CHECKING:
    from tools.case_study_studio.run_summary_queue import RunSummaries


def execute(queue: RunSummaries, job_id: int) -> None:
    job = queue.row(job_id)
    if job["status"] != "running" or job["pid"] != os.getpid():
        return
    output = queue.output(job_id)
    try:
        context = read_json(output / "context.json")
        if input_key(context) != job["input_key"]:
            raise ReviewError("Run 总结固定输入校验失败，未调用模型")
        result = run_summary(context, output)
        with queue.control.connect() as db:
            db.execute("UPDATE run_summaries SET status='completed',finished_at=?,usage=?,error=NULL "
                "WHERE id=? AND status='running' AND pid=?", (now(), json.dumps(result["usage"]), job_id, os.getpid()))
    except Exception as exc:
        message = str(exc) if isinstance(exc, ReviewError) else f"Run 总结未完成（{type(exc).__name__}）"
        with queue.control.connect() as db:
            db.execute("UPDATE run_summaries SET status='failed',finished_at=?,error=?,usage=? "
                "WHERE id=? AND status='running' AND pid=?",
                (now(), message, json.dumps(usage(output / "calls.jsonl")), job_id, os.getpid()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    control = Control(args.root, args.state)
    execute(control.run_summaries, args.job_id)


if __name__ == "__main__":
    main()
