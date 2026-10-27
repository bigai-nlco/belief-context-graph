"""Run: python -m tools.ai_review --input TASK_DIR --output REVIEW_DIR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.ai_review.client import ReviewConfig, ReviewError
from tools.ai_review.pipeline import run_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence-grounded BCG AI annotation via LiteLLM"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One task directory, result.json or single-case bundle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Separate directory for prompts, citations, usage and review",
    )
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".bcg/.env")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare inputs and prompt preview without an API call",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Resume a failed/interrupted review; reuse valid completed batches",
    )
    parser.add_argument("--chunk-turns", type=int, default=6)
    parser.add_argument("--chunk-chars", type=int, default=60000)
    parser.add_argument("--max-batches", type=int, default=64)
    args = parser.parse_args()
    try:
        result = run_review(
            args.input,
            args.output,
            config=ReviewConfig.from_env(args.env_file),
            dry_run=args.dry_run,
            retry_failed=args.retry,
            chunk_chars=args.chunk_chars,
            chunk_turns=args.chunk_turns,
            max_batches=args.max_batches,
        )
    except (ReviewError, ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result.get("status", "completed"),
                "usage": result.get("usage"),
                "task_id": result["task_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
