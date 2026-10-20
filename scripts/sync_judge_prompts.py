#!/usr/bin/env python3
"""Copy the official judge prompts from local upstream clones into bcg.

    python scripts/sync_judge_prompts.py --drb2 ~/code/DeepResearch-Bench-II --drbench ~/code/drbench
    python scripts/sync_judge_prompts.py --drb2 ... --drbench ... --check   # verify only

Both upstream repositories are Apache-2.0 licensed; SOURCES.json records the
commit and file of every prompt for attribution and reproducibility.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bcg.apps.benchmark.judges.official_sources import (  # noqa: E402
    PROMPT_SOURCES,
    PROMPTS_DIR,
    SOURCES_FILE,
    extract_prompts,
    sha256_text,
)

REPOSITORIES = {
    "DeepResearch-Bench-II": "https://github.com/imlrz/DeepResearch-Bench-II",
    "drbench": "https://github.com/ServiceNow/drbench",
}


def commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--drb2", type=Path, required=True)
    parser.add_argument("--drbench", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Compare without writing.")
    args = parser.parse_args()

    prompts = extract_prompts(args.drb2, args.drbench)
    if args.check:
        mismatched = [
            name
            for name, text in prompts.items()
            if (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8") != text
        ]
        print(
            "All prompts match upstream."
            if not mismatched
            else f"Mismatched: {mismatched}"
        )
        return 1 if mismatched else 0

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    roots = {"DeepResearch-Bench-II": args.drb2, "drbench": args.drbench}
    sources = {
        "license": "Apache-2.0 (both upstream repositories)",
        "synced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repositories": {
            name: {"url": url, "commit": commit(roots[name])}
            for name, url in REPOSITORIES.items()
        },
        "prompts": {},
    }
    for name, text in prompts.items():
        (PROMPTS_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
        repository, path = PROMPT_SOURCES[name]
        sources["prompts"][name] = {
            "repository": repository,
            "path": path,
            "sha256": sha256_text(text),
        }
        print(f"{name}: {len(text)} chars")
    SOURCES_FILE.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {SOURCES_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
