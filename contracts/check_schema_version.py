#!/usr/bin/env python3
"""Guard contract schema versions (step 10 acceptance: CI blocks undeclared breaking changes).

For every versioned contract file: if the working tree differs from HEAD,
the schema_version must have increased. Run as part of ``make check-contracts``
and CI; works both locally (uncommitted edits) and in CI (PR diffs).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSIONED = sorted(
    p
    for p in (ROOT / "contracts").glob("*.json")
    if p.name.endswith(".schema.json") or p.name == "defaults.json"
)


def _head_document(path: Path) -> dict[str, Any] | None:
    try:
        text = subprocess.run(
            ["git", "show", f"HEAD:{path.relative_to(ROOT)}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def _version(document: dict[str, Any]) -> int:
    try:
        return int(document.get("schema_version", 0))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    failures: list[str] = []
    checked = 0
    for path in VERSIONED:
        current = json.loads(path.read_text(encoding="utf-8"))
        head = _head_document(path)
        if head is None:
            continue  # new file in this change set
        checked += 1
        if current == head:
            continue  # unchanged since HEAD
        current_version = _version(current)
        head_version = _version(head)
        if current_version <= head_version:
            failures.append(
                f"{path.relative_to(ROOT)}: content changed but schema_version "
                f"did not increase ({head_version} -> {current_version}); bump it "
                "(additive-optional only, see contracts/README.md)"
            )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print(f"OK: {checked} versioned contract file(s) consistent with HEAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
