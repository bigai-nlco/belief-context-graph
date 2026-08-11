#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
STYLES = ROOT / "assets" / "styles.css"
PAGES_JS = ROOT / "assets" / "pages.js"
errors: list[str] = []

for path in (
    ROOT / "index.html",
    STYLES,
    ROOT / "assets" / "app.js",
    PAGES_JS,
    ROOT / "serve.py",
):
    if not path.is_file():
        errors.append(f"missing required file: {path.relative_to(ROOT)}")

if "gradient(" in STYLES.read_text(encoding="utf-8").lower():
    errors.append("styles.css still contains a gradient fill")

architecture = (SOURCE / "architecture.md").read_text(encoding="utf-8")
for required in (
    "bcg.construct.unified",
    "bcg.construct.hybrid",
    'markerWidth="7"',
    'markerHeight="7"',
):
    if required not in architecture:
        errors.append(f"architecture.md is missing {required!r}")

quickstart = (SOURCE / "quickstart.md").read_text(encoding="utf-8")
for required in ("return memory, result", "memory, result = asyncio.run(main())"):
    if required not in quickstart:
        errors.append(f"quickstart is missing {required!r}")

overview = (SOURCE / "index.md").read_text(encoding="utf-8")
for required in (
    "retrieval memory to belief computation memory",
    "Runnable examples are available",
):
    if required not in overview:
        errors.append(f"index.md is missing {required!r}")

for path in SOURCE.rglob("*.md"):
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines[:-1]):
        next_line = lines[i + 1].strip()
        separator = next_line.startswith("|") and not next_line.replace(
            "|", ""
        ).replace(":", "").replace("-", "").replace(" ", "")
        if line.strip().startswith("|") and separator and "`" in line:
            errors.append(
                f"inline code remains in table heading: {path.relative_to(SOURCE)}:{i + 1}"
            )

match = re.search(
    r"window\.BCG_PAGES = (.*?);\nwindow\.BCG_NAV = (.*?);\nwindow\.BCG_ORDER = (.*?);\s*$",
    PAGES_JS.read_text(encoding="utf-8"),
    re.S,
)
if not match:
    errors.append("pages.js could not be parsed")
else:
    pages = json.loads(match.group(1))
    nav = json.loads(match.group(2))
    order = json.loads(match.group(3))
    for slug in order:
        if slug not in pages:
            errors.append(f"BCG_ORDER references missing page: {slug}")
    for tab in nav:
        for group in tab["groups"]:
            for slug in group["pages"]:
                if slug not in pages:
                    errors.append(f"navigation references missing page: {slug}")
    for slug, page in pages.items():
        if re.search(r"<th[^>]*>\s*<code", page["html"], re.I):
            errors.append(f"compiled table heading contains inline code: {slug}")

source_slugs = {
    str(path.relative_to(SOURCE).with_suffix("")).replace("\\", "/")
    for path in SOURCE.rglob("*.md")
    if path.name != "README.md"
}
for path in SOURCE.rglob("*.md"):
    text = path.read_text(encoding="utf-8")
    for pattern in (
        re.compile(r"\]\(/([^)\s]+)"),
        re.compile(r'href="/([^"]+)"'),
    ):
        for found in pattern.finditer(text):
            target = found.group(1).split("#", 1)[0].rstrip("/")
            if target and target not in source_slugs:
                errors.append(
                    f"broken internal link in {path.relative_to(SOURCE)}: /{target}"
                )

if errors:
    raise SystemExit(
        "Offline documentation validation failed:\n- " + "\n- ".join(errors)
    )

print(
    f"Offline documentation validated: {len(source_slugs)} pages, flat fills, "
    "plain table headings, runnable quick start, and valid navigation."
)
