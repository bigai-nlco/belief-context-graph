"""DeepResearch Bench II (https://github.com/imlrz/DeepResearch-Bench-II).

Each task asks for a long-form research report. Grading material is a set of
expert-derived binary rubrics (``info_recall``, ``analysis``,
``presentation``) plus a ``blocked`` record naming the expert source article.
The official judge scores an item ``-1`` when the supporting sentence cites a
blocked reference, so the Agent must not be able to retrieve that article.

This module loads tasks, builds the per-task retrieval blocklist that the
Agent's ``web_search``/``web_fetch`` tools enforce, and audits finished
reports for blocked citations. Rubric judging (LLM-as-judge) is a pluggable
scorer that is intentionally not run yet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from bcg.apps.benchmark.models import TASK_KIND_REPORT, BenchmarkTask

NAME = "deepresearch_bench_ii"
TASKS_FILENAME = "tasks_and_rubrics.jsonl"
DIRECTORY_CANDIDATES = (
    "deepresearch_bench_ii",
    "DeepResearch-Bench-II",
    "deepresearch_bench_2",
    "drb2",
)
RUBRIC_DIMENSIONS = ("info_recall", "analysis", "presentation")
LANGUAGES = ("zh", "en")

_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"'`，。；）】]+", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)


class DRB2DataError(ValueError):
    """Raised when the DeepResearch Bench II task file is unusable."""


def find_tasks_file(data_root: Path) -> Path:
    for directory in DIRECTORY_CANDIDATES:
        candidate = data_root / directory / TASKS_FILENAME
        if candidate.is_file():
            return candidate
    searched = ", ".join(
        str(data_root / directory / TASKS_FILENAME)
        for directory in DIRECTORY_CANDIDATES
    )
    raise DRB2DataError(
        f"DeepResearch Bench II tasks were not found. Searched: {searched}. "
        f"Run `bcg benchmark prepare {NAME}` or pass "
        f"--data-file {NAME}=PATH/{TASKS_FILENAME}."
    )


def load_tasks(
    data_root: Path,
    *,
    data_file: Path | None = None,
    language: str | None = None,
    theme: str | None = None,
) -> list[BenchmarkTask]:
    source = (
        Path(data_file).expanduser().resolve()
        if data_file is not None
        else find_tasks_file(data_root)
    )
    if source.is_dir():
        source = source / TASKS_FILENAME
    if not source.is_file():
        raise DRB2DataError(f"DeepResearch Bench II task file does not exist: {source}")
    if language is not None and language not in LANGUAGES:
        raise DRB2DataError("--drb2-language must be `zh` or `en`.")

    tasks: list[BenchmarkTask] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DRB2DataError(
                f"{source}:{line_number} is not valid JSON: {exc}"
            ) from exc
        task = _task_from_row(row, source=source, line_number=line_number)
        if language is not None and task.metadata["language"] != language:
            continue
        if theme is not None and task.metadata["theme"].casefold() != theme.casefold():
            continue
        tasks.append(task)
    if not tasks:
        raise DRB2DataError(
            f"No DeepResearch Bench II tasks remain in {source} after filters."
        )
    return tasks


def _task_from_row(row: Any, *, source: Path, line_number: int) -> BenchmarkTask:
    if not isinstance(row, dict):
        raise DRB2DataError(f"{source}:{line_number} is not a JSON object.")
    idx = row.get("idx")
    prompt = row.get("prompt")
    content = row.get("content")
    if not isinstance(idx, int) or not isinstance(prompt, str) or not prompt.strip():
        raise DRB2DataError(f"{source}:{line_number} lacks an integer idx or a prompt.")
    if not isinstance(content, dict) or not isinstance(content.get("rubric"), dict):
        raise DRB2DataError(f"{source}:{line_number} lacks content.rubric.")
    rubric = {
        dimension: [str(item) for item in content["rubric"].get(dimension) or []]
        for dimension in RUBRIC_DIMENSIONS
    }
    blocked = content.get("blocked") if isinstance(content.get("blocked"), dict) else {}
    language = str(row.get("language") or "en").strip().lower()
    original_id = str(row.get("id") or "").strip()
    aliases = tuple(
        value for value in (original_id, f"idx-{idx}") if value and value != str(idx)
    )
    return BenchmarkTask(
        benchmark=NAME,
        task_id=str(idx),
        question=prompt.strip(),
        answers=(),
        kind=TASK_KIND_REPORT,
        aliases=aliases,
        reference={
            "task": str(content.get("task") or prompt),
            "rubric": rubric,
            "blocked": {
                "title": str(blocked.get("title") or ""),
                "authors": [str(value) for value in blocked.get("authors") or []],
                "urls": [str(value) for value in blocked.get("urls") or []],
            },
        },
        metadata={
            "idx": idx,
            "id": original_id,
            "language": language,
            "theme": str(row.get("theme") or ""),
            "description": str(row.get("description") or ""),
            "license": str(row.get("license") or ""),
            "rubric_counts": {key: len(value) for key, value in rubric.items()},
        },
    )


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------


def blocklist_payload(task: BenchmarkTask) -> dict[str, Any]:
    """The per-task record consumed by the Agent's web tools.

    It deliberately contains only what is needed for matching. It is written
    outside the Agent workspace; the Agent never sees the blocked title.
    """

    blocked = task.reference.get("blocked") or {}
    urls = [str(value) for value in blocked.get("urls") or []]
    return {
        "benchmark": task.benchmark,
        "task_id": task.task_id,
        "urls": urls,
        "url_keys": sorted(
            {key for key in (normalize_url(url) for url in urls) if key}
        ),
        "dois": sorted({doi for url in urls for doi in _dois(url)}),
        "titles": [blocked["title"]] if blocked.get("title") else [],
    }


def normalize_url(url: str) -> str:
    """Scheme/``www``/query/fragment-insensitive URL key used for blocking."""

    text = str(url or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"https://{text}"
    try:
        parts = urlsplit(text)
    except ValueError:
        return ""
    host = (parts.hostname or "").casefold()
    if host.startswith("www."):
        host = host[4:]
    path = unquote(parts.path or "").rstrip("/")
    for suffix in ("/pdf", "/full", "/htm", "/html", "/abstract"):
        if path.casefold().endswith(suffix):
            path = path[: -len(suffix)]
    return f"{host}{path}".casefold() if host else ""


def normalize_title(title: str) -> str:
    return re.sub(r"[\W_]+", " ", str(title or "").casefold()).strip()


def _dois(text: str) -> list[str]:
    return [match.rstrip(".,;)").casefold() for match in _DOI_RE.findall(text or "")]


def extract_urls(text: str) -> list[str]:
    return [match.rstrip(".,;:!?") for match in _URL_RE.findall(text or "")]


def audit_blocked_citations(report: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Find report citations that point at the blocked expert source."""

    url_keys = set(payload.get("url_keys") or [])
    dois = set(payload.get("dois") or [])
    hits: list[str] = []
    for url in extract_urls(report):
        key = normalize_url(url)
        if (
            key
            and any(
                key == blocked or key.startswith(f"{blocked}/") for blocked in url_keys
            )
            or any(doi in url.casefold() for doi in dois)
        ):
            hits.append(url)
    for doi in dois:
        if doi in (report or "").casefold() and doi not in " ".join(hits).casefold():
            hits.append(doi)
    folded_report = normalize_title(report)
    title_hits = [
        title
        for title in payload.get("titles") or []
        if len(normalize_title(title)) >= 24 and normalize_title(title) in folded_report
    ]
    return {
        "blocked_url_hits": sorted(set(hits)),
        "blocked_title_hits": title_hits,
        "blocked_citation_hits": len(set(hits)) + len(title_hits),
    }


def official_report_filename(task: BenchmarkTask) -> str:
    return f"idx-{task.metadata.get('idx', task.task_id)}.md"


__all__ = [
    "NAME",
    "DRB2DataError",
    "audit_blocked_citations",
    "blocklist_payload",
    "extract_urls",
    "find_tasks_file",
    "load_tasks",
    "normalize_title",
    "normalize_url",
    "official_report_filename",
]
