"""``--task-ids`` parsing and application.

Accepted forms (combinable):

* ``1,5,7`` — unscoped IDs, matched against every benchmark in the run;
* ``drb2:1,5,60-62;drbench:DR0001-DR0003`` — per-benchmark IDs (``;``-separated);
* ``@path/to/ids.txt`` — read the spec from a file; each line is either a
  scoped group (``drbench:DR0001,DR0002``) or plain IDs; ``#`` starts a comment.

A range ``A-B`` expands when both ends share a non-numeric prefix
(``DR0001-DR0003``, ``60-62``) and no task has that literal ID. Unknown IDs are
an error. Explicitly selected tasks keep the order given and are not shuffled.
Benchmarks that are not named by a scoped group are not filtered by ID.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.public.reports.models import BenchmarkTask

_RANGE = re.compile(r"^(?P<prefix>\D*?)(?P<start>\d+)-(?P<prefix2>\D*?)(?P<end>\d+)$")
_MAX_RANGE = 10_000


class SelectionError(ValueError):
    """Raised for malformed or unknown task IDs."""


@dataclass
class TaskSelection:
    unscoped: list[str] = field(default_factory=list)
    scoped: dict[str, list[str]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.unscoped and not self.scoped

    def ids_for(self, benchmark: str) -> list[str] | None:
        if benchmark in self.scoped:
            return self.scoped[benchmark] + self.unscoped
        return list(self.unscoped) if self.unscoped else None


def parse_task_ids(spec: str | None, *, canonicalize: Callable[[str], str]) -> TaskSelection:
    selection = TaskSelection()
    if not spec or not spec.strip():
        return selection
    text = spec.strip()
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SelectionError(f"Cannot read --task-ids file {path}: {exc}") from exc
        groups = [line.split("#", 1)[0].strip() for line in lines]
    else:
        groups = [part.strip() for part in text.split(";")]
    for group in groups:
        if not group:
            continue
        head, colon, tail = group.partition(":")
        if colon and _looks_like_benchmark(head):
            name = canonicalize(head.strip())
            selection.scoped.setdefault(name, []).extend(_split_ids(tail))
        else:
            selection.unscoped.extend(_split_ids(group))
    return selection


def _looks_like_benchmark(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_\- ]*", value.strip()))


def _split_ids(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]


def expand_token(token: str, tasks: list[BenchmarkTask]) -> list[str]:
    if any(task.matches_id(token) for task in tasks):
        return [token]
    match = _RANGE.match(token)
    if not match:
        return [token]
    prefix = match.group("prefix")
    prefix2 = match.group("prefix2")
    if prefix2 and prefix2 != prefix:
        return [token]
    start, end = int(match.group("start")), int(match.group("end"))
    if end < start or end - start > _MAX_RANGE:
        raise SelectionError(f"Invalid task ID range {token!r}.")
    width = len(match.group("start")) if match.group("start").startswith("0") else 0
    return [f"{prefix}{number:0{width}d}" for number in range(start, end + 1)]


def select_tasks(
    tasks: list[BenchmarkTask], ids: list[str]
) -> tuple[list[BenchmarkTask], list[str]]:
    """Return tasks in the requested order plus the IDs that matched nothing."""

    selected: list[BenchmarkTask] = []
    seen: set[str] = set()
    missing: list[str] = []
    for token in ids:
        for value in expand_token(token, tasks):
            matches = [task for task in tasks if task.matches_id(value)]
            if not matches:
                missing.append(value)
                continue
            for task in matches:
                if task.task_id not in seen:
                    seen.add(task.task_id)
                    selected.append(task)
    return selected, missing


__all__ = [
    "SelectionError",
    "TaskSelection",
    "expand_token",
    "parse_task_ids",
    "select_tasks",
]
