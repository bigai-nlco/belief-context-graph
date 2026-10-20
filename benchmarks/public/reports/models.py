"""Private grading material is separate from the task shown to the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TASK_KIND_REPORT = "report"


@dataclass(frozen=True)
class BenchmarkTask:
    benchmark: str
    task_id: str
    question: str
    answers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    attachment: Path | None = None
    kind: str = TASK_KIND_REPORT
    reference: dict[str, Any] = field(default_factory=dict, repr=False)
    aliases: tuple[str, ...] = ()

    def matches_id(self, value: str) -> bool:
        return value.casefold() in {self.task_id.casefold(), *(x.casefold() for x in self.aliases)}
