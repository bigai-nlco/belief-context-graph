"""Shared data models for benchmark loading, execution, and scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_API_QUOTA_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "quota exceeded",
    "exceeded your current quota",
    "insufficient credit",
    "insufficient balance",
    "credit balance",
    "credits exhausted",
    "out of credits",
    "payment required",
    "budget exceeded",
    "budget has been exceeded",
    "billing limit",
    "key limit exceeded",
    "额度不足",
    "余额不足",
)


@dataclass(frozen=True)
class BenchmarkTask:
    """One normalized benchmark example."""

    benchmark: str
    task_id: str
    question: str
    answers: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    attachment: Path | None = None


@dataclass(frozen=True)
class ScoreResult:
    """The benchmark-specific evaluation of one Agent response."""

    correct: bool
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    extracted_answer: str = ""
    error: str | None = None


@dataclass
class TokenUsage:
    """Token and cost totals, kept separated by direction and cache status."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0
    total: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    total_cost: float = 0.0

    def add_event_usage(self, usage: dict[str, Any]) -> None:
        self.input += _integer(usage.get("input"))
        self.output += _integer(usage.get("output"))
        self.cache_read += _integer(usage.get("cacheRead"))
        self.cache_write += _integer(usage.get("cacheWrite"))
        self.reasoning += _integer(usage.get("reasoning"))
        self.total += _integer(usage.get("totalTokens"))
        cost = usage.get("cost")
        if isinstance(cost, dict):
            self.input_cost += _number(cost.get("input"))
            self.output_cost += _number(cost.get("output"))
            self.cache_read_cost += _number(cost.get("cacheRead"))
            self.cache_write_cost += _number(cost.get("cacheWrite"))
            self.total_cost += _number(cost.get("total"))

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def is_api_quota_error(value: object) -> bool:
    """Return whether a provider error clearly reports exhausted paid quota."""

    normalized = str(value or "").casefold()
    return any(marker in normalized for marker in _API_QUOTA_MARKERS)


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "BenchmarkTask",
    "ScoreResult",
    "TokenUsage",
    "is_api_quota_error",
]
