"""Pluggable scorers for report-type benchmark tasks.

A :class:`ReportScorer` turns ``(task, report, tool trace)`` into a flat dict
of metrics. Deterministic scorers run inline after every report task. LLM
judges implement the same interface with ``requires_llm = True``; they are
registered in :data:`LLM_JUDGES` but deliberately not run yet. Because each
result file records the saved report and trajectory paths, judges can be
added later and applied to finished runs without re-running the Agent.

Metric keys are flat so ``summarize_results`` can average them; list-valued
diagnostics (``files_cited`` and so on) are kept but not averaged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from benchmarks.public.reports import drb2, drbench
from benchmarks.public.reports.models import BenchmarkTask


@dataclass
class ReportScoringContext:
    """Per-task material that scorers may need beyond the report itself."""

    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    blocklist: dict[str, Any] | None = None
    citation_keys: dict[str, list[str]] | None = None


class ReportScorer(Protocol):
    name: str
    requires_llm: bool

    def score(
        self, task: BenchmarkTask, report: str, context: ReportScoringContext
    ) -> dict[str, Any]: ...


class ReportStatistics:
    """Length and citation statistics shared by every report benchmark."""

    name = "report_statistics"
    requires_llm = False

    def score(
        self, task: BenchmarkTask, report: str, context: ReportScoringContext
    ) -> dict[str, Any]:
        text = report or ""
        urls = sorted(set(drb2.extract_urls(text)))
        domains = sorted({(urlsplit(url).hostname or "").removeprefix("www.") for url in urls})
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_words = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))
        return {
            "report_chars": len(text),
            # CJK characters count as words, which matches common practice.
            "report_words": cjk + latin_words,
            "report_headings": len(re.findall(r"(?m)^#{1,6}\s+\S", text)),
            "urls_cited": len(urls),
            "domains_cited": len([domain for domain in domains if domain]),
            "has_references_section": bool(
                re.search(
                    r"(?im)^#{1,6}\s*(references|sources|参考文献|参考资料|来源)\b",
                    text,
                )
            ),
        }


class DRB2BlockedCitationAudit:
    """Counts report citations of the blocked expert source (DRB2 ``-1``)."""

    name = "drb2_blocked_citation_audit"
    requires_llm = False

    def score(
        self, task: BenchmarkTask, report: str, context: ReportScoringContext
    ) -> dict[str, Any]:
        payload = context.blocklist or drb2.blocklist_payload(task)
        return drb2.audit_blocked_citations(report, payload)


class DRBenchDeterministicRecall:
    """File-level read/citation recall from DRBench file labels."""

    name = "drbench_deterministic_recall"
    requires_llm = False

    def score(
        self, task: BenchmarkTask, report: str, context: ReportScoringContext
    ) -> dict[str, Any]:
        return drbench.deterministic_recall(
            task,
            report,
            tool_trace=context.tool_trace,
            citation_keys=context.citation_keys,
        )


class LLMReportJudge:
    """Base for the next round's LLM judges; not run in this round.

    Implementations should read their judge configuration at construction,
    return flat metrics, and raise on transport failure so the runner can mark
    the task ``score_error`` (fail closed), like :class:`scoring.LLMJudge`.
    """

    name = "llm_report_judge"
    requires_llm = True

    def score(
        self, task: BenchmarkTask, report: str, context: ReportScoringContext
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} is not implemented yet.")


class DRB2RubricJudge(LLMReportJudge):
    """Official three-way rubric judge (GPT-5.5, 50 rubric items per batch)."""

    name = "drb2_rubric_judge"


class DRBenchInsightsRecallJudge(LLMReportJudge):
    """Official DRBench insights_recall / distractor_recall LLM judge."""

    name = "drbench_insights_recall"


DETERMINISTIC_SCORERS: dict[str, tuple[ReportScorer, ...]] = {
    drb2.NAME: (ReportStatistics(), DRB2BlockedCitationAudit()),
    drbench.NAME: (ReportStatistics(), DRBenchDeterministicRecall()),
}

LLM_JUDGES: dict[str, tuple[type[LLMReportJudge], ...]] = {
    drb2.NAME: (DRB2RubricJudge,),
    drbench.NAME: (DRBenchInsightsRecallJudge,),
}


def score_report_task(
    task: BenchmarkTask,
    report: str,
    context: ReportScoringContext,
    *,
    scorers: Sequence[ReportScorer] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Run scorers; return merged metrics and any scorer error messages."""

    selected = DETERMINISTIC_SCORERS.get(task.benchmark, ()) if scorers is None else scorers
    metrics: dict[str, Any] = {"scorers": []}
    errors: list[str] = []
    for scorer in selected:
        try:
            values = scorer.score(task, report, context)
        except Exception as exc:  # A scorer bug must not lose the report.
            errors.append(f"{scorer.name}: {exc}")
            continue
        metrics["scorers"].append(scorer.name)
        metrics.update(values)
    return metrics, errors


__all__ = [
    "DETERMINISTIC_SCORERS",
    "LLM_JUDGES",
    "DRB2BlockedCitationAudit",
    "DRB2RubricJudge",
    "DRBenchDeterministicRecall",
    "DRBenchInsightsRecallJudge",
    "LLMReportJudge",
    "ReportScorer",
    "ReportScoringContext",
    "ReportStatistics",
    "score_report_task",
]
