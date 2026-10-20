"""DRBench report metrics, aligned with the official ``score_report``.

Ported from https://github.com/ServiceNow/drbench (Apache-2.0), commit in
``prompts/SOURCES.json``:

* ``score_report`` truncates the report to 60,000 characters and extracts
  claims with ``break_report_to_insights`` (verbatim prompt, ``temperature=0``,
  3 attempts, JSON array between the first ``[`` and last ``]``, claims shorter
  than 10 characters dropped, ``[]`` when every attempt fails);
* ``insights_recall`` / ``distractor_recall`` (``QASimilarityV2``): one call
  per gold QA of that ``qa_type`` with ``insight_scoring.txt``, sent through
  ``AIAgentManager`` defaults ``temperature=0.7`` and ``max_tokens=1000``; 3
  attempts; ``"yes"`` scores 1; a QA whose every attempt fails scores 0; the
  metric is the mean over QAs (0.0 when there are none);
* ``report_quality``: five criteria scored 1-10 and divided by 10, verbatim
  prompt then the verbatim retry prompt, ``temperature=0``, 3 attempts, at
  least 5 parsed criteria required, neutral 0.5 when every attempt fails.

Deviations, recorded in every judgement:

* ``report_quality`` receives the report text. Official ``score_report``
  builds its report dict without ``report_text``, so it scores an empty report.
* Failed judge calls are scored as officially (0, or 0.5 for quality) but are
  counted in ``judge_failures`` so they can be found and re-judged.
* A scoring reply that is JSON but lacks a key (``answer``, ...) is treated as
  a failed attempt and retried; officially the ``KeyError`` aborts the run.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bcg.apps.benchmark.judges.client import JudgeClient, JudgeRequestError, add_usage
from bcg.apps.benchmark.judges.official_sources import load_prompt
from bcg.apps.benchmark.models import BenchmarkTask

NAME = "drbench_official"
DEFAULT_MODEL = "gpt-4o-mini"  # score_report() default
MAX_REPORT_LENGTH = 60_000
METRICS = ("insights_recall", "distractor_recall", "report_quality")
DEFAULT_METRICS = METRICS
EXTRACTION_PARAMS: dict[str, Any] = {"temperature": 0}
SCORING_PARAMS: dict[str, Any] = {"temperature": 0.7, "max_tokens": 1000}
QUALITY_PARAMS: dict[str, Any] = {"temperature": 0}
OFFICIAL_PARAMS = {
    "extraction": EXTRACTION_PARAMS,
    "scoring": SCORING_PARAMS,
    "quality": QUALITY_PARAMS,
}
MAX_RETRIES = 3
QUALITY_KEYS = (
    "depth_quality",
    "relevance_to_question",
    "persona_consistency",
    "coherence_conciseness",
    "contradictions",
)
DEVIATIONS = (
    "report_quality scores the report text; official score_report passes an empty report_text.",
    "Failed judge calls keep the official score (0, or 0.5 for report_quality) and are counted in judge_failures.",
    "A scoring reply missing a JSON key is retried; officially the KeyError aborts scoring.",
)


class _Call:
    """One logical judge call with the official attempt budget."""

    def __init__(self, client: JudgeClient, usage: dict[str, int]) -> None:
        self.client = client
        self.usage = usage
        self.params_used: dict[str, Any] | None = None

    def __call__(self, prompt: str, params: dict[str, Any], attempt: int) -> str:
        try:
            reply = self.client.complete(prompt, params, use_cache=attempt == 0)
        except JudgeRequestError as exc:
            # Officially prompt_llm returns "Error: ..." which then fails to parse.
            return f"Error: {exc}"
        add_usage(self.usage, reply.usage)
        self.params_used = reply.params
        return reply.text


# --- insight extraction -------------------------------------------------------


def extract_insights(
    report: str, call: _Call, params: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = load_prompt("drbench_insight_extraction").format(report_text=report)
    record: dict[str, Any] = {"attempts": 0, "failures": []}
    for attempt in range(MAX_RETRIES):
        record["attempts"] = attempt + 1
        response = call(prompt, params, attempt).strip()
        start, end = response.find("["), response.rfind("]")
        if start == -1 or end == -1 or end <= start:
            record["failures"].append("no JSON array")
            continue
        try:
            insight_list = json.loads(response[start : end + 1])
            if not isinstance(insight_list, list):
                raise ValueError("Response is not a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            record["failures"].append(f"JSON parse failed: {exc}")
            continue
        validated = []
        for insight in insight_list:
            if (
                isinstance(insight, dict)
                and "claim" in insight
                and "citations" in insight
            ):
                claim = (
                    insight["claim"].strip()
                    if isinstance(insight["claim"], str)
                    else ""
                )
                citations = (
                    insight["citations"]
                    if isinstance(insight["citations"], list)
                    else []
                )
                if len(claim) >= 10:
                    validated.append({"claim": claim, "citations": citations})
        record["failed"] = False
        return validated, record
    record["failed"] = True
    return [], record


def format_claims(report_insights: list[dict[str, Any]]) -> str:
    if not report_insights:
        return "No claims found in the report."
    text = ""
    for i, insight in enumerate(report_insights, 1):
        text += f"Insight {i}: {insight.get('claim', '')}\n"
    return text.strip()


def extract_json_from_response(response: str) -> Any:
    match = re.search(r"(\[.*\]|\{.*\})", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        raise ValueError("Could not extract valid JSON from response") from None


# --- QASimilarityV2 -------------------------------------------------------------


def qa_recall(
    qa_list: list[dict[str, Any]],
    claims_text: str,
    call: _Call,
    params: dict[str, Any],
    insight_key: str,
) -> dict[str, Any]:
    template = load_prompt("drbench_insight_scoring")
    results = []
    for qa in [item for item in qa_list if item.get("qa_type") == insight_key]:
        prompt = template.format(
            claims_text=claims_text, question=qa["question"], gold_insight=qa["answer"]
        )
        score, justification, confidence, selected = (
            0.0,
            "Failed to get proper response from model",
            None,
            None,
        )
        failed, attempts, failures = True, 0, []
        for attempt in range(MAX_RETRIES):
            attempts = attempt + 1
            response = call(prompt, params, attempt)
            try:
                parsed = extract_json_from_response(response)
                answer = parsed["answer"].lower()
                score = 1.0 if answer == "yes" else 0.0
                justification = parsed["justification"]
                confidence = parsed["confidence"]
                selected = parsed["selected_insight"]
                failed = False
                break
            except (ValueError, IndexError, TypeError, KeyError, AttributeError) as exc:
                failures.append(str(exc)[:300])
                if attempt == MAX_RETRIES - 1:
                    score = 0.0
                    justification = f"Failed to parse model response after {MAX_RETRIES} retries: {exc}"
        results.append(
            {
                "id": qa.get("id"),
                "type": qa.get("type"),
                "question": qa["question"],
                "expected_insight": qa["answer"],
                "predicted_insight": selected,
                "expected_supporting_paths": (qa.get("supporting_file_paths") or [])
                + (qa.get("supporting_urls") or []),
                "score": score,
                "justification": justification,
                "confidence": confidence,
                "attempts": attempts,
                "failed": failed,
                "failures": failures,
            }
        )
    scores = [item["score"] for item in results]
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "per_question_results": results,
        "judge_failures": sum(item["failed"] for item in results),
    }


# --- ReportQuality --------------------------------------------------------------


def _parse_quality(response: str) -> tuple[dict[str, Any], int]:
    def extract_block(tag: str) -> dict[str, Any]:
        match = re.search(
            rf"<{tag}>\s*<score>\s*(.*?)\s*</score>\s*<justification>\s*(.*?)\s*</justification>\s*</{tag}>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            try:
                score_str, justification = match.groups()
                score = min(10, max(1, int(float(score_str.strip()))))
                return {"score": score / 10.0, "justification": justification.strip()}
            except (ValueError, AttributeError):
                return {
                    "score": 0.0,
                    "justification": f"Failed to parse response for {tag}",
                }
        match = re.search(
            rf"<{tag}>.*?<score>\s*(.*?)\s*</score>.*?</{tag}>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            try:
                score = min(10, max(1, int(float(match.group(1).strip()))))
                return {
                    "score": score / 10.0,
                    "justification": "No justification provided",
                }
            except (ValueError, AttributeError):
                pass
        match = re.search(rf"{tag}.*?(\d+\.?\d*)", response, re.IGNORECASE)
        if match:
            try:
                score = min(10, max(1, int(float(match.group(1)))))
                return {
                    "score": score / 10.0,
                    "justification": "Extracted from unstructured response",
                }
            except (ValueError, AttributeError):
                pass
        return {"score": 0.0, "justification": f"No valid response found for {tag}"}

    metric_result, valid = {}, 0
    for key in QUALITY_KEYS:
        result = extract_block(key)
        metric_result[key] = result
        if result["score"] > 0 or "No valid response" not in result["justification"]:
            valid += 1
    scores = [metric_result[key]["score"] for key in QUALITY_KEYS]
    return {
        "metric_result": metric_result,
        "score": sum(scores) / len(QUALITY_KEYS),
    }, valid


def report_quality(
    report: str, task_config: dict[str, Any], call: _Call, params: dict[str, Any]
) -> dict[str, Any]:
    values = {
        "persona_json": json.dumps(task_config.get("persona"), indent=2),
        "dr_question": task_config.get("dr_question", ""),
        "report_text": report,
    }
    prompts = [load_prompt("drbench_report_quality").format(**values)]
    retry_prompt = load_prompt("drbench_report_quality_retry").format(**values)
    failures = []
    for attempt in range(MAX_RETRIES):
        prompt = prompts[0] if attempt == 0 else retry_prompt
        response = call(prompt, params, attempt)
        if response.startswith("Error: "):
            failures.append(response[:300])
        else:
            evaluation, valid = _parse_quality(response)
            if valid >= len(QUALITY_KEYS):
                return {
                    **evaluation,
                    "attempts": attempt + 1,
                    "failed": False,
                    "failures": failures,
                }
            failures.append(f"only {valid}/{len(QUALITY_KEYS)} criteria parsed")
    default = {
        key: {"score": 0.5, "justification": "Failed to evaluate due to parsing errors"}
        for key in QUALITY_KEYS
    }
    return {
        "metric_result": default,
        "score": 0.5,
        "attempts": MAX_RETRIES,
        "failed": True,
        "failures": failures,
    }


# --- entry point ------------------------------------------------------------------


def judge_report(
    task: BenchmarkTask,
    report: str,
    client: JudgeClient,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    params = {**OFFICIAL_PARAMS, **(params or {})}
    usage: dict[str, int] = {}
    call = _Call(client, usage)
    text = report or ""
    truncated = len(text) > MAX_REPORT_LENGTH
    text = text[:MAX_REPORT_LENGTH]
    qa = task.reference.get("qa") or []
    task_config = task.reference.get("task_config") or {}

    insights, extraction = extract_insights(text, call, params["extraction"])
    claims_text = format_claims(insights)
    scores: dict[str, float] = {}
    details: dict[str, Any] = {}
    failures = int(extraction.get("failed", False))
    for metric in metrics:
        if metric == "insights_recall":
            result = qa_recall(qa, claims_text, call, params["scoring"], "insight")
            enterprise = [
                r
                for r in result["per_question_results"]
                if r.get("type") == "enterprise_fact"
            ]
            if enterprise:
                scores["insights_recall_enterprise"] = sum(
                    r["score"] for r in enterprise
                ) / len(enterprise)
        elif metric == "distractor_recall":
            result = qa_recall(qa, claims_text, call, params["scoring"], "distractor")
        elif metric == "report_quality":
            result = report_quality(text, task_config, call, params["quality"])
            result["judge_failures"] = int(result["failed"])
        else:
            raise ValueError(f"Unknown DRBench metric {metric!r}")
        scores[metric] = result["score"]
        failures += result.get("judge_failures", 0)
        details[metric] = result
    return {
        "status": "ok",
        "error": None,
        "scores": scores,
        # The dict official score_report returns: {metric: score}.
        "official_result": {metric: scores[metric] for metric in metrics},
        "report_truncated": truncated,
        "report_insights": insights,
        "insight_extraction": extraction,
        "details": details,
        "judge_failures": failures,
        "usage": usage,
        "deviations": [
            d
            for d in DEVIATIONS
            if "report_quality" not in d or "report_quality" in metrics
        ],
    }


METRIC_KEYS = (
    "insights_recall",
    "insights_recall_enterprise",
    "distractor_recall",
    "report_quality",
)

__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_MODEL",
    "DEVIATIONS",
    "METRICS",
    "METRIC_KEYS",
    "NAME",
    "OFFICIAL_PARAMS",
    "extract_insights",
    "extract_json_from_response",
    "format_claims",
    "judge_report",
    "qa_recall",
    "report_quality",
]
