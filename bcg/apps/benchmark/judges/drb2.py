"""DeepResearch Bench II rubric judge, aligned with the official evaluator.

Ported from ``run_evaluation.py`` and ``aggregate_scores.py`` of
https://github.com/imlrz/DeepResearch-Bench-II (Apache-2.0), commit recorded
in ``prompts/SOURCES.json``:

* the report (Markdown) is truncated to ``MAX_PAPER_CHARS`` = 150,000 chars;
* rubric items are flattened info_recall -> analysis -> presentation and sent
  in batches of ``CHUNK_SIZE`` = 50, each with the full report, the task, and
  the raw ``blocked`` record, using the verbatim ``PROMPT_TEMPLATE``;
* the request is one Chat Completions user message with
  ``reasoning_effort=medium`` and ``max_completion_tokens=32768``;
* a batch is retried up to ``MAX_RETRIES`` = 10 times on empty text, JSON
  parse failure (fenced block, then full text, with the official key-quote
  cleanup), or a rubric-item mismatch; any failed batch fails the task;
* per-task scores are the share of items scored 1 per dimension, ``total``
  over all items, and ``blocked_rate`` (share scored -1). Tasks that failed
  are excluded from averages, as in the official aggregator.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bcg.apps.benchmark.judges.client import JudgeClient, JudgeRequestError, add_usage
from bcg.apps.benchmark.judges.official_sources import load_prompt
from bcg.apps.benchmark.models import BenchmarkTask

NAME = "drb2_rubric"
DEFAULT_MODEL = "gpt-5.5-luna"  # The official evaluator uses gpt-5.5.
OFFICIAL_PARAMS: dict[str, Any] = {
    "reasoning_effort": "medium",
    "max_completion_tokens": 32768,
}
OFFICIAL_SETTINGS: dict[str, Any] = {
    "chunk_size": 50,
    "max_retries": 10,
    "max_paper_chars": 150_000,
}
DIMENSIONS = ("info_recall", "analysis", "presentation")
FENCED_JSON_PATTERN = r"```json\s*(.*)```"


# --- verbatim ports of the official helpers --------------------------------


def _try_clean_and_load(s: str) -> Any:
    json_clean = re.sub(
        r'"(?P<k>.*?)"(?=\s*:)',
        lambda m: '"' + re.sub(r'(?<!\\)"', r"\"", m.group("k")) + '"',
        s,
    )
    return json.loads(json_clean.strip())


def parse_model_text(text: str) -> tuple[Any, bool]:
    matches = re.findall(FENCED_JSON_PATTERN, text, re.DOTALL)
    if matches:
        try:
            return _try_clean_and_load(matches[0]), True
        except json.JSONDecodeError:
            pass
    try:
        return _try_clean_and_load(text), True
    except json.JSONDecodeError:
        return None, False


def validate_batch_result(rubric_items: list[str], parsed_result: Any) -> bool:
    if not isinstance(parsed_result, dict):
        return False
    results = parsed_result.get("results", [])
    if not isinstance(results, list):
        return False
    if len(results) != len(rubric_items):
        return False
    returned_items = [r.get("rubric_item", "") for r in results]
    return all(expected in returned_items for expected in rubric_items)


def compute_dimension_averages(payload: Any) -> dict[str, float | None]:
    """``aggregate_scores.compute_dimension_averages``."""

    if not isinstance(payload, dict):
        return {}
    out: dict[str, float | None] = {}
    scores = payload.get("scores", {}) or {}
    total_ones = total_minus_ones = total_items = 0
    for json_key, out_key in (
        ("info_recall", "inforecall"),
        ("analysis", "analysis"),
        ("presentation", "presentation"),
    ):
        dim_obj = scores.get(json_key)
        if isinstance(dim_obj, dict) and dim_obj:
            values = [
                item.get("score")
                for item in dim_obj.values()
                if isinstance(item, dict)
                and isinstance(item.get("score"), (int, float))
            ]
            ones = sum(1 for value in values if value == 1)
            minus_ones = sum(1 for value in values if value == -1)
            out[out_key] = ones / len(values) if values else None
            total_ones += ones
            total_minus_ones += minus_ones
            total_items += len(values)
        else:
            out[out_key] = None
    out["total"] = total_ones / total_items if total_items else None
    out["blocked_rate"] = total_minus_ones / total_items if total_items else None
    return out


# --- judge ------------------------------------------------------------------


def judge_report(
    task: BenchmarkTask,
    report: str,
    client: JudgeClient,
    *,
    params: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = dict(OFFICIAL_PARAMS if params is None else params)
    settings = {**OFFICIAL_SETTINGS, **(settings or {})}
    content = task.reference.get("official_content") or {}
    task_text = content.get("task", "")
    rubric = content.get("rubric", {}) or {}
    blocked = content.get("blocked", {})

    all_items: list[str] = []
    dimension_map: dict[str, str] = {}
    for dim in DIMENSIONS:
        items = rubric.get(dim, [])
        if isinstance(items, list):
            for item in items:
                all_items.append(item)
                dimension_map[item] = dim
    usage: dict[str, int] = {}
    if not all_items:
        return _failed("no rubric items", usage, [])

    text = report or ""
    truncated = len(text) > settings["max_paper_chars"]
    text = text[: settings["max_paper_chars"]]
    chunk = int(settings["chunk_size"])
    batches = (
        [all_items]
        if chunk <= 0 or chunk >= len(all_items)
        else [all_items[i : i + chunk] for i in range(0, len(all_items), chunk)]
    )
    template = load_prompt("drb2_rubric")
    batch_records: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    accepted_usage: dict[
        str, int
    ] = {}  # official usage_summary counts accepted replies only
    for number, batch_items in enumerate(batches, start=1):
        rubric_json = json.dumps(
            {"task": task_text, "rubric_items": batch_items, "blocked": blocked},
            ensure_ascii=False,
            indent=2,
        )
        prompt = template.format(paper=text, rubric=rubric_json)
        record: dict[str, Any] = {
            "batch": number,
            "items": len(batch_items),
            "attempts": 0,
            "failures": [],
        }
        accepted = None
        for attempt in range(int(settings["max_retries"])):
            record["attempts"] = attempt + 1
            try:
                # Retries must reach the model again, never a cached bad reply.
                reply = client.complete(prompt, params, use_cache=attempt == 0)
            except JudgeRequestError as exc:
                record["failures"].append(f"request error: {exc}")
                if exc.quota:
                    break
                continue
            add_usage(usage, reply.usage)
            record["params"] = reply.params
            if not reply.text:
                record["failures"].append("empty text")
                continue
            parsed, ok = parse_model_text(reply.text)
            if not ok:
                record["failures"].append("JSON parse failed")
                continue
            try:
                valid = validate_batch_result(batch_items, parsed)
            except AttributeError:  # non-dict entries: the official loop retries too
                valid = False
            if not valid:
                record["failures"].append("rubric_item mismatch or wrong count")
                continue
            accepted = parsed["results"]
            add_usage(accepted_usage, reply.usage)
            record["response"] = reply.text
            record["cached"] = reply.cached
            break
        batch_records.append(record)
        if accepted is None:
            return _failed(
                f"batch {number} failed after {record['attempts']} attempts",
                usage,
                batch_records,
            )
        all_results.extend(accepted)

    scores_by_dimension: dict[str, dict[str, Any]] = {dim: {} for dim in DIMENSIONS}
    for result in all_results:
        item = result.get("rubric_item", "")
        dim = dimension_map.get(item)
        if dim:
            scores_by_dimension[dim][item] = {
                "score": result.get("score", 0),
                "reason": result.get("reason", ""),
                "evidence": result.get("evidence", ""),
            }
    official_result = {
        "task": task_text,
        "scores": scores_by_dimension,
        "usage_summary": {
            "total_tokens": accepted_usage.get("total", 0),
            "input_tokens": accepted_usage.get("input", 0),
            "output_tokens": accepted_usage.get("output", 0),
            "thoughts_tokens": accepted_usage.get("reasoning", 0),
        },
    }
    averages = compute_dimension_averages(official_result)
    return {
        "status": "ok",
        "error": None,
        "scores": averages,
        "official_result": official_result,
        "report_truncated": truncated,
        "rubric_items": len(all_items),
        "batches": batch_records,
        "usage": usage,
        "deviations": [],
    }


def _failed(
    error: str, usage: dict[str, int], batches: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "status": "error",
        "error": error,
        "scores": {},
        "official_result": {"error": error},
        "batches": batches,
        "usage": usage,
        "deviations": [],
    }


METRIC_KEYS = ("total", "inforecall", "analysis", "presentation", "blocked_rate")

__all__ = [
    "DEFAULT_MODEL",
    "METRIC_KEYS",
    "NAME",
    "OFFICIAL_PARAMS",
    "OFFICIAL_SETTINGS",
    "compute_dimension_averages",
    "judge_report",
    "parse_model_text",
    "validate_batch_result",
]
