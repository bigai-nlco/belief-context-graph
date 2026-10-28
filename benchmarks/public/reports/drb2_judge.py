"""DeepResearch Bench II rubric judge, ported from the official evaluator.

Source: https://github.com/imlrz/DeepResearch-Bench-II
commit 7c7a9a8cda9dad4e6728f4d6a89bb04541feb222 (``run_evaluation.py``,
``aggregate_scores.py``, ``gpt_client.py``).
Copyright 2025-2026 Ruizhe Li and DeepResearch Bench II Contributors.
Licensed under the Apache License, Version 2.0 (the same license as this
repository; see ``LICENSE`` at the repository root).

Kept verbatim from upstream (so scores follow the official protocol):
``PROMPT_TEMPLATE``, ``FENCED_JSON_PATTERN``, ``_try_clean_and_load``,
``parse_model_text``, ``validate_batch_result``, the rubric batching /
truncation / per-dimension organisation in :func:`judge_report`, and
``compute_dimension_averages``. Defaults equal upstream ``.env_example``:
50 items per batch, 10 attempts per batch, 150,000 report characters,
``reasoning_effort=medium``, ``max_completion_tokens=32768``, 600 s timeout.

Changed by this repository (Apache-2.0 §4(b) notice):

* ``requests`` -> ``openai.AsyncOpenAI`` with SDK retries disabled, so each
  upstream attempt is exactly one request; threads -> asyncio.
* A transport error sleeps briefly before the next attempt (upstream retries
  immediately). The attempt budget and every scoring rule are unchanged; the
  reasoning effort is never lowered between attempts.
* Configuration uses this repository's ``JUDGE_*`` names. The default model is
  ``gpt-5.5-luna``; the upstream evaluator pins ``gpt-5.5``. The model actually
  used is recorded with every result.
* Upstream ``print`` logging -> :mod:`logging`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

UPSTREAM_REPOSITORY = "https://github.com/imlrz/DeepResearch-Bench-II"
UPSTREAM_COMMIT = "7c7a9a8cda9dad4e6728f4d6a89bb04541feb222"
UPSTREAM_MODEL = "gpt-5.5"
DEFAULT_MODEL = "gpt-5.5-luna"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_TIMEOUT = 600
DEFAULT_CHUNK_SIZE = 50
DEFAULT_MAX_RETRIES = 10
DEFAULT_MAX_PAPER_CHARS = 150000
# Upstream README (Oct 2026): temporarily excluded from leaderboard scoring.
LEADERBOARD_EXCLUDED_IDX = frozenset({112, 114})
DIMENSIONS = ("info_recall", "analysis", "presentation")
# compute_dimension_averages output keys.
SCORE_KEYS = ("inforecall", "analysis", "presentation", "total", "blocked_rate")

REASONING_EFFORT_ENV = "DRB2_JUDGE_REASONING_EFFORT"


# =========================
# Prompt template (three-way classification) -- verbatim from upstream
# =========================
PROMPT_TEMPLATE = """
You will receive an article, a task, and a list of grading rubric items. Your job is to assess whether the article satisfies each rubric item, and provide a THREE-WAY score for EACH rubric item.

Scoring rule per rubric item (strict):
- Score = 1: The article clearly satisfies the rubric item AND the specific supporting sentence(s) do NOT cite any reference listed in "blocked" (match by title/urls). For numerical data, exact values must be explicitly listed and match the rubric.
- Score = 0: The article does NOT mention this rubric item at all.
- Score = -1: The article mentions this rubric item, BUT the supporting sentence(s) cite a blocked reference.

For EACH rubric item, you MUST provide:
1. "score": 1, 0, or -1
2. "reason": A brief explanation
3. "evidence": The specific supporting sentence(s) from the article (empty string if score is 0)

The input format is:
<input_format>
{{
    "task": "...",
    "rubric_items": ["rubric item 1", "rubric item 2", ...],
    "blocked": {{
        "title": "...",
        "authors": ["...", "..."],
        "urls": ["...", "..."]
    }}
}}
</input_format>

Your output MUST strictly follow this JSON format (no extra keys, and the rubric item text MUST match the input EXACTLY):
<output_format>
{{
    "results": [
        {{
            "rubric_item": "rubric item 1",
            "score": 1 or 0 or -1,
            "reason": "brief explanation",
            "evidence": "supporting sentence(s) from the article"
        }},
        {{
            "rubric_item": "rubric item 2",
            "score": 1 or 0 or -1,
            "reason": "brief explanation",
            "evidence": "supporting sentence(s) from the article"
        }},
        ...
    ]
}}
</output_format>

CRITICAL: You MUST return results for ALL rubric items in the input, and the "rubric_item" text MUST match the input text EXACTLY (character-level match).

<passage>
{paper}
</passage>
<task_and_rubric>
{rubric}
</task_and_rubric>
Now, please begin your generation
"""


# =========================
# Configuration and transport (adapted)
# =========================
@dataclass(frozen=True)
class JudgeConfig:
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    api_key: str | None = None
    base_url: str | None = None
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    max_paper_chars: int = DEFAULT_MAX_PAPER_CHARS

    @classmethod
    def from_env(cls) -> JudgeConfig:
        base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            base_url = base_url.rstrip("/").removesuffix("/chat/completions")
        return cls(
            model=(os.environ.get("JUDGE_MODEL") or "").strip() or DEFAULT_MODEL,
            reasoning_effort=(os.environ.get(REASONING_EFFORT_ENV) or "").strip()
            or DEFAULT_REASONING_EFFORT,
            api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or None,
        )

    def public(self) -> dict[str, Any]:
        """Everything that determines a score, without credentials."""
        return {
            "model": self.model,
            "default_model": DEFAULT_MODEL,
            "model_overridden": self.model != DEFAULT_MODEL,
            "upstream_model": UPSTREAM_MODEL,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "timeout": self.timeout,
            "chunk_size": self.chunk_size,
            "max_retries": self.max_retries,
            "max_paper_chars": self.max_paper_chars,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
        }


class ChatClient(Protocol):
    async def complete(self, prompt: str) -> tuple[str, dict[str, int]]:
        """Return ``(text, usage_metadata)`` in upstream's usage key names."""
        ...

    async def probe(self) -> None:
        """Raise if the model or its parameters are unusable."""
        ...


class OpenAIChatClient:
    """Upstream ``GPTClient.query`` on the OpenAI SDK, one request per call."""

    def __init__(self, config: JudgeConfig) -> None:
        from openai import AsyncOpenAI

        if not config.api_key:
            raise ValueError("Missing configuration: JUDGE_API_KEY (or OPENAI_API_KEY)")
        headers = {}
        if session := os.environ.get("JUDGE_SESSION"):
            headers["X-Llmhub-Session"] = session
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
            default_headers=headers or None,
        )

    async def _create(self, prompt: str, max_tokens: int) -> Any:
        return await self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=self.config.reasoning_effort,  # type: ignore[arg-type]
            max_completion_tokens=max_tokens,
            stream=False,
        )

    async def complete(self, prompt: str) -> tuple[str, dict[str, int]]:
        response = await self._create(prompt, self.config.max_output_tokens)
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Invalid Chat Completions response") from exc
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        return content or "", {
            "promptTokenCount": getattr(usage, "prompt_tokens", 0) or 0,
            "candidatesTokenCount": getattr(usage, "completion_tokens", 0) or 0,
            "totalTokenCount": getattr(usage, "total_tokens", 0) or 0,
            "thoughtsTokenCount": getattr(details, "reasoning_tokens", 0) or 0,
        }

    async def probe(self) -> None:
        # Same model and reasoning_effort as grading; tiny output budget.
        await self._create("ok", 16)


# =========================
# JSON fenced block extraction / parsing -- verbatim from upstream
# =========================
FENCED_JSON_PATTERN = r'```json\s*(.*)```'

def _try_clean_and_load(s: str):
    json_clean = re.sub(
        r'"(?P<k>.*?)"(?=\s*:)',
        lambda m: '"' + re.sub(r'(?<!\\)"', r'\"', m.group('k')) + '"',
        s
    )
    return json.loads(json_clean.strip())

def parse_model_text(text: str):
    matches = re.findall(FENCED_JSON_PATTERN, text, re.DOTALL)
    if matches:
        try:
            return _try_clean_and_load(matches[0]), True
        except json.JSONDecodeError as e:
            logger.warning(f"[warn] failed to parse fenced JSON: {e}; trying full text...")
    try:
        return _try_clean_and_load(text), True
    except json.JSONDecodeError as e:
        logger.warning(f"[warn] failed to parse JSON from full text: {e}")
        return None, False


# =========================
# Batched evaluation and validation -- verbatim from upstream
# =========================
def validate_batch_result(rubric_items: list[str], parsed_result: dict) -> bool:
    """
    Validate that the model output contains all rubric_items with exact text match.
    """
    if not isinstance(parsed_result, dict):
        return False
    results = parsed_result.get("results", [])
    if not isinstance(results, list):
        return False
    if len(results) != len(rubric_items):
        return False

    # 检查每个 rubric_item 是否严格匹配
    returned_items = [r.get("rubric_item", "") for r in results]
    for expected in rubric_items:  # noqa: SIM110 -- verbatim upstream
        # Ensure every rubric_item from the input is present in the results
        if expected not in returned_items:
            return False

    return True


async def query_rubric_batch(
    client: ChatClient,
    rubric_items: list[Any],
    task: Any,
    blocked: Any,
    paper_content: str,
    max_retries: int = 5,
    *,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> tuple[list[dict] | None, dict]:
    """Upstream ``query_rubric_batch``; returns ``(results_list, usage_metadata)``."""
    rubric_input = {
        "task": task,
        "rubric_items": rubric_items,
        "blocked": blocked
    }
    rubric_json = json.dumps(rubric_input, ensure_ascii=False, indent=2)

    for attempt in range(max_retries):
        try:
            prompt = PROMPT_TEMPLATE.format(paper=paper_content, rubric=rubric_json)
            text, usage_metadata = await client.complete(prompt)

            if not text:
                logger.warning(f"[warn] batch attempt {attempt+1}/{max_retries} returned empty text")
                continue

            # Parse JSON
            parsed, ok = parse_model_text(text)
            if not ok:
                logger.warning(f"[warn] batch attempt {attempt+1}/{max_retries} JSON parse failed")
                continue

            # Validate result
            if not isinstance(parsed, dict) or not validate_batch_result(rubric_items, parsed):
                logger.warning(f"[warn] batch attempt {attempt+1}/{max_retries} validation failed: rubric_item mismatch or wrong count")
                continue

            # Success
            return parsed["results"], usage_metadata

        except Exception as e:
            logger.warning(f"[warn] batch attempt {attempt+1}/{max_retries} request error: {e}")
            # Local change: back off on transport errors instead of retrying at once.
            if attempt + 1 < max_retries:
                await sleep(min(2.0**attempt, 30.0))
            continue

    # All attempts failed
    return None, {}


async def judge_report(
    content: Any,
    report: str,
    client: ChatClient,
    config: JudgeConfig,
    *,
    label: str = "",
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> dict[str, Any]:
    """Upstream ``process_one_with_chunking`` for one Markdown report.

    ``content`` is the task's raw ``content`` object from
    ``tasks_and_rubrics.jsonl``. Returns upstream's ``result_dict``, or
    ``{"error": ...}`` exactly where upstream returns one.
    """
    rubric_content = content
    chunk_size = config.chunk_size
    max_paper_chars = config.max_paper_chars
    max_retries = config.max_retries

    # Parse rubric_content
    if not isinstance(rubric_content, dict):
        logger.error(f"[err] {label} rubric_content has invalid format")
        return {"error": "invalid rubric_content"}

    task = rubric_content.get("task", "")
    rubric = rubric_content.get("rubric", {})
    blocked = rubric_content.get("blocked", {})

    # Collect all rubric items (across all dimensions)
    all_items = []
    dimension_map = {}  # rubric_item -> dimension
    for dim in ["info_recall", "analysis", "presentation"]:
        items = rubric.get(dim, [])
        if isinstance(items, list):
            for item in items:
                all_items.append(item)
                dimension_map[item] = dim

    if not all_items:
        logger.warning(f"[warn] {label} has no rubric items")
        return {"error": "no rubric items"}

    text_content = report or ""

    # Truncate overly long text
    if text_content and len(text_content) > max_paper_chars:
        logger.info(f"[info] {label} text too long ({len(text_content)}), truncating to {max_paper_chars}")
        text_content = text_content[:max_paper_chars]

    # Batched processing
    if chunk_size <= 0 or chunk_size >= len(all_items):
        # No batching, process all at once
        batches = [all_items]
    else:
        # Split into batches
        batches = [all_items[i:i+chunk_size] for i in range(0, len(all_items), chunk_size)]

    # 累积结果
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_thoughts_tokens = 0
    total_tokens_sum = 0
    all_usage_metadata = []  # save full usageMetadata for each batch

    for batch_idx, batch_items in enumerate(batches):
        logger.info(f"[info] {label} processing batch {batch_idx+1}/{len(batches)} ({len(batch_items)} items)")

        results, usage_metadata = await query_rubric_batch(
            client, batch_items, task, blocked, text_content,
            max_retries, sleep=sleep,
        )

        if results is None:
            logger.error(f"[err] {label} batch {batch_idx+1} failed")
            return {"error": f"batch {batch_idx+1} failed after {max_retries} retries"}

        all_results.extend(results)
        all_usage_metadata.append(usage_metadata)

        # Accumulate token counts
        total_input_tokens += usage_metadata.get("promptTokenCount", 0)
        total_output_tokens += usage_metadata.get("candidatesTokenCount", 0)
        total_thoughts_tokens += usage_metadata.get("thoughtsTokenCount", 0)
        total_tokens_sum += usage_metadata.get("totalTokenCount", 0)

    # Re-organize results by dimension
    scores_by_dimension = {
        "info_recall": {},
        "analysis": {},
        "presentation": {}
    }

    for result in all_results:
        rubric_item = result.get("rubric_item", "")
        score = result.get("score", 0)
        reason = result.get("reason", "")
        evidence = result.get("evidence", "")

        dim = dimension_map.get(rubric_item)
        if dim:
            scores_by_dimension[dim][rubric_item] = {
                "score": score,
                "reason": reason,
                "evidence": evidence
            }

    result_dict = {
        "task": task,
        "scores": scores_by_dimension,
        "usage_summary": {
            "total_tokens": total_tokens_sum,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "thoughts_tokens": total_thoughts_tokens
        },
        "usage_metadata_per_batch": all_usage_metadata  # save full metadata for each batch
    }
    return result_dict


# =========================
# Aggregation -- verbatim from upstream aggregate_scores.py
# =========================
def mean_or_none(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_dimension_averages(payload):
    """
    Compute per-dimension averages for three-way scores and the blocked_rate.

    We treat the score as:
        - 1  -> counted as "correct"
        - 0  -> ignored
        - -1 -> counted as "blocked"
    """
    if not isinstance(payload, dict):
        return {}

    out = {}
    scores = payload.get("scores", {}) or {}

    # Mapping: JSON key -> output dimension name
    dim_map = [("info_recall", "inforecall"), ("analysis", "analysis"), ("presentation", "presentation")]

    total_ones = 0
    total_minus_ones = 0
    total_items = 0

    for json_key, out_key in dim_map:
        dim_obj = scores.get(json_key)
        if isinstance(dim_obj, dict) and dim_obj:
            # Collect all score values for this dimension
            score_values = []
            for item_key, item_val in dim_obj.items():  # noqa: B007 -- verbatim upstream
                if isinstance(item_val, dict):
                    score = item_val.get("score")
                    if isinstance(score, (int, float)):
                        score_values.append(score)

            # Compute the fraction of items with score == 1
            ones_count = sum(1 for s in score_values if s == 1)
            minus_ones_count = sum(1 for s in score_values if s == -1)
            items_count = len(score_values)

            if items_count > 0:
                out[out_key] = ones_count / items_count
            else:
                out[out_key] = None

            total_ones += ones_count
            total_minus_ones += minus_ones_count
            total_items += items_count
        else:
            out[out_key] = None

    # Compute overall score (fraction of items with score == 1 across all dimensions)
    if total_items > 0:
        out["total"] = total_ones / total_items
        out["blocked_rate"] = total_minus_ones / total_items
    else:
        out["total"] = None
        out["blocked_rate"] = None

    return out


# =========================
# Data access
# =========================
def load_official_contents(path: Path) -> dict[int, Any]:
    """Upstream ``load_tasks_data``: ``idx -> content`` exactly as stored."""
    contents: dict[int, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            idx_val = obj.get("idx", None)
            if idx_val is not None:
                contents[int(idx_val)] = obj.get("content", "")
        except Exception:
            continue
    return contents


def content_from_task(task: Any) -> dict[str, Any]:
    """Best-effort ``content`` from a loaded task (normalised by ``drb2.py``).

    The offline judge command reads the raw dataset instead; this is only for
    callers that hold a :class:`BenchmarkTask` and no data file.
    """
    reference = task.reference or {}
    return {
        "task": reference.get("task", ""),
        "rubric": reference.get("rubric", {}),
        "blocked": reference.get("blocked", {}),
    }


__all__ = [
    "DEFAULT_MODEL",
    "LEADERBOARD_EXCLUDED_IDX",
    "PROMPT_TEMPLATE",
    "SCORE_KEYS",
    "UPSTREAM_COMMIT",
    "UPSTREAM_MODEL",
    "ChatClient",
    "JudgeConfig",
    "OpenAIChatClient",
    "compute_dimension_averages",
    "content_from_task",
    "judge_report",
    "load_official_contents",
    "mean_or_none",
    "parse_model_text",
    "query_rubric_batch",
    "validate_batch_result",
]
