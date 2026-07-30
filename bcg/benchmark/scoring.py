"""Official or official-compatible scorers for supported benchmarks."""

from __future__ import annotations

import json
import math
import re
import string
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from bcg.benchmark.models import BenchmarkTask, ScoreResult, is_api_quota_error

_FINAL_PATTERNS = (
    re.compile(r"(?is)\bFINAL\s+ANSWER\s*:\s*(.+?)\s*$"),
    re.compile(r"(?is)\\boxed\{(.+?)\}"),
    re.compile(r"(?im)^\s*(?:answer|final)\s*:\s*(.+?)\s*$"),
)

BROWSECOMP_JUDGE_PROMPT = """\
Judge whether the following response to the question is correct based only on
the precise reference answer.

[question]
{question}

[response]
{response}

[reference answer]
{answer}

Return exactly these fields:
extracted_final_answer: <the exact answer extracted from the response, or None>
reasoning: <brief comparison against the reference answer>
correct: <yes or no>
confidence: <0-100>
"""


@dataclass(frozen=True)
class JudgeConfig:
    """OpenAI-compatible judge endpoint configuration."""

    model: str
    base_url: str
    api_key: str = field(default="", repr=False)
    timeout: float = 120.0
    max_tokens: int = 1024
    max_retries: int = 2


class LLMJudge:
    """Fail-closed BrowseComp-style binary judge."""

    def __init__(
        self,
        config: JudgeConfig,
        completion_fn: Callable[[str], tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.config = config
        self._completion_fn = completion_fn

    def score(self, task: BenchmarkTask, response: str) -> ScoreResult:
        if not task.answers:
            return ScoreResult(
                correct=False,
                score=0.0,
                error="The task has no reference answer.",
            )
        prompt = BROWSECOMP_JUDGE_PROMPT.format(
            question=task.question,
            response=response,
            answer=json.dumps(task.answers, ensure_ascii=False),
        )
        last_error = ""
        started = time.monotonic()
        for attempt in range(max(0, self.config.max_retries) + 1):
            try:
                judge_text, usage = (
                    self._completion_fn(prompt)
                    if self._completion_fn is not None
                    else self._complete(prompt)
                )
                verdicts = re.findall(
                    r"(?im)^\s*correct\s*:\s*(yes|no)\s*$",
                    judge_text,
                )
                if not verdicts:
                    raise ValueError("judge response has no `correct: yes|no` field")
                extracted = _field(judge_text, "extracted_final_answer")
                correct = verdicts[-1].lower() == "yes"
                return ScoreResult(
                    correct=correct,
                    score=float(correct),
                    extracted_answer=(
                        extracted
                        if extracted and extracted.lower() != "none"
                        else extract_final_answer(response)
                    ),
                    metrics={
                        "evaluation_method": "llm_judge",
                        "judge_model": self.config.model,
                        "judge_attempts": attempt + 1,
                        "judge_reasoning": _field(judge_text, "reasoning"),
                        "judge_confidence": _field(judge_text, "confidence"),
                        "judge_response": judge_text,
                        "judge_usage": usage,
                        "judge_wall_time_seconds": time.monotonic() - started,
                    },
                )
            except Exception as exc:  # Fail closed after bounded retries.
                last_error = str(exc)
                if is_api_quota_error(last_error):
                    break
        return ScoreResult(
            correct=False,
            score=0.0,
            extracted_answer=extract_final_answer(response),
            error=f"Judge failed: {last_error}",
            metrics={
                "evaluation_method": "llm_judge",
                "judge_model": self.config.model,
                "judge_attempts": max(0, self.config.max_retries) + 1,
                "judge_wall_time_seconds": time.monotonic() - started,
            },
        )

    def _complete(self, prompt: str) -> tuple[str, dict[str, Any]]:
        endpoint = self.config.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max(64, self.config.max_tokens),
        }
        response = httpx.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=max(0.1, self.config.timeout),
        )
        if response.status_code == 400:
            legacy_payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max(64, self.config.max_tokens),
            }
            response = httpx.post(
                endpoint,
                headers=headers,
                json=legacy_payload,
                timeout=max(0.1, self.config.timeout),
            )
        if response.is_error:
            detail = response.text.strip()[:1000]
            raise RuntimeError(
                f"judge HTTP {response.status_code}{f': {detail}' if detail else ''}"
            )
        payload = response.json()
        try:
            message = payload["choices"][0]["message"]
            text = str(message.get("content") or message.get("reasoning_content") or "")
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                "judge returned an invalid chat-completions payload"
            ) from exc
        if not text.strip():
            raise ValueError("judge returned an empty response")
        usage = payload.get("usage")
        return text.strip(), usage if isinstance(usage, dict) else {}


def score_task(
    task: BenchmarkTask,
    response: str,
    *,
    judge: LLMJudge | None = None,
) -> ScoreResult:
    """Score one response using the benchmark's evaluation protocol."""

    if task.benchmark == "browsecomp":
        if judge is None:
            return ScoreResult(
                correct=False,
                score=0.0,
                extracted_answer=extract_final_answer(response),
                error="An LLM judge is required for this benchmark.",
            )
        return judge.score(task, response)
    if task.benchmark == "gaia":
        return score_gaia(response, task.answers)
    if task.benchmark == "hotpotqa":
        return score_hotpotqa(response, task.answers)
    if task.benchmark == "mmlu_pro":
        return score_mmlu_pro(response, task.answers)
    raise ValueError(f"No scorer is registered for {task.benchmark}.")


def extract_final_answer(response: str) -> str:
    text = str(response or "").strip()
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip().strip("`")
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    return nonempty[-1] if nonempty else ""


def score_gaia(response: str, answers: tuple[str, ...]) -> ScoreResult:
    candidate = extract_final_answer(response)
    matches = [gaia_match(candidate, answer) for answer in answers]
    correct = any(matches)
    return ScoreResult(
        correct=correct,
        score=float(correct),
        extracted_answer=candidate,
        metrics={"evaluation_method": "gaia_official_normalized_exact"},
        error=None if answers else "GAIA test split has no public reference answer.",
    )


def gaia_match(candidate: str, reference: str) -> bool:
    prediction = str(candidate).strip()
    target = str(reference).strip()
    if _looks_numeric(target):
        try:
            return math.isclose(
                _to_number(prediction),
                _to_number(target),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        except ValueError:
            return False

    separator = "," if "," in target else ";" if ";" in target else None
    if separator:
        predicted_items = [item.strip() for item in prediction.split(separator)]
        target_items = [item.strip() for item in target.split(separator)]
        return len(predicted_items) == len(target_items) and all(
            _normalize_text(predicted) == _normalize_text(expected)
            for predicted, expected in zip(predicted_items, target_items, strict=True)
        )
    return _normalize_text(prediction) == _normalize_text(target)


def score_hotpotqa(response: str, answers: tuple[str, ...]) -> ScoreResult:
    candidate = extract_final_answer(response)
    scores = [_hotpot_metrics(candidate, answer) for answer in answers]
    exact = max((item["exact_match"] for item in scores), default=0.0)
    f1 = max((item["f1"] for item in scores), default=0.0)
    precision = max((item["precision"] for item in scores), default=0.0)
    recall = max((item["recall"] for item in scores), default=0.0)
    return ScoreResult(
        correct=bool(exact),
        score=f1,
        extracted_answer=candidate,
        metrics={
            "evaluation_method": "hotpotqa_answer_em_f1",
            "answer_exact_match": exact,
            "answer_f1": f1,
            "answer_precision": precision,
            "answer_recall": recall,
            "supporting_fact_metrics": None,
        },
    )


def score_mmlu_pro(response: str, answers: tuple[str, ...]) -> ScoreResult:
    candidate = extract_multiple_choice(response)
    normalized_targets = {answer.strip().upper() for answer in answers}
    correct = candidate in normalized_targets
    return ScoreResult(
        correct=correct,
        score=float(correct),
        extracted_answer=candidate,
        metrics={"evaluation_method": "mmlu_pro_exact_choice"},
    )


def extract_multiple_choice(response: str) -> str:
    answer = extract_final_answer(response)
    patterns = (
        r"(?i)\b(?:FINAL\s+ANSWER|ANSWER)\s*:\s*\(?([A-J])\)?\b",
        r"(?i)\bTHE\s+ANSWER\s+IS\s*\(?([A-J])\)?\b",
        r"(?i)\\boxed\{\s*([A-J])\s*\}",
        r"^\s*\(?([A-J])\)?[\.\s]*$",
    )
    for text in (response, answer):
        for pattern in patterns:
            matches = re.findall(pattern, text.strip())
            if matches:
                return matches[-1].upper()
    return ""


def _hotpot_metrics(prediction: str, reference: str) -> dict[str, float]:
    predicted_tokens = _normalize_hotpot(prediction).split()
    reference_tokens = _normalize_hotpot(reference).split()
    exact = float(predicted_tokens == reference_tokens)
    if not predicted_tokens or not reference_tokens:
        f1 = float(predicted_tokens == reference_tokens)
        return {
            "exact_match": exact,
            "f1": f1,
            "precision": f1,
            "recall": f1,
        }
    overlap = Counter(predicted_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return {
            "exact_match": exact,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return {
        "exact_match": exact,
        "f1": 2 * precision * recall / (precision + recall),
        "precision": precision,
        "recall": recall,
    }


def _normalize_hotpot(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(normalized.split())


def _looks_numeric(text: str) -> bool:
    try:
        _to_number(text)
    except ValueError:
        return False
    return True


def _to_number(text: str) -> float:
    cleaned = str(text).strip().replace("$", "").replace(",", "")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    value = float(cleaned)
    return value / 100 if percent else value


def _field(text: str, name: str) -> str:
    match = re.search(
        rf"(?ims)^\s*{re.escape(name)}\s*:\s*(.*?)(?=^\s*[a-z_ ]+\s*:|\Z)",
        text,
    )
    return match.group(1).strip() if match else ""
