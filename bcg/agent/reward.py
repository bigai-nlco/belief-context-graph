"""Reward function selection for agent workflow rollouts.

Most agentic benchmarks covered here are open-ended QA with string answers, so
we default to ``rllm.rewards.reward_fn.search_reward_fn`` (Exact Match + F1
against ground-truth strings). Multiple-choice benchmarks such as GPQA need a
stricter grader: normalize the final response to A/B/C/D and compare only that
choice, with no F1 partial-credit path.

BrowseComp follows the benchmark's official LLM-as-a-judge protocol: a separate
chat-completions call compares the full candidate response with the reference
answer and emits a binary ``correct: yes|no`` verdict. Judge failures and
unparseable responses are fail-closed.

Rollout-time answer extraction
------------------------------
For non-BrowseComp string graders we first trust the model's ``\\boxed{...}``
answer. If no boxed span is present we fall back to the raw
(post-``<think>``-strip) response and let EM/F1 decide. BrowseComp instead sends
the full response to its LLM judge, which performs its own final-answer
extraction.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import string
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from rllm.rewards.reward_fn import search_reward_fn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


RewardFn = Callable[[dict, str], RewardOutput]
logger = logging.getLogger(__name__)
MCQ_DATA_SOURCES = {"gpqa", "gpqa_diamond"}
MCQ_LETTERS = tuple("ABCD")
AVERITEC_LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)
AVERITEC_LABEL_ALIASES = {
    "supported": "Supported",
    "support": "Supported",
    "supports": "Supported",
    "true": "Supported",
    "refuted": "Refuted",
    "refute": "Refuted",
    "refutes": "Refuted",
    "false": "Refuted",
    "not enough evidence": "Not Enough Evidence",
    "not enough info": "Not Enough Evidence",
    "notenoughinfo": "Not Enough Evidence",
    "nei": "Not Enough Evidence",
    "insufficient evidence": "Not Enough Evidence",
    "conflicting evidence/cherrypicking": "Conflicting Evidence/Cherrypicking",
    "conflicting evidence": "Conflicting Evidence/Cherrypicking",
    "cherrypicking": "Conflicting Evidence/Cherrypicking",
    "cherry picking": "Conflicting Evidence/Cherrypicking",
    "conflicting": "Conflicting Evidence/Cherrypicking",
}


BROWSECOMP_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


@dataclass(frozen=True)
class BrowseCompJudgeConfig:
    """OpenAI-compatible endpoint configuration for the BrowseComp grader."""

    model: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    timeout: float = 120.0
    max_tokens: int = 2048
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "BrowseCompJudgeConfig":
        return cls(
            model=os.environ.get("BROWSECOMP_GRADER_MODEL")
            or os.environ.get("MODEL", ""),
            base_url=os.environ.get("BROWSECOMP_GRADER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL", ""),
            api_key=os.environ.get("BROWSECOMP_GRADER_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("BROWSECOMP_GRADER_TIMEOUT", "120")),
            max_tokens=int(os.environ.get("BROWSECOMP_GRADER_MAX_TOKENS", "2048")),
            max_retries=int(os.environ.get("BROWSECOMP_GRADER_MAX_RETRIES", "2")),
        )


class BrowseCompLLMJudge:
    """Official-style binary BrowseComp judge using a chat-completions model."""

    def __init__(
        self,
        config: BrowseCompJudgeConfig,
        *,
        completion_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config
        self._completion_fn = completion_fn

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        question = str((task_info or {}).get("question") or "").strip()
        raw_targets = (task_info or {}).get("ground_truth")
        targets = (
            list(raw_targets)
            if isinstance(raw_targets, (list, tuple, set))
            else [raw_targets]
        )
        targets = [str(target).strip() for target in targets if target not in (None, "")]
        response = str(action or "").strip()
        if not question or not targets:
            missing = "question" if not question else "ground_truth"
            return self._failed_result(
                response=response,
                targets=targets,
                error=f"BrowseComp judge missing {missing}",
                attempts=0,
            )
        if not self.config.model or not self.config.base_url:
            return self._failed_result(
                response=response,
                targets=targets,
                error="BrowseComp grader model/base_url is not configured",
                attempts=0,
            )

        correct_answer = targets[0] if len(targets) == 1 else json.dumps(targets, ensure_ascii=False)
        prompt = BROWSECOMP_GRADER_TEMPLATE.format(
            question=question,
            response=response,
            correct_answer=correct_answer,
        )
        judge_response = ""
        last_error = ""
        attempts = 0
        for attempt in range(max(0, int(self.config.max_retries)) + 1):
            attempts = attempt + 1
            try:
                judge_response = (
                    self._completion_fn(prompt)
                    if self._completion_fn is not None
                    else self._call_completion(prompt)
                )
                verdict = self._parse_verdict(judge_response)
                if verdict is None:
                    raise ValueError("grader response did not contain 'correct: yes|no'")
                extracted = self._parse_field(judge_response, "extracted_final_answer")
                if not extracted or extracted.lower() == "none":
                    extracted = _extract_final_answer_span(response)
                reasoning = self._parse_reasoning(judge_response)
                confidence = self._parse_field(judge_response, "confidence")
                is_correct = verdict == "yes"
                logger.info(
                    "[BrowseCompJudge] verdict=%s model=%s attempts=%d",
                    verdict,
                    self.config.model,
                    attempts,
                )
                return RewardOutput(
                    reward=1.0 if is_correct else 0.0,
                    is_correct=is_correct,
                    metadata={
                        "evaluation_method": "browsecomp_llm_judge",
                        "extracted_answer": extracted,
                        "target_answer": correct_answer,
                        "ground_truths": targets,
                        "judge_verdict": verdict,
                        "judge_reasoning": reasoning,
                        "judge_confidence": confidence,
                        "judge_response": judge_response,
                        "judge_model": self.config.model,
                        "judge_attempts": attempts,
                        "exact_match": None,
                        "f1_score": None,
                    },
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "[BrowseCompJudge] attempt %d/%d failed: %s",
                    attempts,
                    max(0, int(self.config.max_retries)) + 1,
                    exc,
                )

        return self._failed_result(
            response=response,
            targets=targets,
            error=last_error or "BrowseComp grader failed",
            attempts=attempts,
            judge_response=judge_response,
        )

    def _call_completion(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max(64, int(self.config.max_tokens)),
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key or 'EMPTY'}",
                "User-Agent": "BeliefTracer BrowseComp grader/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(0.1, float(self.config.timeout))
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise RuntimeError(f"grader HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"grader request failed: {exc}") from exc

        try:
            data = json.loads(raw.decode("utf-8"))
            message = (((data.get("choices") or [{}])[0]).get("message") or {})
            content = str(message.get("content") or "").strip()
            reasoning = str(message.get("reasoning_content") or "").strip()
        except (AttributeError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"grader returned invalid response JSON: {exc}") from exc
        text = content or reasoning
        if not text:
            raise RuntimeError("grader returned an empty response")
        return text

    @staticmethod
    def _parse_verdict(text: str) -> str | None:
        matches = re.findall(r"(?i)\bcorrect\s*:\s*(yes|no)\b", str(text or ""))
        return matches[-1].lower() if matches else None

    @staticmethod
    def _parse_field(text: str, name: str) -> str:
        match = re.search(
            rf"(?im)^\s*{re.escape(name)}\s*:\s*(.*?)\s*$",
            str(text or ""),
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_reasoning(text: str) -> str:
        match = re.search(
            r"(?is)(?:^|\n)\s*reasoning\s*:\s*(.*?)(?=\n\s*correct\s*:)",
            str(text or ""),
        )
        return match.group(1).strip() if match else ""

    def _failed_result(
        self,
        *,
        response: str,
        targets: list[str],
        error: str,
        attempts: int,
        judge_response: str = "",
    ) -> RewardOutput:
        return RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={
                "evaluation_method": "browsecomp_llm_judge",
                "extracted_answer": _extract_final_answer_span(response),
                "target_answer": targets[0] if len(targets) == 1 else targets,
                "ground_truths": targets,
                "judge_verdict": "error",
                "judge_response": judge_response,
                "judge_model": self.config.model,
                "judge_attempts": attempts,
                "judge_error": error,
                "exact_match": None,
                "f1_score": None,
            },
        )


class _BoxedFirstSearchFn(RewardSearchFn):
    """Boxed-first extractor with raw-response fallback.

    Reuses ``RewardSearchFn``'s boxed-unwrapping logic via ``super()`` — it
    already handles ``\\boxed{...}``, ``boxed{...}``, and ``oxed{...}`` with
    nested-brace counting and strips LaTeX wrappers. If super() returns a
    value that looks like it came from a non-boxed fallback (bold, date,
    name, number, "the answer is X", sentence score, etc.), we instead
    return the raw response so the grader scores the agent's final text
    verbatim.
    """

    def extract_answer_from_response(self, response: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""

        if re.search(r"(?:\\boxed|boxed|oxed)\s*\{", cleaned):
            extracted = super().extract_answer_from_response(response)
            if extracted:
                return extracted

        return cleaned


def _boxed_first_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """search_reward_fn variant that uses the boxed-first extractor."""
    reward_config = RewardConfig()
    fn = _BoxedFirstSearchFn(reward_config)
    return fn(RewardInput(task_info=task_info, action=action))


def _strip_thinking(text: Any) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
    return cleaned.strip()


def _extract_braced_span(text: str, marker_start: int) -> str | None:
    brace = text.find("{", marker_start)
    if brace < 0:
        return None
    i = brace + 1
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


def _last_boxed_content(text: str) -> str | None:
    matches = list(re.finditer(r"(?:\\boxed|boxed|oxed)\s*\{", text))
    if not matches:
        return None
    for match in reversed(matches):
        content = _extract_braced_span(text, match.start())
        if content:
            return content
    return None


def _extract_final_answer_span(response: Any) -> str:
    """Extract the model's final-answer span before MCQ normalization."""
    cleaned = _strip_thinking(response)
    if not cleaned:
        return ""

    boxed = _last_boxed_content(cleaned)
    if boxed:
        return boxed

    tail = cleaned[-4000:]
    candidates: list[tuple[int, str]] = []
    answer_marker = (
        r"(?:Final|Correct|Selected|Chosen)\s+"
        r"(?:Answer|Option|Choice)"
    )
    patterns = (
        rf"(?:✅\s*)?(?:{answer_marker})"
        r"\s*(?:\*+)?\s*(?:is\b|[:\-])?\s*(?:\*+\s*)?(?:\$\$)?\s*(?:\\boxed\s*\{)?"
        r"\s*(?=[^\s\.,;:!?\n$])([^\n$]{1,300})",
        r"(?:therefore|thus|so)?\s*,?\s*(?:the\s+)?answer\s+is"
        r"\s*(?:\*+)?\s*[:\-]?\s*(?:\*+\s*)?(?:\\boxed\s*\{)?"
        r"\s*(?=[^\s\.,;:!?\n$])([^\n$]{1,220})",
        r"(?:therefore|thus|so)?\s*,?\s*(?:the\s+)?(?:correct|selected|chosen)\s+"
        r"(?:answer|option|choice)\s+is"
        r"\s*(?:\*+)?\s*[:\-]?\s*(?:\*+\s*)?(?:\\boxed\s*\{)?"
        r"\s*(?=[^\s\.,;:!?\n$])([^\n$]{1,220})",
        r"(?:therefore|thus|so)?\s*,?\s*(?:the\s+)?"
        r"(?:answer|option|choice)\s*(?:\*+)?\s*[:\-]?\s*(?:\*+\s*)?"
        r"(?:option\s*)?([ABCD])\b\s*(?:\*+)?\s+is\s+(?:the\s+)?"
        r"(?:correct|selected|chosen)(?:\s+(?:answer|option|choice))?",
        r"\bAnswer\s*(?:\*+)?\s*[:\-]\s*(?:\*+\s*)?(?:\$\$)?"
        r"\s*(?:\\boxed\s*\{)?\s*(?=[^\s\.,;:!?\n$])([^\n$]{1,220})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, tail, flags=re.IGNORECASE):
            candidates.append((match.start(), match.group(1).strip()))
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return lines[-1] if lines else tail.strip()


def _normalize_text(text: Any) -> str:
    s = str(text or "")
    # Peel simple LaTeX text wrappers before removing braces.
    previous = None
    while previous != s:
        previous = s
        s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("$", "").replace("*", "")
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("π", "pi")
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def _compact_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9.+/%=\- ]+", "", _normalize_text(text)).strip()


def _parse_mcq_choices(question: Any) -> dict[str, str]:
    text = str(question or "")
    markers = list(re.finditer(r"(?<![A-Za-z0-9])([ABCD])\.\s+", text))
    best: list[re.Match[str]] | None = None
    for idx, marker in enumerate(markers):
        if marker.group(1) != "A":
            continue
        seq: list[re.Match[str]] = []
        pos = marker.start()
        for wanted in MCQ_LETTERS:
            found = next(
                (
                    candidate
                    for candidate in markers[idx:]
                    if candidate.start() >= pos and candidate.group(1) == wanted
                ),
                None,
            )
            if found is None:
                break
            seq.append(found)
            pos = found.end()
        if len(seq) == len(MCQ_LETTERS):
            best = seq

    if not best:
        return {}

    choices: dict[str, str] = {}
    for idx, marker in enumerate(best):
        end = best[idx + 1].start() if idx + 1 < len(best) else len(text)
        choices[marker.group(1)] = text[marker.end() : end].strip()
    return choices


def _parse_numeric_value(text: Any) -> tuple[str, float] | float | None:
    raw = str(text or "")
    frac = re.search(r"\\frac\s*\{?\s*(-?\d+(?:\.\d+)?)\s*\}?\s*\{?\s*(-?\d+(?:\.\d+)?)\s*\}?", raw)
    if frac:
        denominator = float(frac.group(2))
        if denominator != 0:
            return float(frac.group(1)) / denominator

    compact = _normalize_text(raw).replace(" ", "")
    slash_frac = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)%?", compact)
    if slash_frac:
        denominator = float(slash_frac.group(2))
        if denominator != 0:
            return float(slash_frac.group(1)) / denominator

    number = re.search(r"-?\d+(?:\.\d+)?", compact)
    if not number:
        return None
    value = float(number.group(0))
    return ("percent", value) if "%" in compact else value


def _numeric_values_match(
    pred: tuple[str, float] | float,
    option: tuple[str, float] | float,
) -> bool:
    if isinstance(pred, tuple) or isinstance(option, tuple):
        if not (isinstance(pred, tuple) and isinstance(option, tuple)):
            return False
        return pred[0] == option[0] and abs(pred[1] - option[1]) <= 0.25
    tolerance = max(1e-6, abs(float(option)) * 0.005)
    return abs(float(pred) - float(option)) <= tolerance


def _map_text_to_choice(text: Any, choices: dict[str, str]) -> str | None:
    compact = _compact_text(text)
    if len(compact) < 4:
        return None
    for letter, option in choices.items():
        option_compact = _compact_text(option)
        if compact == option_compact:
            return letter
        if len(compact) > 10 and (compact in option_compact or option_compact in compact):
            return letter
    return None


def _map_numeric_to_choice(text: Any, choices: dict[str, str]) -> str | None:
    value = _parse_numeric_value(text)
    if value is None:
        return None
    for letter, option in choices.items():
        option_value = _parse_numeric_value(option)
        if option_value is not None and _numeric_values_match(value, option_value):
            return letter
    return None


def _normalize_mcq_choice(text: Any, choices: dict[str, str] | None = None) -> str | None:
    normalized = _normalize_text(text)
    normalized = re.sub(r"^(?:[-*]\s*)+", "", normalized).strip()
    normalized = re.sub(
        r"^(?:(?:therefore|thus|so)\s*,?\s*)?"
        r"(?:(?:the\s+)?answer\s+is|(?:the\s+)?(?:final|correct|selected|chosen)\s+"
        r"(?:answer|option|choice)(?:\s+is)?|answer)"
        r"\s*[:\-]?\s*",
        "",
        normalized,
    ).strip()
    explicit = re.match(
        r"^(?:option\s*)?([abcd])\b(?:[\s\.\:\)\-]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()

    choices = choices or {}
    return _map_text_to_choice(text, choices) or _map_numeric_to_choice(text, choices)


def _target_mcq_choice(ground_truth: Any, choices: dict[str, str]) -> str | None:
    values = ground_truth if isinstance(ground_truth, (list, tuple, set)) else [ground_truth]
    for value in values:
        choice = _normalize_mcq_choice(value, choices)
        if choice:
            return choice
    for value in values:
        mapped = _map_text_to_choice(value, choices) or _map_numeric_to_choice(value, choices)
        if mapped:
            return mapped
    return None


def extract_mcq_answer(action: Any, question: Any = None) -> tuple[str | None, str]:
    """Return ``(A/B/C/D, raw_span)`` from a model response or answer string."""
    choices = _parse_mcq_choices(question)
    raw_span = _extract_final_answer_span(action)
    prediction = _normalize_mcq_choice(raw_span, choices)
    return prediction, raw_span


def _make_mcq_reward_fn(
    correct_reward: float,
    incorrect_reward: float,
) -> RewardFn:
    """Build an exact-choice reward for A/B/C/D multiple-choice tasks."""

    def _fn(task_info: dict, action: str) -> RewardOutput:
        question = (task_info or {}).get("question", "")
        choices = _parse_mcq_choices(question)
        predicted, raw_span = extract_mcq_answer(action, question)
        target = _target_mcq_choice((task_info or {}).get("ground_truth"), choices)
        is_correct = bool(predicted and target and predicted == target)
        reward = correct_reward if is_correct else incorrect_reward
        metadata = {
            "extracted_answer": predicted or "",
            "raw_extracted_answer": raw_span,
            "target_answer": target or "",
            "exact_match": is_correct,
            "f1_score": 1.0 if is_correct else 0.0,
            "choices": choices,
        }
        return RewardOutput(reward=reward, is_correct=is_correct, metadata=metadata)

    return _fn



def _normalize_averitec_label(text: Any) -> str | None:
    raw = str(text or "")
    if not raw.strip():
        return None

    boxed = _last_boxed_content(_strip_thinking(raw))
    candidates = [boxed, _extract_final_answer_span(raw), raw]
    for candidate in candidates:
        normalized = _normalize_text(candidate)
        normalized = normalized.replace("/", " / ")
        normalized = re.sub(r"\s+", " ", normalized).strip(" .,:;-_'\"")
        compact = normalized.replace(" / ", "/")
        if normalized in AVERITEC_LABEL_ALIASES:
            return AVERITEC_LABEL_ALIASES[normalized]
        if compact in AVERITEC_LABEL_ALIASES:
            return AVERITEC_LABEL_ALIASES[compact]
        for alias, label in AVERITEC_LABEL_ALIASES.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
                return label
    return None


def _make_averitec_reward_fn(
    correct_reward: float,
    incorrect_reward: float,
) -> RewardFn:
    def _fn(task_info: dict, action: str) -> RewardOutput:
        predicted = _normalize_averitec_label(action)
        target = _normalize_averitec_label((task_info or {}).get("ground_truth"))
        is_correct = bool(predicted and target and predicted == target)
        reward = correct_reward if is_correct else incorrect_reward
        metadata = {
            "extracted_answer": predicted or "",
            "raw_extracted_answer": _extract_final_answer_span(action),
            "target_answer": target or "",
            "exact_match": is_correct,
            "f1_score": 1.0 if is_correct else 0.0,
            "labels": list(AVERITEC_LABELS),
        }
        return RewardOutput(reward=reward, is_correct=is_correct, metadata=metadata)

    return _fn


def _gaia_is_float(value: Any) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _normalize_gaia_number(value: Any) -> float:
    text = str(value)
    for char in ("$", "%", ","):
        text = text.replace(char, "")
    try:
        return float(text)
    except ValueError:
        return float("inf")


def _normalize_gaia_text(value: Any, *, remove_punctuation: bool = True) -> str:
    text = re.sub(r"\s", "", str(value or "")).lower()
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def gaia_question_scorer(model_answer: Any, ground_truth: Any) -> bool:
    """Match the official GAIA string/number/list evaluation behavior."""
    prediction = str(model_answer or "")
    target = str(ground_truth or "")
    if not target:
        return False

    if _gaia_is_float(target):
        return _normalize_gaia_number(prediction) == float(target)

    if any(separator in target for separator in (",", ";")):
        target_items = re.split(r"[,;]", target)
        prediction_items = re.split(r"[,;]", prediction)
        if len(target_items) != len(prediction_items):
            return False
        for predicted_item, target_item in zip(prediction_items, target_items):
            if _gaia_is_float(target_item):
                if _normalize_gaia_number(predicted_item) != float(target_item):
                    return False
            elif _normalize_gaia_text(
                predicted_item, remove_punctuation=False
            ) != _normalize_gaia_text(target_item, remove_punctuation=False):
                return False
        return True

    return _normalize_gaia_text(prediction) == _normalize_gaia_text(target)


def _make_gaia_reward_fn(
    correct_reward: float,
    incorrect_reward: float,
) -> RewardFn:
    def _fn(task_info: dict, action: str) -> RewardOutput:
        raw_prediction = _extract_final_answer_span(action)
        raw_targets = (task_info or {}).get("ground_truth")
        targets = (
            list(raw_targets)
            if isinstance(raw_targets, (list, tuple, set))
            else [raw_targets]
        )
        targets = [str(target) for target in targets if target not in (None, "", "?")]
        matched_target = next(
            (
                target
                for target in targets
                if gaia_question_scorer(raw_prediction, target)
            ),
            "",
        )
        is_correct = bool(matched_target)
        return RewardOutput(
            reward=correct_reward if is_correct else incorrect_reward,
            is_correct=is_correct,
            metadata={
                "extracted_answer": raw_prediction,
                "target_answer": matched_target or (targets[0] if targets else ""),
                "ground_truths": targets,
                "exact_match": is_correct,
                "f1_score": 1.0 if is_correct else 0.0,
                "evaluation_method": "gaia_question_scorer",
                **({"error": "No public ground truth for this GAIA task"} if not targets else {}),
            },
        )

    return _fn


def _make_configured_reward_fn(
    toolcall_bonus: float,
    correct_reward: float,
    incorrect_reward: float,
) -> RewardFn:
    """Build a boxed-first search reward with shaping terms disabled."""
    reward_kwargs = {
        "toolcall_bonus": toolcall_bonus,
        "apply_repetition_penalty": False,
        "correct_reward": correct_reward,
        "incorrect_reward": incorrect_reward,
        "enable_step_bonus": False,
    }
    accepted = inspect.signature(RewardConfig).parameters
    cfg = RewardConfig(
        **{k: v for k, v in reward_kwargs.items() if k in accepted}
    )

    def _fn(task_info: dict, action: str) -> RewardOutput:
        reward_fn = _BoxedFirstSearchFn(cfg)
        return reward_fn(RewardInput(task_info=task_info, action=action))

    return _fn


REWARD_OVERRIDES: dict[str, RewardFn] = {
    "averitec": _make_averitec_reward_fn(correct_reward=1.0, incorrect_reward=0.0),
    "gaia": _make_gaia_reward_fn(correct_reward=1.0, incorrect_reward=0.0),
}


def build_reward_fn(
    data_source: str,
    *,
    toolcall_bonus: float = 0.0,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    browsecomp_judge_config: BrowseCompJudgeConfig | None = None,
) -> RewardFn:
    """Return the reward function appropriate for ``data_source``.

    BrowseComp is dispatched to its configurable LLM judge. Other open-ended
    tasks use a plain correctness signal with no tool-call bonus or train-time
    shaping; their extractor prefers ``\\boxed{...}`` and falls back to raw text.
    """
    if data_source in MCQ_DATA_SOURCES:
        return _make_mcq_reward_fn(
            correct_reward=correct_reward,
            incorrect_reward=incorrect_reward,
        )

    if data_source in {"browsecomp", "browse_comp"}:
        return BrowseCompLLMJudge(
            browsecomp_judge_config or BrowseCompJudgeConfig.from_env()
        )

    if data_source in REWARD_OVERRIDES:
        return REWARD_OVERRIDES[data_source]

    return _make_configured_reward_fn(
        toolcall_bonus=toolcall_bonus,
        correct_reward=correct_reward,
        incorrect_reward=incorrect_reward,
    )


def build_dispatching_reward_fn(
    *,
    toolcall_bonus: float = 0.0,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    browsecomp_judge_config: BrowseCompJudgeConfig | None = None,
) -> RewardFn:
    """Reward fn that picks the right grader from ``task_info['data_source']``.

    Used by the mixed-rollouts path, where a single workflow engine pools
    rollouts across benchmarks. Per-data-source reward fns are memoized so
    we still build each grader exactly once.
    """
    cache: dict[str, RewardFn] = {}

    def _resolve(data_source: str) -> RewardFn:
        fn = cache.get(data_source)
        if fn is None:
            fn = build_reward_fn(
                data_source,
                toolcall_bonus=toolcall_bonus,
                correct_reward=correct_reward,
                incorrect_reward=incorrect_reward,
                browsecomp_judge_config=browsecomp_judge_config,
            )
            cache[data_source] = fn
        return fn

    def _fn(task_info: dict, action: str) -> RewardOutput:
        data_source = str((task_info or {}).get("data_source") or "")
        return _resolve(data_source)(task_info, action)

    return _fn


__all__ = [
    "BROWSECOMP_GRADER_TEMPLATE",
    "BrowseCompJudgeConfig",
    "BrowseCompLLMJudge",
    "RewardFn",
    "MCQ_DATA_SOURCES",
    "REWARD_OVERRIDES",
    "build_reward_fn",
    "build_dispatching_reward_fn",
    "extract_mcq_answer",
    "gaia_question_scorer",
    "search_reward_fn",
]
