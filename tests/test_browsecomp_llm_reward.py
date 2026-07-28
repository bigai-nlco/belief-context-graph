from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "rllm.rewards.reward_fn",
    reason="BeliefTracer reward integration requires the external rllm runtime",
)

from bcg.agent.benchmark_loader import AgenticTask
from bcg.agent.reward import (
    BrowseCompJudgeConfig,
    BrowseCompLLMJudge,
    build_dispatching_reward_fn,
    build_reward_fn,
)
from bcg.agent.runner import _build_sample_record, _extract_final_answer


def _config(**overrides) -> BrowseCompJudgeConfig:
    values = {
        "model": "grader-model",
        "base_url": "https://grader.test/v1",
        "api_key": "secret",
        "max_retries": 0,
    }
    values.update(overrides)
    return BrowseCompJudgeConfig(**values)


def test_browsecomp_llm_judge_uses_official_prompt_and_yes_verdict() -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return """extracted_final_answer: Paris
reasoning: The response and reference identify the same city.
correct: yes
confidence: 87"""

    judge = BrowseCompLLMJudge(_config(), completion_fn=complete)
    result = judge(
        {
            "question": "Which city is the answer?",
            "ground_truth": "Paris",
            "data_source": "browsecomp",
        },
        "Evidence-based conclusion: \\boxed{Paris}",
    )

    assert result.is_correct is True
    assert result.reward == 1.0
    assert result.metadata["evaluation_method"] == "browsecomp_llm_judge"
    assert result.metadata["judge_verdict"] == "yes"
    assert result.metadata["extracted_answer"] == "Paris"
    assert result.metadata["judge_reasoning"].startswith("The response")
    assert result.metadata["judge_confidence"] == "87"
    assert result.metadata["exact_match"] is None
    assert result.metadata["f1_score"] is None
    assert "[question]: Which city is the answer?" in prompts[0]
    assert "[correct_answer]: Paris" in prompts[0]
    assert "[response]: Evidence-based conclusion" in prompts[0]


def test_browsecomp_grader_no_overrides_lenient_f1() -> None:
    judge_response = """extracted_final_answer: New York
reasoning: The response omits City, which the reference explicitly includes.
correct: no
confidence: 100"""
    config = _config()
    fn = build_reward_fn("browsecomp", browsecomp_judge_config=config)
    fn._completion_fn = lambda _prompt: judge_response  # type: ignore[attr-defined]

    result = fn(
        {
            "question": "Name the place.",
            "ground_truth": "New York City",
            "data_source": "browsecomp",
        },
        "\\boxed{New York}",
    )

    # The previous token-F1 grader treated this as correct with F1=0.8.
    assert result.is_correct is False
    assert result.reward == 0.0
    assert result.metadata["judge_verdict"] == "no"
    assert result.metadata["f1_score"] is None


def test_browsecomp_grader_retries_unparseable_output_then_fails_closed() -> None:
    calls = 0

    def complete(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "I think the answer probably matches."

    judge = BrowseCompLLMJudge(
        _config(max_retries=2),
        completion_fn=complete,
    )
    result = judge(
        {"question": "Question", "ground_truth": "Answer"},
        "\\boxed{Answer}",
    )

    assert calls == 3
    assert result.is_correct is False
    assert result.reward == 0.0
    assert result.metadata["judge_verdict"] == "error"
    assert result.metadata["judge_attempts"] == 3
    assert "correct: yes|no" in result.metadata["judge_error"]


def test_browsecomp_dispatcher_uses_llm_judge_config() -> None:
    config = _config()
    dispatcher = build_dispatching_reward_fn(browsecomp_judge_config=config)
    # Force a deterministic completion on the cached judge created by dispatch.
    original = BrowseCompLLMJudge._call_completion
    BrowseCompLLMJudge._call_completion = lambda self, prompt: (
        "extracted_final_answer: A\nreasoning: mismatch\ncorrect: no\nconfidence: 100"
    )
    try:
        result = dispatcher(
            {
                "question": "Question",
                "ground_truth": "B",
                "data_source": "browsecomp",
            },
            "\\boxed{A}",
        )
    finally:
        BrowseCompLLMJudge._call_completion = original

    assert result.is_correct is False
    assert result.metadata["judge_model"] == "grader-model"


def test_saved_browsecomp_result_preserves_rollout_judge_verdict() -> None:
    metadata = {
        "evaluation_method": "browsecomp_llm_judge",
        "extracted_answer": "semantic equivalent",
        "judge_verdict": "yes",
        "judge_response": "correct: yes",
        "f1_score": None,
        "exact_match": None,
    }
    last_step = SimpleNamespace(
        info={"is_correct": True, "metadata": metadata},
        model_response="Final answer: \\boxed{different wording}",
        action="",
        observation="",
        model_output=None,
        thought="",
        chat_completions=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "finish",
                            "arguments": {
                                "answer": "Final answer: \\boxed{different wording}"
                            },
                        }
                    }
                ],
            }
        ],
    )
    episode = SimpleNamespace(
        trajectories=[SimpleNamespace(steps=[last_step], reward=1.0)],
        termination_reason=SimpleNamespace(value="env_done"),
    )

    extracted, is_correct, saved_metadata = _extract_final_answer(
        episode,
        ground_truth="completely different literal string",
        data_source="browsecomp",
        question="Question",
    )

    assert extracted == "semantic equivalent"
    assert is_correct is True
    assert saved_metadata["evaluation_method"] == "browsecomp_llm_judge"
    assert saved_metadata["judge_verdict"] == "yes"

    record = _build_sample_record(
        problem_id="problem-1",
        task=AgenticTask(
            task_id="browsecomp-1",
            question="Question",
            ground_truth=["completely different literal string"],
            data_source="browsecomp",
        ),
        episode=episode,
        model="remote-model",
        trust_remote_code=False,
    )
    assert record["is_correct"] is True
    assert record["evaluation_metadata"]["judge_verdict"] == "yes"
    assert record["evaluation_metadata"]["judge_response"] == "correct: yes"
