from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field


@dataclass
class _RewardConfig:
    toolcall_bonus: float = 0.0
    apply_repetition_penalty: bool = False
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    enable_step_bonus: bool = False


@dataclass
class _RewardInput:
    task_info: dict
    action: str


@dataclass
class _RewardOutput:
    reward: float = 0.0
    is_correct: bool = False
    metadata: dict = field(default_factory=dict)


class _RewardSearchFn:
    def __init__(self, *_args, **_kwargs):
        pass

    def extract_answer_from_response(self, response: str) -> str:
        return str(response)


def _install_rllm_stubs() -> None:
    rllm = types.ModuleType("rllm")
    rewards = types.ModuleType("rllm.rewards")
    reward_fn = types.ModuleType("rllm.rewards.reward_fn")
    reward_types = types.ModuleType("rllm.rewards.reward_types")
    search_reward = types.ModuleType("rllm.rewards.search_reward")

    reward_fn.search_reward_fn = lambda *_args, **_kwargs: _RewardOutput()
    reward_types.RewardConfig = _RewardConfig
    reward_types.RewardInput = _RewardInput
    reward_types.RewardOutput = _RewardOutput
    search_reward.RewardSearchFn = _RewardSearchFn

    sys.modules.setdefault("rllm", rllm)
    sys.modules.setdefault("rllm.rewards", rewards)
    sys.modules.setdefault("rllm.rewards.reward_fn", reward_fn)
    sys.modules.setdefault("rllm.rewards.reward_types", reward_types)
    sys.modules.setdefault("rllm.rewards.search_reward", search_reward)


_install_rllm_stubs()

from bcg.agent.reward import build_reward_fn, extract_mcq_answer  # noqa: E402


QUESTION = """Which answer is correct?

A. 44%
B. 52%
C. 1/3
D. The long prose option
"""


def _grade(action: str, ground_truth=("A", "44%")) -> _RewardOutput:
    reward_fn = build_reward_fn("gpqa_diamond")
    return reward_fn(
        {
            "question": QUESTION,
            "ground_truth": list(ground_truth),
            "data_source": "gpqa_diamond",
        },
        action,
    )


def test_boxed_text_letter_is_normalized() -> None:
    result = _grade(r"After work, the answer is \boxed{\text{A}}.")

    assert result.is_correct is True
    assert result.reward == 1.0
    assert result.metadata["extracted_answer"] == "A"
    assert result.metadata["f1_score"] == 1.0


def test_markdown_option_line_is_supported() -> None:
    result = _grade("### Final Answer\n**A. 44%**")

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "A"


def test_final_answer_marker_is_supported() -> None:
    result = _grade("Reasoning...\nFinal Answer: A")

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "A"


def test_correct_option_is_marker_is_supported() -> None:
    result = _grade("Reasoning...\n- The correct option is **B**.", ground_truth=("B",))

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "B"


def test_final_choice_is_marker_is_supported() -> None:
    result = _grade("After comparing them, the final choice is **C**.", ground_truth=("C",))

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "C"


def test_option_letter_is_correct_marker_is_supported() -> None:
    result = _grade("Therefore, Option A is the correct choice.")

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "A"


def test_bold_answer_label_is_supported() -> None:
    result = _grade("### **Final Answer**:\n\n**A. 44%**")

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "A"


def test_numeric_answer_maps_to_option() -> None:
    result = _grade(r"The probability is \boxed{43.93\%}.")

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "A"


def test_option_text_maps_to_choice() -> None:
    result = _grade("Final Answer: The long prose option", ground_truth=("D",))

    assert result.is_correct is True
    assert result.metadata["extracted_answer"] == "D"


def test_wrong_choice_gets_no_partial_credit() -> None:
    result = _grade("Final Answer: B")

    assert result.is_correct is False
    assert result.reward == 0.0
    assert result.metadata["f1_score"] == 0.0


def test_extract_mcq_answer_returns_raw_span() -> None:
    prediction, raw_span = extract_mcq_answer(r"Done: \boxed{\text{C. 1/3}}", QUESTION)

    assert prediction == "C"
    assert raw_span == r"\text{C. 1/3}"
