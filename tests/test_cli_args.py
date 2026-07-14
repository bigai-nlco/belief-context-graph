from __future__ import annotations

from bcg.agent.check_qwen_thinking import _parse_args as parse_check_thinking_args
from bcg.agent.rollout import _parse_args as parse_rollout_args


def test_rollout_accepts_max_model_len() -> None:
    cfg = parse_rollout_args(
        [
            "--model",
            "/models/Qwen3.5-4B",
            "--max-model-len",
            "131072",
            "--no-auto-ui",
        ]
    )

    assert cfg.vllm_max_model_len == 131072


def test_rollout_keeps_legacy_vllm_max_model_len_alias() -> None:
    cfg = parse_rollout_args(
        [
            "--model",
            "/models/Qwen3.5-4B",
            "--vllm-max-model-len",
            "65536",
            "--no-auto-ui",
        ]
    )

    assert cfg.vllm_max_model_len == 65536


def test_rollout_accepts_browsecomp_grader_options() -> None:
    cfg = parse_rollout_args(
        [
            "--model",
            "agent-model",
            "--browsecomp-grader-model",
            "grader-model",
            "--browsecomp-grader-base-url",
            "https://grader.test/v1",
            "--browsecomp-grader-timeout",
            "45",
            "--browsecomp-grader-max-tokens",
            "512",
            "--browsecomp-grader-max-retries",
            "1",
            "--no-auto-ui",
        ]
    )

    assert cfg.browsecomp_grader_model == "grader-model"
    assert cfg.browsecomp_grader_base_url == "https://grader.test/v1"
    assert cfg.browsecomp_grader_timeout == 45
    assert cfg.browsecomp_grader_max_tokens == 512
    assert cfg.browsecomp_grader_max_retries == 1


def test_check_thinking_accepts_max_model_len() -> None:
    args = parse_check_thinking_args(
        [
            "--model",
            "/models/Qwen3.5-4B",
            "--max-model-len",
            "131072",
        ]
    )

    assert args.max_model_len == 131072
