from __future__ import annotations

from bcg.agent.check_qwen_thinking import _parse_args as parse_check_thinking_args
from bcg.agent.config import default_rollout_config
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


def test_rollout_uses_config_as_its_default_source(monkeypatch) -> None:
    monkeypatch.setenv("SERPER_ENDPOINT", "https://serper.test/search")
    expected = default_rollout_config()

    cfg = parse_rollout_args(["--model", "agent-model", "--no-auto-ui"])

    assert cfg.max_steps == expected.max_steps
    assert cfg.retrieval_max_results == expected.retrieval_max_results
    assert cfg.serper_endpoint == expected.serper_endpoint
    assert cfg.tonggraph_graph == expected.tonggraph_graph


def test_rollout_preset_can_be_overridden_by_explicit_flags() -> None:
    cfg = parse_rollout_args(
        [
            "--preset",
            "averitec-hero4",
            "--model",
            "agent-model",
            "--max-problems",
            "3",
            "--no-auto-ui",
        ]
    )

    assert cfg.tasks == ["averitec"]
    assert cfg.retrieval_method == "hero4"
    assert cfg.hyde is False
    assert cfg.max_problems == 3
    assert cfg.stage3_rerank_k == 5


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
