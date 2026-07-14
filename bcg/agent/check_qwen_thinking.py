"""Smoke test Qwen3.5 thinking-mode token behavior."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from bcg.cli_help import RichArgumentParser

from bcg.agent.tokenizer_compat import (
    QWEN_THINK_END_TOKEN_ID,
    QWEN_THINK_START_TOKEN_ID,
    build_sampling_params_compat,
    qwen_thinking_token_report,
)


DEFAULT_QUESTION = (
    "What is 2+2? Answer with only the number."
)


def _token_count(tokenizer: Any, text: str | None) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _parse_args(
    argv: list[str] | None = None, prog: str | None = None
) -> argparse.Namespace:
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "Run one local Qwen3.5 generation and assert official thinking "
            "sentinel token behavior."
        )
    )
    parser.add_argument("--model", required=True, help="Local Qwen3.5 model path")
    parser.add_argument(
        "--backend",
        choices=("vllm", "sglang"),
        default="vllm",
        help="Backend used for the token-level smoke test.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode. The assertion expects this flag.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument(
        "--max-model-len",
        "--vllm-max-model-len",
        dest="max_model_len",
        type=int,
        default=None,
        help="Maximum model context length for local vLLM/SGLang backends.",
    )
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    return parser.parse_args(argv)


async def _run_vllm(args: argparse.Namespace) -> dict[str, Any]:
    from bcg.agent.vllm_engine import VLLMEngine

    sampling_params = build_sampling_params_compat(
        args.model,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_new_tokens,
    )
    engine = VLLMEngine(
        model=args.model,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_new_tokens,
        max_model_length=args.max_model_len,
        sampling_params=sampling_params,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.vllm_dtype,
        enforce_eager=args.vllm_enforce_eager,
        disable_thinking=not args.enable_thinking,
    )
    try:
        output = await engine.get_model_response(
            [{"role": "user", "content": args.question}],
            application_id="qwen35-thinking-check",
        )
    finally:
        engine.shutdown()

    report = qwen_thinking_token_report(output.prompt_ids, output.completion_ids)
    report.update(
        {
            "backend": "vllm",
            "model": args.model,
            "enable_thinking": args.enable_thinking,
            "prompt_tokens": output.prompt_length,
            "completion_tokens": output.completion_length,
            "finish_reason": output.finish_reason,
            "reasoning_tokens": _token_count(engine.tokenizer, output.reasoning),
            "content_tokens": _token_count(engine.tokenizer, output.content),
            "text_preview": (output.text or "")[:500],
        }
    )
    return report


def _run_sglang(args: argparse.Namespace) -> dict[str, Any]:
    from bcg.agent.sglang_engine import SGLangEngine

    sampling_params = build_sampling_params_compat(
        args.model,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_new_tokens,
    )
    engine = SGLangEngine(
        model=args.model,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_new_tokens,
        max_model_length=args.max_model_len,
        sampling_params=sampling_params,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_thinking=not args.enable_thinking,
    )
    try:
        output = asyncio.run(
            engine.get_model_response(
                [{"role": "user", "content": args.question}],
                application_id="qwen35-thinking-check",
            )
        )
    finally:
        engine.shutdown()

    report = qwen_thinking_token_report(output.prompt_ids, output.completion_ids)
    report.update(
        {
            "backend": "sglang",
            "model": args.model,
            "enable_thinking": args.enable_thinking,
            "prompt_tokens": output.prompt_length,
            "completion_tokens": output.completion_length,
            "finish_reason": output.finish_reason,
            "reasoning_tokens": _token_count(engine.tokenizer, output.reasoning),
            "content_tokens": _token_count(engine.tokenizer, output.content),
            "text_preview": (output.text or "")[:500],
        }
    )
    return report


def _assert_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.enable_thinking:
        raise SystemExit(
            "Pass --enable-thinking for this smoke test; it verifies thinking mode."
        )
    if not report["prompt_has_think_start"]:
        raise SystemExit(
            f"FAILED: prompt_ids did not contain <think> token "
            f"{QWEN_THINK_START_TOKEN_ID}."
        )
    if not report["completion_has_think_end"]:
        raise SystemExit(
            f"FAILED: completion_ids did not contain </think> token "
            f"{QWEN_THINK_END_TOKEN_ID}. Increase --max-new-tokens if generation "
            "was cut off before the model closed its thinking block."
        )


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    args = _parse_args(argv, prog=prog)
    try:
        if args.backend == "sglang":
            report = _run_sglang(args)
        else:
            report = asyncio.run(_run_vllm(args))
    except ImportError as exc:
        raise SystemExit(
            "The thinking smoke test requires backend runtime dependencies "
            "(vllm or sglang, transformers, rllm). Install the runtime "
            "environment first."
        ) from exc

    print(json.dumps(report, ensure_ascii=False, indent=2))
    _assert_report(args, report)


if __name__ == "__main__":
    main(sys.argv[1:])
