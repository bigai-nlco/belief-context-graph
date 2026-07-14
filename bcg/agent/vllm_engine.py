"""In-process vLLM rollout engine for agent workflow inference.

Implements rllm's ``RolloutEngine`` protocol on top of
``vllm.AsyncLLMEngine``, so the runner loads the model locally and
does not depend on a running OpenAI-compatible HTTP server.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt

from bcg.agent.tokenizer_compat import (
    configure_parser_for_qwen_thinking,
    load_tokenizer_compat,
    prepare_vllm_model_path,
    qwen_completion_text_with_thinking_tags,
)
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.tools.tool_base import Tool
from rllm.workflows import TerminationEvent, TerminationReason


class VLLMEngine(RolloutEngine):
    """Drive an embedded ``AsyncLLMEngine`` via the rllm completion path.

    The engine tokenizes prompts with the model's chat template parser
    (matching rllm's ``VerlEngine`` approach) and streams the final
    ``RequestOutput`` from ``AsyncLLMEngine.generate``.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Any | None = None,
        max_prompt_length: int = 16384,
        max_response_length: int = 8192,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        tools: list[Tool | dict] | None = None,
        accumulate_reasoning: bool = False,
        disable_thinking: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        engine_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        if max_model_length is not None:
            self.max_model_length = max_model_length - 1
        else:
            self.max_model_length = max_prompt_length + max_response_length - 1
        self.sampling_params = dict(sampling_params or {})
        self.tools = list(tools or [])
        self.accumulate_reasoning = accumulate_reasoning
        vllm_model, self._vllm_model_tmpdir = prepare_vllm_model_path(model)

        self.tokenizer = tokenizer or load_tokenizer_compat(
            model, trust_remote_code=trust_remote_code
        )
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = None
        self.chat_parser = ChatTemplateParser.get_parser(
            self.tokenizer, disable_thinking=disable_thinking
        )
        configure_parser_for_qwen_thinking(
            self.chat_parser, model, disable_thinking=disable_thinking
        )

        engine_args = AsyncEngineArgs(
            model=vllm_model,
            tokenizer=vllm_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self.max_model_length + 1,
            dtype=dtype,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            enable_log_requests=False,
            **(engine_kwargs or {}),
        )
        self.llm: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)

        self.validate = False

    def _build_sampling_params(self, overrides: dict) -> tuple[SamplingParams, int]:
        merged = self.sampling_params.copy()
        merged.update(overrides)
        merged.pop("application_id", None)
        merged.pop("validate", None)
        merged.pop("model", None)
        merged.pop("enforce_max_prompt_length", None)
        merged.pop("precomputed_prompt_ids", None)
        merged.pop("accumulate_reasoning", None)
        merged.pop("reasoning_effort", None)
        merged.pop("tools", None)

        max_tokens = merged.pop(
            "max_tokens",
            merged.pop("max_new_tokens", self.max_response_length),
        )

        allowed = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seed",
            "stop",
            "stop_token_ids",
            "skip_special_tokens",
            "ignore_eos",
            "logprobs",
            "n",
        }
        kwargs = {k: v for k, v in merged.items() if k in allowed}
        return SamplingParams(max_tokens=max_tokens, **kwargs), max_tokens

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        tools = kwargs.pop("tools", self.tools)
        accumulate_reasoning = kwargs.pop(
            "accumulate_reasoning", self.accumulate_reasoning
        )

        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
        )
        prompt_ids: list[int] = self.tokenizer.encode(prompt, add_special_tokens=False)

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and (
            prompt_length > self.max_prompt_length
            or prompt_length > self.max_model_length
        ):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        sampling_params, max_tokens = self._build_sampling_params(kwargs)
        remaining = self.max_model_length - prompt_length
        if remaining > 0 and sampling_params.max_tokens > remaining:
            sampling_params.max_tokens = remaining

        request_id = f"{application_id}-{uuid.uuid4().hex[:8]}"
        prompt_input = TokensPrompt(prompt_token_ids=prompt_ids)

        final = None
        async for out in self.llm.generate(
            prompt=prompt_input,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            final = out
        if final is None or not final.outputs:
            raise RuntimeError(f"vLLM returned no outputs for request {request_id}")

        choice = final.outputs[0]
        completion_ids: list[int] = list(choice.token_ids)
        logprobs: list[float] = []
        if choice.logprobs:
            for step in choice.logprobs:
                if not step:
                    continue
                lp = next(iter(step.values()))
                logprobs.append(float(getattr(lp, "logprob", 0.0)))

        finish_reason = choice.finish_reason or "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]
            logprobs = logprobs[:max_tokens]

        parsed = self.chat_parser.parse_completion(completion_ids)
        valid_tool_calls = []
        for tc in parsed.get("tool_calls") or []:
            if not getattr(tc, "name", None) or not getattr(tc, "arguments", None):
                continue
            valid_tool_calls.append(tc)

        raw_completion_text = self.tokenizer.decode(
            completion_ids, skip_special_tokens=False
        )
        completion_text = qwen_completion_text_with_thinking_tags(
            raw_completion_text,
            reasoning=parsed.get("reasoning", ""),
            strip_special_tokens=getattr(self.chat_parser, "_strip_special_tokens", None),
        )

        return ModelOutput(
            text=completion_text,
            content=parsed.get("content", ""),
            reasoning=parsed.get("reasoning", ""),
            tool_calls=valid_tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
            prompt_logprobs=[],
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def wake_up(self) -> None:
        return None

    async def sleep(self) -> None:
        return None

    def shutdown(self) -> None:
        shutdown = getattr(self.llm, "shutdown_background_loop", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass


__all__ = ["VLLMEngine"]
