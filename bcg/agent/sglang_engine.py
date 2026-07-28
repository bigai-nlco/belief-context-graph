"""SGLang rollout engine for BeliefTracer."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser

from bcg.agent.tokenizer_compat import (
    configure_parser_for_qwen_thinking,
    load_tokenizer_compat,
    prepare_vllm_model_path,
    qwen_completion_text_with_thinking_tags,
)


class SGLangEngine(RolloutEngine):
    """Drive SGLang's in-process Engine through the rLLM RolloutEngine API."""

    def __init__(
        self,
        model: str,
        max_prompt_length: int = 16384,
        max_response_length: int = 8192,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        tools: list[Any] | None = None,
        accumulate_reasoning: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        disable_thinking: bool = False,
        engine_kwargs: dict | None = None,
        **_: Any,
    ) -> None:
        os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")

        from sglang import Engine

        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = (
            max_model_length
            if max_model_length is not None
            else max_prompt_length + max_response_length
        )
        self.sampling_params = dict(sampling_params or {})
        self.tools = list(tools or [])
        self.accumulate_reasoning = accumulate_reasoning
        self.disable_thinking = disable_thinking

        model_path, self._model_tmpdir = prepare_vllm_model_path(model)
        self.tokenizer = load_tokenizer_compat(
            model, trust_remote_code=trust_remote_code
        )
        self.chat_parser = ChatTemplateParser.get_parser(
            self.tokenizer, disable_thinking=disable_thinking
        )
        configure_parser_for_qwen_thinking(
            self.chat_parser, model, disable_thinking=disable_thinking
        )
        kwargs = {
            "model_path": model_path,
            "tp_size": tensor_parallel_size,
            "mem_fraction_static": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "context_length": self.max_model_length,
            "max_running_requests": 1,
            "max_total_tokens": self.max_model_length,
            "disable_cuda_graph": True,
            "disable_piecewise_cuda_graph": True,
        }
        kwargs.update(engine_kwargs or {})
        self.engine = Engine(**kwargs)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="BeliefTracerSGLang"
        )

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        request_id = f"{application_id}-{uuid.uuid4()}"
        tools = kwargs.pop("tools", self.tools)
        accumulate_reasoning = kwargs.pop(
            "accumulate_reasoning", self.accumulate_reasoning
        )
        merged = self.sampling_params.copy()
        merged.update(kwargs)
        for ignored in (
            "validate",
            "model",
            "enforce_max_prompt_length",
            "precomputed_prompt_ids",
            "reasoning_effort",
        ):
            merged.pop(ignored, None)
        max_tokens = int(
            merged.pop("max_tokens", merged.pop("max_new_tokens", self.max_response_length))
        )
        sampling_params = {
            "temperature": merged.get("temperature", 0.6),
            "top_p": merged.get("top_p", 0.95),
            "max_new_tokens": max_tokens,
        }
        for key in (
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
        ):
            if key in merged:
                sampling_params[key] = merged[key]

        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.max_prompt_length or len(prompt_ids) > self.max_model_length:
            from rllm.workflows import TerminationEvent, TerminationReason

            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)
        remaining_tokens = self.max_model_length - len(prompt_ids) - 1
        if remaining_tokens <= 0:
            from rllm.workflows import TerminationEvent, TerminationReason

            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)
        max_tokens = min(max_tokens, remaining_tokens)
        sampling_params["max_new_tokens"] = max_tokens

        result = await self._generate(prompt, sampling_params, request_id)
        completion_ids = _extract_output_ids(result)
        if completion_ids is None:
            completion_ids = self.tokenizer.encode(
                _extract_text(result), add_special_tokens=False
            )
        completion_ids = completion_ids[:max_tokens]
        finish_reason = _extract_finish_reason(result)
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"

        parsed = self.chat_parser.parse_completion(completion_ids)
        valid_tool_calls = []
        for tc in parsed.get("tool_calls") or []:
            if not getattr(tc, "name", None) or not getattr(tc, "arguments", None):
                continue
            valid_tool_calls.append(tc)

        raw_completion_text = self.tokenizer.decode(
            completion_ids, skip_special_tokens=False
        )
        text = qwen_completion_text_with_thinking_tags(
            raw_completion_text,
            reasoning=parsed.get("reasoning", ""),
            strip_special_tokens=getattr(self.chat_parser, "_strip_special_tokens", None),
        )

        return ModelOutput(
            text=text,
            content=parsed.get("content", ""),
            reasoning=parsed.get("reasoning", ""),
            tool_calls=valid_tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=[],
            prompt_logprobs=[],
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _generate(
        self, prompt: str, sampling_params: dict[str, Any], application_id: str
    ) -> Any:
        import asyncio

        loop = asyncio.get_running_loop()
        generate = partial(
            self.engine.generate,
            prompt=prompt,
            sampling_params=sampling_params,
            rid=application_id,
        )
        return await loop.run_in_executor(
            self._executor,
            generate,
        )


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("text", "output", "content"):
            if key in result and result[key] is not None:
                return str(result[key])
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if isinstance(choice, dict):
                return str(choice.get("text") or choice.get("message", {}).get("content") or "")
    return str(result or "")


def _extract_output_ids(result: Any) -> list[int] | None:
    if isinstance(result, dict):
        output_ids = result.get("output_ids")
        if isinstance(output_ids, list):
            return [int(token_id) for token_id in output_ids]
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if isinstance(choice, dict):
                choice_ids = choice.get("output_ids") or choice.get("token_ids")
                if isinstance(choice_ids, list):
                    return [int(token_id) for token_id in choice_ids]
    return None


def _extract_finish_reason(result: Any) -> str:
    if isinstance(result, dict):
        meta = result.get("meta_info") or result.get("meta") or {}
        reason = meta.get("finish_reason") or result.get("finish_reason")
        if reason:
            return str(reason)
    return "stop"


__all__ = ["SGLangEngine"]
