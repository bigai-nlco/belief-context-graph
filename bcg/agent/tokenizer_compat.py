"""Tokenizer compatibility helpers for local checkpoints."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    AutoTokenizer = None


_QWEN_35_36_RE = re.compile(r"qwen[-_/]?3[._-][56](?=[^0-9]|$)", re.IGNORECASE)
_QWEN_3_RE = re.compile(
    r"qwen[-_/]?3(?![._-][56])(?=[^0-9]|$)", re.IGNORECASE
)
_QWEN_FUNCTION_RE = re.compile(
    r"^<function=([^\n>]+)>\s*(.*?)\s*</function>$", re.DOTALL
)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([^\n>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
QWEN_THINK_START_TOKEN_ID = 248068
QWEN_THINK_END_TOKEN_ID = 248069
QWEN_THINK_START = "<think>"
QWEN_THINK_END = "</think>"


def _model_names(model_or_tokenizer: Any) -> list[str]:
    names: list[str] = []
    if isinstance(model_or_tokenizer, str):
        names.append(model_or_tokenizer)
    else:
        for attr in ("name_or_path", "model_name", "model_path"):
            value = getattr(model_or_tokenizer, attr, None)
            if isinstance(value, str):
                names.append(value)
    return names


def is_qwen35_or_qwen36(model_or_tokenizer: Any) -> bool:
    """Return whether a model/tokenizer belongs to the Qwen3.5/3.6 series."""

    return any(_QWEN_35_36_RE.search(name) for name in _model_names(model_or_tokenizer))


def is_qwen3_family(model_or_tokenizer: Any) -> bool:
    """Return whether a model/tokenizer belongs to the Qwen3 family."""

    names = _model_names(model_or_tokenizer)
    return any(_QWEN_3_RE.search(name) or _QWEN_35_36_RE.search(name) for name in names)


def qwen_thinking_chat_template_kwargs(
    model_or_tokenizer: Any, *, disable_thinking: bool
) -> dict[str, bool]:
    """Build chat-template kwargs for Qwen3-family thinking control.

    HF tokenizers receive these values as template variables; OpenAI-compatible
    servers commonly expose the same dictionary as ``chat_template_kwargs``.
    """

    if not is_qwen3_family(model_or_tokenizer):
        return {}
    return {"enable_thinking": not disable_thinking}


def qwen35_sampling_defaults(*, enable_thinking: bool) -> dict[str, float | int]:
    """Return Qwen3.5/3.6 recommended general-task sampling defaults."""

    return {
        "temperature": 1.0 if enable_thinking else 0.7,
        "top_p": 0.95 if enable_thinking else 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }


def qwen_thinking_token_report(
    prompt_ids: list[int] | None, completion_ids: list[int] | None
) -> dict[str, Any]:
    """Summarize whether official Qwen thinking sentinels are present.

    Qwen3.5's official template pre-fills ``<think>`` when thinking is
    enabled, so token 248068 is normally found in ``prompt_ids``. The model
    then generates reasoning and closes it with ``</think>`` token 248069,
    which should appear in ``completion_ids`` for a completed thinking answer.
    """

    prompt_ids = prompt_ids or []
    completion_ids = completion_ids or []
    return {
        "think_start_token_id": QWEN_THINK_START_TOKEN_ID,
        "think_end_token_id": QWEN_THINK_END_TOKEN_ID,
        "prompt_has_think_start": QWEN_THINK_START_TOKEN_ID in prompt_ids,
        "prompt_has_think_end": QWEN_THINK_END_TOKEN_ID in prompt_ids,
        "completion_has_think_start": QWEN_THINK_START_TOKEN_ID in completion_ids,
        "completion_has_think_end": QWEN_THINK_END_TOKEN_ID in completion_ids,
    }


def qwen_completion_text_with_thinking_tags(
    raw_completion_text: str,
    *,
    reasoning: str | None = None,
    strip_special_tokens: Any | None = None,
) -> str:
    """Return completion text with visible Qwen thinking delimiters.

    In official Qwen3.5 thinking mode, ``<think>`` is a generation prompt
    prefix rather than a generated completion token. vLLM therefore returns a
    completion that typically starts with reasoning text and later contains the
    generated ``</think>`` token. For trajectory readability and parser
    compatibility, reattach the opening delimiter when reasoning is present.
    """

    text = raw_completion_text or ""
    if callable(strip_special_tokens):
        text = strip_special_tokens(text)
    has_thinking = bool(reasoning) or QWEN_THINK_END in text
    if has_thinking and text and not text.lstrip().startswith(QWEN_THINK_START):
        return f"{QWEN_THINK_START}\n{text}"
    return text


def build_sampling_params_compat(
    model_or_tokenizer: Any,
    *,
    enable_thinking: bool,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    """Build sampling params with Qwen3.5 mode-aware defaults.

    Explicit values override model-family defaults. Non-Qwen3.5 models keep
    BeliefTracer's historical temperature/top-p defaults unless overridden.
    """

    if is_qwen35_or_qwen36(model_or_tokenizer):
        params: dict[str, Any] = qwen35_sampling_defaults(
            enable_thinking=enable_thinking
        )
    else:
        params = {"temperature": 0.6, "top_p": 0.95}

    overrides = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    params.update(
        {key: value for key, value in overrides.items() if value is not None}
    )

    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    return params


def _parse_parameter_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class Qwen35ToolParserCompat:
    """Parse both legacy Qwen and Qwen3.5/3.6 native tool-call formats."""

    tool_call_begin = "<tool_call>"
    tool_call_end = "</tool_call>"
    tool_calls_begin = ""
    tool_calls_end = ""
    tool_output_begin = "<tool_response>"
    tool_output_end = "</tool_response>"

    def parse(self, model_response: str) -> list[Any]:
        from rllm.tools.tool_base import ToolCall

        return [
            ToolCall(name=call["name"], arguments=call["arguments"])
            for call in self.parse_qwen_tool_calls(model_response)
        ]

    def parse_qwen_tool_calls(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        cursor = 0

        while True:
            start = text.find(self.tool_call_begin, cursor)
            if start == -1:
                break
            content_start = start + len(self.tool_call_begin)
            end = text.find(self.tool_call_end, content_start)
            if end == -1:
                break

            raw_call = text[content_start:end].strip()
            parsed = self._parse_one_call(raw_call)
            if parsed is not None:
                tool_calls.append(parsed)
            cursor = end + len(self.tool_call_end)

        return tool_calls

    def _parse_one_call(self, raw_call: str) -> dict[str, Any] | None:
        function_match = _QWEN_FUNCTION_RE.match(raw_call)
        if function_match:
            name = function_match.group(1).strip()
            parameters_body = function_match.group(2)
            arguments = {
                match.group(1).strip(): _parse_parameter_value(match.group(2))
                for match in _QWEN_PARAMETER_RE.finditer(parameters_body)
            }
            return {"name": name, "arguments": arguments}

        try:
            call_data = json.loads(raw_call)
        except json.JSONDecodeError:
            return None

        if not isinstance(call_data, dict):
            return None
        name = call_data.get("name")
        if not isinstance(name, str) or not name:
            return None
        arguments = call_data.get("arguments", {})
        return {"name": name, "arguments": arguments}

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""

# Tools

You have access to the following functions:

<tools>
{tools_schema}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
""".rstrip()


def apply_chat_template_compat(
    tokenizer: Any,
    messages: list[dict],
    *,
    model: str | None = None,
    disable_thinking: bool = False,
    **kwargs: Any,
) -> str:
    """Apply a tokenizer chat template with model-specific compatibility kwargs."""

    template_kwargs = qwen_thinking_chat_template_kwargs(
        model or tokenizer, disable_thinking=disable_thinking
    )
    return tokenizer.apply_chat_template(messages, **kwargs, **template_kwargs)


def configure_parser_for_qwen_thinking(
    parser: Any, model_or_tokenizer: Any, *, disable_thinking: bool
) -> None:
    """Patch rLLM's hand-written Qwen parser for Qwen3-family compatibility.

    rLLM's built-in Qwen parser predates the Qwen3 chat template. The Qwen3
    family uses the tokenizer's official template, which controls thinking via
    ``enable_thinking`` and omits the legacy default system preamble. Qwen3.5/
    3.6 additionally switch to the structured ``<function=...>`` tool-call
    format, so accept both that and rLLM's legacy JSON-in-``<tool_call>``
    format.
    """

    if not is_qwen3_family(model_or_tokenizer):
        return

    tokenizer = getattr(parser, "tokenizer", None)
    has_chat_template = callable(getattr(tokenizer, "apply_chat_template", None))

    if has_chat_template:
        def _parse(
            self,
            messages: list[dict],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
            tools: list | None = None,
            accumulate_reasoning: bool = False,
            **kwargs: Any,
        ) -> str:
            del is_first_msg, accumulate_reasoning
            prompt_kwargs = dict(kwargs)
            prompt_kwargs.pop("tools", None)
            prompt_kwargs["tokenize"] = False
            prompt_kwargs["add_generation_prompt"] = add_generation_prompt
            if tools is not None:
                prompt_kwargs["tools"] = tools
            return apply_chat_template_compat(
                self.tokenizer,
                messages,
                model=model_or_tokenizer,
                disable_thinking=disable_thinking,
                **prompt_kwargs,
            )

        parser.parse = MethodType(_parse, parser)

    if hasattr(parser, "tool_parser") and is_qwen35_or_qwen36(model_or_tokenizer):
        parser.tool_parser = Qwen35ToolParserCompat()

    if not has_chat_template:
        return

    stub_messages = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
    with_prompt = apply_chat_template_compat(
        tokenizer,
        stub_messages,
        model=model_or_tokenizer,
        disable_thinking=disable_thinking,
        tokenize=False,
        add_generation_prompt=True,
    )
    without_prompt = apply_chat_template_compat(
        tokenizer,
        stub_messages,
        model=model_or_tokenizer,
        disable_thinking=disable_thinking,
        tokenize=False,
        add_generation_prompt=False,
    )
    parser.generation_prompt = with_prompt[len(without_prompt) :]


def get_tool_parser_compat(parser_name: str, model: str | None = None) -> Any:
    """Return an rLLM tool parser, overriding Qwen3.5/3.6 with compat parsing."""

    if parser_name == "qwen" and model and is_qwen35_or_qwen36(model):
        return Qwen35ToolParserCompat()

    from rllm.parser import get_tool_parser

    return get_tool_parser(parser_name)()


def _read_tokenizer_config(model: str) -> dict[str, Any] | None:
    path = Path(model)
    try:
        if not path.is_dir():
            return None
    except (OSError, PermissionError):
        return None

    config_path = path / "tokenizer_config.json"
    try:
        if not config_path.is_file():
            return None
    except (OSError, PermissionError):
        return None

    try:
        with config_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def _legacy_extra_special_tokens(model: str) -> list[str] | None:
    config = _read_tokenizer_config(model)
    if not config:
        return None

    tokens = config.get("extra_special_tokens")
    if not isinstance(tokens, list):
        return None

    return [token for token in tokens if isinstance(token, str)]


def load_tokenizer_compat(model: str, *args: Any, **kwargs: Any):
    """Load tokenizers with legacy list-valued ``extra_special_tokens``.

    Transformers 4.57 expects ``extra_special_tokens`` to be a mapping, while
    some Qwen-style local checkpoints store a legacy list there. Keep those
    tokens registered as special tokens via ``additional_special_tokens``.
    """

    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required to load local tokenizers. "
            "Install BCG with `uv tool install .` or `uv sync`."
        )

    tokens = _legacy_extra_special_tokens(model)
    if tokens:
        kwargs.setdefault("extra_special_tokens", {})
        kwargs.setdefault("additional_special_tokens", tokens)

    return AutoTokenizer.from_pretrained(model, *args, **kwargs)


def prepare_vllm_model_path(
    model: str,
) -> tuple[str, tempfile.TemporaryDirectory | None]:
    """Return a model path whose tokenizer config is accepted by vLLM/HF.

    vLLM loads its own tokenizer from the model directory and does not expose a
    tokenizer-kwargs hook for this field. For affected local checkpoints, build
    a temporary directory of symlinks with only ``tokenizer_config.json``
    rewritten. The caller must keep the returned TemporaryDirectory alive.
    """

    tokens = _legacy_extra_special_tokens(model)
    if not tokens:
        return model, None

    src = Path(model)
    config = _read_tokenizer_config(model)
    if config is None:
        return model, None

    tmpdir = tempfile.TemporaryDirectory(prefix="belief_tracer-vllm-model-")
    dst = Path(tmpdir.name)

    for child in src.iterdir():
        target = dst / child.name
        if child.name == "tokenizer_config.json":
            continue
        os.symlink(child, target, target_is_directory=child.is_dir())

    patched_config = dict(config)
    patched_config["extra_special_tokens"] = {}
    patched_config.setdefault("additional_special_tokens", tokens)
    with (dst / "tokenizer_config.json").open("w") as f:
        json.dump(patched_config, f, indent=2)
        f.write("\n")

    return str(dst), tmpdir
