"""Dedicated LiteLLM configuration; never inherits the Agent's API routing."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values


@dataclass(frozen=True)
class ReviewConfig:
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "medium"
    base_url: str = "https://litellm.mybigai.ac.cn/"
    api_key: str = field(default="", repr=False)
    timeout: float = 180
    max_output_tokens: int = 8192

    def public(self) -> dict[str, Any]:
        return {"provider": "litellm", "model": self.model, "reasoning_effort": self.reasoning_effort,
                "base_url": self.base_url, "timeout": self.timeout, "max_output_tokens": self.max_output_tokens}

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> ReviewConfig:
        values = {**dotenv_values(env_file or Path.home() / ".bcg/.env"), **os.environ}
        return cls(api_key=values.get("LITELLM_OPENAI_API_KEY") or "",
                   base_url=values.get("LITELLM_OPENAI_BASE_URL") or values.get("LITELLM_BASE_URL") or cls.base_url)


class ReviewError(RuntimeError):
    """Safe-to-display diagnostic without response bodies or credentials."""


class LiteLLMReviewer:
    def __init__(self, config: ReviewConfig, journal: Path, *, transport: httpx.BaseTransport | None = None) -> None:
        self.config, self.journal = config, journal
        if not config.api_key:
            raise ReviewError("缺少 LITELLM_OPENAI_API_KEY，未发起请求")
        parsed = httpx.URL(config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.userinfo or parsed.query:
            raise ReviewError("LiteLLM base URL 格式无效")
        base = config.base_url.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.http = httpx.Client(timeout=httpx.Timeout(config.timeout, connect=20),
                                 trust_env=False, follow_redirects=False, transport=transport)

    def close(self) -> None:
        self.http.close()

    def complete(self, messages: list[dict], stage: str) -> str:
        body = {"model": self.config.model, "reasoning_effort": self.config.reasoning_effort,
                "messages": messages, "response_format": {"type": "json_object"},
                "max_completion_tokens": self.config.max_output_tokens, "stream": False}
        for attempt in range(3):
            started = time.monotonic()
            entry: dict[str, Any] = {"stage": stage, "attempt": attempt + 1, "request": body,
                                     "started_at": time.time()}
            retry = False
            try:
                response = self.http.post(self.endpoint, json=body,
                    headers={"Authorization": "Bearer " + self.config.api_key})
                entry["http_status"] = response.status_code
                if response.status_code >= 400:
                    retry = response.status_code in {429, 502, 503, 504}
                    raise ReviewError(f"LiteLLM 请求失败（HTTP {response.status_code}）")
                if response.status_code != 200:
                    raise ReviewError("LiteLLM 返回非预期状态，不跟随重定向")
                data = response.json()
                entry["usage"] = data.get("usage", {})
                entry["response_id"] = data.get("id")
                entry["actual_model"] = data.get("model")
                choice = data["choices"][0]
                entry["finish_reason"] = choice.get("finish_reason")
                answer = choice["message"].get("content")
                entry["response_text"] = answer
                if choice.get("finish_reason") == "length":
                    raise ReviewError("AI Review 输出达到 Token 上限，未保存为有效标注")
                if not isinstance(answer, str) or not answer.strip():
                    raise ReviewError("AI Review 没有返回可见的 JSON 内容")
                return answer
            except httpx.TransportError as exc:
                retry = True
                error = ReviewError(f"LiteLLM 网络请求失败（{type(exc).__name__}）")
            except ReviewError as exc:
                error = exc
            except (ValueError, KeyError, IndexError, TypeError):
                error = ReviewError("LiteLLM 返回内容格式异常")
            finally:
                entry["duration_seconds"] = round(time.monotonic() - started, 3)
                self.journal.parent.mkdir(parents=True, exist_ok=True)
                with self.journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.journal.chmod(0o600)
            if not retry or attempt == 2:
                raise error
            time.sleep(2 ** attempt)
        raise ReviewError("LiteLLM 请求未完成")
