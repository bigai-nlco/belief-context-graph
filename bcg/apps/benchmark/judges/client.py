"""OpenAI-compatible Chat Completions client for the official report judges.

Each official judge sends a single user message with fixed request
parameters (``temperature``, ``max_tokens``, ``reasoning_effort``, ...). This
client sends exactly those parameters. When an endpoint rejects one of them
(reasoning models reject ``temperature``; some reject ``max_tokens`` in favour
of ``max_completion_tokens``), the request is retried once without (or with
the renamed) parameter and the adaptation is recorded, so every judgement
states the parameters that were actually used.

Responses can be cached on disk by request hash; a re-run then reuses
completed calls (useful for resuming a large judge run).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bcg.apps.benchmark.models import is_api_quota_error


@dataclass(frozen=True)
class JudgeEndpoint:
    model: str
    base_url: str
    api_key: str = field(default="", repr=False)
    timeout: float = 600.0


@dataclass
class JudgeReply:
    text: str
    usage: dict[str, int]
    params: dict[str, Any]
    cached: bool = False
    seconds: float = 0.0


class JudgeRequestError(RuntimeError):
    """A judge request failed (transport, HTTP, or malformed response)."""

    @property
    def quota(self) -> bool:
        return is_api_quota_error(str(self))


# Parameter adaptations: name -> (pattern in the error, replacement or None to drop).
_ADAPTATIONS = (
    ("temperature", None),
    ("max_tokens", "max_completion_tokens"),
    ("max_completion_tokens", "max_tokens"),
    ("reasoning_effort", None),
)


class JudgeClient:
    def __init__(
        self,
        endpoint: JudgeEndpoint,
        *,
        cache_dir: Path | None = None,
        transport: Any = None,
    ) -> None:
        self.endpoint = endpoint
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._transport = transport  # tests inject a callable(url, headers, payload) -> (status, json|text)
        self._lock = threading.Lock()
        # Adaptations learned from endpoint errors, shared by every call.
        self.adaptations: dict[str, str | None] = {}

    # -- public -----------------------------------------------------------
    def complete(
        self, prompt: str, params: dict[str, Any], *, use_cache: bool = True
    ) -> JudgeReply:
        effective = self._apply_adaptations(dict(params))
        key = self._cache_key(prompt, params)
        if use_cache:
            cached = self._read_cache(key)
            if cached is not None:
                return JudgeReply(
                    cached["text"],
                    cached.get("usage", {}),
                    cached.get("params", effective),
                    True,
                )
        started = time.monotonic()
        for _ in range(len(_ADAPTATIONS) + 1):
            status, body = self._post(prompt, effective)
            if status < 400:
                text, usage = self._parse(body)
                reply = JudgeReply(
                    text, usage, effective, False, time.monotonic() - started
                )
                self._write_cache(
                    key, {"text": text, "usage": usage, "params": effective}
                )
                return reply
            detail = body if isinstance(body, str) else json.dumps(body)[:2000]
            adapted = self._adapt(status, detail, effective) if status == 400 else None
            if adapted is None:
                raise JudgeRequestError(f"judge HTTP {status}: {detail[:1000]}")
            effective = adapted
        raise JudgeRequestError("judge request kept failing after parameter adaptation")

    # -- internals --------------------------------------------------------
    def _apply_adaptations(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            adaptations = dict(self.adaptations)
        for name, replacement in adaptations.items():
            if name in params:
                value = params.pop(name)
                if replacement:
                    params.setdefault(replacement, value)
        return params

    def _adapt(
        self, status: int, detail: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        folded = detail.casefold()
        for name, replacement in _ADAPTATIONS:
            if (
                name in params
                and name in folded
                and (
                    "unsupported" in folded
                    or "not support" in folded
                    or "does not support" in folded
                    or "unrecognized" in folded
                    or "invalid" in folded
                    or "not allowed" in folded
                    or "only the default" in folded
                    or "use '" in folded
                    or "instead" in folded
                )
            ):
                with self._lock:
                    self.adaptations[name] = replacement
                return self._apply_adaptations(dict(params))
        return None

    def _post(self, prompt: str, params: dict[str, Any]) -> tuple[int, Any]:
        url = self.endpoint.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        payload = {
            "model": self.endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
            **params,
        }
        if self._transport is not None:
            return self._transport(url, headers, payload)
        try:
            response = httpx.post(
                url, headers=headers, json=payload, timeout=self.endpoint.timeout
            )
        except httpx.HTTPError as exc:
            raise JudgeRequestError(f"judge request failed: {exc}") from exc
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text

    @staticmethod
    def _parse(body: Any) -> tuple[str, dict[str, int]]:
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeRequestError(
                f"invalid Chat Completions response: {str(body)[:500]}"
            ) from exc
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return str(content or ""), {
            "input": int(usage.get("prompt_tokens") or 0),
            "output": int(usage.get("completion_tokens") or 0),
            "reasoning": int(details.get("reasoning_tokens") or 0),
            "total": int(usage.get("total_tokens") or 0),
        }

    def _cache_key(self, prompt: str, params: dict[str, Any]) -> str:
        material = json.dumps(
            {"model": self.endpoint.model, "params": params, "prompt": prompt},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / key[:2] / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / key[:2] / f"{key}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
            temporary.write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError:
            pass


def add_usage(total: dict[str, int], usage: dict[str, int]) -> dict[str, int]:
    for key in ("input", "output", "reasoning", "total"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0))
    total["calls"] = total.get("calls", 0) + 1
    return total


__all__ = [
    "JudgeClient",
    "JudgeEndpoint",
    "JudgeReply",
    "JudgeRequestError",
    "add_usage",
]
