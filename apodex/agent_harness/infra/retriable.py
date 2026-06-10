"""Error-classification helper for the LLM-call retry loop.

Only one classifier is needed in OSS: ``is_transient_network``. It
recognises the proxy-wrapped 5xx envelopes (``bad_response_status_code``
/ ``new_api_error``) and the obvious connection / timeout failures that
warrant a backoff-then-retry on the same key, and excludes overload /
capacity errors so they aren't double-counted.
"""

from __future__ import annotations

import re


_OVERLOAD_PATTERNS = (
    re.compile(r"overload", re.IGNORECASE),
    re.compile(r"capacity", re.IGNORECASE),
    re.compile(r"529", re.IGNORECASE),
    re.compile(r"overloaded_error", re.IGNORECASE),
)


_TRANSIENT_NETWORK_PATTERNS = (
    re.compile(r"\btimeout\b", re.IGNORECASE),
    re.compile(r"timed[\s_]*out", re.IGNORECASE),
    re.compile(
        r"connection[\s_]*(?:reset|refused|aborted|error|closed)",
        re.IGNORECASE,
    ),
    re.compile(r"bad_response_status_code", re.IGNORECASE),
    re.compile(r"new_api_error", re.IGNORECASE),
    re.compile(r"gateway[\s_]*time[\s_]*out", re.IGNORECASE),
    re.compile(r"upstream[\s_]*(?:timeout|error)", re.IGNORECASE),
)


def _stringify(err: BaseException) -> str:
    """Concatenate every signal an LLM SDK might surface."""
    parts: list[str] = [type(err).__name__, str(err)]
    for attr in ("status_code", "response", "body", "message"):
        val = getattr(err, attr, None)
        if val is not None:
            parts.append(str(val))
    return " | ".join(parts)


def _get_status_code(err: BaseException) -> int | None:
    """Best-effort integer HTTP status extraction."""
    for attr in ("status_code", "status", "code"):
        val = getattr(err, attr, None)
        if isinstance(val, int):
            return val
    return None


def is_transient_network(err: BaseException) -> bool:
    """True if ``err`` is a request-level transient (timeout / connection
    reset / upstream 5xx / proxy-wrapped upstream blip).

    Used by ``core/runtime/loop/llm_client.py`` to keep proxy-wrapped 5xx
    envelopes on the same key (retry-with-backoff) rather than surfacing
    them as non-transient failures.
    """
    blob = _stringify(err)
    if any(p.search(blob) for p in _OVERLOAD_PATTERNS):
        return False
    status = _get_status_code(err)
    if status is not None and 500 <= status < 600:
        return True
    return any(p.search(blob) for p in _TRANSIENT_NETWORK_PATTERNS)


__all__ = ["is_transient_network"]
