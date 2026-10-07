"""Stable documented hook fields only; no transcript scraping or hidden reasoning."""

from __future__ import annotations

import hashlib
import json
import re

from .config import Config


def redact(text: str, secrets: list[str]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if len(secret) >= 8:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[redacted]", text)
    return text


def visible_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        return "\n".join(
            part.get("text", "")
            for part in value["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return json.dumps(value, ensure_ascii=False)


def normalize_event(raw: dict, config: Config, secrets: list[str]) -> dict | None:
    if not config.enabled or not config.contains(str(raw.get("cwd", ""))):
        return None
    session = raw.get("session_id")
    if not isinstance(session, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,160}", session
    ):
        raise ValueError("invalid Codex session id")
    kind = raw.get("hook_event_name")
    if kind not in {"UserPromptSubmit", "PostToolUse", "Stop"}:
        return None

    def clean(text: str) -> str:
        text = redact(text, secrets)
        if len(text) > config.max_event_chars:
            text = text[: config.max_event_chars] + "\n[BCG capture truncated here]"
        return text

    if kind == "PostToolUse":
        name = str(raw.get("tool_name", ""))
        # Never feed recall output back into the graph, including code-mode
        # nested calls whose canonical names use the same MCP server prefix.
        if "bcg_memory" in name.casefold():
            return None
        call_id = str(raw.get("tool_use_id", ""))
        if not name or not call_id:
            raise ValueError("missing native tool name or call id")
        payload = {
            "name": name[:200],
            "call_id": call_id[:240],
            "arguments": clean(visible_text(raw.get("tool_input", {}))),
            "text": clean(visible_text(raw.get("tool_response", ""))),
            "is_error": bool(
                isinstance(raw.get("tool_response"), dict)
                and raw["tool_response"].get("isError")
            ),
        }
        identity = call_id
    else:
        text = raw.get(
            "prompt" if kind == "UserPromptSubmit" else "last_assistant_message"
        )
        if not isinstance(text, str) or not text.strip():
            return None
        payload = {"text": clean(text)}
        identity = hashlib.sha256(payload["text"].encode()).hexdigest()
    turn = str(raw.get("turn_id") or "unknown")[:200]
    key = hashlib.sha256(
        json.dumps([session, turn, kind, identity]).encode()
    ).hexdigest()
    return {
        "session": session,
        "turn": turn,
        "kind": kind,
        "key": key,
        "payload": payload,
    }


def graph_turns(event: dict) -> list[dict]:
    data = event["payload"]
    if event["kind"] != "PostToolUse":
        return [
            {
                "role": "user" if event["kind"] == "UserPromptSubmit" else "assistant",
                "content": data["text"],
            }
        ]
    try:
        arguments = json.loads(data["arguments"])
    except ValueError:
        arguments = {"input": data["arguments"]}
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}
    body = data["text"]
    # Search output already has ranked snippets. For other native tools make
    # ONE observation item, so the existing bounded semantic extractor is used
    # instead of its opaque-tool fallback. This is transport metadata, not a
    # new belief or a fabricated source URL; the raw event stays in SQLite.
    if (
        not re.search(r"(?m)^\[\d+\] .+", body)
        and body.strip()
        and not data["is_error"]
    ):
        body = f"[1] Observation from {data['name']}\nSnippet: " + " ".join(
            body.split()
        )

    def block(tag: str, payload: dict) -> str:
        # JSON escaping prevents a source string closing our parser envelope.
        encoded = (
            json.dumps(payload, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )
        return f"<{tag}>\n{encoded}\n</{tag}>"

    return [
        {
            "role": "assistant",
            "content": block(
                "tool_call",
                {"id": data["call_id"], "name": data["name"], "arguments": arguments},
            ),
        },
        {
            "role": "tool",
            "content": block(
                "tool_result",
                {
                    "tool_call_id": data["call_id"],
                    "name": data["name"],
                    "content": body,
                    "is_error": data["is_error"],
                },
            ),
        },
    ]
