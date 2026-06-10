"""Web fetch tool — Jina scrape + SUMMARY_LLM extraction.

Tool behaviour:

- Tool ``name`` is ``"web_fetch"``.
- Argument schema accepts ``url: str | list[str]`` and
  ``info_to_extract: str | list[str]`` — multi-URL fetch in **one turn**
  via ``asyncio.gather``.
- ``info_to_extract`` is required (no default).
- Pipeline is plain Jina → fallback to direct httpx → SUMMARY_LLM
  extraction. No academic backend routing, no scrape-cache.
- Output format ``[N] URL: ...\\n    Info: ...`` for both single and
  multi-URL calls.

Used by :mod:`workflows.react_base`.

Required env vars:
  JINA_API_KEY / JINA_BASE_URL — primary scraper
  SUMMARY_LLM_BASE_URL / SUMMARY_LLM_MODEL_NAME / SUMMARY_LLM_API_KEY
    — cheap LLM that does ``info_to_extract`` post-processing. Without
      this the tool returns ``[ERROR]: Extraction failed:
      SUMMARY_LLM_BASE_URL not set``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from agent_harness.core.tool import tool

logger = logging.getLogger(__name__)


# Env read at call time, not import time.


def _jina_api_key() -> str:
    return os.getenv("JINA_API_KEY", "")


def _jina_base_url() -> str:
    return os.getenv("JINA_BASE_URL", "https://r.jina.ai")


def _summary_llm_base_url() -> str | None:
    raw = os.environ.get("SUMMARY_LLM_BASE_URL")
    if not raw:
        return None
    url = raw.rstrip("/")
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def _summary_llm_model_name() -> str | None:
    return os.environ.get("SUMMARY_LLM_MODEL_NAME")


def _summary_llm_api_key() -> str | None:
    return os.environ.get("SUMMARY_LLM_API_KEY")


def _short_error_body(response: httpx.Response | None, limit: int = 1200) -> str:
    if response is None:
        return ""
    body = " ".join((response.text or "").split())
    if not body:
        return ""
    return body if len(body) <= limit else body[:limit].rstrip() + "..."


def _looks_like_context_overflow(response: httpx.Response | None) -> bool:
    if response is None:
        return False
    text = (response.text or "").lower()
    return any(
        marker in text
        for marker in (
            "exceeds the model's maximum context length",
            "longer than the model's context length",
            "context_length_exceeded",
            "maximum context",
            "context window",
            "too many tokens",
            "token limit",
            "input is too long",
            "prompt is too long",
            "request too large",
        )
    )


_BANNED_URL_PATTERNS: tuple[str, ...] = (
    "huggingface.co/datasets",
    "huggingface.co/spaces",
)


def _ensure_list(val: Any) -> Any:
    """Unwrap doubly-serialised JSON list payloads (mirrors web_search)."""
    if isinstance(val, str) and val.startswith("["):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return val


# ── Jina scraping ─────────────────────────────────────────────────────


async def _scrape_url_with_jina(
    url: str,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = 102400 * 4,
) -> dict[str, Any]:
    """Scrape via Jina reader API.

    Retries on connect/read timeouts and 5xx/408/409/425/429 with delays
    [1, 2, 4, 8] s. Detects Jina ``InsufficientBalanceError`` JSON bodies.
    """
    api_key = _jina_api_key()
    if not api_key:
        return {"success": False, "content": "", "error": "JINA_API_KEY not set"}

    # Strip ``r.jina.ai/`` prefix if caller already wrapped the URL.
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]

    jina_url = f"{_jina_base_url()}/{url}"
    headers = {"Authorization": f"Bearer {api_key}"}
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4, 8]
    response: httpx.Response | None = None

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    jina_url,
                    headers=headers,
                    timeout=httpx.Timeout(None, connect=20, read=60),
                    follow_redirects=True,
                )
            response.raise_for_status()
            break
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except httpx.HTTPStatusError as e:
            sc = e.response.status_code
            if (sc >= 500 or sc in [408, 409, 425, 429]) and attempt < len(
                retry_delays,
            ):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}

    if response is None:
        return {"success": False, "content": "", "error": "No response received"}

    content = response.text
    if not content:
        return {"success": False, "content": "", "error": "Empty response from Jina"}

    # Detect Jina balance exhaustion — the body is JSON like
    # ``{"name": "InsufficientBalanceError", ...}`` rather than the page.
    try:
        maybe_err = json.loads(content)
        if (
            isinstance(maybe_err, dict)
            and maybe_err.get("name") == "InsufficientBalanceError"
        ):
            return {
                "success": False,
                "content": "",
                "error": "Jina insufficient balance",
            }
    except json.JSONDecodeError:
        pass

    return {"success": True, "content": content[:max_chars], "error": ""}


async def _scrape_url_with_python(
    url: str,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = 102400 * 4,
) -> dict[str, Any]:
    """Direct httpx GET fallback when Jina fails. Same retry policy."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4]

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=headers,
                    timeout=httpx.Timeout(None, connect=20, read=60),
                    follow_redirects=True,
                )
            response.raise_for_status()
            content = response.text
            if not content:
                return {"success": False, "content": "", "error": "Empty response"}
            return {"success": True, "content": content[:max_chars], "error": ""}
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except httpx.HTTPStatusError as e:
            sc = e.response.status_code
            if (sc >= 500 or sc in [408, 409, 425, 429]) and attempt < len(
                retry_delays,
            ):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}

    return {"success": False, "content": "", "error": "All retries exhausted"}


# ── LLM extraction ───────────────────────────────────────────────────

_EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


async def _extract_info_with_llm(
    content: str,
    info_to_extract: str,
    truncate_last_num_chars: int = -1,
) -> dict[str, Any]:
    """Call the cheap SUMMARY_LLM to focus-extract from the scraped page."""
    if not content or not content.strip():
        return {"success": False, "extracted_info": "", "error": "Empty content"}

    base_url = _summary_llm_base_url()
    if not base_url:
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_BASE_URL not set",
        }

    text = content
    if truncate_last_num_chars > 0:
        text = content[:-truncate_last_num_chars] + "[...truncated]"

    prompt = _EXTRACT_INFO_PROMPT.format(info_to_extract, text)
    model = _summary_llm_model_name() or "default"

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    # GPT 4/5-style models require max_completion_tokens.
    if "gpt" in model:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        if "gpt-5" in model.lower() or "gpt5" in model.lower():
            payload["service_tier"] = "flex"
            payload["reasoning_effort"] = "minimal"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = _summary_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    retry_delays = [1, 2, 4, 8]
    response: httpx.Response | None = None

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    base_url,
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(None, connect=30, read=300),
                )

            # Context-overflow recovery: truncate tail and retry (40K × attempt).
            if _looks_like_context_overflow(response):
                payload["messages"][0]["content"] = _EXTRACT_INFO_PROMPT.format(
                    info_to_extract,
                    content[: -(40960 * attempt)] + "[...truncated]",
                )
                continue

            response.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            # GPT-5 sometimes rejects ``service_tier`` — drop and retry.
            if (
                "gpt-5" in model.lower() or "gpt5" in model.lower()
            ) and "service_tier" in payload:
                payload.pop("service_tier", None)
            if _looks_like_context_overflow(e.response):
                payload["messages"][0]["content"] = _EXTRACT_INFO_PROMPT.format(
                    info_to_extract,
                    content[: -(40960 * attempt)] + "[...truncated]",
                )
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            body = _short_error_body(e.response)
            suffix = f" Response body: {body}" if body else ""
            return {
                "success": False,
                "extracted_info": "",
                "error": f"{e}{suffix}",
            }
        except httpx.HTTPError as e:
            # GPT-5 sometimes rejects ``service_tier`` — drop and retry.
            if (
                "gpt-5" in model.lower() or "gpt5" in model.lower()
            ) and "service_tier" in payload:
                payload.pop("service_tier", None)
            # Retry any transient HTTP/network error within the budget.
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "extracted_info": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "extracted_info": "", "error": str(e)}

    if response is None:
        return {"success": False, "extracted_info": "", "error": "No response"}

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "extracted_info": "",
            "error": f"JSON parse error: {e}",
        }

    if "choices" in data and data["choices"]:
        try:
            return {
                "success": True,
                "extracted_info": data["choices"][0]["message"]["content"],
                "error": "",
            }
        except (KeyError, IndexError) as e:
            return {"success": False, "extracted_info": "", "error": str(e)}

    return {
        "success": False,
        "extracted_info": "",
        "error": f"Unexpected response: {data}",
    }


# ── Single-URL fetch + extract ────────────────────────────────────────


async def _fetch_single(
    url: str,
    info_to_extract: str,
    custom_headers: dict[str, str] | None = None,
) -> str:
    """Scrape one URL (Jina → fallback to direct) then run LLM extraction."""
    if any(pat in url for pat in _BANNED_URL_PATTERNS):
        return "Blocked: scraping Hugging Face datasets/spaces is not allowed."

    scrape = await _scrape_url_with_jina(url, custom_headers)
    if not scrape["success"]:
        logger.warning("Jina failed for %s: %s, trying direct", url, scrape["error"])
        scrape = await _scrape_url_with_python(url, custom_headers)
        if not scrape["success"]:
            return f"[ERROR]: Scraping failed: {scrape['error']}"

    result = await _extract_info_with_llm(scrape["content"], info_to_extract)
    if not result["success"]:
        return f"[ERROR]: Extraction failed: {result['error']}"
    return result["extracted_info"]


# ── Tool ──────────────────────────────────────────────────────────────


@tool(name="web_fetch")
async def web_fetch(
    url: str | list[str],
    info_to_extract: str | list[str],
    custom_headers: dict[str, str] | None = None,
) -> str:
    """Fetch content from a URL and extract specific types of information.

    Args:
        url: The URL to fetch, or a list of URLs to fetch in parallel
        info_to_extract: The specific types of information to extract (usually a question), or a list of extraction prompts (one per URL)
        custom_headers (Dict[str, str]): Additional headers to include in the request

    Returns:
        Extracted information as plain text. For multiple URLs, results are numbered
    """
    # Accept JSON-encoded list payloads for both fields.
    url = _ensure_list(url)
    info_to_extract = _ensure_list(info_to_extract)

    urls = url if isinstance(url, list) else [url]
    urls = [u for u in urls if u and u.strip()]
    if not urls:
        return "[ERROR]: url is required and cannot be empty."

    # info_to_extract broadcast: 1:1 if len matches, single → broadcast,
    # mismatched (>=2 != N) → join into one prompt and broadcast.
    if isinstance(info_to_extract, list):
        if len(info_to_extract) == len(urls):
            infos = info_to_extract
        elif len(info_to_extract) == 1:
            infos = info_to_extract * len(urls)
        else:
            infos = [" ".join(info_to_extract)] * len(urls)
    else:
        infos = [info_to_extract] * len(urls)

    # Dedup identical (url, info) pairs to save extraction tokens.
    seen: set[tuple[str, str]] = set()
    deduped_urls: list[str] = []
    deduped_infos: list[str] = []
    for u, info in zip(urls, infos, strict=False):
        key = (u, info)
        if key in seen:
            continue
        seen.add(key)
        deduped_urls.append(u)
        deduped_infos.append(info)
    urls = deduped_urls
    infos = deduped_infos

    try:
        results = await asyncio.gather(
            *[
                _fetch_single(u, info, custom_headers)
                for u, info in zip(urls, infos, strict=False)
            ],
        )

        # ``[N] URL: <u>\n    Info: <text>`` — same format for single & multi-URL.
        lines: list[str] = []
        for i, (u, text) in enumerate(zip(urls, results, strict=False), 1):
            lines.append(f"[{i}] URL: {u}")
            lines.append(f"    Info: {text}")
        return "\n".join(lines)

    except Exception as e:
        return f"[ERROR]: Unexpected error: {str(e)}"


__all__ = ["web_fetch"]
