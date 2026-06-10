"""Web search tool — Serper-backed, plain-text output.

Tool behaviour:

- Tool ``name`` is ``"web_search"``.
- Banlist is ``unifuncs`` + ``huggingface.co/datasets|spaces`` only.
- Output is plain text ``[i] Title: ...\\n    Date: ...\\n    Snippet:
  ...\\n    URL: ...`` — no Markdown ``##`` headers, no ``answerBox``,
  no ``knowledgeGraph``.
- Multi-query calls merge organic results into one numbered list.
- Empty result + quoted query → strip ``"`` and retry once.

Used by :mod:`workflows.react_base`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib.parse import unquote

import httpx
from agent_harness.core.tool import tool
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# Read env at call time — profile loader may run load_dotenv after import.
def _serper_base_url() -> str:
    return os.getenv("SERPER_BASE_URL", "https://google.serper.dev")


def _serper_api_key() -> str:
    return os.getenv("SERPER_API_KEY", "")


_BANNED_URL_PATTERNS: tuple[str, ...] = (
    "unifuncs",
    "huggingface.co/datasets",
    "huggingface.co/spaces",
)

# RFC 3986 reserved encodings preserved during URL unquote — decoding any of
# these would alter URL semantics. Used to keep dedup keys stable.
_RESERVED_PERCENT_ENCODINGS: frozenset[str] = frozenset({
    "%2f", "%2F", "%3f", "%3F", "%23", "%26", "%3d", "%3D", "%40",
    "%3a", "%3A", "%5b", "%5B", "%5d", "%5D", "%21", "%24", "%27",
    "%28", "%29", "%2a", "%2A", "%2b", "%2B", "%2c", "%2C", "%3b",
    "%3B", "%25", "%20",
})


def _safe_unquote(url: str) -> str:
    """Decode percent-encoded URL chars without breaking URL structure.

    Reserved chars (``/``, ``?``, ``#``, ``&``, ``=``, etc.) stay
    encoded so dedup keys and request URLs match exactly.
    """
    if not url:
        return url

    out: list[str] = []
    i, n = 0, len(url)
    while i < n:
        if url[i] == "%" and i + 2 < n:
            hex_chars = url[i + 1 : i + 3]
            if all(c in "0123456789ABCDEFabcdef" for c in hex_chars):
                triple = url[i : i + 3]
                if triple in _RESERVED_PERCENT_ENCODINGS:
                    out.append(triple)
                    i += 3
                    continue
                # Collect consecutive %XX bytes (UTF-8 multi-byte handling).
                seq = triple
                j = i + 3
                while j + 2 < n and url[j] == "%":
                    nxt_hex = url[j + 1 : j + 3]
                    if not all(c in "0123456789ABCDEFabcdef" for c in nxt_hex):
                        break
                    nxt = url[j : j + 3]
                    if nxt in _RESERVED_PERCENT_ENCODINGS:
                        break
                    seq += nxt
                    j += 3
                try:
                    out.append(unquote(seq))
                    i = j
                    continue
                except Exception:
                    out.append(triple)
                    i += 3
                    continue
        out.append(url[i])
        i += 1
    return "".join(out)


def _decode_http_urls_in_obj(data: Any) -> Any:
    """Recursively unquote http(s) URL strings in nested dict/list payloads."""
    if isinstance(data, str):
        if "%" in data and "http" in data:
            return _safe_unquote(data)
        return data
    if isinstance(data, list):
        return [_decode_http_urls_in_obj(item) for item in data]
    if isinstance(data, dict):
        return {k: _decode_http_urls_in_obj(v) for k, v in data.items()}
    return data


def _ensure_list(val: Any) -> Any:
    """Unwrap doubly-serialised JSON list payloads (``'["a","b"]'`` → ``["a","b"]``)."""
    if isinstance(val, str) and val.startswith("["):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return val


def _is_banned_url(url: str) -> bool:
    if not url:
        return False
    return any(pat in url for pat in _BANNED_URL_PATTERNS)


def _is_retryable_serper_error(exc: BaseException) -> bool:
    """Retry only transient failures.

    Serper returns useful 4xx JSON bodies for account/config problems. Retrying
    those hides the real error behind tenacity's RetryError wrapper.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status in {408, 409, 425, 429}
    return False


def _serper_error_message(response: httpx.Response) -> str:
    """Extract a concise, user-safe error message from a Serper response."""
    text = response.text.strip()
    if not text:
        return response.reason_phrase or "empty error response"

    try:
        data = response.json()
    except ValueError:
        return text[:500]

    if isinstance(data, dict):
        for key in ("message", "error", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
            if value:
                return json.dumps(value, ensure_ascii=False)[:500]

    return text[:500]


def _format_serper_status_error(exc: httpx.HTTPStatusError) -> str:
    response = exc.response
    status = f"{response.status_code} {response.reason_phrase}".strip()
    return f"[ERROR]: Serper request failed ({status}): {_serper_error_message(response)}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(_is_retryable_serper_error),
    reraise=True,
)
async def _serper_request(
    payload: dict[str, Any], headers: dict[str, str],
) -> httpx.Response:
    """One Serper POST. ``raise_for_status()`` feeds non-2xx into the tenacity retry chain."""
    base_url = _serper_base_url().rstrip("/")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/search", json=payload, headers=headers,
        )
        resp.raise_for_status()
        return resp


async def _search_single_query(
    query: str,
    gl: str,
    hl: str,
    location: str | None,
    num: int | None,
    tbs: str | None,
    page: int | None,
    autocorrect: bool | None,
) -> list[dict]:
    """Run one Serper search; transparent quote-stripping retry on empty results."""
    payload: dict[str, Any] = {"q": query.strip(), "gl": gl, "hl": hl}
    if location:
        payload["location"] = location
    payload["num"] = num if num is not None else 10
    if tbs:
        payload["tbs"] = tbs
    if page is not None:
        payload["page"] = page
    if autocorrect is not None:
        payload["autocorrect"] = autocorrect

    headers = {"X-API-KEY": _serper_api_key(), "Content-Type": "application/json"}

    resp = await _serper_request(payload, headers)
    data = _decode_http_urls_in_obj(resp.json())
    organic = [
        item for item in data.get("organic", [])
        if not _is_banned_url(item.get("link", ""))
    ]

    # Empty + contains quotes → strip quotes and try once more. Helps with
    # over-specific quoted phrase queries that Google returns nothing for.
    if not organic and '"' in query:
        clean_q = query.replace('"', "").strip()
        if clean_q:
            payload["q"] = clean_q
            resp = await _serper_request(payload, headers)
            data = _decode_http_urls_in_obj(resp.json())
            organic = [
                item for item in data.get("organic", [])
                if not _is_banned_url(item.get("link", ""))
            ]

    return organic


def _format_results_plaintext(organic_results: list[dict]) -> str:
    """Plain-text format. Indentation & field labels are load-bearing."""
    lines: list[str] = []
    for i, item in enumerate(organic_results, 1):
        lines.append(f"[{i}] Title: {item.get('title', '')}")
        if item.get("date"):
            lines.append(f"    Date: {item['date']}")
        if item.get("snippet"):
            lines.append(f"    Snippet: {item['snippet']}")
        lines.append(f"    URL: {item.get('link', '')}")
    return "\n".join(lines)


@tool(name="web_search")
async def web_search(
    q: str | list[str],
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    tbs: str | None = None,
    page: int | None = None,
    autocorrect: bool | None = None,
) -> str:
    """Perform web searches and retrieve rich results.

    Args:
        q: Search query string, or a list of query strings to execute in parallel
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 10)
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)
        page: Page number of results to return (default: 1)
        autocorrect: Whether to autocorrect spelling in query

    Returns:
        Numbered plain-text list of search results, each with Title, Snippet, and URL
    """
    if not _serper_api_key():
        return "[ERROR]: SERPER_API_KEY environment variable not set."

    # Accept Union[str, List[str]] and JSON-encoded list payloads.
    q = _ensure_list(q)
    queries = q if isinstance(q, list) else [q]
    queries = [qry for qry in queries if qry and qry.strip()]
    if not queries:
        return "[ERROR]: Search query 'q' is required and cannot be empty."

    try:
        all_organic: list[list[dict]] = await asyncio.gather(
            *[
                _search_single_query(
                    qry, gl, hl, location, num, tbs, page, autocorrect,
                )
                for qry in queries
            ],
        )

        # Multi-query: merge into one numbered list, dedup by URL.
        # Empty-URL items are kept (short-circuited on the dedup check).
        merged: list[dict] = []
        seen_urls: set[str] = set()
        for results in all_organic:
            for item in results:
                url = item.get("link", "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged.append(item)

        if not merged:
            return "No search results found."

        return _format_results_plaintext(merged)

    except httpx.HTTPStatusError as e:
        return _format_serper_status_error(e)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return f"[ERROR]: Serper request failed after retries: {str(e)}"
    except Exception as e:
        return f"[ERROR]: Unexpected error: {str(e)}"


__all__ = ["web_search"]
