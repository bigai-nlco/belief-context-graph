"""Real web search backed by the Serper Google Search API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from bcg.env import load_project_env
from rllm.tools.tool_base import Tool, ToolOutput


_DEFAULT_ENDPOINT = "https://google.serper.dev/search"
_MISSING_KEY_VALUES = {
    "",
    "changeme",
    "replace-me",
    "replace_me",
    "your-api-key",
    "your_api_key",
    "put-your-key-here",
}


def resolve_serper_api_key(
    api_key: str | None = None,
) -> str:
    """Resolve a key with priority constructor > root .env environment."""

    load_project_env()
    value = api_key if api_key is not None else os.environ.get("SERPER_API_KEY")
    value = str(value or "").strip()
    if value.lower() in _MISSING_KEY_VALUES:
        return ""
    return value


class SerperSearchTool(Tool):
    """Search the live public web through ``google.serper.dev``."""

    NAME = "serper_search"
    FEEDS_MEMORY = True
    DESCRIPTION = (
        "Search the live web with Google via Serper and return source titles, URLs, "
        "and result snippets. Use follow-up queries to verify important facts."
    )

    def __init__(
        self,
        name: str = NAME,
        description: str | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        country: str | None = None,
        language: str | None = None,
        max_results: int = 10,
        max_output_chars: int = 12000,
        timeout: float = 30.0,
        http_open: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = resolve_serper_api_key(api_key)
        self.endpoint = str(
            endpoint or os.environ.get("SERPER_ENDPOINT") or _DEFAULT_ENDPOINT
        ).strip()
        self.country = str(
            country if country is not None else os.environ.get("SERPER_COUNTRY", "us")
        ).strip()
        self.language = str(
            language if language is not None else os.environ.get("SERPER_LANGUAGE", "en")
        ).strip()
        self.max_results = max(1, min(int(max_results), 20))
        self.max_output_chars = max(1000, int(max_output_chars))
        self.timeout = max(0.1, float(timeout))
        self._http_open = http_open or urllib.request.urlopen
        super().__init__(name=name, description=description or self.DESCRIPTION)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A focused Google web-search query. Include names, dates, "
                                "domains, or quoted phrases when useful."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": (
                                f"Number of search results to return (default: "
                                f"{self.max_results}, maximum: 20)."
                            ),
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def usage_prompt(self, hyde: bool = False, detail: bool = True) -> str:
        if not detail:
            return (
                "- serper_search: search the live web and return titles, URLs, and "
                "snippets. Call as:\n"
                "  <tool_call>\n"
                '  {"name": "serper_search", "arguments": {"query": "..."}}\n'
                "  </tool_call>"
            )
        return (
            "- serper_search(query: string, top_k?: integer): search the live web via "
            "Google/Serper. Use a focused natural-language query, inspect the returned "
            "source URLs and snippets, and issue follow-up searches to cross-check facts. "
            "A snippet is evidence from a search result, not a guarantee that the source "
            "supports every inferred claim.\n"
            "  Example:\n"
            "  <tool_call>\n"
            '  {"name": "serper_search", "arguments": {"query": "site:usgs.gov '
            'clown anemonefish nonindigenous Florida"}}\n'
            "  </tool_call>"
        )

    def forward(self, query: str, top_k: int | None = None) -> ToolOutput:
        query = str(query or "").strip()
        if not query:
            return ToolOutput(name=self.name, error="query must be a non-empty string")
        if not self.api_key:
            return ToolOutput(
                name=self.name,
                error=(
                    "SERPER_API_KEY is not configured. Add it to the project-root "
                    ".env or export SERPER_API_KEY."
                ),
            )

        result_limit = self.max_results if top_k is None else max(1, min(int(top_k), 20))
        payload: dict[str, Any] = {"q": query, "num": result_limit}
        if self.country:
            payload["gl"] = self.country
        if self.language:
            payload["hl"] = self.language
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "BeliefTracer/1.0",
            },
            method="POST",
        )

        try:
            with self._http_open(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500].strip()
            except Exception:
                pass
            suffix = f": {detail}" if detail else ""
            return ToolOutput(
                name=self.name,
                error=f"Serper request failed with HTTP {exc.code}{suffix}",
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return ToolOutput(name=self.name, error=f"Serper request failed: {exc}")

        try:
            response_payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return ToolOutput(name=self.name, error=f"Serper returned invalid JSON: {exc}")
        if not isinstance(response_payload, dict):
            return ToolOutput(name=self.name, error="Serper returned a non-object response")
        if response_payload.get("message") and not response_payload.get("organic"):
            return ToolOutput(
                name=self.name,
                error=f"Serper API error: {response_payload.get('message')}",
            )

        evidences = self._extract_evidences(response_payload, result_limit)
        output = self._format_evidences(evidences)
        return ToolOutput(
            name=self.name,
            output=output,
            metadata={
                "provider": "serper",
                "endpoint": self.endpoint,
                "query": query,
                "num_results": len(evidences),
                "search_parameters": {
                    "country": self.country,
                    "language": self.language,
                    "top_k": result_limit,
                },
                "evidences": evidences,
            },
        )

    @staticmethod
    def _extract_evidences(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        candidates: list[dict[str, str]] = []
        answer_box = payload.get("answerBox")
        if isinstance(answer_box, dict):
            answer_text = (
                answer_box.get("answer")
                or answer_box.get("snippet")
                or answer_box.get("result")
            )
            if answer_text:
                candidates.append(
                    {
                        "source_type": "answer_box",
                        "title": str(answer_box.get("title") or "Google answer box"),
                        "url": str(answer_box.get("link") or ""),
                        "text": str(answer_text),
                    }
                )

        knowledge = payload.get("knowledgeGraph")
        if isinstance(knowledge, dict):
            description = str(knowledge.get("description") or "")
            attributes = knowledge.get("attributes")
            if isinstance(attributes, dict) and attributes:
                attribute_text = "; ".join(
                    f"{key}: {value}" for key, value in attributes.items()
                )
                description = "\n".join(x for x in (description, attribute_text) if x)
            if description:
                candidates.append(
                    {
                        "source_type": "knowledge_graph",
                        "title": str(knowledge.get("title") or "Google knowledge graph"),
                        "url": str(knowledge.get("website") or ""),
                        "text": description,
                    }
                )

        organic = payload.get("organic")
        if isinstance(organic, list):
            for item in organic:
                if not isinstance(item, dict):
                    continue
                snippet = item.get("snippet") or item.get("description")
                if not snippet:
                    continue
                candidates.append(
                    {
                        "source_type": "organic",
                        "title": str(item.get("title") or "Untitled result"),
                        "url": str(item.get("link") or ""),
                        "text": str(snippet),
                    }
                )

        evidences: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (candidate["url"].strip(), candidate["text"].strip())
            if key in seen:
                continue
            seen.add(key)
            title = candidate["title"].strip()
            url = candidate["url"].strip()
            snippet = candidate["text"].strip()
            text_parts = [f"Title: {title}"]
            if url:
                text_parts.append(f"URL: {url}")
            text_parts.append(f"Snippet: {snippet}")
            evidences.append(
                {
                    "rank": len(evidences) + 1,
                    "source_type": candidate["source_type"],
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    # The shared tool-result protocol renders only `text`, so
                    # embed source identity here as well as keeping structured
                    # title/url fields for archive and result consumers.
                    "text": "\n".join(text_parts),
                }
            )
            if len(evidences) >= limit:
                break
        return evidences

    def _format_evidences(self, evidences: list[dict[str, Any]]) -> str:
        if not evidences:
            return "No web search results found. Try a broader or differently phrased query."
        blocks: list[str] = []
        for evidence in evidences:
            lines = [f"[{evidence['rank']}] {evidence['title']}"]
            if evidence.get("url"):
                lines.append(f"URL: {evidence['url']}")
            lines.append(f"Snippet: {evidence.get('snippet') or evidence['text']}")
            blocks.append("\n".join(lines))
        rendered = "\n\n".join(blocks)
        if len(rendered) > self.max_output_chars:
            rendered = rendered[: self.max_output_chars].rstrip() + "\n[results truncated]"
        return rendered


__all__ = [
    "SerperSearchTool",
    "resolve_serper_api_key",
]
