"""Conservative, auditable source modifiers for individual tool-result facts.

These are provisional routing priors, never a verification of a page's claims.
Only an explicitly selected result rank can attribute a URL to a semantic fact.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from .._shared.tool_results import extract_tool_results

def _matches(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def classify_url(url: str, config: dict[str, Any] | None = None) -> tuple[str, str | None]:
    if config is None:
        from .confidence import normalize_confidence_config
        config = normalize_confidence_config()
    domains = config["source_domains"]
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    if not host or parsed.scheme not in {"http", "https"}:
        return "mixed_or_unattributed", None
    if _matches(host, set(domains["primary_official"])):
        return "primary_official", host
    if _matches(host, set(domains["scholarly_publisher"])):
        return "scholarly_publisher", host
    if _matches(host, set(domains["encyclopedia"])):
        return "encyclopedia", host
    if _matches(host, set(domains["community"])):
        return "community", host
    if _matches(host, set(domains["tabloid_or_entertainment"])):
        return "tabloid_or_entertainment", host
    return "general_web", host


def tool_source_metadata(
    node: dict[str, Any], evidence: list[dict[str, Any]] | None = None,
    raw_content: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an individual search hit or a cached page's own header."""
    if config is None:
        from .confidence import normalize_confidence_config
        config = normalize_confidence_config()
    coefficients = config["source_coefficients"]
    tool_name = node.get("tool_name")
    # The cache tools print the SourceStore header before any page text. Its
    # source_id and URL identify the same page that web_fetch previewed; a
    # numbered match in the body is not a new web_search result.
    if tool_name in {"read_source", "find_in_source"}:
        parsed_results = extract_tool_results(raw_content)
        index = node.get("tool_result_index", 0)
        if parsed_results:
            bodies = ([parsed_results[index].body] if isinstance(index, int)
                      and 0 <= index < len(parsed_results)
                      and parsed_results[index].tool_name == tool_name else [])
        else:
            bodies = [raw_content]
        if len(bodies) == 1:
            header = re.match(r"\ASource: (https?://[^\s]+)\nSource ID: (S[1-9]\d*)\n", bodies[0])
            if header:
                url = header.group(1)
                category, host = classify_url(url, config)
                if host:
                    return {"source_category": category,
                            "source_coefficient": coefficients[category],
                            "source_url": url, "source_host": host,
                            "source_id": header.group(2)}
        return {"source_category": "mixed_or_unattributed",
                "source_coefficient": coefficients["mixed_or_unattributed"]}
    items = node.get("tool_result_items") or []
    rank = node.get("source_rank")
    matches = ([item for item in items if isinstance(item, dict) and item.get("rank") == rank]
               if tool_name in {None, "web_search"} else [])
    if not isinstance(rank, int) or len(matches) != 1:
        # A web_fetch result can state its own single fetched page. This is
        # stronger attribution than an arbitrary URL in the body of a page.
        source_urls: set[str] = set()
        if node.get("tool_name") == "web_fetch":
            parsed_results = extract_tool_results(raw_content)
            index = node.get("tool_result_index", 0)
            if parsed_results and isinstance(index, int) and 0 <= index < len(parsed_results):
                bodies = [parsed_results[index].body]
            elif parsed_results:
                bodies = []  # never attribute a mixed batch to an arbitrary page
            else:
                bodies = [raw_content]
            for record in evidence or []:
                text = str(record.get("text") or "")
                if not parsed_results:
                    bodies.append(text)
            for body in bodies:
                source_urls.update(
                    re.findall(r"(?m)^Source:\s*(https?://[^\s]+)", body)
                )
        if len(source_urls) == 1:
            url = next(iter(source_urls))
            category, host = classify_url(url, config)
            if host:
                return {"source_category": category,
                        "source_coefficient": coefficients[category],
                        "source_url": url, "source_host": host}
        return {"source_category": "mixed_or_unattributed",
                "source_coefficient": coefficients["mixed_or_unattributed"]}
    url = matches[0].get("url")
    category, host = classify_url(url if isinstance(url, str) else "", config)
    result: dict[str, Any] = {"source_category": category,
                              "source_coefficient": coefficients[category],
                              "source_rank": rank}
    if host:
        result.update(source_url=url, source_host=host)
    return result
