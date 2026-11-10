"""Conservative, auditable source modifiers for individual tool-result facts.

These are provisional routing priors, never a verification of a page's claims.
Only an explicitly selected result rank can attribute a URL to a semantic fact.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from .._shared.tool_results import extract_tool_results

COEFFICIENTS = {
    "primary_official": 0.88,
    "scholarly_publisher": 0.78,
    "encyclopedia": 0.68,
    "general_web": 0.58,
    "community": 0.42,
    "tabloid_or_entertainment": 0.38,
    "mixed_or_unattributed": 0.50,
}

PRIMARY_DOMAINS = {
    "cboe.com", "sec.gov", "fred.stlouisfed.org", "unesco.org",
    "openai.com", "who.int", "cdc.gov", "data.gov", "worldbank.org",
}
SCHOLARLY_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "arxiv.org",
    "journals.plos.org", "sciencedirect.com", "link.springer.com",
    "tandfonline.com", "dl.acm.org", "mdpi.com",
}
ENCYCLOPEDIA_DOMAINS = {"wikipedia.org", "britannica.com", "investopedia.com"}
COMMUNITY_DOMAINS = {"reddit.com", "quora.com", "researchgate.net"}
TABLOID_DOMAINS = {"tmz.com", "pagesix.com", "perezhilton.com"}


def _matches(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def classify_url(url: str) -> tuple[str, str | None]:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    if not host or parsed.scheme not in {"http", "https"}:
        return "mixed_or_unattributed", None
    if _matches(host, PRIMARY_DOMAINS):
        return "primary_official", host
    if _matches(host, SCHOLARLY_DOMAINS):
        return "scholarly_publisher", host
    if _matches(host, ENCYCLOPEDIA_DOMAINS):
        return "encyclopedia", host
    if _matches(host, COMMUNITY_DOMAINS):
        return "community", host
    if _matches(host, TABLOID_DOMAINS):
        return "tabloid_or_entertainment", host
    return "general_web", host


def tool_source_metadata(
    node: dict[str, Any], evidence: list[dict[str, Any]] | None = None,
    raw_content: str = "",
) -> dict[str, Any]:
    """Resolve one fact's declared result rank, never guess from a mixed list."""
    items = node.get("tool_result_items") or []
    rank = node.get("source_rank")
    matches = [item for item in items if isinstance(item, dict) and item.get("rank") == rank]
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
            category, host = classify_url(url)
            if host:
                return {"source_category": category,
                        "source_coefficient": COEFFICIENTS[category],
                        "source_url": url, "source_host": host}
        return {"source_category": "mixed_or_unattributed",
                "source_coefficient": COEFFICIENTS["mixed_or_unattributed"]}
    url = matches[0].get("url")
    category, host = classify_url(url if isinstance(url, str) else "")
    result: dict[str, Any] = {"source_category": category,
                              "source_coefficient": COEFFICIENTS[category],
                              "source_rank": rank}
    if host:
        result.update(source_url=url, source_host=host)
    return result
