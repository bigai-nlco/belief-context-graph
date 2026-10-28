"""Evidence and consistency guards for search judgments and grouped findings."""
from __future__ import annotations

import re

from tools.ai_review.schema import Finding, Summary


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def validate_concise_summary(value: Summary, pool: dict) -> None:
    # This catches copied passages, not semantic equivalence. Semantic grouping
    # remains a model judgment with original occurrences retained for inspection.
    narratives = [normalized(pool[ref][key]) for group in value.process_groups for ref in group.member_ids
                  for key in ("expected", "observed", "impact")]
    short_sections = [r.note for r in value.ratings] + [value.conclusion.benefit, value.conclusion.problem]
    for text in short_sections:
        compact = normalized(text)
        if len(compact) >= 40 and any(compact in original or (len(original) >= 40 and original in compact)
                                     for original in narratives):
            raise ValueError("ratings and conclusion must reference findings, not copy their analysis")
    conclusions = [normalized(text) for key, text in value.conclusion.model_dump().items() if key != "evidence"]
    for index, text in enumerate(conclusions):
        if len(text) >= 40 and text in conclusions[:index]:
            raise ValueError("conclusion fields must not repeat the same passage")


def validate_search(finding: Finding, step: dict) -> None:
    review = finding.search_assessment
    if review is None:
        if "A2" in finding.dimensions:
            raise ValueError("A2 requires an explicit search_assessment")
        return
    if any(not getattr(review, key).strip() for key in ("information_need", "expected_gain", "basis")):
        raise ValueError("search assessment requires a concrete need, expected gain and basis")
    results = step["tool_results"]
    if review.result_quote and (len(review.result_quote.strip()) < 8 or review.result_quote not in results["text"]):
        raise ValueError("result_quote must exactly quote the current tool results")
    if review.observed_gain != "不可见" and not review.result_quote.strip():
        raise ValueError("observed search gain requires a tool result quote")
    if review.verdict == "合理核验" and review.verification_reason in {"无", "不明"}:
        raise ValueError("reasonable verification requires a specific verification reason")
    if review.verdict == "无效重复" and (review.graph_sufficiency != "充分" or review.verification_reason != "无"):
        raise ValueError("redundant search requires sufficient Graph evidence and no verification gap")
    if review.verdict in {"无效重复", "无效工具重试"}:
        if review.observed_gain != "无新增证据" or results["truncated"]:
            raise ValueError("negative search verdict requires complete results with no new evidence")
        if finding.kind == "benefit" or finding.strength == "证据不足":
            raise ValueError("negative search verdict cannot be a benefit or have insufficient evidence")
    if review.verdict == "无效工具重试":
        # Only pre-action evidence counts. A first failure cannot retroactively
        # establish that the Agent already knew the tool was unavailable.
        before = [step["action_text"]["text"]]
        before += [p[key]["text"] for p in step["previous_calls"] for key in ("action_text", "tool_results")]
        if len(review.unavailable_quote.strip()) < 8 or not any(review.unavailable_quote in text for text in before):
            raise ValueError("unavailable_quote must quote evidence known before the current tool result")
    elif review.unavailable_quote:
        raise ValueError("unavailable_quote is only used for knowingly repeating unavailable tools")
    if review.verdict in {"合理核验", "新增信息搜索", "证据不足"} and "A2" in finding.dimensions and finding.kind != "benefit":
        raise ValueError("reasonable or uncertain search cannot be classified as a negative A2 finding")
    if review.verdict == "证据不足" and finding.strength != "证据不足":
        raise ValueError("uncertain search cannot claim strong evidence")


def validate_groups(value: Summary, pool: dict) -> dict[str, set[str]]:
    if [g.representative_id for g in value.process_groups] != value.selected_finding_ids:
        raise ValueError("process groups must match selected representatives in priority order")
    seen: set[str] = set()
    dimensions: dict[str, set[str]] = {}
    for group in value.process_groups:
        members = set(group.member_ids)
        if not group.reason.strip() or group.representative_id not in members:
            raise ValueError("each group needs a reason and must include its representative")
        if len(members) != len(group.member_ids) or members & seen or not members <= pool.keys():
            raise ValueError("group members must be unique, existing and assigned to one group only")
        seen.update(members)
        kinds = {pool[ref]["kind"] for ref in members}
        verdicts = {(pool[ref].get("search_assessment") or {}).get("verdict") for ref in members}
        if len(kinds) != 1 or len(verdicts) != 1:
            raise ValueError("cannot group benefits with problems or different search verdicts")
        dimensions[group.representative_id] = {d for ref in members for d in pool[ref]["dimensions"]}
    return dimensions
