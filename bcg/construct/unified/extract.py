"""
extract.py
==========
Single-call incremental graph update for the streaming engine.

`update_graph(...)` runs one LLM call per turn. The model returns:
  * new belief nodes,
  * new decision nodes, and
  * typed relations among new nodes and existing nodes.

Valid relation types: depends_on, supplements, contradicts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .._shared.tool_queries import (
    extract_pure_tool_calls,
    extract_tool_calls,
    rule_tool_call_belief,
    strip_valid_tool_calls,
)
from .._shared.tool_results import (
    compact_tool_result,
    extract_tool_results,
    parse_tool_result,
)
from . import llm
from .prompts import (
    build_assistant_tool_result_extraction_prompt,
    build_layered_relation_extraction_prompt,
    build_node_extraction_prompt,
    build_relation_extraction_prompt,
    build_update_prompt,
    format_clustered_sentences_for_prompt,
    format_sentences_for_prompt,
)

VALID_STANCES = {"asserted", "recalled", "speculated", "judged"}
VALID_RELATION_TYPES = {"depends_on", "supplements", "contradicts"}


def _clean_str(v: Any) -> str | None:
    if (
        isinstance(v, str)
        and v.strip()
        and v.strip().lower() not in ("null", "none", "n/a")
    ):
        return v.strip()
    return None


def _clean_stance(value: Any) -> str:
    stance = str(value or "").strip().lower()
    return stance if stance in VALID_STANCES else "asserted"


def _clean_entities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    entities: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            continue
        entity = raw.strip()
        if not entity or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
    return entities


def _clean_node(
    raw: Any,
    mode: str,
    n_sentences: int,
    ordinal: int,
    *,
    node_type: str,
    preserve_model_tmp_id: bool = False,
) -> dict[str, Any] | None:
    """Validate / coerce one belief or decision object from the model."""
    if not isinstance(raw, dict):
        return None

    text_key = "decision" if node_type == "decision" else "belief"
    text = raw.get(text_key)
    if not isinstance(text, str) or not text.strip():
        # Allow decision objects that used the historical `belief` key by mistake.
        if node_type == "decision" and isinstance(raw.get("belief"), str):
            text = raw.get("belief")
        else:
            return None

    # Two-phase extraction does not need the model to invent temporary ids:
    # relations are generated later against code-owned global ids.  Keep the
    # model-provided value only for the legacy single-response node+edge API,
    # where relations in that same response must reference temporary ids.
    prefix = "d" if node_type == "decision" else "n"
    tmp = f"{prefix}{ordinal}"
    if preserve_model_tmp_id:
        model_tmp = raw.get("tmp_id")
        if isinstance(model_tmp, str) and model_tmp.strip():
            tmp = model_tmp.strip()

    primary_text_key = "decision" if node_type == "decision" else "belief"
    out: dict[str, Any] = {
        "tmp_id": tmp,
        "node_type": node_type,
        primary_text_key: text.strip(),
        "stance": _clean_stance(raw.get("stance")),
        "entities": _clean_entities(raw.get("entities")),
    }
    query = _clean_str(raw.get("query"))
    tool_name = _clean_str(raw.get("tool_name"))
    if query is not None:
        out["query"] = query
    if tool_name is not None:
        out["tool_name"] = tool_name

    if mode != "excerpt":
        idx_in = raw.get("supporting_sentence_indices")
        indices: list[int] | None = None
        if isinstance(idx_in, list):
            cleaned = sorted(
                {
                    int(i)
                    for i in idx_in
                    if isinstance(i, (int, float)) and 0 <= int(i) < n_sentences
                }
            )
            if cleaned:
                indices = cleaned
        out["supporting_sentence_indices"] = indices
    else:
        excerpts_in = raw.get("supporting_excerpts") or []
        excerpts = [e.strip() for e in excerpts_in if isinstance(e, str) and e.strip()]
        if not excerpts:
            return None
        out["supporting_excerpts"] = excerpts
    return out


def _attach_and_complete_tool_call_nodes(
    beliefs: list[dict[str, Any]],
    *,
    role: str,
    content: str,
    mode: str,
    sentences: list[str],
    seen_tmp: set[str],
    ordinal: int,
) -> int:
    """Guarantee one code-owned node per tool call in a mixed Assistant turn.

    The model decides the semantic wording. The source tool-call JSON remains
    authoritative for ``tool_name`` and ``query``; unmatched model metadata is
    discarded, and omitted query nodes are synthesized deterministically.
    """

    if role.strip().lower() != "assistant":
        for node in beliefs:
            node.pop("query", None)
            node.pop("tool_name", None)
        return ordinal

    calls = extract_tool_calls(content)
    unmatched = list(enumerate(calls))

    for node in beliefs:
        model_query = _clean_str(node.get("query"))
        belief_text = str(node.get("belief") or "")
        match_position: int | None = None
        for position, (_, call) in enumerate(unmatched):
            if model_query == call.query or (call.query and call.query in belief_text):
                match_position = position
                break
        if match_position is None:
            node.pop("query", None)
            node.pop("tool_name", None)
            continue
        call_index, call = unmatched.pop(match_position)
        # Tool use is represented as an ordinary belief with code-owned
        # natural-language text. ``query`` remains metadata; it is never a
        # separate graph node type.
        node["belief"] = rule_tool_call_belief(call)
        node["tool_name"] = call.name
        node["tool_arguments"] = dict(call.arguments)
        node["tool_call_index"] = call_index
        node["extraction_method"] = "rule_tool_call"
        if call.tool_call_id is not None:
            node["tool_call_id"] = call.tool_call_id
        if call.query is not None:
            node["query"] = call.query
        else:
            node.pop("query", None)

    for call_index, call in unmatched:
        while f"n{ordinal}" in seen_tmp:
            ordinal += 1
        node: dict[str, Any] = {
            "tmp_id": f"n{ordinal}",
            "node_type": "belief",
            "belief": rule_tool_call_belief(call),
            "stance": "asserted",
            "entities": [call.name],
            "tool_name": call.name,
            "tool_arguments": dict(call.arguments),
            "tool_call_index": call_index,
            "extraction_method": "rule_tool_call",
        }
        if call.tool_call_id is not None:
            node["tool_call_id"] = call.tool_call_id
        if call.query is not None:
            node["query"] = call.query
        if mode == "excerpt":
            node["supporting_excerpts"] = [call.excerpt]
        else:
            supporting = [
                index
                for index, sentence in enumerate(sentences)
                if (call.query is not None and call.query in sentence)
                or (sentence.strip() and sentence.strip() in call.excerpt)
            ]
            node["supporting_sentence_indices"] = supporting or None
        beliefs.append(node)
        seen_tmp.add(node["tmp_id"])
        ordinal += 1
    return ordinal


def _rule_extract_tool_call_nodes(
    *,
    role: str,
    content: str,
    mode: str,
    sentences: list[str],
) -> dict[str, Any] | None:
    """Return complete model-shaped nodes for a pure tool-call assistant turn."""

    if role.strip().lower() != "assistant":
        return None
    calls = extract_pure_tool_calls(content)
    if calls is None:
        return None

    nodes: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        node: dict[str, Any] = {
            "tmp_id": f"n{index}",
            "node_type": "belief",
            "belief": rule_tool_call_belief(call),
            "stance": "asserted",
            "entities": [call.name],
            "tool_name": call.name,
            "tool_arguments": dict(call.arguments),
            "tool_call_index": index,
            "extraction_method": "rule_tool_call",
        }
        if call.tool_call_id is not None:
            node["tool_call_id"] = call.tool_call_id
        if call.query is not None:
            node["query"] = call.query
        if mode == "excerpt":
            node["supporting_excerpts"] = [call.excerpt]
        else:
            supporting = [
                sentence_index
                for sentence_index, sentence in enumerate(sentences)
                if sentence.strip() and sentence.strip() in call.excerpt
            ]
            node["supporting_sentence_indices"] = supporting or None
        nodes.append(node)

    raw_output = json.dumps(
        {"extraction_method": "rule_tool_call", "beliefs": nodes, "decisions": []},
        ensure_ascii=False,
    )
    return {
        "nodes": nodes,
        "beliefs": nodes,
        "decisions": [],
        "relations": [],
        "raw_output": raw_output,
        "skipped": False,
        "extraction_method": "rule_tool_call",
    }


def extract_rule_tool_result_nodes(
    *,
    role: str,
    content: str,
    mode: str,
    sentences: list[str] | None = None,
    max_results: int = 5,
    max_snippet_chars: int = 240,
) -> dict[str, Any] | None:
    """Create one bounded, provenance-rich node without calling a model.

    Canonical ID-bearing result groups always use this path so parallel results
    retain exact call provenance. Legacy result strings remain controlled by
    the configured streaming construction policy.
    """

    if role.strip().lower() not in {"tool", "function"}:
        return None
    parsed_results = extract_tool_results(content)
    grouped_wire = bool(parsed_results)
    if not parsed_results:
        parsed = parse_tool_result(content)
        parsed_results = [parsed] if parsed is not None else []
    if not parsed_results:
        return None
    nodes: list[dict[str, Any]] = []
    for index, parsed in enumerate(parsed_results):
        belief, result_items, entities = compact_tool_result(
            parsed,
            max_results=max_results,
            max_snippet_chars=max_snippet_chars,
        )
        node: dict[str, Any] = {
            "tmp_id": f"n{index}",
            "node_type": "belief",
            "belief": belief,
            "stance": "recalled",
            "entities": entities,
            "tool_name": parsed.tool_name,
            "tool_result_count": len(parsed.results),
            "tool_result_items": result_items,
            "tool_result_truncated_count": max(
                0, len(parsed.results) - len(result_items)
            ),
            "tool_result_index": index,
            "extraction_method": "rule_tool_result",
        }
        if parsed.tool_call_id is not None:
            node["tool_call_id"] = parsed.tool_call_id
        if mode == "excerpt":
            node["supporting_excerpts"] = [
                (parsed.body or content) if grouped_wire else content
            ]
        else:
            node["supporting_sentence_indices"] = (
                list(range(len(sentences or []))) or None
            )
        nodes.append(node)
    raw_output = json.dumps(
        {"extraction_method": "rule_tool_result", "beliefs": nodes, "decisions": []},
        ensure_ascii=False,
    )
    return {
        "nodes": nodes,
        "beliefs": nodes,
        "decisions": [],
        "relations": [],
        "raw_output": raw_output,
        "skipped": False,
        "extraction_method": "rule_tool_result",
    }


def extract_compact_tool_result_nodes(
    client,
    model: str,
    *,
    role: str,
    content: str,
    mode: str,
    query: str | None,
    sentences: list[str] | None = None,
    max_results: int = 10,
    max_snippet_chars: int = 240,
    max_facts: int = 3,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any] | None:
    """Distill one structured tool result with a small, history-free prompt.

    Empty and unparseable results stay on the zero-LLM rule path. A failed or
    empty model response also falls back to the deterministic compact node.
    """

    rule_result = extract_rule_tool_result_nodes(
        role=role,
        content=content,
        mode=mode,
        sentences=sentences,
        max_results=max_results,
        max_snippet_chars=max_snippet_chars,
    )
    if rule_result is None:
        return None
    rule_node = rule_result["nodes"][0]
    if int(rule_node.get("tool_result_count") or 0) == 0:
        return rule_result

    result_items = list(rule_node.get("tool_result_items") or [])
    # Full URLs and source metadata remain on the graph node/evidence. Titles
    # plus bounded snippets carry the extraction semantics at lower token cost.
    prompt_items = [
        {key: item[key] for key in ("rank", "title", "snippet") if key in item}
        for item in result_items
    ]
    prompt = f"""Extract at most {max(1, int(max_facts))} self-contained facts useful for this query:
{query or "(query unavailable)"}

Rules:
- Use only these results; preserve exact names, dates, quantities, and versions.
- Every fact must name its specific subject or source title. Never emit a generic
  claim such as "a tower" or combine claims from different results.
- Skip irrelevant results and do not infer facts that the title/snippet does not state.
- Prefer answer-relevant facts that prevent repeating this search.
- For each fact, classify stance as asserted, recalled, speculated, or judged.
- For each fact, list only specific salient entities explicitly present in it.
- JSON only: {{"beliefs":[{{"belief":"...","stance":"asserted","entities":["..."]}}]}}.

Results:
{json.dumps(prompt_items, ensure_ascii=False)}
"""
    try:
        raw = llm.call_model(
            client,
            model,
            prompt,
            max_tokens=min(int(max_tokens or 768), 768),
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        return rule_result
    parsed = llm.parse_json_response(raw)
    raw_beliefs = parsed.get("beliefs") if isinstance(parsed, dict) else None
    if not isinstance(raw_beliefs, list):
        return rule_result

    nodes: list[dict[str, Any]] = []
    for raw_node in raw_beliefs[: max(1, int(max_facts))]:
        if not isinstance(raw_node, dict):
            continue
        belief = _clean_str(raw_node.get("belief"))
        if not belief:
            continue
        node: dict[str, Any] = {
            "tmp_id": f"n{len(nodes)}",
            "node_type": "belief",
            "belief": belief,
            "stance": _clean_stance(raw_node.get("stance")),
            "entities": _clean_entities(raw_node.get("entities")),
            "tool_name": rule_node.get("tool_name") or "tool",
            "tool_result_count": rule_node.get("tool_result_count") or 0,
            "tool_result_items": result_items,
            "tool_result_truncated_count": rule_node.get("tool_result_truncated_count")
            or 0,
            "extraction_method": "compact_llm_tool_result",
        }
        if rule_node.get("tool_call_id"):
            node["tool_call_id"] = str(rule_node["tool_call_id"])
        node["tool_result_index"] = int(rule_node.get("tool_result_index") or 0)
        node["tool_result_fact_index"] = len(nodes)
        if mode == "excerpt":
            node["supporting_excerpts"] = [content]
        else:
            node["supporting_sentence_indices"] = (
                list(range(len(sentences or []))) or None
            )
        nodes.append(node)
    if not nodes:
        return rule_result
    return {
        "nodes": nodes,
        "beliefs": nodes,
        "decisions": [],
        "relations": [],
        "raw_output": raw,
        "skipped": False,
        "extraction_method": "compact_llm_tool_result",
    }


def extract_compact_tool_result_nodes_batch(
    client,
    model: str,
    *,
    items: list[dict[str, Any]],
    mode: str,
    max_results: int = 10,
    max_snippet_chars: int = 240,
    max_facts: int = 3,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> list[dict[str, Any] | None]:
    """Distill independent tool results with one history-free model call.

    ``item_index`` is code-owned and is the only way model output is associated
    with an input result.  The model is explicitly forbidden from combining
    evidence across items.  A malformed/missing item falls back independently
    to the same deterministic node used by the single-result path.
    """

    results: list[dict[str, Any] | None] = []
    candidates: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        content = str(item.get("content") or "")
        sentences = item.get("sentences")
        if not isinstance(sentences, list):
            sentences = None
        rule_result = extract_rule_tool_result_nodes(
            role="tool",
            content=content,
            mode=mode,
            sentences=sentences,
            max_results=max_results,
            max_snippet_chars=max_snippet_chars,
        )
        results.append(rule_result)
        if rule_result is None:
            continue
        rule_node = rule_result["nodes"][0]
        if int(rule_node.get("tool_result_count") or 0) == 0:
            continue
        result_items = list(rule_node.get("tool_result_items") or [])
        candidates.append(
            {
                "item_index": item_index,
                "query": item.get("query") or "(query unavailable)",
                "tool_name": rule_node.get("tool_name") or "tool",
                "results": [
                    {
                        key: result_item[key]
                        for key in ("rank", "title", "snippet")
                        if key in result_item
                    }
                    for result_item in result_items
                ],
            }
        )

    if not candidates:
        return results

    fact_limit = max(1, int(max_facts))
    prompt = f"""Extract at most {fact_limit} self-contained facts for EACH item below.

Rules:
- Treat every item independently. Never combine, compare, or transfer information
  between different item_index values.
- Use only that item's results and query; preserve exact names, dates, quantities,
  and versions.
- Every fact must name its specific subject or source title. Never emit a generic
  claim such as "a tower".
- Skip irrelevant results and do not infer facts the title/snippet does not state.
- Prefer answer-relevant facts that prevent repeating the corresponding search.
- Return every input item_index exactly once, even when its beliefs list is empty.
- For every fact, classify stance as asserted, recalled, speculated, or judged,
  and list only specific salient entities explicitly present in that fact.
- JSON only: {{"items":[{{"item_index":0,"beliefs":[{{"belief":"...","stance":"asserted","entities":["..."]}}]}}]}}.

Items:
{json.dumps(candidates, ensure_ascii=False)}
"""
    try:
        raw = llm.call_model(
            client,
            model,
            prompt,
            max_tokens=min(
                int(max_tokens or 4096),
                max(768, 256 * len(candidates)),
            ),
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        return results

    parsed = llm.parse_json_response(raw)
    raw_items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(raw_items, list):
        return results
    output_by_index: dict[int, dict[str, Any]] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        item_index = raw_item.get("item_index")
        if not isinstance(item_index, int) or item_index in output_by_index:
            continue
        if 0 <= item_index < len(items):
            output_by_index[item_index] = raw_item

    for candidate in candidates:
        item_index = int(candidate["item_index"])
        raw_item = output_by_index.get(item_index)
        raw_beliefs = raw_item.get("beliefs") if raw_item is not None else None
        if not isinstance(raw_beliefs, list):
            continue
        rule_result = results[item_index]
        if rule_result is None:
            continue
        rule_node = rule_result["nodes"][0]
        content = str(items[item_index].get("content") or "")
        sentences = items[item_index].get("sentences")
        if not isinstance(sentences, list):
            sentences = []
        result_items = list(rule_node.get("tool_result_items") or [])
        nodes: list[dict[str, Any]] = []
        for raw_node in raw_beliefs[:fact_limit]:
            if not isinstance(raw_node, dict):
                continue
            belief = _clean_str(raw_node.get("belief"))
            if not belief:
                continue
            node: dict[str, Any] = {
                "tmp_id": f"n{len(nodes)}",
                "node_type": "belief",
                "belief": belief,
                "stance": _clean_stance(raw_node.get("stance")),
                "entities": _clean_entities(raw_node.get("entities")),
                "tool_name": rule_node.get("tool_name") or "tool",
                "tool_result_count": rule_node.get("tool_result_count") or 0,
                "tool_result_items": result_items,
                "tool_result_truncated_count": rule_node.get(
                    "tool_result_truncated_count"
                )
                or 0,
                "extraction_method": "compact_llm_tool_result",
            }
            if rule_node.get("tool_call_id"):
                node["tool_call_id"] = str(rule_node["tool_call_id"])
            node["tool_result_index"] = int(
                rule_node.get("tool_result_index") or item_index
            )
            node["tool_result_fact_index"] = len(nodes)
            if mode == "excerpt":
                node["supporting_excerpts"] = [content]
            else:
                node["supporting_sentence_indices"] = (
                    list(range(len(sentences))) or None
                )
            nodes.append(node)
        # An empty/malformed per-item response falls back only that item. This
        # makes a partial batch response no worse than the old sequential path.
        if not nodes:
            continue
        results[item_index] = {
            "nodes": nodes,
            "beliefs": nodes,
            "decisions": [],
            "relations": [],
            "raw_output": raw,
            "skipped": False,
            "extraction_method": "compact_llm_tool_result",
            "batch_item_index": item_index,
            "batch_size": len(items),
        }
    return results


def extract_assistant_tool_result_nodes_batch(
    client,
    model: str,
    *,
    assistant_content: str,
    assistant_sentences: list[str] | None,
    graph_nodes_str: str,
    items: list[dict[str, Any]],
    mode: str,
    max_results: int = 10,
    max_snippet_chars: int = 240,
    max_facts: int = 3,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any] | None]]:
    """Extract one Assistant layer and its Tool Result layer in one call.

    The returned Assistant result and per-item Tool results remain model-shaped
    but separate.  ``StreamingBeliefBuilder`` consumes them at their original
    turn indices, preserving source/evidence/edge provenance.
    """

    tool_results: list[dict[str, Any] | None] = []
    candidates: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        content = str(item.get("content") or "")
        sentences = item.get("sentences")
        if not isinstance(sentences, list):
            sentences = None
        rule_result = extract_rule_tool_result_nodes(
            role="tool",
            content=content,
            mode=mode,
            sentences=sentences,
            max_results=max_results,
            max_snippet_chars=max_snippet_chars,
        )
        tool_results.append(rule_result)
        if rule_result is None:
            continue
        rule_node = rule_result["nodes"][0]
        if int(rule_node.get("tool_result_count") or 0) == 0:
            continue
        result_items = list(rule_node.get("tool_result_items") or [])
        candidates.append(
            {
                "item_index": item_index,
                "fact_limit": max(1, int(max_facts)),
                "query": item.get("query") or "(query unavailable)",
                "tool_name": rule_node.get("tool_name") or "tool",
                "results": [
                    {
                        key: result_item[key]
                        for key in ("rank", "title", "snippet")
                        if key in result_item
                    }
                    for result_item in result_items
                ],
            }
        )

    calls = extract_tool_calls(assistant_content)
    semantic_content = (
        strip_valid_tool_calls(assistant_content) if calls else assistant_content
    )
    source_sentences = list(assistant_sentences or [])
    semantic_sentences = [
        strip_valid_tool_calls(sentence) if calls else sentence
        for sentence in source_sentences
    ]
    prompt = build_assistant_tool_result_extraction_prompt(
        mode=mode,
        assistant_content=semantic_content,
        assistant_sentences_block=format_sentences_for_prompt(semantic_sentences),
        tool_items=json.dumps(candidates, ensure_ascii=False),
        graph_nodes=graph_nodes_str,
    )
    raw = llm.call_model(
        client,
        model,
        prompt,
        temperature=0.0,
        # This call returns one bounded Assistant partition plus at most
        # ``max_facts`` facts for each Tool Result item.  Do not inherit the
        # graph model's potentially very large global completion budget.
        max_tokens=min(
            int(max_tokens or 4096),
            max(1536, 1024 + 256 * len(candidates)),
        ),
        reasoning_effort=reasoning_effort,
    )
    parsed = llm.parse_json_response(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("assistant"), dict):
        raise ValueError("combined extraction response is missing assistant partition")

    raw_assistant = parsed["assistant"]
    n_sentences = len(source_sentences)
    assistant_nodes: list[dict[str, Any]] = []
    assistant_beliefs: list[dict[str, Any]] = []
    assistant_decisions: list[dict[str, Any]] = []
    seen_tmp: set[str] = set()
    ordinal = 0
    for belief in raw_assistant.get("beliefs") or []:
        cleaned = _clean_node(belief, mode, n_sentences, ordinal, node_type="belief")
        if cleaned is None:
            continue
        if cleaned["tmp_id"] in seen_tmp:
            cleaned["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cleaned["tmp_id"])
        assistant_nodes.append(cleaned)
        assistant_beliefs.append(cleaned)
        ordinal += 1

    semantic_belief_count = len(assistant_beliefs)
    ordinal = _attach_and_complete_tool_call_nodes(
        assistant_beliefs,
        role="assistant",
        content=assistant_content,
        mode=mode,
        sentences=source_sentences,
        seen_tmp=seen_tmp,
        ordinal=ordinal,
    )
    assistant_nodes.extend(assistant_beliefs[semantic_belief_count:])
    if re.search(
        r"<thinking>.*?</thinking>", semantic_content, re.DOTALL | re.IGNORECASE
    ):
        for node in assistant_beliefs:
            if node.get("extraction_method") != "rule_tool_call":
                node["source_component"] = "thinking"

    for decision in raw_assistant.get("decisions") or []:
        cleaned = _clean_node(
            decision, mode, n_sentences, ordinal, node_type="decision"
        )
        if cleaned is None:
            continue
        if cleaned["tmp_id"] in seen_tmp:
            cleaned["tmp_id"] = f"d{ordinal}"
        seen_tmp.add(cleaned["tmp_id"])
        assistant_nodes.append(cleaned)
        assistant_decisions.append(cleaned)
        ordinal += 1

    raw_tool_items = parsed.get("tool_items")
    output_by_index: dict[int, dict[str, Any]] = {}
    if isinstance(raw_tool_items, list):
        for raw_item in raw_tool_items:
            if not isinstance(raw_item, dict):
                continue
            item_index = raw_item.get("item_index")
            if (
                isinstance(item_index, int)
                and 0 <= item_index < len(items)
                and item_index not in output_by_index
            ):
                output_by_index[item_index] = raw_item

    fact_limit = max(1, int(max_facts))
    for candidate in candidates:
        item_index = int(candidate["item_index"])
        raw_item = output_by_index.get(item_index)
        raw_beliefs = raw_item.get("beliefs") if raw_item is not None else None
        rule_result = tool_results[item_index]
        if not isinstance(raw_beliefs, list) or rule_result is None:
            continue
        rule_node = rule_result["nodes"][0]
        content = str(items[item_index].get("content") or "")
        sentences = items[item_index].get("sentences")
        if not isinstance(sentences, list):
            sentences = []
        result_items = list(rule_node.get("tool_result_items") or [])
        nodes: list[dict[str, Any]] = []
        for raw_node in raw_beliefs[:fact_limit]:
            if not isinstance(raw_node, dict):
                continue
            belief = _clean_str(raw_node.get("belief"))
            if not belief:
                continue
            node: dict[str, Any] = {
                "tmp_id": f"n{len(nodes)}",
                "node_type": "belief",
                "belief": belief,
                "stance": _clean_stance(raw_node.get("stance")),
                "entities": _clean_entities(raw_node.get("entities")),
                "tool_name": rule_node.get("tool_name") or "tool",
                "tool_result_count": rule_node.get("tool_result_count") or 0,
                "tool_result_items": result_items,
                "tool_result_truncated_count": rule_node.get(
                    "tool_result_truncated_count"
                )
                or 0,
                "extraction_method": "compact_llm_tool_result",
                "tool_result_index": int(
                    rule_node.get("tool_result_index") or item_index
                ),
                "tool_result_fact_index": len(nodes),
            }
            if rule_node.get("tool_call_id"):
                node["tool_call_id"] = str(rule_node["tool_call_id"])
            if mode == "excerpt":
                node["supporting_excerpts"] = [content]
            else:
                node["supporting_sentence_indices"] = (
                    list(range(len(sentences))) or None
                )
            nodes.append(node)
        if nodes:
            tool_results[item_index] = {
                "nodes": nodes,
                "beliefs": nodes,
                "decisions": [],
                "relations": [],
                "raw_output": raw,
                "skipped": False,
                "extraction_method": "compact_llm_tool_result",
                "batch_item_index": item_index,
                "batch_size": len(items),
            }

    assistant_result = {
        "nodes": assistant_nodes,
        "beliefs": assistant_beliefs,
        "decisions": assistant_decisions,
        "relations": [],
        "raw_output": raw,
        "skipped": False,
        "extraction_method": "combined_assistant_tool_result",
    }
    return assistant_result, tool_results


def _clean_relations(raw: Any) -> list[dict[str, Any]]:
    """Keep unresolved typed relations. Endpoints may be existing ids or tmp ids."""
    out: list[dict[str, Any]] = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        rtype = (r.get("type") or "").strip()
        if rtype not in VALID_RELATION_TYPES:
            continue
        frm = r.get("from", r.get("from_id"))
        to = r.get("to", r.get("to_id"))
        if frm is None or to is None:
            continue
        note = r.get("note", "") or ""
        if not isinstance(note, str):
            note = str(note)
        out.append({"from": frm, "to": to, "type": rtype, "note": note.strip()})
    return out


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _node_line(b: dict[str, Any]) -> str:
    src = b.get("source") or {}
    line = {
        "id": b.get("id"),
        "node_type": b.get("node_type", "belief"),
        "role": b.get("role") or src.get("role") or src.get("type") or "?",
        "turn": src.get("turn_id", src.get("turn_index")),
        "stance": b.get("stance"),
        "conf": b.get("confidence"),
        "entities": b.get("entities") or [],
    }
    if b.get("event_time"):
        line["time"] = b.get("event_time")
    content = b.get("decision") if b.get("node_type") == "decision" else b.get("belief")
    content = content or b.get("belief") or b.get("decision") or ""
    line["content"] = content if len(content) <= 240 else content[:220] + " …"
    return json.dumps(line, ensure_ascii=False)


def format_graph_nodes(
    nodes: list[dict[str, Any]], char_budget: int | None = 9000
) -> str:
    """Compact JSON view of graph nodes.

    A ``None`` budget preserves every node; a numeric budget keeps the most
    recent nodes when the rendered context would otherwise be too large.
    """
    if not nodes:
        return "[]"
    ordered = sorted(nodes, key=lambda b: b.get("id", 0))
    lines = [_node_line(b) for b in ordered]
    total = sum(len(s) + 4 for s in lines)
    omitted = 0
    while char_budget is not None and lines and total > char_budget:
        total -= len(lines[0]) + 4
        lines.pop(0)
        omitted += 1
    items = (
        [f"  (... {omitted} earlier node(s) omitted for length ...)"] if omitted else []
    )
    items += ["  " + s for s in lines]
    return "[\n" + ",\n".join(items) + "\n]"


def format_extraction_nodes(
    nodes: list[dict[str, Any]], char_budget: int | None = 9000
) -> str:
    """Render only prior semantic content needed during node extraction."""
    if not nodes:
        return "[]"
    ordered = sorted(nodes, key=lambda b: b.get("id", 0))
    lines: list[str] = []
    for node in ordered:
        content = (
            node.get("decision")
            if node.get("node_type") == "decision"
            else node.get("belief")
        )
        content = content or node.get("belief") or node.get("decision") or ""
        if len(content) > 240:
            content = content[:220] + " …"
        lines.append(json.dumps({"content": content}, ensure_ascii=False))

    total = sum(len(line) + 4 for line in lines)
    omitted = 0
    while char_budget is not None and lines and total > char_budget:
        total -= len(lines[0]) + 4
        lines.pop(0)
        omitted += 1
    items = (
        [f"  (... {omitted} earlier node(s) omitted for length ...)"] if omitted else []
    )
    items += ["  " + line for line in lines]
    return "[\n" + ",\n".join(items) + "\n]"


def format_relation_nodes(
    nodes: list[dict[str, Any]], char_budget: int | None = 9000
) -> str:
    """Render only the semantic fields needed for relation judgment."""
    if not nodes:
        return "[]"
    ordered = sorted(nodes, key=lambda b: b.get("id", 0))
    lines: list[str] = []
    for node in ordered:
        content = (
            node.get("decision")
            if node.get("node_type") == "decision"
            else node.get("belief")
        )
        content = content or node.get("belief") or node.get("decision") or ""
        if len(content) > 240:
            content = content[:220] + " …"
        lines.append(
            json.dumps({"id": node.get("id"), "content": content}, ensure_ascii=False)
        )

    total = sum(len(line) + 4 for line in lines)
    omitted = 0
    while char_budget is not None and lines and total > char_budget:
        total -= len(lines[0]) + 4
        lines.pop(0)
        omitted += 1
    items = (
        [f"  (... {omitted} earlier node(s) omitted for length ...)"] if omitted else []
    )
    items += ["  " + line for line in lines]
    return "[\n" + ",\n".join(items) + "\n]"


def format_relation_node_sets(
    previous_nodes: list[dict[str, Any]],
    current_nodes: list[dict[str, Any]],
    *,
    layer_by_id: dict[int, int] | None = None,
    char_budget: int | None = 9000,
) -> tuple[str, str, set[int]]:
    """Render a self-contained layered relation window.

    Historical entries carry ``id`` and semantic ``content`` plus ``layer``
    when ``layer_by_id`` is provided; current entries carry ``id`` and
    ``content``. A shared budget is applied before the two JSON lists are split
    so every id shown elsewhere in the relation prompt has visible content.
    """

    entries: list[tuple[int, bool, str]] = []
    for node in previous_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, int) or (
            layer_by_id is not None and node_id not in layer_by_id
        ):
            continue
        content = (
            node.get("decision")
            if node.get("node_type") == "decision"
            else node.get("belief")
        )
        content = content or node.get("belief") or node.get("decision") or ""
        if len(content) > 240:
            content = content[:220] + " …"
        rendered = {"id": node_id}
        if layer_by_id is not None:
            rendered["layer"] = layer_by_id[node_id]
        rendered["content"] = content
        entries.append(
            (node_id, False, json.dumps(rendered, ensure_ascii=False))
        )
    for node in current_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, int):
            continue
        content = (
            node.get("decision")
            if node.get("node_type") == "decision"
            else node.get("belief")
        )
        content = content or node.get("belief") or node.get("decision") or ""
        if len(content) > 240:
            content = content[:220] + " …"
        entries.append(
            (
                node_id,
                True,
                json.dumps({"id": node_id, "content": content}, ensure_ascii=False),
            )
        )

    entries.sort(key=lambda item: item[0])
    total = sum(len(line) + 4 for _node_id, _is_current, line in entries)
    while char_budget is not None and entries and total > char_budget:
        _node_id, _is_current, line = entries.pop(0)
        total -= len(line) + 4

    retained_ids = {node_id for node_id, _is_current, _line in entries}
    previous_lines = [line for _id, current, line in entries if not current]
    current_lines = [line for _id, current, line in entries if current]

    def render(lines: list[str]) -> str:
        return "[\n" + ",\n".join("  " + line for line in lines) + "\n]"

    return render(previous_lines), render(current_lines), retained_ids


def format_graph_edges(
    relations: list[dict[str, Any]],
    keep_ids: set | None = None,
    max_edges: int = 400,
    *,
    include_id: bool = True,
) -> str:
    """Compact view of existing typed relations so the model won't duplicate them."""
    if not relations:
        return "[]"
    rels = relations
    if keep_ids is not None:
        rels = [
            r
            for r in rels
            if r.get("from_id") in keep_ids and r.get("to_id") in keep_ids
        ]
    if not rels:
        return "[]"
    rels = sorted(rels, key=lambda r: (r.get("from_id", 0), r.get("to_id", 0)))[
        -max_edges:
    ]
    lines = []
    for relation in rels:
        rendered = {
            "from": relation.get("from_id"),
            "to": relation.get("to_id"),
            "type": relation.get("type"),
        }
        if include_id:
            rendered = {"id": relation.get("id"), **rendered}
        lines.append("  " + json.dumps(rendered, ensure_ascii=False))
    return "[\n" + ",\n".join(lines) + "\n]"


# ---------------------------------------------------------------------------
# Update entry point
# ---------------------------------------------------------------------------


def update_graph(
    client,
    model: str,
    *,
    role: str,
    mode: str = "sentences",
    content: str | None = None,
    sentences: list[str] | None = None,
    clusters: list[list[int]] | None = None,
    graph_nodes_str: str = "[]",
    graph_edges_str: str = "[]",
    current_date: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """
    One LLM call. Returns cleaned unresolved nodes and typed relations:
        {
          "nodes": [...], "beliefs": [...], "decisions": [...],
          "relations": [...], "raw_output": str|None,
          "skipped": bool, "skip_reason"?: str
        }
    """
    n_sentences = len(sentences or [])
    rule_result = _rule_extract_tool_call_nodes(
        role=role,
        content=content or "\n".join(sentences or []),
        mode=mode,
        sentences=sentences or [],
    )
    if rule_result is not None:
        return rule_result
    if mode != "excerpt":
        if clusters:
            sentences_block = format_clustered_sentences_for_prompt(
                sentences or [], clusters
            )
        else:
            sentences_block = format_sentences_for_prompt(sentences or [])
        prompt = build_update_prompt(
            role,
            mode="sentences",
            sentences_block=sentences_block,
            graph_nodes=graph_nodes_str,
            graph_edges=graph_edges_str,
            current_date=current_date,
        )
    else:
        prompt = build_update_prompt(
            role,
            mode="excerpt",
            content=content or "",
            graph_nodes=graph_nodes_str,
            graph_edges=graph_edges_str,
            current_date=current_date,
        )

    if prompt is None:
        return {
            "nodes": [],
            "beliefs": [],
            "decisions": [],
            "relations": [],
            "raw_output": None,
            "skipped": True,
            "skip_reason": f"unknown role {role!r}",
        }

    try:
        raw = llm.call_model(
            client,
            model,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        return {
            "nodes": [],
            "beliefs": [],
            "decisions": [],
            "relations": [],
            "raw_output": f"[ERROR] {e}",
            "skipped": True,
            "skip_reason": str(e),
        }

    parsed = llm.parse_json_response(raw)
    parsed = parsed if isinstance(parsed, dict) else {}

    out_nodes: list[dict[str, Any]] = []
    out_beliefs: list[dict[str, Any]] = []
    out_decisions: list[dict[str, Any]] = []
    seen_tmp: set = set()
    ordinal = 0

    for b in parsed.get("beliefs", []) or []:
        cb = _clean_node(
            b,
            mode,
            n_sentences,
            ordinal,
            node_type="belief",
            preserve_model_tmp_id=True,
        )
        if cb is None:
            continue
        if cb["tmp_id"] in seen_tmp:
            cb["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cb["tmp_id"])
        out_nodes.append(cb)
        out_beliefs.append(cb)
        ordinal += 1

    parsed_belief_count = len(out_beliefs)
    ordinal = _attach_and_complete_tool_call_nodes(
        out_beliefs,
        role=role,
        content=content or "\n".join(sentences or []),
        mode=mode,
        sentences=sentences or [],
        seen_tmp=seen_tmp,
        ordinal=ordinal,
    )
    out_nodes.extend(out_beliefs[parsed_belief_count:])

    for d in parsed.get("decisions", []) or []:
        cd = _clean_node(
            d,
            mode,
            n_sentences,
            ordinal,
            node_type="decision",
            preserve_model_tmp_id=True,
        )
        if cd is None:
            continue
        if cd["tmp_id"] in seen_tmp:
            cd["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cd["tmp_id"])
        out_nodes.append(cd)
        out_decisions.append(cd)
        ordinal += 1

    relations = _clean_relations(parsed.get("relations", []) or [])
    return {
        "nodes": out_nodes,
        "beliefs": out_beliefs,
        "decisions": out_decisions,
        "relations": relations,
        "raw_output": raw,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Two-phase extraction (Phase 1: nodes only, Phase 2: relations only)
# ---------------------------------------------------------------------------


def extract_nodes(
    client,
    model: str,
    *,
    role: str,
    mode: str = "sentences",
    content: str | None = None,
    sentences: list[str] | None = None,
    clusters: list[list[int]] | None = None,
    graph_nodes_str: str = "[]",
    graph_edges_str: str = "[]",
    current_date: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Phase 1: one LLM call to extract beliefs + decisions only (no relations)."""
    n_sentences = len(sentences or [])
    source_content = content or "\n".join(sentences or [])
    rule_result = _rule_extract_tool_call_nodes(
        role=role,
        content=source_content,
        mode=mode,
        sentences=sentences or [],
    )
    if rule_result is not None:
        return rule_result
    calls = (
        extract_tool_calls(source_content)
        if role.strip().lower() == "assistant"
        else []
    )
    semantic_content = (
        strip_valid_tool_calls(source_content) if calls else source_content
    )
    semantic_sentences = [
        strip_valid_tool_calls(sentence) if calls else sentence
        for sentence in (sentences or [])
    ]
    if mode != "excerpt":
        if clusters:
            sentences_block = format_clustered_sentences_for_prompt(
                semantic_sentences, clusters
            )
        else:
            sentences_block = format_sentences_for_prompt(semantic_sentences)
        prompt = build_node_extraction_prompt(
            role,
            mode="sentences",
            sentences_block=sentences_block,
            graph_nodes=graph_nodes_str,
            graph_edges=graph_edges_str,
            current_date=current_date,
        )
    else:
        prompt = build_node_extraction_prompt(
            role,
            mode="excerpt",
            content=semantic_content,
            graph_nodes=graph_nodes_str,
            graph_edges=graph_edges_str,
            current_date=current_date,
        )

    if prompt is None:
        return {
            "nodes": [],
            "beliefs": [],
            "decisions": [],
            "raw_output": None,
            "skipped": True,
            "skip_reason": f"unknown role {role!r}",
        }

    try:
        raw = llm.call_model(
            client,
            model,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        return {
            "nodes": [],
            "beliefs": [],
            "decisions": [],
            "raw_output": f"[ERROR] {e}",
            "skipped": True,
            "skip_reason": str(e),
        }

    parsed = llm.parse_json_response(raw)
    parsed = parsed if isinstance(parsed, dict) else {}

    out_nodes: list[dict[str, Any]] = []
    out_beliefs: list[dict[str, Any]] = []
    out_decisions: list[dict[str, Any]] = []
    seen_tmp: set = set()
    ordinal = 0

    for b in parsed.get("beliefs", []) or []:
        cb = _clean_node(b, mode, n_sentences, ordinal, node_type="belief")
        if cb is None:
            continue
        if cb["tmp_id"] in seen_tmp:
            cb["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cb["tmp_id"])
        out_nodes.append(cb)
        out_beliefs.append(cb)
        ordinal += 1

    parsed_belief_count = len(out_beliefs)
    ordinal = _attach_and_complete_tool_call_nodes(
        out_beliefs,
        role=role,
        content=content or "\n".join(sentences or []),
        mode=mode,
        sentences=sentences or [],
        seen_tmp=seen_tmp,
        ordinal=ordinal,
    )
    out_nodes.extend(out_beliefs[parsed_belief_count:])

    if role.strip().lower() == "assistant" and re.search(
        r"<thinking>.*?</thinking>", semantic_content, re.DOTALL | re.IGNORECASE
    ):
        for node in out_beliefs:
            if node.get("extraction_method") != "rule_tool_call":
                node["source_component"] = "thinking"

    for d in parsed.get("decisions", []) or []:
        cd = _clean_node(d, mode, n_sentences, ordinal, node_type="decision")
        if cd is None:
            continue
        if cd["tmp_id"] in seen_tmp:
            cd["tmp_id"] = f"n{ordinal}"
        seen_tmp.add(cd["tmp_id"])
        out_nodes.append(cd)
        out_decisions.append(cd)
        ordinal += 1

    return {
        "nodes": out_nodes,
        "beliefs": out_beliefs,
        "decisions": out_decisions,
        "raw_output": raw,
        "skipped": False,
    }


def extract_relations(
    client,
    model: str,
    *,
    role: str,
    content: str,
    graph_nodes_str: str = "[]",
    graph_edges_str: str = "[]",
    previous_nodes_str: str | None = None,
    current_nodes_str: str | None = None,
    new_node_ids: set,
    current_date: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Phase 2: one LLM call to extract relations on the post-merge graph."""
    import json as _json

    prompt = build_relation_extraction_prompt(
        role=role,
        content=content or "",
        graph_nodes=graph_nodes_str,
        graph_edges=graph_edges_str,
        new_node_ids=_json.dumps(sorted(new_node_ids)),
        previous_nodes=previous_nodes_str,
        current_nodes=current_nodes_str,
        current_date=current_date,
    )

    try:
        raw = llm.call_model(
            client,
            model,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        return {"relations": [], "raw_output": f"[ERROR] {e}", "skipped": True}

    parsed = llm.parse_json_response(raw)
    parsed = parsed if isinstance(parsed, dict) else {}
    relations = _clean_relations(parsed.get("relations", []) or [])
    return {"relations": relations, "raw_output": raw, "skipped": False}


def extract_layered_relations(
    client,
    model: str,
    *,
    role: str,
    content: str,
    previous_nodes_str: str = "[]",
    graph_edges_str: str = "[]",
    current_nodes_str: str = "[]",
    new_node_ids: set,
    candidate_layers: list[dict[str, Any]],
    validation_feedback: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Extract Assistant relations while selecting at most one prior layer."""
    prompt = build_layered_relation_extraction_prompt(
        role=role,
        content=content or "",
        previous_nodes=previous_nodes_str,
        graph_edges=graph_edges_str,
        current_nodes=current_nodes_str,
        validation_feedback=validation_feedback,
    )

    try:
        raw = llm.call_model(
            client,
            model,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        return {
            "selected_previous_layer": None,
            "relations": [],
            "raw_output": f"[ERROR] {e}",
            "skipped": True,
        }

    parsed = llm.parse_json_response(raw)
    parsed = parsed if isinstance(parsed, dict) else {}
    return {
        "selected_previous_layer": parsed.get("selected_previous_layer"),
        "relations": _clean_relations(parsed.get("relations", []) or []),
        "raw_output": raw,
        "skipped": False,
    }
