"""Async HTTP client for the Belief Graph service."""

from __future__ import annotations

import json
import logging
from typing import Any
from xml.etree import ElementTree as ET

import aiohttp

logger = logging.getLogger(__name__)


def _n_relations(snapshot: dict) -> int:
    """Count edges in a graph snapshot. The service exposes them under
    ``relations`` (current) or ``forward_relations`` (older); prefer whichever
    is populated."""
    rels = snapshot.get("relations") or snapshot.get("forward_relations") or []
    return len(rels)


class BeliefGraphClient:
    """Thin async wrapper around the Belief Graph HTTP API."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        """GET /health — verify service is reachable."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning("Belief graph health check failed: %s", exc)
            return False

    async def push_turn(
        self,
        problem_id: str,
        role: str,
        content: str,
        is_message_end: bool = True,
        is_trajectory_end: bool = False,
        meta: dict | None = None,
    ) -> dict | None:
        """POST /turn — push a single turn, return the graph snapshot.

        Extra keys in ``meta`` (e.g. {"timings": {...}}) are merged into the payload;
        the online server preserves them verbatim in trajectory_stream.jsonl, so live
        consumers can read per-turn side-channel data.
        """
        payload = {
            "problem_id": problem_id,
            "role": role,
            "content": content,
            "is_message_end": is_message_end,
            "is_trajectory_end": is_trajectory_end,
        }
        if meta:
            for k, v in meta.items():
                if k not in payload:
                    payload[k] = v
        logger.info("[BeliefGraph] >>> POST /turn  role=%s", role)
        result = await self._post("/turn", payload)
        if result:
            n = result.get("n_beliefs", 0)
            logger.info(
                "[BeliefGraph] <<< /turn returned %d beliefs, keys=%s, relations=%d",
                n, list(result.keys()), _n_relations(result),
            )
        return result

    async def push_turns(self, turns: list[dict]) -> dict | None:
        """POST /turns — batch push, return latest snapshot.

        Response format: {"pushed": N, "finalized": [...], "latest": {problem_id: snapshot}}
        We extract the first snapshot from "latest".
        """
        roles = [t.get("role", "?") for t in turns]
        logger.info("[BeliefGraph] >>> POST /turns  roles=%s", roles)
        result = await self._post("/turns", turns)
        if result is None:
            return None
        latest = result.get("latest", {})
        if latest:
            snapshot = next(iter(latest.values()))
            logger.info(
                "[BeliefGraph] <<< /turns returned %d beliefs, keys=%s, relations=%d",
                snapshot.get("n_beliefs", 0),
                list(snapshot.keys()),
                _n_relations(snapshot),
            )
            return snapshot
        return result

    async def get_graph(self, problem_id: str) -> dict | None:
        """GET /graph?problem_id=... — fetch the full graph."""
        logger.info("[BeliefGraph] >>> GET /graph  problem_id=%s", problem_id)
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/graph", params={"problem_id": problem_id}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(
                        "[BeliefGraph] <<< /graph returned %d beliefs, finalized=%s",
                        result.get("n_beliefs", 0), result.get("finalized"),
                    )
                    return result
                logger.warning("GET /graph returned %d", resp.status)
                return None
        except Exception as exc:
            logger.warning("GET /graph failed: %s", exc)
            return None

    async def finalize(self, problem_id: str) -> dict | None:
        """POST /finalize — trigger backward pass + merge."""
        return await self._post("/finalize", {"problem_id": problem_id})

    async def _post(self, path: str, payload: Any) -> dict | None:
        try:
            session = await self._get_session()
            sanitized = self._sanitize_payload(payload)
            async with session.post(
                f"{self.base_url}{path}",
                json=sanitized,
                headers={"content-type": "application/json"},
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning("POST %s returned %d", path, resp.status)
                return None
        except Exception as exc:
            logger.warning("POST %s failed: %s", path, exc)
            return None

    @staticmethod
    def _sanitize_payload(payload: Any) -> Any:
        """Escape problematic characters in string values before sending."""
        if isinstance(payload, str):
            return payload.replace("'", "\\'")
        if isinstance(payload, dict):
            return {k: BeliefGraphClient._sanitize_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [BeliefGraphClient._sanitize_payload(item) for item in payload]
        return payload

    @staticmethod
    def format_graph_for_prompt(
        snapshot: dict,
        fmt: str = "structured",
        include_relations: bool = True,
        deepseek_v4_payload_format: str = "json",
    ) -> str:
        """Format beliefs + relations into a text block for model context.

        Args:
            snapshot: Graph snapshot from the belief graph service.
            fmt: Format version name. Supported:
                - "structured" (default): Belief Nodes + directed relation tuples.
                - "narrative": Prose paragraph summarising beliefs and links.
                - "markdown": Markdown list with relation arrows.
                - "xml": XML-tagged format.
                - "triplet": KG-style (subject, relation, object) triples.
                - "yaml": YAML-structured beliefs and relations.
                - "json": Dialogue-like JSON messages converted from beliefs.
                - "deepseek_v4": DeepSeek-V4 encoded dialogue-like belief context.
            include_relations: Whether to include edges in the output.
            deepseek_v4_payload_format: Serialization used for each belief inside
                DeepSeek-V4 role markers: "json", "xml", or "markdown".
        """
        beliefs = snapshot.get("beliefs", [])
        relations = (
            snapshot.get("forward_relations") or snapshot.get("relations") or []
        ) if include_relations else []
        if not beliefs:
            return ""

        if fmt == "deepseek_v4":
            return _fmt_deepseek_v4(
                beliefs,
                relations,
                payload_format=deepseek_v4_payload_format,
            )

        formatter = _GRAPH_FORMATTERS.get(fmt)
        if formatter is None:
            logger.warning("Unknown graph format '%s', falling back to 'structured'", fmt)
            formatter = _GRAPH_FORMATTERS["structured"]
        return formatter(beliefs, relations)


_GRAPH_PREFIX = (
    "The following belief graph captures your reasoning trajectory so far. "
    "These are preliminary beliefs derived from prior turns — they may contain "
    "errors or incomplete information. Use them to guide what to search next, "
    "but do NOT treat them as verified evidence. You must still use the search "
    "tool to confirm key claims before reaching a final verdict."
)


def _fmt_structured(beliefs: list[dict], relations: list[dict]) -> str:
    lines: list[str] = [
        _GRAPH_PREFIX,
        "",
        "Belief Nodes:",
    ]
    for b in beliefs:
        bid = b.get("id", "?")
        lines.append(f"  [{bid}] {b.get('belief', '')}")
    if relations:
        lines.append("")
        lines.append("Relations:")
        for r in relations:
            rtype = r.get("type", "informs")
            note = r.get("note", "")
            reason = f", reason: {note}" if note else ""
            lines.append(
                f"  Belief [{r.get('from_id', '?')}] {rtype} "
                f"Belief [{r.get('to_id', '?')}]{reason}"
            )
    return "\n".join(lines)


def _fmt_narrative(beliefs: list[dict], relations: list[dict]) -> str:
    parts = [f"[{b.get('id','?')}] {b.get('belief','')}" for b in beliefs]
    text = _GRAPH_PREFIX + "\n\nCurrent beliefs: " + "; ".join(parts) + "."
    if relations:
        links = []
        for r in relations:
            links.append(
                f"belief {r.get('from_id','?')} {r.get('type','informs')} "
                f"belief {r.get('to_id','?')}"
                + (f" ({r.get('note','')})" if r.get("note") else "")
            )
        text += " Relations: " + "; ".join(links) + "."
    return text


def _fmt_markdown(beliefs: list[dict], relations: list[dict]) -> str:
    lines = ["## Belief Graph", "", _GRAPH_PREFIX, ""]
    for b in beliefs:
        lines.append(f"- **[{b.get('id','?')}]** {b.get('belief','')}")
    if relations:
        lines.append("")
        lines.append("### Relations")
        for r in relations:
            note = f" — {r.get('note','')}" if r.get("note") else ""
            lines.append(
                f"- [{r.get('from_id','?')}] → [{r.get('to_id','?')}] "
                f"({r.get('type','informs')}){note}"
            )
    return "\n".join(lines)


def _fmt_xml(beliefs: list[dict], relations: list[dict]) -> str:
    lines = [f"<!-- {_GRAPH_PREFIX} -->", "<belief_graph>", "  <beliefs>"]
    for b in beliefs:
        lines.append(
            f"    <belief id=\"{b.get('id','?')}\">{b.get('belief','')}</belief>"
        )
    lines.append("  </beliefs>")
    if relations:
        lines.append("  <relations>")
        for r in relations:
            lines.append(
                f"    <relation from=\"{r.get('from_id','?')}\" "
                f"to=\"{r.get('to_id','?')}\" "
                f"type=\"{r.get('type','informs')}\">"
                f"{r.get('note','')}</relation>"
            )
        lines.append("  </relations>")
    lines.append("</belief_graph>")
    return "\n".join(lines)


def _fmt_triplet(beliefs: list[dict], relations: list[dict]) -> str:
    lines: list[str] = [_GRAPH_PREFIX, ""]
    for b in beliefs:
        lines.append(f"({b.get('id', '?')}, \"{b.get('belief', '')}\")")
    if relations:
        lines.append("")
        for r in relations:
            note = r.get("note", "")
            reason = f", \"{note}\"" if note else ""
            lines.append(
                f"(Belief {r.get('from_id', '?')}, {r.get('type', 'informs')}, "
                f"Belief {r.get('to_id', '?')}{reason})"
            )
    return "\n".join(lines)


def _fmt_yaml(beliefs: list[dict], relations: list[dict]) -> str:
    lines: list[str] = [f"# {_GRAPH_PREFIX}", "", "beliefs:"]
    for b in beliefs:
        lines.append(f"  - id: {b.get('id', '?')}")
        lines.append(f"    text: \"{b.get('belief', '')}\"")
    if relations:
        lines.append("relations:")
        for r in relations:
            lines.append(f"  - from: {r.get('from_id', '?')}")
            lines.append(f"    to: {r.get('to_id', '?')}")
            lines.append(f"    type: {r.get('type', 'informs')}")
            if r.get("note"):
                lines.append(f"    reason: \"{r.get('note', '')}\"")
    return "\n".join(lines)


def _belief_dialogue_messages(beliefs: list[dict], relations: list[dict]) -> list[dict]:
    by_belief_id: dict = {}
    for b in beliefs:
        bid = b.get("id", "?")
        source = b.get("source") or {}
        by_belief_id[bid] = {
            "id": bid,
            "role": b.get("role") or source.get("role") or "assistant",
            "content": b.get("belief", ""),
            "relations": [],
            "confidence": b.get("confidence"),
        }

    for r in relations:
        from_id = r.get("from_id", "?")
        to_id = r.get("to_id", "?")
        rtype = r.get("type", "informs")
        note = r.get("note", "")
        if from_id in by_belief_id:
            rel = {
                "direction": "outgoing",
                "to": to_id,
                "type": rtype,
            }
            if note:
                rel["reason"] = note
            by_belief_id[from_id]["relations"].append(rel)
        if to_id in by_belief_id:
            rel = {
                "direction": "incoming",
                "from": from_id,
                "type": rtype,
            }
            if note:
                rel["reason"] = note
            by_belief_id[to_id]["relations"].append(rel)

    return [by_belief_id[b.get("id", "?")] for b in beliefs]


def _fmt_json(beliefs: list[dict], relations: list[dict]) -> str:
    obj: dict = {"messages": _belief_dialogue_messages(beliefs, relations)}
    return json.dumps(obj, ensure_ascii=False, indent=2)


_DSV4_BOS = "<｜begin▁of▁sentence｜>"
_DSV4_EOS = "<｜end▁of▁sentence｜>"
_DSV4_USER = "<｜User｜>"
_DSV4_ASSISTANT = "<｜Assistant｜>"
_DSV4_VALID_ROLES = {"system", "user", "assistant"}


def _encode_deepseek_v4_context(messages: list[dict]) -> str:
    """Encode messages using DeepSeek-V4 role markers as an embedded context.

    This intentionally omits the final generation marker that the official
    encoder appends for inference prompts, because the result is nested inside
    this agent's system prompt rather than sent as the whole model prompt.
    """
    parts: list[str] = [_DSV4_BOS]
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role == "system":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"{_DSV4_ASSISTANT}{content}{_DSV4_EOS}")
        else:
            parts.append(f"{_DSV4_USER}{content}")
    return "".join(parts)


def _deepseek_v4_payload_json(msg: dict) -> str:
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


def _deepseek_v4_payload_xml(msg: dict) -> str:
    root = ET.Element("belief", {"id": str(msg.get("id", "?"))})
    ET.SubElement(root, "content").text = str(msg.get("content") or "")
    relations_el = ET.SubElement(root, "relations")
    for relation in msg.get("relations") or []:
        attrs = {
            str(key): str(value)
            for key, value in relation.items()
            if key != "reason" and value is not None
        }
        relation_el = ET.SubElement(relations_el, "relation", attrs)
        if relation.get("reason"):
            ET.SubElement(relation_el, "reason").text = str(relation["reason"])
    confidence = msg.get("confidence")
    ET.SubElement(root, "confidence").text = (
        "" if confidence is None else str(confidence)
    )
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


def _deepseek_v4_payload_markdown(msg: dict) -> str:
    lines = [
        f"### Belief {msg.get('id', '?')}",
        f"**Content:** {msg.get('content') or ''}",
        "**Relations:**",
    ]
    relations = msg.get("relations") or []
    if relations:
        for relation in relations:
            fields = [
                f"{key}={value}"
                for key, value in relation.items()
                if value is not None and value != ""
            ]
            lines.append("- " + "; ".join(fields))
    else:
        lines.append("- None")
    confidence = msg.get("confidence")
    lines.append(f"**Confidence:** {'' if confidence is None else confidence}")
    return "\n".join(lines)


_DSV4_PAYLOAD_FORMATTERS = {
    "json": _deepseek_v4_payload_json,
    "xml": _deepseek_v4_payload_xml,
    "markdown": _deepseek_v4_payload_markdown,
}


def _fmt_deepseek_v4(
    beliefs: list[dict],
    relations: list[dict],
    payload_format: str = "json",
) -> str:
    payload_formatter = _DSV4_PAYLOAD_FORMATTERS.get(payload_format)
    if payload_formatter is None:
        logger.warning(
            "Unknown DeepSeek-V4 payload format '%s', falling back to 'json'",
            payload_format,
        )
        payload_formatter = _DSV4_PAYLOAD_FORMATTERS["json"]

    messages: list[dict] = []
    for msg in _belief_dialogue_messages(beliefs, relations):
        role = msg.get("role")
        if role not in _DSV4_VALID_ROLES:
            role = "user"
        content_obj = {
            "id": msg.get("id", "?"),
            "content": msg.get("content", ""),
            "relations": msg.get("relations", []),
            "confidence": msg.get("confidence"),
        }
        messages.append({
            "role": role,
            "content": payload_formatter(content_obj),
        })
    return _encode_deepseek_v4_context(messages)


_GRAPH_FORMATTERS = {
    "structured": _fmt_structured,
    "narrative": _fmt_narrative,
    "markdown": _fmt_markdown,
    "xml": _fmt_xml,
    "triplet": _fmt_triplet,
    "yaml": _fmt_yaml,
    "json": _fmt_json,
}


__all__ = ["BeliefGraphClient"]
