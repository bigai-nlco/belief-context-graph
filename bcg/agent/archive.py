"""Two-layer archive writer for tool results.

Layer 1 (per-tool index, e.g. ``averitec_search.json``): a query-centric
index grouped by tool. Each query entry carries the query text, turn number,
and a list of results with ``raw_url`` + ``summary``.

Layer 2 (``raw/tN_eNNN.json``): the full tool-call result for one item.

Files live under ``<file_root>/archives/<thread_id>/`` so the sandboxed
``read_file`` tool (same root) can read both layers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bcg.agent.tools.file_read_tool import resolve_file_root

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArchiveWriter:
    """Append tool results to a per-thread two-layer archive."""

    def __init__(self, thread_id: str, root: str | Path | None = None) -> None:
        self.thread_id = str(thread_id)
        safe_id = self.thread_id.replace("/", "_").replace(":", "_")
        self.root = resolve_file_root(root)
        self.thread_dir = self.root / "archives" / safe_id
        self.raw_dir = self.thread_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._safe_id = safe_id
        self._seq = 0
        # {tool_name: [{"query": str, "turn": int, "results": [{"raw_url", "summary"}]}]}
        self._tool_queries: dict[str, list[dict[str, Any]]] = {}

    @property
    def tool_index_urls(self) -> dict[str, str]:
        """Sandbox-relative URLs of each per-tool index file."""
        return {
            tool_name: f"file://archives/{self._safe_id}/{tool_name}.json"
            for tool_name in self._tool_queries
        }

    def _next_item_id(self, turn: int) -> str:
        self._seq += 1
        return f"t{turn}_e{self._seq:03d}"

    def _raw_url(self, item_id: str) -> str:
        return f"file://archives/{self._safe_id}/raw/{item_id}.json"

    @staticmethod
    def _summarize(content: str, limit: int = 200) -> str:
        text = " ".join(str(content or "").split())
        return text[:limit] + ("..." if len(text) > limit else "")

    def _find_or_create_query_entry(
        self, tool_name: str, query: str, turn: int
    ) -> dict[str, Any]:
        """Get existing query entry for this turn+query, or create a new one."""
        entries = self._tool_queries.setdefault(tool_name, [])
        for entry in entries:
            if entry["turn"] == turn and entry["query"] == query:
                return entry
        new_entry: dict[str, Any] = {"query": query, "turn": turn, "results": []}
        entries.append(new_entry)
        return new_entry

    def add(
        self,
        *,
        turn: int,
        tool_name: str,
        tool_arguments: dict[str, Any] | None,
        tool_result: str,
        related_belief_nodes: list[str] | None = None,
        call_id: str | None = None,
        global_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Write one layer-2 raw file and append to the tool's query index.

        ``call_id``/``global_call_id`` are optional and archive/log-only (the
        local ``call_id`` -- e.g. "call_1" -- identifies which <tool_call> in
        the turn this result belongs to; ``global_call_id`` additionally
        scopes it to the problem/round for cross-run archive lookups). Not
        passing them preserves the pre-existing raw_record shape exactly.
        """
        item_id = self._next_item_id(turn)
        args = dict(tool_arguments or {})

        raw_record = {
            "item_id": item_id,
            "turn": turn,
            "tool_name": tool_name,
            "tool_arguments": args,
            "tool_result": {"content": str(tool_result or "")},
            "created_at": _now_iso(),
        }
        if call_id:
            raw_record["call_id"] = call_id
        if global_call_id:
            raw_record["global_call_id"] = global_call_id
        self._write_json(self.raw_dir / f"{item_id}.json", raw_record)

        query = str(args.get("query") or args.get("path") or "")
        summary = self._summarize(tool_result)
        result_entry = {"raw_url": self._raw_url(item_id), "summary": summary}

        qe = self._find_or_create_query_entry(tool_name, query, turn)
        qe["results"].append(result_entry)
        self._write_tool_index(tool_name)
        return result_entry

    def add_evidences(
        self,
        *,
        turn: int,
        tool_name: str,
        query: str,
        evidences: list[dict[str, Any]],
        call_id: str | None = None,
        global_call_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Archive each retrieved evidence as its own layer-2 file.

        Groups all evidences under one query entry in the tool index.
        ``call_id``/``global_call_id`` are optional and archive/log-only --
        see ``add()`` docstring. Omitting them keeps the raw_record shape
        identical to before this parameter existed.
        """
        added: list[dict[str, Any]] = []
        qe = self._find_or_create_query_entry(tool_name, query, turn)

        for ev in evidences or []:
            item_id = self._next_item_id(turn)
            text = str(ev.get("text") or "")
            summary = str(ev.get("summary") or "") or self._summarize(text)
            raw_record = {
                "item_id": item_id,
                "turn": turn,
                "tool_name": tool_name,
                "query": query,
                "rank": ev.get("rank"),
                "score": ev.get("score"),
                "url": ev.get("url", ""),
                "evidence": text,
                "summary": summary,
                "created_at": _now_iso(),
            }
            if call_id:
                raw_record["call_id"] = call_id
            if global_call_id:
                raw_record["global_call_id"] = global_call_id
            self._write_json(self.raw_dir / f"{item_id}.json", raw_record)

            result_entry = {"raw_url": self._raw_url(item_id), "summary": summary}
            qe["results"].append(result_entry)
            added.append(result_entry)

        if added:
            self._write_tool_index(tool_name)
        return added

    def _covered_turns(self, tool_name: str) -> list[int]:
        entries = self._tool_queries.get(tool_name, [])
        turns = [e["turn"] for e in entries]
        return [min(turns), max(turns)] if turns else []

    def _write_tool_index(self, tool_name: str) -> None:
        index = {
            "archive_id": f"arch_{self._safe_id}",
            "thread_id": self.thread_id,
            "tool_name": tool_name,
            "covered_turns": self._covered_turns(tool_name),
            "queries": self._tool_queries.get(tool_name, []),
        }
        self._write_json(self.thread_dir / f"{tool_name}.json", index)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("[Archive] Failed to write %s: %s", path, exc)

    @property
    def num_items(self) -> int:
        return self._seq


__all__ = ["ArchiveWriter"]
