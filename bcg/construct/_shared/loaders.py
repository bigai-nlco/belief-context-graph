"""
loaders.py  (v3)
================
Input adapters. There are NO scenarios anymore: every input is normalised into
a flat list of items, each a flat list of role-tagged turns.

Accepted inputs
---------------
* A trajectory:  {"trajectory": [ {role, content}, ... ]}  (also a bare list of
  messages, or {"messages": [...]}). → ONE item.
* Multi-session QA data: a list of items, each with `sessions` (a list of
  sessions, each a list of {role, content, has_answer}), plus parallel
  `session_ids` / `dates` arrays and question metadata. Each item's sessions are
  FLATTENED (chronologically by date, by default) into ONE turn stream; each
  turn carries its session's date so time attribution still works.

Output item shape
-----------------
    {
      "item_id": "<id>",
      "meta":    { ...carried metadata... },
      "turns":   [ {"role": str, "content": str,
                    "date": str|None, "has_answer": bool|None}, ... ],
      "order_sorted": bool,   # whether date-sorting was applied
    }
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any


def load_input_file(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Date parsing (for chronological flattening of multi-session items)
# ---------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d",
)
_PAREN_DAY_RE = re.compile(r"\s*\([A-Za-z]{3,}\)\s*")


def parse_date(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    for cand in (s.strip(), _PAREN_DAY_RE.sub(" ", s).strip()):
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(cand, fmt)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Shape detection (internal only — not a user-facing "scenario")
# ---------------------------------------------------------------------------


def _is_sessioned(data: Any) -> bool:
    probe: dict[str, Any] | None = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        probe = data[0]
    elif isinstance(data, dict):
        probe = data
    return bool(probe is not None and "sessions" in probe)


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


def _turns_from_messages(messages: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append(
                {
                    "role": m.get("role") or "user",
                    "content": m.get("content", "") or "",
                    "date": m.get("date"),
                    "has_answer": m.get("has_answer"),
                }
            )
    return out


def _flatten_sessions(
    raw: dict[str, Any], keep_order: bool
) -> tuple[list[dict[str, Any]], bool]:
    sessions_raw = raw.get("sessions") or []
    raw.get("session_ids") or []
    dates = raw.get("dates") or []

    sess: list[dict[str, Any]] = []
    for i, turns_raw in enumerate(sessions_raw):
        date = dates[i] if i < len(dates) else None
        sess.append(
            {
                "date": date,
                "date_parsed": parse_date(date),
                "turns": [
                    {
                        "role": t.get("role") or "user",
                        "content": t.get("content", "") or "",
                        "date": date,
                        "has_answer": t.get("has_answer"),
                    }
                    for t in (turns_raw or [])
                    if isinstance(t, dict)
                ],
            }
        )

    order_sorted = False
    if (not keep_order) and sess and all(s["date_parsed"] is not None for s in sess):
        sess = sorted(sess, key=lambda s: s["date_parsed"])
        order_sorted = True

    flat: list[dict[str, Any]] = []
    for s in sess:
        flat.extend(s["turns"])
    return flat, order_sorted


def iter_items(data: Any, keep_order: bool = False) -> list[dict[str, Any]]:
    """Normalise ANY accepted input into a list of items (see module docstring)."""
    if _is_sessioned(data):
        raw_items = data if isinstance(data, list) else [data]
        items: list[dict[str, Any]] = []
        for idx, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("question_id") or raw.get("item_id") or f"item_{idx}")
            turns, order_sorted = _flatten_sessions(raw, keep_order)
            meta = {
                k: raw.get(k)
                for k in (
                    "question_id",
                    "question_type",
                    "question",
                    "answer",
                    "question_date",
                    "answer_session_ids",
                )
                if k in raw
            }
            items.append(
                {
                    "item_id": item_id,
                    "meta": meta,
                    "turns": turns,
                    "order_sorted": order_sorted,
                }
            )
        return items

    # trajectory / bare list / {"messages": ...}
    if isinstance(data, dict):
        messages = data.get("trajectory") or data.get("messages") or []
        item_id = str(data.get("item_id") or data.get("problem_id") or "trajectory")
    elif isinstance(data, list):
        messages = data
        item_id = "trajectory"
    else:
        messages, item_id = [], "trajectory"
    return [
        {
            "item_id": item_id,
            "meta": {},
            "turns": _turns_from_messages(messages),
            "order_sorted": False,
        }
    ]


def select_items(
    items: list[dict[str, Any]], selector: str | None
) -> list[dict[str, Any]]:
    """Filter items by id or by 0-based index. None selects all."""
    if selector is None or selector == "":
        return items
    by_id = [it for it in items if it["item_id"] == selector]
    if by_id:
        return by_id
    try:
        i = int(selector)
        if 0 <= i < len(items):
            return [items[i]]
    except ValueError:
        pass
    raise KeyError(
        f"--item {selector!r} matches no item id and is not a valid index "
        f"(0..{len(items) - 1})"
    )


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "item"
