"""Task-local source snapshots, with deterministic bounded reading and literal search.

The harness owns this directory outside the model's writable worktree. Only
already-vetted fetch results enter it; neither tool accepts URLs or file paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

MAX_SOURCE_CHARS = 2_000_000
MAX_TOTAL_CHARS = 20_000_000
MAX_SOURCES = 200
MAX_READ_CHARS = 20_000
MAX_READ_LINES = 400


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    title: str
    text: str
    truncated: bool

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines() or [""]

    def metadata(self) -> dict:
        return {
            "source_id": self.source_id, "url": self.url, "title": self.title,
            "chars": len(self.text), "lines": len(self.lines), "truncated": self.truncated,
            "sha256": hashlib.sha256(self.text.encode()).hexdigest(),
            "file": f"/sources/{self.source_id}.txt",
        }

    def header(self) -> str:
        note = "partial source (upstream/cache limit)" if self.truncated else "cached source"
        return (
            f"Source: {self.url}\nSource ID: {self.source_id}\nTitle: {self.title}\n"
            f"{len(self.lines)} lines, {len(self.text)} characters; {note}.\n"
            f"Read-only Python file: /sources/{self.source_id}.txt\n"
        )


class SourceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources: dict[str, Source] = {}
        self._keys: dict[tuple[str, str, bool], str] = {}
        self._chars = 0
        self._seen: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._reads: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=4))
        self._queries: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=8))
        self.seen_chars = 0

    def _observe(self, source: Source, spans: list[tuple[int, int]]) -> int:
        """Union only text actually returned, not the whole fetched document."""
        previous = self._seen[source.source_id]
        before = sum(end - start for start, end in previous)
        merged: list[tuple[int, int]] = []
        for start, end in sorted([*previous, *spans]):
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        self._seen[source.source_id] = merged
        fresh = sum(end - start for start, end in merged) - before
        self.seen_chars += fresh
        return fresh

    @staticmethod
    def _offsets(text: str) -> list[int]:
        offsets, cursor = [], 0
        for line in text.splitlines(keepends=True) or [""]:
            offsets.append(cursor)
            cursor += len(line)
        return offsets

    def access_records(self) -> list[dict]:
        return [{"source_id": s.source_id, "url": s.url, "lines": len(s.lines),
                 "returned_chars": sum(b - a for a, b in self._seen[s.source_id]),
                 "returned_spans": self._seen[s.source_id],
                 "recent_reads": list(self._reads[s.source_id]),
                 "recent_queries": list(self._queries[s.source_id])}
                for s in self.sources.values()]

    def _save_access(self) -> None:
        temp = self.root / "access.pending"
        temp.write_text(json.dumps(self.access_records(), ensure_ascii=False), encoding="utf-8")
        temp.replace(self.root / "access.json")

    def add(self, url: str, text: str, *, title: str = "", truncated: bool = False) -> Source:
        if not text.strip():
            raise ValueError("No readable source content.")
        truncated = truncated or len(text) > MAX_SOURCE_CHARS
        text = text[:MAX_SOURCE_CHARS]
        key = (url, hashlib.sha256(text.encode()).hexdigest(), truncated)
        if key in self._keys:
            return self.sources[self._keys[key]]
        if len(self.sources) >= MAX_SOURCES or self._chars + len(text) > MAX_TOTAL_CHARS:
            raise ValueError("Task source cache limit reached. Read existing sources instead.")
        source = Source(f"S{len(self.sources) + 1}", url, (title or url)[:300], text, truncated)
        (self.root / f"{source.source_id}.txt").write_text(text, encoding="utf-8")
        sources = {**self.sources, source.source_id: source}
        # Never include benchmark reference data or host paths in the index.
        index = self.root / "index.json"
        temp = self.root / "index.pending"
        temp.write_text(json.dumps([s.metadata() for s in sources.values()], ensure_ascii=False,
                                   indent=2), encoding="utf-8")
        temp.replace(index)
        self.sources = sources
        self._keys[key] = source.source_id
        self._chars += len(text)
        return source

    def get(self, source_id: str) -> Source:
        source = self.sources.get(source_id)
        if source is None:
            available = ", ".join(self.sources) or "none yet"
            raise ValueError(f"Unknown source_id for this task. Available IDs: {available}. Use an ID returned by web_fetch.")
        return source

    def preview(self, source: Source, chars: int) -> str:
        size = max(1, min(chars, MAX_READ_CHARS))
        result = source.header() + "\n" + source.text[:size]
        self._observe(source, [(0, min(size, len(source.text)))])
        self._save_access()
        if len(source.text) > size:
            result += (
                "\n[Preview truncated, not the cached source. Use find_in_source to locate "
                "evidence and read_source to continue reading.]"
            )
        return result

    def read(
        self, source_id: str, start_line: int = 1, end_line: int = 0, start_column: int = 0,
    ) -> str:
        source = self.get(source_id)
        lines = source.lines
        offsets = self._offsets(source.text)
        if start_line < 1 or start_line > len(lines) or start_column < 0:
            raise ValueError(f"{source_id} has lines 1–{len(lines)}; column offsets start at 0.")
        if start_column > len(lines[start_line - 1]):
            raise ValueError("start_column exceeds the starting line length.")
        requested_end = end_line or start_line + 99
        if requested_end < start_line:
            raise ValueError("end_line must be at least start_line (or 0 for 100 lines).")
        end = min(requested_end, len(lines), start_line + MAX_READ_LINES - 1)
        out = [source.header(), "Cached text (1-based lines; 0-based column offsets):"]
        remaining = MAX_READ_CHARS
        cursor = None
        spans = []
        last_returned_line = start_line
        for number in range(start_line, end + 1):
            column = start_column if number == start_line else 0
            content = lines[number - 1][column:]
            prefix = f"L{number}" + (f"@{column}" if column else "") + ": "
            room = remaining - len(prefix) - 1
            if room <= 0:
                cursor = (number, column)
                break
            excerpt = content[:room]
            out.append(prefix + excerpt)
            last_returned_line = number
            spans.append((offsets[number - 1] + column, offsets[number - 1] + column + len(excerpt)))
            remaining -= len(prefix) + len(excerpt) + 1
            if len(excerpt) < len(content):
                cursor = (number, column + len(excerpt))
                break
        if cursor is None and end < len(lines):
            cursor = (end + 1, 0)
        if cursor is not None:
            out.append(
                f'Continue: read_source(source_id="{source_id}", '
                f"start_line={cursor[0]}, start_column={cursor[1]})."
            )
        else:
            out.append("[End of cached source.]" if not source.truncated else
                       "[End of partial cached source; uncaptured content is unavailable.]")
        fresh = self._observe(source, spans)
        self._reads[source_id].append({"start_line": start_line, "start_column": start_column,
                                       "last_line": last_returned_line, "new_chars": fresh})
        self._save_access()
        out.append(f"Access record: {fresh} newly returned characters; "
                   + ("this text was already returned earlier. Re-reading may recover context, not new evidence."
                      if fresh == 0 else "returned text is recorded across turns; this does not establish task completion."))
        return "\n".join(out)

    def find(self, source_id: str, query: str | list[str], start_line: int = 1,
             max_matches: int = 10, *, match_mode: str = "phrase", offset: int = 0) -> str:
        source = self.get(source_id)
        if match_mode not in {"phrase", "any", "all"}:
            raise ValueError("match_mode must be phrase, any, or all.")
        if match_mode == "phrase":
            if not isinstance(query, str):
                raise ValueError("phrase mode takes one literal string.")
            terms = [query.strip()]
        elif isinstance(query, str):
            terms = query.split()
        elif isinstance(query, list) and all(isinstance(t, str) for t in query):
            terms = [t.strip() for t in query]
        else:
            raise ValueError("query must be a string or a list of keyword/phrase strings.")
        terms = list(dict.fromkeys(t.casefold() for t in terms))
        if not terms or len(terms) > 16 or any(not t or "\n" in t or "\r" in t for t in terms) or sum(map(len, terms)) > 500:
            raise ValueError("Use 1–16 nonempty single-line terms, at most 500 characters in total.")
        lines = source.lines
        if start_line < 1 or start_line > len(lines):
            raise ValueError("start_line is outside this source.")
        if offset < 0:
            raise ValueError("offset must be nonnegative.")
        limit = max(1, min(max_matches, 20))
        patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in terms]
        matches = []
        for number, line in enumerate(lines[start_line - 1:], start_line):
            found = [(term, m) for term, pattern in zip(terms, patterns, strict=True) if (m := pattern.search(line))]
            if not found or (match_mode == "all" and len(found) != len(terms)):
                continue
            first = min(m.start() for _, m in found)
            left, right = max(0, first - 160), min(len(line), first + 500 + 160)
            matches.append((number, left, right, first, [t for t, _ in found]))
        # Exact match evidence only. Prefer lines covering more supplied terms;
        # offset pagination avoids skipping lower-ranked earlier lines.
        matches.sort(key=lambda m: (-len(m[4]), m[0]))
        selected = matches[offset:offset + limit]
        positions = self._offsets(source.text)
        fresh = self._observe(source, [(positions[n - 1] + left, positions[n - 1] + right)
                                        for n, left, right, _, _ in selected])
        query_record = {"terms": terms, "mode": match_mode, "start_line": start_line,
                        "matches": len(matches), "offset": offset}
        repeated = any(all(old.get(k) == query_record[k] for k in ("terms", "mode", "start_line"))
                       for old in self._queries[source_id])
        self._queries[source_id].append(query_record)
        self._save_access()
        hits = [f"L{n} column {column} [{len(found)}/{len(terms)} terms match the full line]: {lines[n - 1][left:right]}"
                for n, left, right, column, found in selected]
        if offset + limit < len(matches):
            hits.append(f"More matches: repeat the same query/mode/start_line with offset={offset + limit}.")
            if match_mode == "phrase" and selected:
                hits.append(f"Alternatively continue phrase lookup at start_line={selected[-1][0] + 1}, offset=0.")
        if not selected:
            hits.append("No matching text in this page of cached-source matches. This does not establish absence of the topic.")
            if match_mode == "phrase":
                hits.append("This was exact phrase matching, not keyword search. Try match_mode=any with separate keywords.")
        return (source.header() + f"\nSearch mode: {match_mode}; {len(matches)} matching lines; "
                f"offset={offset}; newly returned characters={fresh}."
                + (" This source/query/mode was searched before." if repeated else "") + "\n" + "\n".join(hits))
