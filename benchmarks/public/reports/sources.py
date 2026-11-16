"""Task-local source snapshots, with deterministic bounded reading and literal search.

The harness owns this directory outside the model's writable worktree. Only
already-vetted fetch results enter it; neither tool accepts URLs or file paths.
"""

from __future__ import annotations

import hashlib
import json
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
            raise ValueError("Unknown source_id for this task. Use an ID returned by web_fetch.")
        return source

    def preview(self, source: Source, chars: int) -> str:
        size = max(1, min(chars, MAX_READ_CHARS))
        result = source.header() + "\n" + source.text[:size]
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
        if start_line < 1 or start_line > len(lines) or start_column < 0:
            raise ValueError("Line numbers start at 1; column offsets start at 0.")
        if start_column > len(lines[start_line - 1]):
            raise ValueError("start_column exceeds the starting line length.")
        requested_end = end_line or start_line + 99
        if requested_end < start_line:
            raise ValueError("end_line must be at least start_line (or 0 for 100 lines).")
        end = min(requested_end, len(lines), start_line + MAX_READ_LINES - 1)
        out = [source.header(), "Cached text (1-based lines; 0-based column offsets):"]
        remaining = MAX_READ_CHARS
        cursor = None
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
        return "\n".join(out)

    def find(self, source_id: str, query: str, start_line: int = 1, max_matches: int = 10) -> str:
        source = self.get(source_id)
        if not query.strip() or len(query) > 500 or "\n" in query or "\r" in query:
            raise ValueError("query must be a nonempty single-line literal phrase (at most 500 chars).")
        if start_line < 1 or start_line > len(source.lines):
            raise ValueError("start_line is outside this source.")
        limit = max(1, min(max_matches, 20))
        hits = []
        # Regex is escaped; no user-controlled patterns or expensive fuzzy/LLM search.
        import re

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for number, line in enumerate(source.lines[start_line - 1:], start_line):
            match = pattern.search(line)
            if match is None:
                continue
            column = match.start()
            left, right = max(0, column - 160), min(len(line), match.end() + 160)
            hits.append(f"L{number} column {column}: {line[left:right]}")
            if len(hits) >= limit:
                if number < len(source.lines):
                    hits.append(f"More matches may exist; continue find_in_source at start_line={number + 1}.")
                break
        return source.header() + "\n" + ("\n".join(hits) if hits else "No matching text in the cached source.")
