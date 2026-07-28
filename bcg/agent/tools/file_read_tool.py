"""Sandboxed local file reader tool.

Exposes a ``read_file`` tool that lets the agent read UTF-8 text files from a
single sandboxed workspace directory, addressed by a ``file://`` style URL that
is *relative to the sandbox root*. All access is confined to the root; any
attempt to escape it (absolute paths, ``..`` traversal, ``~`` expansion, or a
symlink pointing outside the root) is rejected.

The same root is used by the two-layer archive (see ``archive.py``), so the
agent can read ``file://archives/<thread>/manifest.json`` and the raw evidence
files it points to.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput

_DEFAULT_ROOT = "ai_workspace"
_DEFAULT_MAX_BYTES = 256 * 1024


def resolve_file_root(root: str | Path | None = None) -> Path:
    """Resolve the sandbox root with priority: arg > env > default.

    The directory is created if it does not exist so downstream archive writes
    and reads have a stable home.
    """
    value = (
        root
        or os.environ.get("BELIEF_TRACER_FILE_ROOT")
        or _DEFAULT_ROOT
    )
    resolved = Path(value).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


class FileReadTool(Tool):
    """Read a UTF-8 text file from the sandboxed AI workspace by URL."""

    NAME = "read_file"
    # This tool only recalls already-stored content (e.g. archived evidence the
    # search tool already pushed to memory). Its results must NOT be re-archived
    # or re-pushed to the belief graph — that would be redundant and bias/slow
    # the graph. The model still sees the result in the live context.
    FEEDS_MEMORY = False
    DESCRIPTION = (
        "Read a UTF-8 text file from the local AI workspace by URL. Only files "
        "under the sandboxed workspace root are accessible. Use this to open an "
        "archive manifest (file://archives/<thread>/manifest.json) and then the "
        "raw_url of any item you need in full."
    )

    def __init__(
        self,
        name: str = NAME,
        description: str | None = None,
        root: str | Path | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.root = resolve_file_root(root)
        self.max_bytes = int(max_bytes if max_bytes is not None else _DEFAULT_MAX_BYTES)
        super().__init__(name=name, description=description or self.DESCRIPTION)

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
                        "url": {
                            "type": "string",
                            "description": (
                                "file://<path-relative-to-workspace>, e.g. "
                                "file://archives/<thread>/raw/e001.json. Bare "
                                "relative paths are also accepted."
                            ),
                        },
                        "max_bytes": {
                            "type": "integer",
                            "description": (
                                f"Truncate the file after this many bytes "
                                f"(default {self.max_bytes})."
                            ),
                            "minimum": 1,
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    def set_task(self, task: dict[str, Any]) -> None:
        # No per-task state needed; method exists so MultiTool's set_task fan-out
        # does not skip this tool.
        return None

    def usage_prompt(self, hyde: bool = False, detail: bool = True) -> str:
        """Self-describing usage text for this tool (dynamically injected).

        ``detail=False`` returns the brief variant (one-line purpose + tool-call
        JSON shape) for the system prompt's tool listing; ``detail=True`` (the
        default) returns the full guidance for the user message.
        """
        if not detail:
            return (
                "- read_file: read a file from the AI workspace (e.g. an archive "
                "manifest). Call as:\n"
                "  <tool_call>\n"
                '  {"name": "read_file", "arguments": {"url": "file://..."}}\n'
                "  </tool_call>"
            )
        return (
            "- read_file(url: string): read a UTF-8 text file from the AI workspace, "
            "e.g. an archive manifest (file://archives/<thread>/manifest.json) or an "
            "item's raw_url. Only files under the sandboxed workspace are accessible.\n"
            "  Example:\n"
            "  <tool_call>\n"
            '  {"name": "read_file", "arguments": {"url": "file://archives/<thread>/manifest.json"}}\n'
            "  </tool_call>"
        )

    def _resolve(self, url: str) -> Path | None:
        """Map a sandbox URL to an absolute path, or None if it escapes the root.

        Rejects: non-string/empty input, http(s) URLs, absolute paths,
        ``file:///abs`` forms, ``~`` expansion, and any path (or symlink target)
        that resolves outside the sandbox root.
        """
        if not isinstance(url, str) or not url.strip():
            return None
        rel = url.strip()

        # Reject network schemes outright.
        lowered = rel.lower()
        if lowered.startswith(("http://", "https://", "ftp://")):
            return None

        # Strip a leading file:// (but not file:///abs, which is an absolute path).
        if lowered.startswith("file://"):
            rel = rel[len("file://"):]
            if rel.startswith("/"):
                # file:///etc/passwd -> absolute, not allowed.
                return None

        # Reject absolute paths and home expansion before joining.
        if rel.startswith("~") or os.path.isabs(rel):
            return None

        root = self.root.resolve()
        candidate = (root / rel).resolve()

        # Containment check (handles ../ traversal and absolute escapes).
        if not candidate.is_relative_to(root):
            return None

        # Symlink-escape check: the realpath of the final file must also be
        # inside the root, so a symlink under the root can't point outside it.
        try:
            real = Path(os.path.realpath(candidate))
        except OSError:
            return None
        if not real.is_relative_to(root):
            return None

        return candidate

    def forward(self, url: str, max_bytes: int | None = None) -> ToolOutput:
        path = self._resolve(url)
        if path is None:
            return ToolOutput(
                name=self.name,
                error=f"Path not allowed or escapes workspace: {url!r}",
            )
        if not path.is_file():
            return ToolOutput(name=self.name, error=f"File not found: {url}")

        limit = int(max_bytes) if max_bytes is not None else self.max_bytes
        limit = max(1, limit)
        try:
            data = path.read_text("utf-8", errors="replace")
        except OSError as exc:
            return ToolOutput(name=self.name, error=f"Failed to read {url}: {exc}")

        truncated = len(data) > limit
        if truncated:
            data = data[:limit]

        # Wrap content so the model treats it as untrusted data, not instructions.
        body = f'<file_content url="{url}">\n{data}\n</file_content>'
        if truncated:
            body += f"\n[truncated at {limit} bytes]"
        return ToolOutput(
            name=self.name,
            output=body,
            metadata={"url": url, "bytes": len(data), "truncated": truncated},
        )


__all__ = ["FileReadTool", "resolve_file_root"]
