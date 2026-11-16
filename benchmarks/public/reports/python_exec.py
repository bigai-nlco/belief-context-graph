"""Offline, task-scoped Python using the existing executor and strict bubblewrap."""

from __future__ import annotations

import asyncio
from pathlib import Path

from plugins.tools._sandbox import BwrapSandbox
from plugins.tools.run_python_code import execute_python_code


class ReportPython:
    def __init__(self, workspace: Path, sources: Path) -> None:
        self.workspace = workspace
        self.sources = sources
        self.lock = asyncio.Lock()

    async def run(self, code: str, timeout: int = 0) -> str:
        if not code.strip():
            return "Error: empty code provided."
        if len(code) > 100_000:
            return "Error: code exceeds 100000 characters."
        timeout = min(timeout, 120) if timeout > 0 else 60
        # Share this task's worktree across calls, but use a fresh private /tmp,
        # PID namespace and network namespace for every execution. Only one
        # Python execution per task at a time, including team subagents.
        async with self.lock:
            sandbox = None
            try:
                # Construction is synchronous so cancellation cannot lose a
                # sandbox allocated in a still-running background constructor.
                sandbox = BwrapSandbox(
                    workspace=self.workspace,
                    binds=((str(self.sources), "/sources", True),),
                    mem_limit_mb=1024, strict_isolation=True,
                )
                result = await execute_python_code(code, timeout, sandbox=sandbox, allow_net=False)
                return result[:20_000] + ("\n[Output truncated.]" if len(result) > 20_000 else "")
            except (RuntimeError, OSError) as exc:
                return f"Error: isolated Python sandbox unavailable ({type(exc).__name__}). No code ran on the host."
            finally:
                if sandbox is not None:
                    await asyncio.to_thread(sandbox.kill)
