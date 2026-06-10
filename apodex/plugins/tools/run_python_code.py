"""Run Python code tool — E2B sandbox executor.

Tool behaviour:

- Tool ``name`` is ``"run_python_code"``.
- Argument is ``code_block`` (not ``code``).
- Default timeout is 300s.
- Output: ``stdout`` only on success, ``Error: <name+value+traceback>``
  on failure. No stderr append, no exit-code prefix.
- E2B template ``bbi714anprk1mspoq12b`` (preinstalled scientific stack).

Used by :mod:`workflows.react_base`.

Required env: ``E2B_API_KEY``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from e2b_code_interpreter import Sandbox
from agent_harness.core.tool import tool

logger = logging.getLogger(__name__)


_E2B_TEMPLATE = "bbi714anprk1mspoq12b"  # pre-baked image with scientific stack
_DEFAULT_TIMEOUT = 300  # seconds


def _format_execution_result(execution: Any) -> str:
    """Format an e2b ``Execution`` object.

    On success returns ``logs.stdout`` joined; on error returns the
    ``execution.error`` repr. No stderr append, no exit-code prefix.
    """
    err = getattr(execution, "error", None)
    if err is not None:
        return f"Error: {err}"

    logs = getattr(execution, "logs", None)
    stdout = ""
    if logs is not None:
        stdout_list = getattr(logs, "stdout", None) or []
        stdout = "".join(stdout_list)

    return stdout if stdout else "Execution completed with no output."


@tool(name="run_python_code")
async def run_python_code(code_block: str) -> str:
    """Run short, safe python code in a sandbox and return the execution result.

    Args:
        code_block: The python code to run.

    Returns:
        On success, the stdout of the executed code.
        On failure, the error details (exception name, value, and traceback).
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        sandbox: Sandbox | None = None
        try:
            # e2b-code-interpreter 2.x: new sandboxes come from Sandbox.create.
            # Offload the sync SDK to a worker thread so a stalled sandbox
            # cannot freeze the event loop.
            sandbox = await asyncio.to_thread(
                Sandbox.create,
                template=_E2B_TEMPLATE,
                timeout=_DEFAULT_TIMEOUT,
            )
            execution = await asyncio.to_thread(sandbox.run_code, code_block)
            return _format_execution_result(execution)
        except Exception as e:
            if attempt == max_attempts:
                return f"[ERROR]: Code execution failed: {str(e)}"
        finally:
            if sandbox:
                try:
                    await asyncio.to_thread(sandbox.kill)
                except Exception:
                    pass

    # Unreachable in practice — the last attempt's ``except`` returns first.
    return "[ERROR]: Code execution failed after all retries."


__all__ = ["run_python_code"]
