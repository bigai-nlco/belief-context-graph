"""Collect each BCG task's Graph Construction artifacts into its run directory.

The Graph server is a shared, long-lived process that writes every session to
``<graph store>/<problem_id>/`` (by default ``~/.bcg/graphs``), where the
Agent's ``problem_id`` is ``<agent session id>:<random uuid>`` and ``:``
becomes ``_`` on disk. Those names say nothing about the benchmark or task.

The Agent's JSON-mode trajectory starts with its session header, so the
session id identifies the task's Graph directory exactly. After the Agent has
released its Graph session (it disappears from the server's ``/health``
``all`` list), the directory is moved to::

    <output-dir>/graphs/<benchmark>/<mode>/<task>/

next to ``<output-dir>/<benchmark>/<mode>/``, together with a
``benchmark_link.json`` naming the task result, model-I/O, and trajectory
files (paths relative to the Graph directory). Nothing on the Graph server
side changes, so already-running servers keep working.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

GRAPHS_DIRNAME = "graphs"
LINK_NAME = "benchmark_link.json"
REQUIRED_FILES = ("trajectory_stream.jsonl", "belief_graph.jsonl")


def default_graph_store() -> Path:
    from bcg.apps.agent_runtime import _state_root

    return _state_root() / "graphs"


def session_id_from_trajectory(text: str) -> str | None:
    """The Agent session id from the JSON-mode header line."""

    for line in text.splitlines()[:5]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("type") == "session"
            and event.get("id")
        ):
            return str(event["id"])
    return None


def find_graph_dirs(store: Path, session_id: str) -> list[Path]:
    if not store.is_dir():
        return []
    return sorted(
        (path for path in store.glob(f"{session_id}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )


def _live_problem_ids(graph_url: str) -> list[str] | None:
    try:
        with urlopen(f"{graph_url.rstrip('/')}/health", timeout=2.0) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError):
        return None
    ids = value.get("all") if isinstance(value, dict) else None
    return [str(item) for item in ids] if isinstance(ids, list) else None


def wait_for_release(graph_url: str | None, session_id: str, timeout: float) -> bool:
    """True once no server session for ``session_id`` remains (or no server)."""

    if not graph_url:
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    prefix = f"{session_id}:"
    while True:
        live = _live_problem_ids(graph_url)
        if live is None or not any(item.startswith(prefix) for item in live):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def _relative(target: Path | str | None, start: Path) -> str | None:
    if not target:
        return None
    import os

    return os.path.relpath(Path(target).resolve(), start.resolve()).replace("\\", "/")


def collect_task_graph(
    *,
    trajectory_text: str,
    output_dir: Path,
    benchmark: str,
    mode: str,
    task_id: str,
    task_dir_name: str,
    graph_store: Path | None = None,
    graph_url: str | None = None,
    release_timeout: float = 60.0,
    copy: bool = False,
    links: dict[str, Path | str | None] | None = None,
) -> dict[str, Any]:
    """Move (or copy) one task's Graph directory into the run directory.

    Returns ``{"status", "graph_dir", "source", "note"}``. ``status`` is one of
    ``moved``, ``copied``, ``copied_active`` (the server still held the
    session, so the source was left in place), ``no_session_id``,
    ``not_found``, or ``error``.
    """

    store = Path(graph_store or default_graph_store()).expanduser()
    session_id = session_id_from_trajectory(trajectory_text)
    if not session_id:
        return {
            "status": "no_session_id",
            "graph_dir": None,
            "source": None,
            "note": "The trajectory has no Agent session header.",
        }
    sources = find_graph_dirs(store, session_id)
    if not sources:
        return {
            "status": "not_found",
            "graph_dir": None,
            "source": None,
            "note": f"No {session_id}_* directory under {store}.",
        }
    source = sources[-1]
    note = (
        f"{len(sources)} Graph directories matched; collected the newest."
        if len(sources) > 1
        else None
    )
    released = wait_for_release(graph_url, session_id, release_timeout)
    target = (
        Path(output_dir).expanduser().resolve()
        / GRAPHS_DIRNAME
        / benchmark
        / mode
        / task_dir_name
    )
    try:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy or not released:
            shutil.copytree(source, target)
            status = "copied" if released else "copied_active"
        else:
            shutil.move(str(source), str(target))
            status = "moved"
    except OSError as exc:
        return {
            "status": "error",
            "graph_dir": None,
            "source": str(source),
            "note": str(exc),
        }
    link = {
        "benchmark": benchmark,
        "mode": mode,
        "task_id": task_id,
        "agent_session_id": session_id,
        "graph_problem_dir": source.name,
        "source": str(source),
        "collection": status,
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # Relative to this directory, so the run directory can be moved.
        **{key: _relative(value, target) for key, value in (links or {}).items()},
    }
    (target / LINK_NAME).write_text(
        json.dumps(link, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        note = "; ".join(filter(None, [note, f"missing {', '.join(missing)}"]))
    return {
        "status": status,
        "graph_dir": str(target),
        "source": str(source),
        "note": note,
    }


def collect_run_graphs(
    run_dir: Path,
    *,
    graph_store: Path | None = None,
    graph_url: str | None = None,
    copy: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Backfill Graph directories for every BCG task result in ``run_dir``."""

    run_dir = Path(run_dir).expanduser().resolve()
    outcomes: list[dict[str, Any]] = []
    for result_path in sorted(run_dir.glob("*/*/tasks/*.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("mode") != "bcg":
            continue
        benchmark = result_path.parents[2].name
        mode = result_path.parents[1].name
        task_dir_name = result_path.stem
        target = run_dir / GRAPHS_DIRNAME / benchmark / mode / task_dir_name
        if target.is_dir() and not force:
            outcomes.append(
                {
                    "task": f"{benchmark}/{mode}/{result.get('task_id')}",
                    "status": "already_collected",
                    "graph_dir": str(target),
                }
            )
            continue
        trajectory = Path(str(result.get("trajectory") or ""))
        if not trajectory.is_file():
            # The run directory may have moved since the run.
            candidate = result_path.parents[1] / "trajectories" / trajectory.name
            trajectory = candidate if candidate.is_file() else trajectory
        if not trajectory.is_file():
            outcomes.append(
                {
                    "task": f"{benchmark}/{mode}/{result.get('task_id')}",
                    "status": "no_trajectory",
                    "graph_dir": None,
                }
            )
            continue
        outcome = collect_task_graph(
            trajectory_text=trajectory.read_text(encoding="utf-8", errors="replace"),
            output_dir=run_dir,
            benchmark=benchmark,
            mode=mode,
            task_id=str(result.get("task_id")),
            task_dir_name=task_dir_name,
            graph_store=graph_store,
            graph_url=graph_url,
            release_timeout=0.0,
            copy=copy,
            links=task_links(result_path, result),
        )
        if outcome.get("graph_dir"):
            result["graph_dir"] = outcome["graph_dir"]
            result["graph_collection"] = outcome["status"]
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        outcomes.append(
            {"task": f"{benchmark}/{mode}/{result.get('task_id')}", **outcome}
        )
    return outcomes


def task_links(
    result_path: Path, result: dict[str, Any]
) -> dict[str, Path | str | None]:
    """Files a viewer needs next to a Graph, resolved against the mode directory."""

    mode_dir = result_path.parents[1]

    def local(key: str, subdir: str) -> Path | None:
        value = result.get(key)
        if not value:
            return None
        path = Path(str(value))
        if path.is_file():
            return path
        candidate = mode_dir / subdir / path.name
        return candidate if candidate.is_file() else None

    return {
        "task_result": result_path,
        "model_io": local("model_io_trace", "model-io"),
        "trajectory": local("trajectory", "trajectories"),
        "graph_contexts": local("graph_context_trace", "graph-contexts"),
        "report": local("report", "reports"),
    }


__all__ = [
    "GRAPHS_DIRNAME",
    "LINK_NAME",
    "collect_run_graphs",
    "collect_task_graph",
    "default_graph_store",
    "find_graph_dirs",
    "session_id_from_trajectory",
    "task_links",
]
