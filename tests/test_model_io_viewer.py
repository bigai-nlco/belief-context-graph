from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import pytest

from bcg.apps.model_io_viewer import (
    MODEL_IO_SCHEMA,
    ModelIoViewerError,
    create_model_io_viewer_server,
    discover_model_io_traces,
    load_model_io_trace,
    render_model_io_viewer,
    resolve_model_io_trace,
)


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "schema": MODEL_IO_SCHEMA,
            "type": "request",
            "call_id": 1,
            "timestamp": "2026-08-18T00:00:00Z",
            "model": {"id": "test-model", "provider": "benchmark"},
            "payload": {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Question"},
                ],
                "tools": [{"type": "function", "function": {"name": "search"}}],
            },
        },
        {
            "schema": MODEL_IO_SCHEMA,
            "type": "response",
            "call_id": 1,
            "timestamp": "2026-08-18T00:00:01Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer"}],
                "stopReason": "stop",
                "usage": {"totalTokens": 42},
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_load_model_io_trace_pairs_request_and_response(tmp_path: Path) -> None:
    trace = tmp_path / "model-io" / "task.jsonl"
    _write_trace(trace)

    calls = load_model_io_trace(trace)

    assert len(calls) == 1
    assert calls[0]["request"]["payload"]["messages"][0]["content"] == "System prompt"
    assert calls[0]["response"]["message"]["content"][0]["text"] == "Answer"


def test_viewer_resolves_task_result_and_renders_portable_html(tmp_path: Path) -> None:
    trace = tmp_path / "run" / "benchmark" / "default" / "model-io" / "task.jsonl"
    _write_trace(trace)
    task_result = tmp_path / "task.json"
    task_result.write_text(json.dumps({"model_io_trace": str(trace)}), encoding="utf-8")

    output = render_model_io_viewer(task_result, output=tmp_path / "viewer.html")
    html = output.read_text(encoding="utf-8")

    assert resolve_model_io_trace(task_result) == trace.resolve()
    assert "BCG Model I/O Viewer" in html
    assert "System prompt" in html
    assert "Answer" in html
    assert "https://" not in html


def test_static_resolver_requires_task_filter_when_multiple_traces(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path / "a" / "model-io" / "task-a.jsonl")
    _write_trace(tmp_path / "b" / "model-io" / "task-b.jsonl")

    with pytest.raises(ModelIoViewerError, match="use --task"):
        resolve_model_io_trace(tmp_path)

    selected = resolve_model_io_trace(tmp_path, task="task-b")
    assert selected.name == "task-b.jsonl"


def test_directory_viewer_lists_tasks_and_loads_selected_trace_lazily(
    tmp_path: Path,
) -> None:
    default_trace = (
        tmp_path
        / "browsecomp"
        / "default"
        / "model-io"
        / "browsecomp-default-browsecomp-0001.jsonl"
    )
    bcg_trace = (
        tmp_path
        / "browsecomp"
        / "bcg"
        / "model-io"
        / "browsecomp-bcg-browsecomp-0002.jsonl"
    )
    _write_trace(default_trace)
    _write_trace(bcg_trace)

    entries = discover_model_io_traces(tmp_path)
    assert [(entry["task_id"], entry["mode"]) for entry in entries] == [
        ("browsecomp-0001", "default"),
        ("browsecomp-0002", "bcg"),
    ]

    server, url = create_model_io_viewer_server(
        tmp_path,
        task="browsecomp-0002",
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(url, timeout=2) as response:  # noqa: S310 - local test server
            page = response.read().decode("utf-8")
        assert "Task ID" in page
        assert "browsecomp-0001 · default / browsecomp" in page
        assert '"initial_task_key":"0"' in page

        key = next(
            entry["key"] for entry in entries if entry["task_id"] == "browsecomp-0002"
        )
        with urlopen(  # noqa: S310 - local test server
            f"{url}api/task?key={key}", timeout=2
        ) as response:
            selected = json.loads(response.read())
        assert selected["task_id"] == "browsecomp-0002"
        assert selected["mode"] == "bcg"
        assert selected["calls"][0]["call_id"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
