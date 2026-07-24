"""Tests for the layered-context / archive / four-stage retrieval additions.

These avoid importing rllm (not installed in CI for this package) by stubbing the
minimal ``rllm.tools.tool_base`` surface that ``file_read_tool`` / ``archive``
depend on. The CLI tests use the real ``rollout._parse_args`` path.
"""

from __future__ import annotations

import sys
import types
import json
import tempfile
from pathlib import Path

import pytest


def _stub_rllm() -> None:
    # If a usable rllm.tools.tool_base is already importable (real package or a
    # prior stub), keep it. Only stub when Tool/ToolOutput can't be imported —
    # this avoids a half-initialized rllm module (left by other tests) breaking
    # `from rllm.tools.tool_base import Tool`.
    try:
        from rllm.tools.tool_base import Tool, ToolOutput  # noqa: F401
        return
    except Exception:
        pass
    rllm = types.ModuleType("rllm")
    tools = types.ModuleType("rllm.tools")
    tool_base = types.ModuleType("rllm.tools.tool_base")

    class Tool:
        def __init__(self, name=None, description=None):
            self.name = name
            self.description = description

    class ToolOutput:
        def __init__(self, name=None, output=None, error=None, metadata=None):
            self.name = name
            self.output = output
            self.error = error
            self.metadata = metadata

    tool_base.Tool = Tool
    tool_base.ToolOutput = ToolOutput
    sys.modules["rllm"] = rllm
    sys.modules["rllm.tools"] = tools
    sys.modules["rllm.tools.tool_base"] = tool_base


@pytest.fixture()
def workspace():
    _stub_rllm()
    return Path(tempfile.mkdtemp())


def test_file_read_sandbox_blocks_escape(workspace):
    from bcg.agent.tools.file_read_tool import FileReadTool

    (workspace / "ok.txt").write_text("hello", encoding="utf-8")
    tool = FileReadTool(root=workspace)

    assert tool.forward(url="file://ok.txt").output is not None
    for bad in (
        "file://../../../etc/passwd",
        "../etc/passwd",
        "/etc/passwd",
        "file:///etc/passwd",
        "~/secret",
        "http://evil.com/x",
        "",
    ):
        assert tool.forward(url=bad).error is not None, bad


def test_file_read_blocks_symlink_escape(workspace):
    import os
    from bcg.agent.tools.file_read_tool import FileReadTool

    outside = Path(tempfile.mkdtemp()) / "secret.txt"
    outside.write_text("SECRET", encoding="utf-8")
    os.symlink(outside, workspace / "link.txt")

    tool = FileReadTool(root=workspace)
    assert tool.forward(url="file://link.txt").error is not None


def test_archive_two_layers_and_readback(workspace):
    from bcg.agent.archive import ArchiveWriter
    from bcg.agent.tools.file_read_tool import FileReadTool

    aw = ArchiveWriter("averitec:abcd1234", root=workspace)
    aw.add(turn=1, tool_name="averitec_search",
           tool_arguments={"query": "q" * 200, "top_k": 10},
           tool_result="[Evidence 1] " + "lorem " * 60)
    aw.add(turn=2, tool_name="averitec_search",
           tool_arguments={"query": "second"}, tool_result="another")

    index = json.loads((aw.thread_dir / "averitec_search.json").read_text())
    assert index["covered_turns"] == [1, 2]
    assert len(index["queries"]) == 2
    first_result = index["queries"][0]["results"][0]
    assert first_result["raw_url"].endswith("raw/t1_e001.json")
    raw = json.loads((aw.raw_dir / "t1_e001.json").read_text())
    assert len(raw["tool_arguments"]["query"]) == 200

    tool = FileReadTool(root=workspace)
    assert tool.forward(url=aw.tool_index_urls["averitec_search"]).error is None
    assert tool.forward(url=first_result["raw_url"]).error is None


def test_archive_per_evidence(workspace):
    from bcg.agent.archive import ArchiveWriter

    aw = ArchiveWriter("5:0:abcd", root=workspace)
    evs = [
        {"rank": 1, "score": 0.99, "text": "France cancelled visas of 183 Pakistani citizens.",
         "url": "http://a", "summary": "France revoked 183 Pakistani visas."},
        {"rank": 2, "score": 0.5, "text": "Unrelated weather text.", "url": "http://b", "summary": ""},
    ]
    aw.add_evidences(turn=1, tool_name="averitec_search",
                     query="Did France cancel 183 visas?", evidences=evs)

    index = json.loads((aw.thread_dir / "averitec_search.json").read_text())
    # One query entry, with one result per evidence (not per search call).
    assert len(index["queries"]) == 1
    results = index["queries"][0]["results"]
    assert len(results) == 2
    raws = sorted(aw.raw_dir.glob("*.json"))
    assert len(raws) == 2
    # First evidence carries its own summary (judge summary when present).
    assert results[0]["summary"] == "France revoked 183 Pakistani visas."
    # Second evidence had no judge summary -> falls back to a snippet (non-empty).
    assert results[1]["summary"]
    # Each raw holds a single evidence string, not a concatenated blob.
    r0 = json.loads(raws[0].read_text())
    assert isinstance(r0["evidence"], str)
    assert "183 Pakistani" in r0["evidence"]


def test_archive_add_evidences_with_call_id(workspace):
    from bcg.agent.archive import ArchiveWriter

    aw = ArchiveWriter("claim469:abcd1234", root=workspace)
    evs = [{"rank": 1, "score": 0.9, "text": "evidence text", "url": "http://a", "summary": ""}]
    aw.add_evidences(
        turn=1, tool_name="averitec_search", query="q", evidences=evs,
        call_id="call_1", global_call_id="claim469:abcd1234_round1_call_1",
    )

    raws = sorted(aw.raw_dir.glob("*.json"))
    assert len(raws) == 1
    raw = json.loads(raws[0].read_text())
    assert raw["call_id"] == "call_1"
    assert raw["global_call_id"] == "claim469:abcd1234_round1_call_1"


def test_archive_add_evidences_without_call_id_unchanged(workspace):
    """Omitting call_id/global_call_id (the pre-existing call signature)
    must produce a raw_record identical in shape to before those params
    existed -- no empty/None call_id key leaking in."""
    from bcg.agent.archive import ArchiveWriter

    aw = ArchiveWriter("legacy:xyz", root=workspace)
    evs = [{"rank": 1, "score": 0.9, "text": "evidence text", "url": "http://a", "summary": ""}]
    aw.add_evidences(turn=1, tool_name="averitec_search", query="q", evidences=evs)

    raws = sorted(aw.raw_dir.glob("*.json"))
    raw = json.loads(raws[0].read_text())
    assert "call_id" not in raw
    assert "global_call_id" not in raw


def test_archive_add_with_call_id(workspace):
    from bcg.agent.archive import ArchiveWriter

    aw = ArchiveWriter("legacy:abc", root=workspace)
    aw.add(
        turn=1, tool_name="read_file", tool_arguments={"url": "file://x"},
        tool_result="content", call_id="call_2", global_call_id="legacy:abc_round1_call_2",
    )
    raws = sorted(aw.raw_dir.glob("*.json"))
    raw = json.loads(raws[0].read_text())
    assert raw["call_id"] == "call_2"
    assert raw["global_call_id"] == "legacy:abc_round1_call_2"


def test_cli_four_stage_and_archive_flags():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args([
        "--model", "/m/Qwen", "--no-auto-ui", "--retrieval-method", "hero4",
        "--stage1-bm25-k", "500", "--stage2-embed-k", "32", "--stage3-rerank-k", "8",
        "--no-judge", "--enable-archive", "--recent-turns", "3",
        "--file-tool-root", "/tmp/ws",
    ])
    assert cfg.retrieval_method == "hero4"
    assert (cfg.stage1_bm25_k, cfg.stage2_embed_k, cfg.stage3_rerank_k) == (500, 32, 8)
    assert cfg.enable_judge is False
    assert cfg.enable_archive is True
    assert cfg.enable_file_read is True  # implied by --enable-archive
    assert cfg.recent_turns == 3


def test_cli_defaults_enable_archive_with_two_recent_turns():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args(["--model", "/m/Qwen", "--no-auto-ui"])
    assert cfg.retrieval_method == "bm25"
    assert cfg.enable_archive is True
    assert cfg.enable_file_read is True
    assert cfg.recent_turns == 2
    # Default keeps sequential tool-call execution (no behavior change).
    assert cfg.max_tool_workers == 1
    # Retained for legacy --no-archive runs; archive mode aligns updates to eviction.
    assert cfg.belief_graph_interval == 1
    # Default keeps layered graph placement as the historical user message.
    assert cfg.belief_graph_placement == "user"


def test_cli_can_disable_default_archive():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args(["--model", "/m/Qwen", "--no-auto-ui", "--no-archive"])

    assert cfg.enable_archive is False
    assert cfg.enable_file_read is False
    assert cfg.layered_context is False


def test_cli_belief_graph_interval():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args([
        "--model", "/m/Qwen", "--no-auto-ui", "--belief-graph-interval", "3",
    ])
    assert cfg.belief_graph_interval == 3


def test_cli_recent_turns_zero_is_graph_only_and_minus_one_is_unbounded():
    from bcg.agent.rollout import _parse_args

    graph_only = _parse_args([
        "--model", "/m/Qwen", "--no-auto-ui", "--recent-turns", "0",
    ])
    unbounded = _parse_args([
        "--model", "/m/Qwen", "--no-auto-ui", "--recent-turns", "-1",
    ])

    assert graph_only.recent_turns == 0
    assert unbounded.recent_turns == -1


def test_cli_rejects_recent_turns_below_minus_one():
    from bcg.agent.rollout import _parse_args

    with pytest.raises(SystemExit):
        _parse_args([
            "--model", "/m/Qwen", "--no-auto-ui", "--recent-turns", "-2",
        ])


def test_cli_belief_graph_placement():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args([
        "--model", "/m/Qwen", "--no-auto-ui", "--belief-graph-placement", "system",
    ])
    assert cfg.belief_graph_placement == "system"


def test_cli_max_tool_workers():
    from bcg.agent.rollout import _parse_args

    cfg = _parse_args([
        "--model", "/m/Qwen", "--no-auto-ui", "--max-tool-workers", "4",
    ])
    assert cfg.max_tool_workers == 4
