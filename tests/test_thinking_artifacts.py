from __future__ import annotations

import json

from bcg.agent.config import model_output_dir_name
from bcg.agent.ui import _result_summary


def test_model_output_dir_name_separates_thinking_modes() -> None:
    thinking = model_output_dir_name(
        "/data/share/models/Qwen3.5-9B",
        enable_thinking=True,
    )
    no_thinking = model_output_dir_name(
        "/data/share/models/Qwen3.5-9B",
        enable_thinking=False,
    )

    assert thinking == "Qwen3.5-9B_thinking"
    assert no_thinking == "Qwen3.5-9B_no-thinking"
    assert thinking != no_thinking


def test_result_summary_title_uses_thinking_mode(tmp_path) -> None:
    model_dir = tmp_path / "Qwen3.5-9B_thinking"
    result_dir = model_dir / "gpqa_diamond"
    result_dir.mkdir(parents=True)
    (model_dir / "overall_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "model": "/data/share/models/Qwen3.5-9B",
                    "enable_thinking": True,
                },
                "summaries": [],
            }
        ),
        encoding="utf-8",
    )
    result_path = result_dir / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "summary": {
                    "benchmark": "gpqa_diamond",
                    "model": "/data/share/models/Qwen3.5-9B",
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    summary = _result_summary(result_path)

    assert summary["model"] == "Qwen3.5-9B (thinking)"


def test_result_summary_title_uses_no_thinking_mode(tmp_path) -> None:
    model_dir = tmp_path / "Qwen3.5-9B_no-thinking"
    result_dir = model_dir / "gpqa_diamond"
    result_dir.mkdir(parents=True)
    (model_dir / "overall_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "model": "/data/share/models/Qwen3.5-9B",
                    "enable_thinking": False,
                },
                "summaries": [],
            }
        ),
        encoding="utf-8",
    )
    result_path = result_dir / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "summary": {
                    "benchmark": "gpqa_diamond",
                    "model": "/data/share/models/Qwen3.5-9B",
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    summary = _result_summary(result_path)

    assert summary["model"] == "Qwen3.5-9B (no thinking)"
