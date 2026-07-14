from __future__ import annotations

import base64
import json
from pathlib import Path

from bcg.agent.benchmark_loader import load_benchmark
from bcg.agent.reward import build_reward_fn, gaia_question_scorer
from scripts.prepare_web_benchmarks import convert_browsecomp_rows, derive_key


def _encrypt_browsecomp(value: str, canary: str) -> str:
    raw = value.encode("utf-8")
    key = derive_key(canary, len(raw))
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode("ascii")


def test_convert_official_browsecomp_encrypted_rows() -> None:
    canary = "unit-test-canary"
    converted = convert_browsecomp_rows(
        [
            {
                "canary": canary,
                "problem": _encrypt_browsecomp("Who wrote the work?", canary),
                "answer": _encrypt_browsecomp("Example Author", canary),
            }
        ]
    )

    assert converted == [
        {
            "task_id": "browsecomp-0000",
            "input": "Who wrote the work?",
            "ground_truth_answer": "Example Author",
            "extra_info": {
                "source": "openai/simple-evals BrowseComp",
                "source_row": 0,
            },
        }
    ]


def test_browsecomp_alias_loads_normalized_data(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "browse_comp"
    benchmark_dir.mkdir()
    (benchmark_dir / "data.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "browsecomp-0042",
                    "input": "Question text",
                    "ground_truth_answer": "Exact answer",
                }
            ]
        ),
        encoding="utf-8",
    )

    for name in ("browse_comp", "browsecomp"):
        tasks = load_benchmark(name, artifacts_dir=tmp_path, shuffle=False)
        assert len(tasks) == 1
        assert tasks[0].task_id == "browsecomp-0042"
        assert tasks[0].question == "Question text"
        assert tasks[0].ground_truth == ["Exact answer"]
        assert tasks[0].data_source == "browsecomp"


def test_gaia_loader_reads_metadata_filters_level_and_preserves_attachment(
    tmp_path: Path, monkeypatch
) -> None:
    validation = tmp_path / "gaia" / "2023" / "validation"
    validation.mkdir(parents=True)
    attachment = validation / "evidence.txt"
    attachment.write_text("attachment contents", encoding="utf-8")
    rows = [
        {
            "task_id": "gaia-one",
            "Question": "Find the answer.",
            "Level": 1,
            "Final answer": "42",
            "file_name": "evidence.txt",
            "Annotator Metadata": {"Steps": "must not enter the prompt"},
        },
        {
            "task_id": "gaia-two",
            "Question": "A harder question.",
            "Level": 2,
            "Final answer": "other",
            "file_name": "",
        },
    ]
    (validation / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setenv("GAIA_SPLIT", "validation")
    monkeypatch.setenv("GAIA_LEVEL", "1")

    tasks = load_benchmark("gaia", artifacts_dir=tmp_path, shuffle=False)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "gaia-one"
    assert task.ground_truth == ["42"]
    assert task.extra_info["file_path"] == str(attachment.resolve())
    assert task.extra_info["file_url"] == "file://gaia/2023/validation/evidence.txt"
    assert task.extra_info["file_url"] in task.question
    assert "must not enter the prompt" not in task.question


def test_gaia_official_scorer_and_reward() -> None:
    assert gaia_question_scorer("$1,234", "1234")
    assert gaia_question_scorer("34689, 12345", "34689,12345")
    assert gaia_question_scorer("Sea gull!", "seagull")
    assert not gaia_question_scorer("34689", "34689,12345")

    reward = build_reward_fn("gaia")(
        {"ground_truth": "34689", "data_source": "gaia"},
        "The evidence points to \\boxed{34689}.",
    )
    assert reward.is_correct
    assert reward.reward == 1.0
    assert reward.metadata["evaluation_method"] == "gaia_question_scorer"
