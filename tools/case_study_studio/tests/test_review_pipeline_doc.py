"""Keep the public methodology page tied to runtime contracts, without API calls."""
import json
from unittest.mock import patch

import pytest

from tools.ai_review.pipeline import validate_findings, validate_summary
from tools.ai_review.prompts import FIND, PROMPT_VERSION, SYNTHESIZE, SYSTEM, messages
from tools.ai_review.schema import Findings, Summary
from tools.case_study_studio.build_review_pipeline import DEST, ROOT, SOURCE_TARGETS, build


@pytest.fixture(scope="module")
def contract() -> dict:
    with patch("tools.ai_review.client.ReviewConfig.from_env", side_effect=AssertionError("must not read credentials")):
        return build()


def test_snapshot_is_current_and_deterministic(contract: dict) -> None:
    assert json.loads(DEST.read_text()) == contract
    assert contract["prompt_version"] == PROMPT_VERSION
    assert contract["format"] == "bcg-review-method-contract"
    assert build() == contract


def test_templates_and_schemas_are_runtime_exports(contract: dict) -> None:
    assert contract["prompts"] == {"system": SYSTEM, "findings": FIND, "summary": SYNTHESIZE}
    assert contract["schemas"] == {"findings": Findings.model_json_schema(), "summary": Summary.model_json_schema()}
    assert contract["defaults"] == {"chunk_chars": 60000, "chunk_turns": 6, "max_batches": 64}
    assert "api_key" not in contract["model"]


def test_example_uses_real_preparation_validation_and_serialization(contract: dict) -> None:
    e = contract["example"]
    assert "非真实" in e["label"]
    assert len(e["case"]["steps"]) == 3
    validate_findings(Findings.model_validate(e["findings"]), e["packets"][0])
    validate_summary(Summary.model_validate(e["summary"]), e["candidates"], e["summary_input"])
    assert e["summary"]["selected_finding_ids"] == ["F002", "F001"]
    assert e["summary"]["process_groups"][0]["member_ids"] == ["F002", "F003"]
    assert e["requests"]["findings"] == messages("findings", e["packets"][0])
    assert e["requests"]["summary"] == messages("summary", e["summary_input"])
    assert "steps" not in e["summary_input"], "global synthesis sees candidates, not all raw turns"
    for stage in ("findings", "summary"):
        assert [m["role"] for m in e["requests"][stage]] == ["system", "user"]


def test_source_links_use_an_explicit_allowlist_and_real_lines(contract: dict) -> None:
    assert set(contract["sources"]) == set(SOURCE_TARGETS)
    for key, source in contract["sources"].items():
        name, symbol = SOURCE_TARGETS[key]
        assert source["path"] == name and source["symbol"] == symbol
        lines = (ROOT / name).read_text().splitlines()
        assert source["text"] == "\n".join(lines[source["line"] - 1:source["end_line"]])
        assert not name.startswith("/") and ".." not in name
