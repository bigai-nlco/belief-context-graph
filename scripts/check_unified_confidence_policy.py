"""Dependency-free regression for Unified source attribution and propagation.

Run from the project root: python scripts/check_unified_confidence_policy.py
"""

from __future__ import annotations

import importlib.util
import argparse
import json
import math
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("bcg", "bcg.core", "bcg.construct", "bcg.construct.unified", "bcg.construct._shared"):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / name.replace(".", "/"))]
    sys.modules[name] = package


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("bcg.core.confidence", "bcg/core/confidence.py")
confidence = load("bcg.construct.unified.confidence", "bcg/construct/unified/confidence.py")
tool_results = load("bcg.construct._shared.tool_results", "bcg/construct/_shared/tool_results.py")
sources = load("bcg.construct.unified.source_policy", "bcg/construct/unified/source_policy.py")

wrapped = "<tool_result>" + json.dumps({
    "tool_call_id": "one", "name": "web_search",
    "content": "## Search Results\n[1] Cboe index\nURL: https://www.cboe.com/vix\n"
               "[2] Forum thread\nURL: https://www.reddit.com/r/finance",
}) + "</tool_result>"
parsed = tool_results.extract_tool_results(wrapped)
assert len(parsed) == 1
assert [item.url for item in parsed[0].results] == [
    "https://www.cboe.com/vix", "https://www.reddit.com/r/finance"
]

items = [
    {"rank": 1, "url": "https://www.cboe.com/vix"},
    {"rank": 2, "url": "https://www.reddit.com/r/finance"},
]
official = sources.tool_source_metadata({"source_rank": 1, "tool_result_items": items})
community = sources.tool_source_metadata({"source_rank": 2, "tool_result_items": items})
unattributed = sources.tool_source_metadata({"tool_result_items": items})
assert official["source_category"] == "primary_official"
assert community["source_category"] == "community"
assert unattributed["source_category"] == "mixed_or_unattributed"
fetch = sources.tool_source_metadata(
    {"tool_name": "web_fetch"},
    [{"text": "Source: https://fred.stlouisfed.org/series/VIXCLS\nObserved: 19.42"}],
)
assert fetch["source_host"] == "fred.stlouisfed.org"
policy = confidence.normalize_confidence_config()
confirmed = policy["stance_quality"]["confirmed"]
unknown = policy["stance_quality"]["unknown"]
assert core.initial_confidence("tool", "confirmed", official["source_coefficient"]) == round(official["source_coefficient"] * confirmed, 3)
assert core.initial_confidence("tool", "confirmed", community["source_coefficient"]) == round(community["source_coefficient"] * confirmed, 3)
assert core.initial_confidence("user", "confirmed") == round(policy["source_reliability"]["user"] * confirmed, 3)
assert core.initial_confidence("assistant", "confirmed") == round(policy["source_reliability"]["assistant"] * confirmed, 3)
assert core.initial_confidence("tool", "confirmed") == round(policy["source_reliability"]["tool"] * confirmed, 3)
assert core.initial_confidence("user", "unknown") == unknown
assert core.initial_confidence("tool", "asserted") == unknown  # no four-label alias

# Verify a caller override reaches source attribution, priors, and evidence.
custom = confidence.normalize_confidence_config({
    "source_reliability": {"user": 0.40},
    "stance_quality": {"confirmed": 0.70},
    "source_coefficients": {"primary_official": 0.55},
    "source_domains": {"primary_official": ["my-institution.example"]},
    "evidence_log_odds_weight": 0.10,
})
assert core.initial_confidence("user", "confirmed", config=custom) == 0.28
custom_source = sources.tool_source_metadata(
    {"source_rank": 1, "tool_result_items": items}, config=custom
)
assert custom_source["source_category"] == "general_web"
assert sources.classify_url("https://my-institution.example/study", custom)[0] == "primary_official"
assert custom["source_coefficients"]["primary_official"] == 0.55
assert core.initial_confidence("tool", "confirmed", 0.55, custom) == 0.385
assert confidence.evidence_contribution(
    {"role": "tool", "stance": "confirmed", "source_coefficient": 0.55}, custom
) == 0.0
merged = {
    "source": {"item_id": "case", "turn_id": 1},
    "evidence_ids": [1, 2], "initial_confidence": 0.50,
    "evidence_confidence": 0.0, "factor_confidence": 0.0, "confidence": 0.50,
}
confidence.record_evidence_merge_confidence(
    merged, added_evidence_ids=[2], added_evidence_records=[], absorbed_ids=[9],
    newest_id=9, evidence_by_id={
        1: {"role": "user", "stance": "confirmed", "source": {"item_id": "case", "turn_id": 1}},
        2: {"role": "tool", "stance": "confirmed", "source_coefficient": 0.88,
            "source": {"item_id": "case", "turn_id": 2}},
    }, config=custom,
)
assert math.isclose(merged["evidence_confidence"], round(0.10 * core.logit(0.616), 6))

# Exercise the real layered YAML loader and a user file without any API call.
from bcg.config.loader import load_settings
settings, _ = load_settings(explicit=str(ROOT / "bcg/config/config.example.yaml"))
assert settings.backend == "unified"
with tempfile.TemporaryDirectory() as tmp:
    override = Path(tmp) / "bcg.yaml"
    override.write_text("pipeline:\n  confidence:\n    stance_quality:\n      confirmed: 0.70\n    source_coefficients:\n      primary_official: 0.55\n", encoding="utf-8")
    configured, _ = load_settings(explicit=str(override))
    resolved = confidence.normalize_confidence_config(
        configured.pipeline.confidence.model_dump(exclude_none=True)
    )
    assert core.initial_confidence("tool", "confirmed", 0.55, resolved) == 0.385

evidence = {"role": "tool", "stance": "confirmed", **community}
community_prior = core.initial_confidence("tool", "confirmed", community["source_coefficient"])
official_prior = core.initial_confidence("tool", "confirmed", official["source_coefficient"])
assert confidence.evidence_contribution(evidence) == max(0.0, policy["evidence_log_odds_weight"] * core.logit(community_prior))
assert confidence.evidence_contribution({"role": "tool", "stance": "confirmed", **official}) == max(0.0, policy["evidence_log_odds_weight"] * core.logit(official_prior))

propagation_config = confidence.normalize_confidence_config({
    "relation_propagation": {
        "default_relation_weight": 0.5,
        "input_confidence_threshold": 0.8,
        "min_confidence_delta": 0.001,
        "max_iterations": 3,
    }
})
nodes = {
    1: {"initial_confidence": 0.90, "evidence_confidence": 0, "factor_confidence": 0,
        "confidence": 0.90},
    2: {"initial_confidence": 0.85, "evidence_confidence": 0, "factor_confidence": 0,
        "confidence": 0.85},
    3: {"initial_confidence": 0.60, "evidence_confidence": 0, "factor_confidence": 0,
        "confidence": 0.60},
}
edges = [
    {"id": 0, "from_id": 2, "to_id": 1, "type": "depends_on"},
    {"id": 1, "from_id": 3, "to_id": 2, "type": "depends_on"},
]
report = confidence.propagate_relation_confidences(
    nodes, edges, seed_output_node_ids=[2], config=propagation_config
)
assert report["updated_node_ids"] == [2, 3], report
assert nodes[2]["confidence"] > 0.85 and nodes[3]["confidence"] > 0.60
weak = confidence.relation_factor_contribution(
    {"id": 2, "from_id": 3, "to_id": 4, "type": "depends_on"},
    {3: nodes[3], 4: {"confidence": 0.395}}, config=propagation_config,
)
assert weak is None  # a forum claim cannot activate the 0.8 propagation gate
print("PASS: source attribution, stance × role priors, evidence, and two-hop propagation")


def inspect_case(path: Path) -> None:
    """Read a case-study export without making any model or network calls."""
    cases = json.loads(path.read_text(encoding="utf-8")).get("cases", [])
    for case in cases:
        sources = case.get("sources") or []
        trajectory = next((source for source in sources if source.get("kind") == "graph_trajectory"), {})
        counts: Counter[str] = Counter()
        turns = 0
        search_urls = 0
        fetched_urls = 0
        for record in trajectory.get("records") or []:
            if record.get("role") != "tool":
                continue
            turns += 1
            for result in tool_results.extract_tool_results(str(record.get("text") or "")):
                for item in result.results:
                    if item.url:
                        category, _ = sources_policy_classify(item.url)
                        counts[category] += 1
                        search_urls += 1
                if result.tool_name == "web_fetch":
                    import re
                    for url in re.findall(r"(?m)^Source:\s*(https?://[^\s]+)", result.body):
                        category, _ = sources_policy_classify(url)
                        counts[category] += 1
                        fetched_urls += 1
        print(f"CASE {path.name}: tool_turns={turns} search_urls={search_urls} fetch_sources={fetched_urls}")
        print("  source_categories=" + json.dumps(dict(sorted(counts.items())), ensure_ascii=False))


sources_policy_classify = sources.classify_url
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-key Unified confidence and case-source check")
    parser.add_argument("--case-study", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    for case_path in args.case_study:
        inspect_case(case_path)
