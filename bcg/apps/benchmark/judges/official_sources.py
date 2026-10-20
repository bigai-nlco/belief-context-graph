"""Extract the official judge prompts from upstream source trees.

The judges in this package must use the official prompts byte for byte. Most
DRBench prompts are Python f-strings with indentation and ``{{ }}`` escapes,
which are easy to corrupt when copied by hand. This module parses the upstream
sources with :mod:`ast` and turns each f-string into an equivalent
``str.format`` template (literal braces doubled, each interpolation replaced
by a named field), so ``template.format(**values)`` reproduces the f-string
exactly.

``scripts/sync_judge_prompts.py`` writes the templates into ``prompts/`` and
records their provenance in ``prompts/SOURCES.json``; the tests re-extract
from local clones (``BCG_DRB2_REPO``, ``BCG_DRBENCH_REPO``) when available and
assert byte equality.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SOURCES_FILE = PROMPTS_DIR / "SOURCES.json"

# Template name -> (repository, path inside repository).
PROMPT_SOURCES = {
    "drb2_rubric": ("DeepResearch-Bench-II", "run_evaluation.py"),
    "drbench_insight_extraction": ("drbench", "drbench/agents/utils.py"),
    "drbench_insight_scoring": (
        "drbench",
        "drbench/prompts/eval_metrics/insight_scoring.txt",
    ),
    "drbench_report_quality": ("drbench", "drbench/metrics/report_quality.py"),
    "drbench_report_quality_retry": ("drbench", "drbench/metrics/report_quality.py"),
}

# f-string interpolation source -> template field name.
_FIELD_NAMES = {
    "report_text": "report_text",
    "dr_question": "dr_question",
    "json.dumps(persona, indent=2)": "persona_json",
}


def _fstring_to_template(node: ast.JoinedStr, source: str) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value).replace("{", "{{").replace("}", "}}"))
        elif isinstance(value, ast.FormattedValue):
            if value.conversion != -1 or value.format_spec is not None:
                raise ValueError("Unsupported f-string conversion in official prompt.")
            expression = ast.get_source_segment(source, value.value) or ""
            field = _FIELD_NAMES.get(expression.strip())
            if field is None:
                raise ValueError(
                    f"Unknown interpolation {expression!r} in official prompt."
                )
            parts.append("{" + field + "}")
        else:  # pragma: no cover - defensive
            raise ValueError("Unsupported node in official prompt f-string.")
    return "".join(parts)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Function {name} not found.")


def _assigned_fstrings(function: ast.FunctionDef, target: str) -> list[ast.JoinedStr]:
    found = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == target
            and isinstance(node.value, ast.JoinedStr)
        ):
            found.append(node.value)
    return sorted(found, key=lambda value: value.lineno)


def extract_prompts(drb2_repo: Path, drbench_repo: Path) -> dict[str, str]:
    drb2_repo, drbench_repo = Path(drb2_repo), Path(drbench_repo)
    prompts: dict[str, str] = {}

    source = (drb2_repo / "run_evaluation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PROMPT_TEMPLATE"
        ):
            prompts["drb2_rubric"] = node.value.value  # already a .format template
    if "drb2_rubric" not in prompts:
        raise ValueError("PROMPT_TEMPLATE not found in DeepResearch Bench II.")

    source = (drbench_repo / "drbench/agents/utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    (extraction,) = _assigned_fstrings(
        _function(tree, "break_report_to_insights"), "insight_claims_prompt"
    )
    prompts["drbench_insight_extraction"] = _fstring_to_template(extraction, source)

    prompts["drbench_insight_scoring"] = (
        drbench_repo / "drbench/prompts/eval_metrics/insight_scoring.txt"
    ).read_text(encoding="utf-8")

    source = (drbench_repo / "drbench/metrics/report_quality.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    first, retry = _assigned_fstrings(_function(tree, "compute"), "prompt")
    prompts["drbench_report_quality"] = _fstring_to_template(first, source)
    prompts["drbench_report_quality_retry"] = _fstring_to_template(retry, source)
    return prompts


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def load_sources() -> dict:
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def verify_bundled_prompts() -> list[str]:
    """Names of bundled templates whose sha256 differs from SOURCES.json."""

    recorded = load_sources()["prompts"]
    return [
        name
        for name, meta in recorded.items()
        if sha256_text(load_prompt(name)) != meta["sha256"]
    ]


__all__ = [
    "PROMPTS_DIR",
    "PROMPT_SOURCES",
    "extract_prompts",
    "load_prompt",
    "load_sources",
    "sha256_text",
    "verify_bundled_prompts",
]
