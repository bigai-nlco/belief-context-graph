"""Dependency-free smoke check for the task-relevance confidence adjustment."""
from __future__ import annotations

import ast
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "bcg/construct/_shared/context_selection.py"
module = ast.parse(SOURCE.read_text(encoding="utf-8"))
selection_function = next(
    node
    for node in module.body
    if isinstance(node, ast.FunctionDef) and node.name == "_task_adjusted_confidence"
)
namespace: dict[str, object] = {}
exec(
    compile(ast.Module(body=[selection_function], type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)
score = namespace["_task_adjusted_confidence"]
assert math.isclose(score(0.8, 1.0), 0.8)
assert math.isclose(score(0.8, 0.0), 0.4)
assert math.isclose(score(0.8, 0.5), 0.6)
assert math.isclose(score(1.2, 1.0), 1.0)
assert math.isclose(score(0.8, -1.0), 0.4)

# Test Unified evidence math without importing the project or its dependencies.
CONFIDENCE_SOURCE = ROOT / "bcg/construct/unified/confidence.py"
confidence_module = ast.parse(CONFIDENCE_SOURCE.read_text(encoding="utf-8"))
evidence_function = next(
    node
    for node in confidence_module.body
    if isinstance(node, ast.FunctionDef) and node.name == "evidence_contribution"
)
math_namespace = {
    "_role_from_record": lambda record: record.get("role", ""),
    "source_reliability": lambda role: {"tool": 0.80, "assistant": 0.65}.get(role, 0.55),
    "stance_quality": lambda stance: {"asserted": 0.75}.get(stance, 0.35),
    "logit": lambda p: math.log(p / (1.0 - p)),
}
exec(
    compile(ast.Module(body=[evidence_function], type_ignores=[]), str(CONFIDENCE_SOURCE), "exec"),
    math_namespace,
)
contribution = math_namespace["evidence_contribution"]
expected = math.log(0.775 / 0.225)
assert math.isclose(contribution({"role": "tool", "stance": "asserted"}), expected)
assert math.isclose(contribution({"role": "assistant", "stance": "speculated"}), 0.0)
print("PASS: relevance affects selection only; Unified evidence adds log-likelihood ratios")
