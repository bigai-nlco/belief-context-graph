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

print("PASS: relevance affects selection only")
