"""Dependency-free check for task-specific graph contribution scoring."""
from __future__ import annotations

import ast
from pathlib import Path

source = Path(__file__).parents[1] / "workflows/stateful_react_agent/bcg/pcst.py"
module = ast.parse(source.read_text(encoding="utf-8"))
function = next(
    node
    for node in module.body
    if isinstance(node, ast.FunctionDef) and node.name == "_dependency_contribution"
)
namespace: dict[str, object] = {}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
    namespace,
)
score = namespace["_dependency_contribution"]
nodes = {
    10: {"id": 10, "source": {"turn_id": 2}},
    20: {"id": 20, "source": {"turn_id": 2}},
    30: {"id": 30, "source": {"turn_id": 1}},
}
relations = [
    {"from_id": 10, "to_id": 20, "type": "depends_on"},
    {"from_id": 30, "to_id": 20, "type": "depends_on"},
]
values = score(nodes, relations, {10: 0.8, 20: 0.4, 30: 1.0})
assert abs(values[20] - 2 / 3) < 1e-12
assert values[10] == 0.0 and values[30] == 0.0
assert score(nodes, relations, {10: 0.0, 20: 0.0, 30: 1.0})[20] == 0.0
print("PASS: relevant dependent claims reward their premise; pinned input adds no reward")
