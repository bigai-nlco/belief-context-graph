"""
factor.py
=========
Factor schema helpers and relation -> factor activation rules.

Concept used by this codebase
-----------------------------
A Relation is the concrete semantic edge between two nodes. A Factor is the
reusable computation rule/template that a relation can activate.

This distinction keeps the two graphs in the project design separate:

* semantic graph: ``relation(from_id, to_id, type, note, activated_factor_ids)``
* computation graph: ``factor(factor_type, weight, activation_condition,
  embedding)``

The concrete input/output binding for one activation is read from the relation,
not from a unique per-edge Factor object. Multiple relations can therefore point
to the same factor id when their factor-type matches and their semantic
activation text embeddings are similar enough.

For computation we keep the project-level convention stable:
``input_variables`` affect ``output_variables``. Therefore:

* ``A depends_on B``   -> B supports A, so input=B and output=A.
* ``A contradicts B`` -> A refutes B, so input=A and output=B.
* ``supplements``     -> semantic-only edge; no factor is activated.

``input_variables`` and ``output_variables`` are pair-aligned aggregate
bookkeeping over relation activations: index ``i`` in both lists describes one
input -> output activation. Repeated node ids are intentional when multiple
relations activate the same reusable factor template.

``activation_condition["note"]`` stores the concise mechanism-level natural
language meaning of the factor. It is generated separately from relation.note,
embedded, and used for factor reuse. The numeric gate
``valid_only_if_input_confidence`` controls whether an input node is confident
enough to activate the factor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

FactorType = Literal["support", "refute"]

DEFAULT_FACTOR_WEIGHT = 0.5
DEFAULT_FACTOR_INPUT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_FACTOR_SIMILARITY_THRESHOLD = 0.85

GENERIC_FACTOR_TEXT: Dict[str, str] = {
    "depends_on": "one belief supports another belief as a required premise",
    "contradicts": "one belief refutes another belief by semantic contradiction",
}

RELATION_FACTOR_TYPES: Dict[str, FactorType] = {
    "depends_on": "support",
    "contradicts": "refute",
}

_GENERIC_NOTES = {
    "depends on",
    "supports",
    "support",
    "is related",
    "related",
    "contradicts",
    "conflicts",
    "refutes",
    "supplements",
    "adds detail",
}


def _coerce_int_list(value: Any, *, dedupe: bool = True) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        values: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]

    out: List[int] = []
    seen: set[int] = set()
    for raw in values:
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if dedupe and i in seen:
            continue
        if dedupe:
            seen.add(i)
        out.append(i)
    return out


def _coerce_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    if not isinstance(value, Iterable):
        return []
    out: List[float] = []
    for raw in value:
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            return []
    return out


@dataclass
class Factor:
    id: int
    factor_type: FactorType
    input_variables: List[int] = field(default_factory=list)
    output_variables: List[int] = field(default_factory=list)
    weight: float = DEFAULT_FACTOR_WEIGHT
    activation_condition: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    activated_relation_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.factor_type not in {"support", "refute"}:
            raise ValueError("factor_type must be 'support' or 'refute'")
        self.id = int(self.id)
        self.input_variables = _coerce_int_list(self.input_variables, dedupe=False)
        self.output_variables = _coerce_int_list(self.output_variables, dedupe=False)
        self.activated_relation_ids = _coerce_int_list(self.activated_relation_ids)
        self.embedding = _coerce_float_list(self.embedding)
        self.weight = float(self.weight)
        if self.activation_condition is None:
            self.activation_condition = {}
        if not isinstance(self.activation_condition, dict):
            raise ValueError("activation_condition must be a dict")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": "factor",
            "factor_type": self.factor_type,
            "input_variables": list(self.input_variables),
            "output_variables": list(self.output_variables),
            "weight": self.weight,
            "activation_condition": dict(self.activation_condition),
            "embedding": list(self.embedding),
            "activated_relation_ids": list(self.activated_relation_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Factor":
        return cls(
            id=int(data.get("id", 0)),
            factor_type=data.get("factor_type", "support"),
            input_variables=_coerce_int_list(data.get("input_variables"), dedupe=False),
            output_variables=_coerce_int_list(data.get("output_variables"), dedupe=False),
            weight=float(data.get("weight", DEFAULT_FACTOR_WEIGHT)),
            activation_condition=dict(data.get("activation_condition") or {}),
            embedding=_coerce_float_list(data.get("embedding")),
            activated_relation_ids=_coerce_int_list(data.get("activated_relation_ids")),
        )


def relation_factor_endpoint_ids(relation: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return ``(input_id, output_id)`` for a factor-activating relation.

    Factor direction is always input -> output:
      * A depends_on B: B supports A, so input=B and output=A.
      * A contradicts B: A refutes B, so input=A and output=B.
      * supplements: no factor.
    """
    try:
        from_id = int(relation.get("from_id"))
        to_id = int(relation.get("to_id"))
    except (TypeError, ValueError):
        return None

    rtype = relation.get("type")
    if rtype == "depends_on":
        return to_id, from_id
    if rtype == "contradicts":
        return from_id, to_id
    return None


def _node_text(node: Optional[Dict[str, Any]]) -> str:
    if not isinstance(node, dict):
        return ""
    for key in ("belief", "decision", "content", "text"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _shorten_text(text: str, *, max_chars: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() or text[:max_chars]


def _informative_note(note: Any) -> str:
    text = _shorten_text(str(note or ""), max_chars=160)
    if not text:
        return ""
    low = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    low = re.sub(r"\s+", " ", low).strip()
    if low in _GENERIC_NOTES:
        return ""
    if len(low.split()) <= 2 and low in _GENERIC_NOTES:
        return ""
    return text


def factor_semantic_text_from_relation(
    relation: Dict[str, Any],
    nodes_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    factor_note: Optional[str] = None,
) -> Optional[str]:
    """Return the mechanism-level natural language for a factor.

    Preferred path: a separate factor-note abstraction step supplies
    ``factor_note`` from the concrete input node, relation type/direction,
    output node, and relation.note. That note should be slot-level /
    mechanism-level, not a field concatenation and not a relation.note rewrite.

    Fallback path: when no factor-note generator is configured or generation
    fails, use a conservative relation-derived note so factor activation still
    works.
    """
    rtype = str(relation.get("type") or "")
    if rtype not in RELATION_FACTOR_TYPES:
        return None

    generated = _informative_note(factor_note)
    if generated:
        return generated

    # Fallback only. This keeps the pipeline usable without the factor-note LLM,
    # but normal runs should provide a generated mechanism-level factor_note.
    note = _informative_note(relation.get("note"))
    if note:
        return note

    endpoints = relation_factor_endpoint_ids(relation)
    if endpoints is None:
        return GENERIC_FACTOR_TEXT.get(rtype)

    nodes_by_id = nodes_by_id or {}
    input_id, output_id = endpoints
    input_text = _shorten_text(_node_text(nodes_by_id.get(input_id)), max_chars=90)
    output_text = _shorten_text(_node_text(nodes_by_id.get(output_id)), max_chars=90)
    if input_text and output_text:
        verb = "supports" if rtype == "depends_on" else "refutes"
        return f"{input_text} {verb} {output_text}"
    return GENERIC_FACTOR_TEXT.get(rtype)


def factor_spec_from_relation(
    relation: Dict[str, Any],
    nodes_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    input_confidence_threshold: float = DEFAULT_FACTOR_INPUT_CONFIDENCE_THRESHOLD,
    factor_note: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build the reusable factor template spec activated by one relation.

    The returned spec deliberately does NOT bind concrete input/output variables;
    the relation supplies that binding during confidence computation. Factor
    reuse is decided by semantic embedding similarity over
    ``activation_condition["note"]`` plus matching factor_type.
    """
    rtype = str(relation.get("type") or "")
    factor_type = RELATION_FACTOR_TYPES.get(rtype)
    if factor_type is None or relation_factor_endpoint_ids(relation) is None:
        return None

    semantic_text = factor_semantic_text_from_relation(
        relation, nodes_by_id, factor_note=factor_note
    )
    if not semantic_text:
        return None

    return {
        "node_type": "factor",
        "factor_type": factor_type,
        "input_variables": [],
        "output_variables": [],
        "weight": DEFAULT_FACTOR_WEIGHT,
        "activation_condition": {
            "note": semantic_text,
            "valid_only_if_input_confidence": float(input_confidence_threshold),
        },
        "embedding": [],
        "activated_relation_ids": [],
    }
