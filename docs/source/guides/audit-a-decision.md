---
title: "Audit a Decision"
description: "Trace a final decision from output text back through beliefs, evidence, and confidence."
icon: "magnifying-glass-chart"
---

A decision audit should answer four questions.

## 1. What was decided?

Find nodes with:

```python
decisions = [
    belief
    for belief in result.graph.beliefs()
    if belief.node_type == "decision"
]
```

Confirm the canonical decision text and source turn.

## 2. What did it depend on?

Traverse incoming or outgoing `depends_on` relations according to the decision's graph position.

```python
relations = result.graph.relation_dicts("depends_on")
```

Inspect each relation note, weight, and activation condition.

## 3. Which evidence supports each belief?

For every node in the decision path:

```python
for excerpt in belief.evidence:
    print(excerpt.text, excerpt.start, excerpt.end)
```

Compare the highlighted excerpt against the original trajectory turn.

## 4. How did confidence change?

Review:

- `initial_confidence`
- `evidence_confidence`
- `factor_confidence`
- every `confidence_history` entry
- contradictions and their activation state
- merges that changed the canonical node

## Viewer workflow

1. replay the sample to the final turn
2. click **Explain final decision**
3. inspect the decision path
4. open each node's evidence
5. review confidence history
6. inspect merge records for rewritten nodes
7. export or retain `result.json`, prompt logs, and timing artifacts

<Warning>
A clear graph path is evidence of system traceability, not proof that the decision is correct. Human or domain-specific validation remains necessary.
</Warning>
