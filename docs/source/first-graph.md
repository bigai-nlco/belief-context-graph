---
title: "Your First Graph"
description: "Read the result of a BCG construction run."
icon: "circle-nodes"
---

After a run completes, start with four views of the same state.

## 1. Belief nodes

```python
beliefs = result.graph.beliefs()

for belief in beliefs:
    print({
        "id": belief.id,
        "type": belief.node_type,
        "text": belief.belief,
        "stance": belief.stance,
        "confidence": belief.confidence,
        "source_role": belief.source.role,
        "turn": belief.source.turn_index,
    })
```

Look for:

- propositions that matter to later reasoning
- decision nodes
- stance appropriate to the wording
- source role and turn index
- evidence excerpts that actually support the proposition

## 2. Relations

```python
for relation in result.graph.relations():
    print(
        relation.from_id,
        relation.type,
        relation.to_id,
        relation.weight,
    )
```

The direction matters. For `depends_on`, the source belief depends on the target belief.

## 3. Confidence history

```python
belief = beliefs[0]

print(belief.initial_confidence)
print(belief.evidence_confidence)
print(belief.factor_confidence)

for update in belief.confidence_history:
    print(update.step, update.value, update.reason)
```

The final score should be explainable through those components and updates.

## 4. Provenance

```python
for excerpt in belief.evidence:
    print(excerpt.text)
    print(excerpt.start, excerpt.end, excerpt.match, excerpt.via)
    print(excerpt.source)
```

Exact offsets let an inspector highlight the supporting span in the original turn.

## Use the viewer

The static visualizer is best for one result:

```bash
bcg construct visualize outputs/example/result.json
```

The Live Stream Viewer is best for replaying turn-by-turn graph growth and investigating the final decision. See [Live Stream Viewer](/guides/live-stream-viewer).
