---
title: "Confidence"
description: "How BCG computes an auditable posterior for every belief."
icon: "gauge-high"
---

BCG confidence is represented as explicit components rather than a single unexplained score.

## Posterior formula

```text
confidence = sigmoid(
  logit(initial_confidence)
  + evidence_confidence
  + factor_confidence
)
```

Scores are clamped away from exact zero and one before logit conversion.

## Components

### Initial confidence

The prior is derived from:

- source reliability
- stance quality

The default configuration uses a weighted average of the two. Example policy values assign higher reliability to tool results than to LLM reasoning, and lower stance quality to speculation than to assertion.

### Evidence confidence

This component changes as evidence is attached or merged. Evidence remains separately inspectable through `evidence_ids`, excerpt text, and source offsets.

### Factor confidence

This component is produced by active graph relations. It records how supporting dependencies or contradictions changed the target posterior.

## Confidence history

Each update may record:

- step and resulting value
- reason and delta
- confidence configuration
- iteration
- evidence added or scored
- absorbed beliefs
- factor details
- formula and method
- fallback use
- model name

```python
for entry in belief.confidence_history:
    print(entry.model_dump())
```

## Relation activation

Confidence-carrying edges can have:

```json
{
  "weight": 0.7,
  "activated_condition": {
    "input_conf_threshold": 0.8
  }
}
```

The relation only contributes after the relevant input belief crosses the threshold.

## Propagation semantics

- `depends_on`: positive support flows from `to_id` to `from_id`
- `contradicts`: negative support flows from `from_id` to `to_id`
- `supplements`: semantic-only; no confidence weight or activation condition

The configuration limits minimum deltas and propagation iterations.

<Warning>
Confidence is a policy-controlled graph score, not a calibrated probability guarantee. Validate the policy for your domain before using it as an action threshold.
</Warning>
