---
title: "Beliefs & Decisions"
description: "The typed nodes that carry BCG's memory state."
icon: "brain"
---

A BCG node wraps a `BeliefPayload`. The payload is the stable application-level record; the outer `BCGNode` supplies a UUID, timestamps, generic kind, and extension payload.

## Belief payload

Important fields include:

| Field | Meaning |
|---|---|
| `id` | Monotonic integer identifier within the graph |
| `node_type` | `belief` or `decision` |
| `belief` | Canonical proposition text |
| `decision` | Optional decision-specific text |
| `stance` | `asserted`, `recalled`, `judged`, or `speculated` |
| `layer` | `io` or `reasoning` |
| `role` | Source message role |
| `entities` | Extracted entity labels |
| `source` | Turn, segment, session, and scenario provenance |
| `evidence_ids` | References into graph-level evidence records |
| `factor_ids` | References into factor records |
| `merged_from` | IDs absorbed into this canonical node |
| `confidence` | Current posterior |
| `confidence_history` | Reproducible update audit |

## Belief vs decision

A decision is represented by the same core payload with:

```json
{
  "node_type": "decision",
  "belief": "Prioritize a reliability review for Acme.",
  "decision": "Prioritize a reliability review for Acme."
}
```

This keeps decisions connected to the same evidence, confidence, relation, and temporal machinery as other beliefs.

## Stance

Stance captures how the speaker presents a proposition:

| Stance | Typical language |
|---|---|
| `asserted` | “The service is down.” |
| `recalled` | “I remember the service failed last week.” |
| `judged` | “This pattern suggests a capacity issue.” |
| `speculated` | “The database might be overloaded.” |

Stance contributes to the prior. It is not a truth label.

## Layers

- `io` usually holds externally visible interaction state and propositions grounded in turns, tool outputs, or function I/O.
- `reasoning` usually holds latent reasoning-layer propositions, model-side synthesis, and decision-oriented summaries.

The layer is useful for filtering context and auditing how an output was derived from inputs. It is orthogonal to node type and provenance: decisions are a specialized kind of belief node, and evidence excerpts can be attached regardless of whether the host turn came from a user, assistant, tool, or function.
