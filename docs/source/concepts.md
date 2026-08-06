---
title: "Core Concepts"
description: "The small set of ideas that define BCG's belief-native memory model."
icon: "book-open"
---

BCG models the *state of belief* that an agent accumulates while acting. The graph is built from a trajectory, but it is not merely a transcript graph.

## The model at a glance

| Concept | Purpose |
|---|---|
| **Belief** | A proposition the system can retain, query, merge, and score |
| **Decision** | A belief node whose semantic role is an action or conclusion |
| **Evidence** | An exact excerpt and source reference supporting a node |
| **Confidence** | A deterministic posterior built from explicit components |
| **Relation** | A directional semantic or confidence-propagating connection |
| **Merge** | Canonicalization of semantically duplicate beliefs |
| **Session** | A temporal boundary grouping related turns in one run |
| **Context policy** | The rule deciding which raw turns remain and which graph state is injected |

## Belief, not just text

A belief contains the proposition text plus:

- `node_type`: `belief` or `decision`
- `stance`: `asserted`, `recalled`, `judged`, or `speculated`
- `layer`: `io` or `reasoning`
- source role and turn metadata
- exact evidence excerpts
- entities and optional event time
- merge lineage
- current confidence and full confidence history

This makes the node independently inspectable. Retrieval is only one operation over it.

## Confidence is a graph calculation

BCG stores the prior and the two update channels separately:

```text
confidence = sigmoid(
  logit(initial_confidence)
  + evidence_confidence
  + factor_confidence
)
```

The current score can therefore be reproduced from graph fields rather than accepted as an opaque model judgment.

## Relations have different semantics

The active relation vocabulary is:

- `depends_on` — the source belief depends on the target; strong target confidence can support the source.
- `supplements` — adds context without confidence propagation.
- `contradicts` — the source conflicts with the target and can reduce target confidence when active.

Legacy names remain readable during compatibility migration: `informs`, `extends`, and `confirms`.

## A graph is also an audit record

BCG preserves not only the final nodes and edges but also:

- source offsets
- merge groups and rewiring
- confidence updates
- per-turn snapshots
- engine events
- prompts and embedding calls
- timing and token usage

<CardGroup cols={2}>
<Card title="Beliefs & decisions" icon="circle-nodes" href="/concepts/beliefs-and-decisions" />
<Card title="Confidence" icon="gauge" href="/concepts/confidence" />
<Card title="Evidence & provenance" icon="quote-left" href="/concepts/evidence-and-provenance" />
<Card title="Relations" icon="share-nodes" href="/concepts/relations" />
<Card title="Merging" icon="code-merge" href="/concepts/merging" />
<Card title="Sessions & time" icon="clock" href="/concepts/sessions-and-time" />
</CardGroup>
