---
title: "Evidence & Provenance"
description: "Trace a belief back to the exact text that produced it."
icon: "highlighter"
---

Evidence is stored both as graph-level records and as typed excerpts attached to belief payloads.

## Evidence excerpt

```python
from bcg.graph import EvidenceExcerpt

excerpt = EvidenceExcerpt(
    text="Acme is threatening to churn",
    start=0,
    end=29,
    match="exact",
    via="split_sentence",
    source=belief_source,
)
```

Supported provenance fields include:

- excerpt `text`
- start and exclusive end offsets
- match quality: `exact`, `normalized`, `fuzzy`, or `not_found`
- extraction path: `llm_excerpt`, `split_sentence`, `semantic_chunk`, or `manual`
- role, trajectory index, segment index, session, and turn metadata

## Evidence modes

<Tabs>

<Tab title="Sentence">

The `unified` backend uses whole-sentence evidence with offsets. This is the CLI default.

</Tab>

<Tab title="Excerpt">

The model quotes a free span verbatim. Use:

```bash
bcg construct run unified \
  --input trajectory.json \
  --evidence-mode excerpt
```

</Tab>

<Tab title="Semantic chunk">

The `hybrid` backend can use embedding-distance semantic chunks and isolate tool calls before extraction.

</Tab>

</Tabs>

## Why offsets matter

Exact offsets enable:

- evidence highlighting in the viewer
- reproducible review against the original turn
- detection of unsupported summaries
- human correction without rereading the full trajectory
- audit records that survive node rewrites during merge

The live viewer renders evidence with a warm yellow highlight and links it to the corresponding graph node.

## Missing evidence

`BCGMemory.context()` and `search()` can retain nodes with missing-evidence flags when `include_missing_evidence=True`. This helps applications distinguish a semantically relevant node from a well-supported one.
