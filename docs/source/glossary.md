---
title: "Glossary"
description: "Terms used throughout the BCG codebase and documentation."
icon: "list"
---

| Term | Meaning |
|---|---|
| **Belief** | A proposition represented as a typed graph node |
| **Decision** | A node marked `node_type="decision"` |
| **Trajectory** | Ordered messages supplied to graph construction |
| **Turn** | One trajectory message or a fully assembled streamed message |
| **Segment** | A sentence, semantic chunk, or isolated tool-call unit |
| **Evidence** | Source text excerpt with optional exact offsets |
| **Stance** | `asserted`, `recalled`, `judged`, or `speculated` |
| **Layer** | `io` for user/tool-facing content, `reasoning` for assistant reasoning |
| **Prior** | `initial_confidence`, derived from source reliability and stance quality |
| **Evidence contribution** | `evidence_confidence`, updated as evidence accumulates |
| **Factor contribution** | `factor_confidence`, propagated through active relations |
| **Canonical node** | The surviving node after a merge |
| **Merge verification** | Optional model gate and rewrite for embedding-proposed duplicates |
| **Activation condition** | Minimum confidence required before an edge propagates a factor |
| **Snapshot** | Graph state emitted after a turn or finalization |
| **Run** | One construction lifecycle with a stable `run_id` |
| **Session** | A logical temporal group of turns inside a run |
| **Problem ID** | HTTP streaming key selecting one active graph session |
| **BCG context mode** | Reference Agent policy that retains a small raw window and injects graph memory |
