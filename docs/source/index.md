---
title: "Belief Context Graph"
description: "Belief-native graph memory for LLM agents: probabilistic, temporal, evidence-grounded, and auditable."
icon: "brain"
---

```bash
curl -LsSf https://raw.githubusercontent.com/bigai-nlco/belief-context-graph/main/install.sh | sh
bcg --version
```

Most agent memory systems are optimized for retrieval: preserving conversation history, finding similar fragments, linking entities, or recording what happened. An agent can retrieve the right passage and still act on the wrong belief.

Those systems answer retrieval questions—*what is relevant, what is related, and what happened?* BCG addresses the next layer: what the agent should believe and whether confidence is sufficient to act.

The missing questions are:

- Should the agent believe this claim?
- How confident should it be?
- Which exact evidence supports it?
- What contradicts or qualifies it?
- Was it formed in this session or inherited from an earlier one?
- Is uncertainty high enough to block action?

**Belief Context Graph (`BCG`) upgrades agent memory from retrieval memory to belief computation memory.** It turns an agent trajectory into a graph of beliefs, decisions, evidence, confidence, and typed relations. BCG is agent-independent: use the included terminal Agent, embed the Python SDK, or call the streaming HTTP server from your own runtime.

## What BCG adds

<CardGroup cols={2}>

<Card title="Belief-native memory" icon="brain">
Store beliefs and decisions as first-class typed nodes rather than anonymous chunks.
</Card>

<Card title="Deterministic confidence" icon="gauge-high">
Recompute posterior confidence from an explicit prior, merged evidence, and active graph factors.
</Card>

<Card title="Exact provenance" icon="highlighter">
Link every extracted belief back to character offsets in its source turn.
</Card>

<Card title="Temporal state" icon="clock-rotate-left">
Track sessions, turns, event time, merge history, and confidence history.
</Card>

<Card title="Conflict-aware relations" icon="diagram-project">
Represent dependency, supplementation, and contradiction explicitly.
</Card>

<Card title="Auditable execution" icon="magnifying-glass-chart">
Inspect prompts, embedding calls, merge decisions, timing, token usage, and graph evolution.
</Card>

</CardGroup>

## The core loop

```text
conversation turn
      │
      ▼
split or semantic chunk
      │
      ▼
extract beliefs + decisions + evidence
      │
      ▼
initialize confidence
      │
      ▼
merge duplicates before linking
      │
      ▼
create typed relations
      │
      ▼
propagate relation confidence
      │
      ▼
queryable BCG + audit artifacts
```

<Note>
BCG does not replace an agent framework. It supplies graph construction, a typed graph model, context assembly, HTTP endpoints, artifacts, and reference integrations beneath your existing agent loop.
</Note>

## Start by use case

<CardGroup cols={2}>

<Card title="Try the reference Agent" icon="terminal" href="/reference-agent">
Launch the bundled terminal Agent and switch between Default and BCG context modes.
</Card>

<Card title="Embed the Python SDK" icon="python" href="/guides/python-sdk">
Construct graphs inside an application with `BCG`, `BCGMemory`, and `BCGRunner`.
</Card>

<Card title="Run a graph service" icon="server" href="/guides/streaming-server">
Push turns over HTTP and retrieve the latest graph per `problem_id`.
</Card>

<Card title="Inspect a live run" icon="eye" href="/guides/live-stream-viewer">
Replay graph growth, inspect evidence, select subgraphs, and inspect the final decision path.
</Card>

</CardGroup>

Runnable examples are available in [Quick Start](/quickstart), [Manual beliefs](/guides/manual-beliefs), [Streaming HTTP server](/guides/streaming-server), and [Batch construction](/guides/batch-construction).

## Current project boundaries

BCG currently persists complete run artifacts to files and keeps each active construction session in process memory. The repository does **not** include a production graph-database adapter, a multi-node session coordinator, or a hosted control plane. Deploy those concerns around BCG when your application requires them.

<Tip>
For the shortest working path, install BCG, run `bcg setup`, and launch `bcg`. For framework integration, start with the Python SDK or HTTP server instead.
</Tip>
