---
title: "Architecture"
description: "How trajectories become canonical, confidence-bearing belief graphs."
icon: "diagram-project"
---

BCG separates public interfaces from its two core construction engines: `bcg.construct.api_based` and `bcg.construct.light`.


<div class="arch-diagram" aria-label="BCG architecture overview">
  <svg role="img" viewBox="0 0 1040 560" aria-labelledby="archTitle archDesc">
    <title id="archTitle">BCG architecture overview</title>
    <desc id="archDesc">Public interfaces feed BCGRunner or SessionManager, then converge on the selected bcg.construct api_based or light StreamingTrajectorySession, which executes split, extract, confidence, merge, relation, and audit stages.</desc>
    <defs>
      <marker id="archArrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M1 1 L11 6 L1 11 Z" class="arch-arrowhead"></path>
      </marker>
    </defs>

    <rect class="arch-card arch-card-top" x="180" y="24" width="680" height="64" rx="18"></rect>
    <text class="arch-text arch-title" x="520" y="64" text-anchor="middle">Agent / application / batch file / benchmark adapter</text>

    <line class="arch-link" x1="520" y1="88" x2="520" y2="126"></line>
    <line class="arch-link" x1="520" y1="126" x2="300" y2="126"></line>
    <line class="arch-link" x1="520" y1="126" x2="740" y2="126"></line>

    <text class="arch-label" x="300" y="118" text-anchor="middle">Python SDK</text>
    <text class="arch-label" x="740" y="118" text-anchor="middle">HTTP / CLI</text>

    <line class="arch-link" x1="300" y1="126" x2="300" y2="152" marker-end="url(#archArrow)"></line>
    <line class="arch-link" x1="740" y1="126" x2="740" y2="152" marker-end="url(#archArrow)"></line>

    <rect class="arch-card arch-card-left" x="180" y="156" width="240" height="72" rx="18"></rect>
    <rect class="arch-card arch-card-right" x="620" y="156" width="240" height="72" rx="18"></rect>
    <text class="arch-text" x="300" y="198" text-anchor="middle">BCGRunner</text>
    <text class="arch-text" x="740" y="198" text-anchor="middle">SessionManager</text>

    <line class="arch-link" x1="300" y1="228" x2="300" y2="258"></line>
    <line class="arch-link" x1="740" y1="228" x2="740" y2="258"></line>
    <line class="arch-link" x1="300" y1="258" x2="520" y2="258"></line>
    <line class="arch-link" x1="740" y1="258" x2="520" y2="258"></line>
    <line class="arch-link" x1="520" y1="258" x2="520" y2="286" marker-end="url(#archArrow)"></line>

    <rect class="arch-card arch-card-center" x="320" y="288" width="400" height="78" rx="20"></rect>
    <text class="arch-text" x="520" y="320" text-anchor="middle">StreamingTrajectorySession</text>
    <text class="arch-subtext" x="520" y="345" text-anchor="middle">bcg.construct.api_based · bcg.construct.light</text>

    <line class="arch-link" x1="520" y1="366" x2="520" y2="392" marker-end="url(#archArrow)"></line>

    <rect class="arch-stage" x="84" y="410" width="128" height="50" rx="15"></rect>
    <rect class="arch-stage" x="236" y="410" width="128" height="50" rx="15"></rect>
    <rect class="arch-stage" x="388" y="410" width="128" height="50" rx="15"></rect>
    <rect class="arch-stage" x="540" y="410" width="128" height="50" rx="15"></rect>
    <rect class="arch-stage" x="692" y="410" width="128" height="50" rx="15"></rect>
    <rect class="arch-stage" x="844" y="410" width="112" height="50" rx="15"></rect>

    <text class="arch-stage-text" x="148" y="440" text-anchor="middle">split</text>
    <text class="arch-stage-text" x="300" y="440" text-anchor="middle">extract</text>
    <text class="arch-stage-text" x="452" y="440" text-anchor="middle">confidence</text>
    <text class="arch-stage-text" x="604" y="440" text-anchor="middle">merge</text>
    <text class="arch-stage-text" x="756" y="440" text-anchor="middle">relations</text>
    <text class="arch-stage-text" x="900" y="440" text-anchor="middle">audit</text>

    <line class="arch-link" x1="212" y1="435" x2="236" y2="435" marker-end="url(#archArrow)"></line>
    <line class="arch-link" x1="364" y1="435" x2="388" y2="435" marker-end="url(#archArrow)"></line>
    <line class="arch-link" x1="516" y1="435" x2="540" y2="435" marker-end="url(#archArrow)"></line>
    <line class="arch-link" x1="668" y1="435" x2="692" y2="435" marker-end="url(#archArrow)"></line>
    <line class="arch-link" x1="820" y1="435" x2="844" y2="435" marker-end="url(#archArrow)"></line>

    <line class="arch-link" x1="520" y1="460" x2="520" y2="488" marker-end="url(#archArrow)"></line>

    <rect class="arch-card arch-card-bottom" x="336" y="492" width="368" height="52" rx="16"></rect>
    <text class="arch-text" x="520" y="523" text-anchor="middle">BCG graph + run artifacts</text>
  </svg>
  <p class="arch-caption">The public entry paths converge on the selected backend session. The core stage implementations live in <code>bcg.construct.api_based</code> and <code>bcg.construct.light</code>; the surrounding SDK, HTTP, and CLI layers select and orchestrate one of them.</p>
</div>


## Public layers

### Graph model

`bcg.graph` defines the typed Pydantic models:

- `BCG`, `BCGNode`, `BCGEdge`
- `BeliefPayload`, `BeliefSource`, `EvidenceExcerpt`
- `RelationPayload`, `RelationActivationCondition`
- `ConfidenceHistoryEntry`

### Memory facade

`BCGMemory` offers application-facing operations:

- manually observe an already-formed belief
- find matching beliefs
- search current graph text
- assemble task context with conflicts and missing-evidence flags

### Construction lifecycle

`BCGRunner` owns one run at a time, selects either `bcg.construct.api_based` or `bcg.construct.light`, synchronizes snapshots into the public `BCG` model, tracks sessions, and writes compatibility artifacts. The split, extraction, confidence, merge, and relation logic remains inside the selected `bcg.construct` package.

### CLI and HTTP

The `bcg construct` command family wraps the same construction backends for batch, server, replay, and visualization workflows.

## Incremental construction pipeline

<Steps>

<Step title="Segment the turn">

`api_based` uses sentence-oriented evidence or free excerpts. `light` can use semantic breakpoint chunking and isolates tool calls.

</Step>

<Step title="Extract nodes">

The backend produces zero or more belief and decision candidates for the current turn.

</Step>

<Step title="Initialize confidence">

Source reliability and stance quality determine the prior. Evidence and graph factors remain separate components.

</Step>

<Step title="Merge before linking">

Embedding candidates identify possible duplicates. Optional model verification can accept, reject, and rewrite the surviving canonical node.

</Step>

<Step title="Generate relations">

Relations are created against canonical nodes, preventing new edges from pointing at nodes removed in the same turn.

</Step>

<Step title="Propagate confidence">

Active `depends_on` and `contradicts` edges contribute deterministic factors until convergence limits are reached.

</Step>

<Step title="Persist artifacts">

The final graph, per-turn snapshots, audit logs, timing, token usage, and merge details are written beneath the run output directory.

</Step>

</Steps>

## Backend boundary

The core construction logic is implemented in the sibling packages `bcg.construct.api_based` and `bcg.construct.light`. They implement the same graph contract but make different infrastructure tradeoffs:

| Area | api_based | light |
|---|---|---|
| Node and relation generation | One OpenAI-compatible chat model | Small generative model via compatible endpoint |
| Embeddings | Configured embedding endpoint or local provider | Local sentence-transformers |
| Stance | Inferred through API pipeline | Required local four-class classifier |
| NER | Pipeline extraction | Local spaCy / rules / Hugging Face options |
| Tuning surface | CLI flags and stream options | Full `belief_graph` config sections |

See [Choosing a backend](/backends/choosing-a-backend).

## Concurrency model

The HTTP server uses `ThreadingHTTPServer`. Each `problem_id` owns a separate session and lock:

- turns for the same problem are processed in arrival order
- different problem IDs can execute concurrently
- batch endpoints fan independent items across a worker pool
- `POST /release` removes a completed session from memory

<Warning>
This is process-local coordination. A load-balanced multi-process deployment needs sticky routing or an external session coordinator.
</Warning>
