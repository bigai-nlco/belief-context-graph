---
title: "Context Policy"
description: "How the reference Agent combines recent raw turns with graph memory."
icon: "shield-check"
---

Graph construction is only half of memory. An agent also needs a policy for deciding what enters the next model request.

## Reference BCG policy

The included Agent:

1. retains the initial user input permanently
2. retains the latest two completed turns as raw messages
3. sends older evicted messages to Graph Construction
4. encodes the current graph with a role-marked dialogue context template and injects it into the system prompt
5. disables conventional conversation compaction

```text
<｜begin▁of▁sentence｜><｜User｜>### Belief 1
**Content:** The user is looking for a specific person.
**Relations:**
- None
**Confidence:** 0.9
```

Each Markdown payload contains the belief ID, content, outgoing relations, and confidence. Every relation is rendered once on its source belief rather than duplicated as an incoming relation on its target. The guide tells the Agent that earlier raw turns were omitted, confidence should influence trust, and searches already represented by the graph should not be repeated. The encoded block is appended directly to the system prompt; it is not wrapped in `<belief_graph>` tags.

This policy gives the model:

- exact recent interaction text
- stable task intent
- compressed long-horizon belief state
- confidence and contradiction signals

## Why a hybrid window

Keeping only graph state can lose phrasing, formatting, or immediate conversational intent. Keeping all raw turns makes context cost grow without bound. A small raw window plus structured graph memory balances the two.

## Customize the window

For the reference Agent:

```bash
export BCG_RECENT_TURNS=4
bcg
```

For a custom agent, choose the window and graph rendering yourself.

## Integration questions

Before adopting a policy, decide:

- Which roles are safe to convert to graph memory?
- Should tool outputs remain verbatim longer than assistant reasoning?
- Which belief confidence is required for prompt injection?
- Should contradictions be included or resolved first?
- How much graph text fits the model context?
- Which nodes must remain hidden from a user-facing response?

<Warning>
BCG supplies graph state and provenance. Your application remains responsible for authorization, prompt construction, data minimization, and action gating.
</Warning>
