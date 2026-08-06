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
4. injects the current graph as Markdown into the system prompt
5. disables conventional conversation compaction

```xml
<belief_graph format="markdown">
# Beliefs
...
</belief_graph>
```

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
