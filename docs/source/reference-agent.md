---
title: "Reference Agent"
description: "Use the bundled terminal Agent to experience BCG context management."
icon: "terminal"
---

The repository includes an interactive terminal Agent as a reference integration. It demonstrates one context policy; it is not a required BCG runtime.

## Configure

```bash
bcg setup
```

The setup guide asks for:

1. Agent authentication and endpoint
2. Agent model
3. Default context mode: `BCG` or `Default`
4. Managed local graph server or an existing server
5. `api_based` or `light` when BCG manages the server

Configuration is stored under `~/.bcg`:

```text
~/.bcg/config.json
~/.bcg/.env
~/.bcg/model_config.json
~/.bcg/agent/
~/.bcg/logs/graph-server.log
~/.bcg/graphs/
```

## Launch

```bash
bcg
```

`bcg agent` is an explicit alias.

## Context modes

<Tabs>

<Tab title="BCG">

BCG mode:

- permanently retains the initial user input
- keeps the latest two completed turns as raw messages by default
- sends messages leaving that window to graph construction
- injects the current Markdown graph into the system prompt
- disables traditional conversation compaction

The graph is wrapped in:

```xml
<belief_graph format="markdown">
...
</belief_graph>
```

</Tab>

<Tab title="Default">

Default mode keeps the normal Agent conversation and uses automatic compaction. No graph context is injected.

</Tab>

</Tabs>

The context mode is fixed after the first user message. Start a new session to switch.

## Terminal commands

| Command | Purpose |
|---|---|
| `/help` | Commands and keyboard controls |
| `/model` | Select the inference model |
| `/mode` | Choose Default or BCG before the first message |
| `/login` / `/logout` | Configure or remove credentials |
| `/new` / `/resume` | Start or restore a session |
| `/graph` | Check graph connectivity and context policy |
| `/exit` | Exit |

## Adjust the raw context window

```bash
export BCG_RECENT_TURNS=4
bcg
```

<Warning>
The reference policy is intentionally opinionated. Custom agents should choose retention and graph-injection rules based on their own latency, safety, and task requirements.
</Warning>
