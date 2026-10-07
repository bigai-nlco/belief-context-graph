# BCG inside a continuing native Codex session

This integration lives in the isolated `belief-context-graph-codex` checkout. It does not replace Codex with the benchmark agent, and it does not start a new Codex process for every model call or user message. Nothing is installed into your global Codex configuration automatically.

## Three selectable approaches

| Approach | Start the agent | What changes in actual model context | Important limit |
| --- | --- | --- | --- |
| `assist` | Ordinary `codex` after project setup | Hooks add bounded, selected graph memory as **developer context**; MCP supplies recall and evidence tools. | Does not delete history, rewrite the system prompt, or promise token savings. |
| `compact` | Ordinary `codex` after project setup | Everything in `assist`, plus a bounded pre-compaction queue drain, a compaction handoff prompt and graph rehydration after native compaction. An optional token threshold controls when Codex compacts. | Codex still owns compaction. This is not an exact two-step window or guaranteed verbatim retention of every user input. Provider/version behavior can affect custom compaction prompts. |
| `gateway` | `bcg codex run --mode gateway -- …` | One continuing native Codex TUI, with outgoing Responses requests edited by a local gateway: original user/system/developer input retained, recent two completed assistant/tool steps retained, evicted steps built into a graph and selected memory appended to `instructions`. | Experimental, HTTP-only, native ChatGPT login in this launcher. An opaque/native compaction item or unsupported history falls back to unmodified native context. No claim of permanent control over hidden state. |

Start with `assist` for compatibility. Use `compact` to experiment with a supported, partial context intervention. Use `gateway` only when request-level trimming is worth its compatibility and latency trade-offs. These are selectable integrations in one codebase, not three forks of the graph algorithm.

## Install and enable

Prerequisites: Python 3.11–3.13, the BCG graph dependencies, and a native Codex installation with lifecycle Hooks and MCP support. The live verification target is Codex CLI **0.160.1** on Linux. Older releases without these hook events are not supported by the automatic-capture path. Other operating systems are not yet verified; the worker lock currently uses POSIX `flock`.

From this checkout:

```sh
uv sync
uv run bcg codex init \
  --project /path/to/your/project \
  --mode assist \
  --env-file ~/.bcg/.env \
  --allow-graph-upload

cd /path/to/your/project
codex
```

Alternatively, install this checkout with `uv tool install .` and use `bcg codex init` without `uv run`. There is no new published package/release implied by this prototype.

**Review the data transfer before enabling it.** The upload flag opts this project into sending captured user text, native tool arguments/results and final assistant text to your configured graph provider. Known configured credentials and common token patterns are redacted, but this is **not** a complete secret/DLP scanner. Do not enable it in a sensitive project without reviewing that provider's data policy and your organization's rules. Images, binary attachments, hidden/encrypted reasoning, and arbitrary transcript files are not collected.

Trust the project in Codex, then open **`/hooks`**, review the BCG commands and trust their exact definitions. Installing BCG does **not** bypass Codex's hook trust. Review the `bcg_memory` entry in `/mcp` too. Native sandbox, tool permissions, tools and approvals remain under Codex's control. If hooks are disabled by policy, do not bypass it; use ordinary Codex without automatic capture.

The installation creates only:

```text
your-project/
├── .codex/config.toml       # named MCP entry; marked compaction block if chosen
├── .codex/hooks.json        # BCG entries merged with existing hooks
└── .bcg-codex/              # private config, queues, graphs; do not commit
    ├── config.json
    ├── graph-model.json
    ├── compact-prompt.txt
    ├── install.json
    └── state/
        ├── memory.sqlite3
        ├── graph-worker.log
        └── graphs/<graph_id>/
```

Keep `.bcg-codex/` out of version control. Generated `.codex` wiring contains machine-local absolute paths; do not publish it as portable project configuration. Existing non-BCG hooks and MCP servers are preserved. `~/.codex/config.toml` and `auth.json` are not edited by installation.

## Models and credentials

Agent authentication/model/effort are whatever native Codex uses. For this experiment, launch `codex -m gpt-6-luna -c model_reasoning_effort=low`. BCG never reads Codex auth for `assist` or `compact`, and no agent key is copied into BCG config.

The generated `graph-model.json` defaults to:

```json
{
  "graph-model": {
    "model": "qwen/qwen3.8-flash",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_OPENAI_API_KEY",
    "reasoning_effort": "none",
    "max_tokens": 4096
  }
}
```

Put `OPENROUTER_OPENAI_API_KEY=…` in the selected private env file, not the project TOML. The complete generated JSON also configures local `sentence-transformers/all-MiniLM-L6-v2` embeddings, `context_chars=12000`, two extraction-history graph turns and three previous relation layers. Pass `--embedding-model /path/to/all-MiniLM-L6-v2` to use existing local weights. Stance/entities, relations and confidence propagation use the existing unified graph implementation. Embeddings are local/CPU, and weights may download on first use unless a local directory is supplied.

No separate HTTP graph server is required for native modes: the MCP process loads the existing unified BCG SDK in its background worker. MCP discovery remains responsive while graph models load. Search is **not** bundled into production BCG memory; use Codex's usual search/file/tools. The live smoke test adds a separate, read-only Serper MCP adapter for its BrowseComp question only.

## Native event and memory flow

1. `UserPromptSubmit` records the user's submitted text. `PostToolUse` records the real native/nested tool name, arguments, call ID and returned text. `Stop` records the final assistant text, including answers not previously ingested. Only documented event fields are consumed; intermediate assistant thinking is not available through these hooks and is not invented.
2. Hooks write a private local SQLite inbox and return promptly. They do not call the graph model, block a tool, approve a request, replace a tool output, or continue a finished turn. BCG's own MCP responses are excluded from capture to avoid a memory feedback loop.
3. One background worker per project builds the graph. Exact tool-call IDs connect each result to its call. Search snippets use the existing semantic extraction path. Other textual tool outputs are represented as one bounded observation item so the same semantic extractor can extract facts, rather than saving an opaque `exec` string. The original captured event stays in SQLite. Capture is limited to 12,000 characters per text field and 500 events per session by default; truncation and failures are explicit.
4. Automatic context is capped at 4,400 characters including guidance. It includes original selected belief text, propagated confidence and original edges, with graph-qualified IDs. A lexical relevance ranking (including Chinese bigrams) picks seeds and up to two real neighbouring nodes. Space is reserved for edges. Original user beliefs and decisions are eligible because native compaction can omit their source messages. There are at most two fresh graph injections during one native user turn; stale snapshots are not repeatedly injected on every tool call.
5. The model can call `bcg_memory.recall(session_id, query, wait_seconds)`, `inspect(session_id, graph_id, node_id)` and `status(session_id)`. These tools are read-only. Hooks supply the current session ID. `inspect` returns original evidence and incident relations, not an invented summary.

Graph capture is **event-based**, not a byte-identical audit of all Codex model input/output. Parallel tool completions are recorded in their observed completion order. Hosted search/image tools without native tool hooks are not directly captured; their visible final answer can still be captured. Tool input/results are never automatically treated as trusted instructions.

On restart/resume, acknowledged graph snapshots remain available without replaying paid model calls. New work starts a separate graph segment; node IDs are qualified by segment and no cross-segment edges are fabricated. Unacknowledged in-flight events are retried in a new segment. Failed events are visible in status and require an explicit `retry`; there is no unbounded automatic provider retry loop at the integration level. The graph SDK still has its own request retry behavior.

## Compaction variant

```sh
bcg codex init --project /path/to/your/project --mode compact \
  --env-file ~/.bcg/.env --allow-graph-upload
cd /path/to/your/project
codex
```

Restart Codex after changing modes and review changed hooks if prompted. Use `/compact` normally. Optionally specify `--compact-at 24000` during initialization to request an earlier native token-based compaction threshold; the default leaves the model's normal threshold unchanged. Existing user-owned compaction settings cause initialization to stop for reconciliation instead of silently replacing them.

Before compaction the hook waits at most 15 seconds for pending graph work. It never vetoes compaction if the graph is slow/offline. The `SessionStart(source=compact)` hook reintroduces relevant stored beliefs as developer context before continuation, alongside Codex's own summary. Pending/failed counts are included, so the model is not told the graph is complete when it is not.

## Experimental request gateway

After the same project initialization:

```sh
cd /path/to/your/project
bcg codex run --mode gateway -- -m gpt-6-luna -c model_reasoning_effort=low
```

The wrapper launches **one** native interactive Codex process and a loopback-only proxy for its lifetime. Codex reads its own normal login; only an ephemeral local provider override is passed on the command line. Login bearer credentials are forwarded only to the pinned official Codex endpoint, not to OpenRouter or the graph model. Other Codex tools, permissions and configuration are preserved. The native memory hooks/MCP are disabled for this invocation only to avoid duplicate building.

Here a step means a complete assistant/tool-call batch plus paired results, not a user-visible conversation turn or two individual messages. The UI/local history stays native; the actual model request is trimmed. On graph failure or unsupported input, the full original input is forwarded instead of being lost. Native compaction is passed through unmodified; opaque compaction content cannot be safely dissected, so later requests containing it use the native-context fallback. Tool-call/response pairing and original retained reasoning items are preserved. Encrypted reasoning is never decrypted.

This variant may synchronously wait for graph construction before forwarding a request. The original gateway's tool extraction does not yet unwrap every Code Mode `exec` into its nested tools; native Hooks are better at that. It is not a drop-in replacement for every Codex transport, auth provider, UI or remote environment. `--record-model-io` is optional and sensitive; by default only bounded metadata is logged.

## Inspect, pause and remove

```sh
bcg codex status
bcg codex disable      # stop new capture/building; an in-flight request may finish
bcg codex enable
bcg codex retry        # explicitly retry failed events; may incur model cost
bcg codex uninstall   # remove only generated wiring; private memories stay on disk
```

Run from the initialized project root, or pass `--config /project/.bcg-codex/config.json` (`--project` for uninstall). Restart Codex after uninstalling. Use normal filesystem tools to delete the private `.bcg-codex` directory if you also want to erase the memories. No unrelated server/tmux process is stopped.

## Official interface references

The supported native capabilities and trust requirements are documented in [Codex Hooks](https://learn.chatgpt.com/docs/hooks). Hook-added context is developer context; the API does not offer arbitrary history deletion or direct system-message replacement. Context/compaction settings are in the [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference). The stronger gateway uses the [custom model-provider interface](https://learn.chatgpt.com/docs/config-file/config-advanced#custom-model-providers), but its history transformation remains a BCG experiment rather than an official Codex context-management extension.
