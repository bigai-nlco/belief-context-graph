# BCG context gateway for native Codex

This checkout provides one Codex integration: the request-side BCG gateway. It launches one continuing native Codex session, not the reference benchmark Agent or a new Codex process per user turn. The terminal, tools, approvals, and visible conversation stay native; the gateway changes outgoing model context.

See [gateway validation](codex-native-validation.md) for the actual continuing-session checks and their limits.

## Install and initialize

Requirements: Python 3.11–3.13, `uv`, a native Codex installation, and a ChatGPT login. Linux with Codex CLI 0.162.1 is the most recently verified environment; other operating systems are not yet verified. From this checkout:

```sh
uv tool install --python 3.13 .
codex login

cd /path/to/your/project
bcg codex init --env-file ~/.bcg/.env --allow-graph-upload
```

`--mode gateway` remains accepted, but can be omitted because gateway is the only mode. Initialization is offline. It does not install hooks/MCP, read model credentials, contact providers, download models, or modify any project/global Codex configuration or authentication.

The upload flag acknowledges that captured conversation and tool text will be sent to your configured Graph provider. Review that provider's data policy and your organization's rules. Do not assume complete secret filtering.

Initialization creates:

```text
your-project/
└── .bcg-codex/
    ├── .gitignore
    ├── config.json
    ├── graph-model.json
    └── install.json
```

No `.codex/` wiring or compaction prompt is generated. The private-data ignore file excludes configuration and session records from Git. Reinitializing preserves existing Graph model settings, record paths, and memories, updates the selected env-file path, and re-enables the gateway.

A legacy installation that owns hooks/MCP must first be removed using its original or archived installer. This gateway-only checkout refuses to overwrite that wiring or treat it as a gateway installation. Existing manually created gateway configuration without a manifest can be adopted in place.

## Models and credentials

The Agent uses Codex's existing ChatGPT login. The gateway forwards Agent authentication only to the pinned official Codex endpoint, never to the Graph provider. The launcher currently targets this login path, not arbitrary Agent API providers.

Graph Construction uses a separate key and is billed independently. The generated `graph-model.json` includes this section:

```json
{
  "graph-model": {
    "model": "openai/gpt-6-luna",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_OPENAI_API_KEY",
    "reasoning_effort": "none",
    "max_tokens": 4096
  }
}
```

Put the named variable in your private env file:

```dotenv
OPENROUTER_OPENAI_API_KEY=your-graph-api-key
```

For another provider, edit `model`, `base_url`, `api_key_env`, and the supported reasoning setting in `graph-model.json`. Preserve its other sections. Initialization does not create or modify the env file, and never copies the key into project configuration.

The complete configuration also selects local CPU `sentence-transformers/all-MiniLM-L6-v2` embeddings. Weights may download on first use. For a new configuration, `init --embedding-model /path/to/local/model` selects existing weights.

The gateway loads the unified Graph SDK lazily in-process. Its runtime keeps the existing settings: `context_chars=12000`, two extraction-history Graph turns, and three previous relation layers. No separate Graph HTTP server, event queue, or background hook worker is required.

## Experimental request gateway

From the initialized project root, start Codex with BCG:

```sh
bcg codex
```

BCG automatically adds `--no-daemon` so an existing shared Codex daemon cannot reuse older provider settings instead of this invocation's configuration. You do not need to pass it yourself. Optional Codex arguments go after `--`, for example:

```sh
bcg codex -- -m gpt-6-luna -c 'model_reasoning_effort="low"'
```

To resume a conversation with BCG still enabled:

```sh
bcg codex resume                 # Open the session picker
bcg codex resume --last          # Continue the latest session in this directory
bcg codex resume SESSION_ID      # Continue a specific session
```

Add `--all` to include sessions from other directories. Resume restores the native Codex conversation, but BCG currently creates a new Graph from the history replayed in subsequent requests, not from the previous Graph state. This can add Graph API costs; history already removed by native compaction cannot be reconstructed from the replay alone. The older `bcg codex run` and `bcg codex run -- resume ...` forms remain compatible.

The local gateway listens on loopback only and exits when Codex exits. It uses an ephemeral token and command-line provider overrides, not persistent Codex configuration changes. Plain `codex` does not start BCG. Any legacy BCG memory MCP registration is disabled for this invocation only to prevent duplicate Graph building; no new MCP server is registered.

For each supported model request:

1. Retain the original user/system/developer input and the latest two completed Assistant/tool steps, plus unfinished tool work.
2. Write older completed steps to the Graph before removing them from the outgoing request.
3. Select relevant Graph beliefs and original relations using focused selection with a 6,600-character node budget.
4. Append the selected Graph memory to the original Responses `instructions`.
5. Forward the request and stream the response back to the same Codex process.

A step means an Assistant/tool-call batch and its paired results, not one individual message or a user-visible conversational turn. Retained items, tool definitions, call IDs, and original instructions are preserved. Generic text tool outputs use bounded observation extraction; parallel results remain one Graph turn. The original retained tool outputs are not rewritten.

Graph construction can delay a request because it happens before forwarding. When Graph building or selection fails, selection is empty, or history cannot safely be interpreted, the original input is forwarded unchanged. UI history is never deleted. This is not a guarantee of lossless memory or exact two-step control after every form of native compaction.

Native title generation and compaction requests bypass Graph transformation. Opaque compaction/history items use the native-context fallback. Original retained encrypted reasoning items are preserved, but encrypted reasoning is never decrypted. Only exposed reasoning text or summaries can become Graph input.

The launcher uses the HTTP Responses transport. Remote Codex environments, other auth paths, hidden provider history, and every possible tool format are not guaranteed compatible. Code Mode outputs are not always unwrapped into their nested tools.

## Diagnostics

`bcg codex status` reports the mode, enabled flag, and local audit/Graph paths. It is not a live server health check.

Runtime records are written under `.bcg-codex/state/`:

```text
state/
├── gateway-graph.log
├── gateway/
│   ├── requests.jsonl
│   └── model_io/<request_id>.jsonl
└── graphs/<problem_id>/
```

Request audits include `input_items_before`, `input_items_after`, `evicted_steps`, `graph_chars`, `selected_node_ids`, and `selected_relation_ids`. These count protocol items and complete steps, not user-visible turns. `fallback` and `fallback_detail` identify conservative pass-through without exposing provider exception messages.

Full model I/O is opt-in:

```sh
bcg codex --record-model-io
```

Each I/O file contains `incoming`, `forwarded`, and `context`, followed by the response. Logging is capped at 100 MiB per directory; `model_io_enabled` does not guarantee every request was recorded after the cap. Earlier unrecorded requests cannot be recovered. Logs may contain private code, user text, and encrypted reasoning.

Graph trajectories and snapshots contain only messages ingested into the graph, not the complete Codex transcript. Keep them private.

## Disable or remove

```sh
bcg codex disable
bcg codex enable
bcg codex uninstall
```

Disable blocks future gateway launches. Exit Codex to stop an already-running gateway. Uninstall disables the gateway and removes its installation manifest, while preserving configuration and memories. It does not touch any `.codex` files. Delete private records separately only if you intend to erase them.

Run from the initialized project root, or pass `--config /project/.bcg-codex/config.json` when launching/resuming or using status/enable/disable, and `--project /project` for init/uninstall. BCG's `--config` selects its project configuration; use native Codex's `-c key=value` for Codex setting overrides.

## Official interface

The integration uses Codex's [custom model-provider configuration](https://learn.chatgpt.com/docs/config-file/config-advanced#custom-model-providers) and per-invocation [command-line overrides](https://learn.chatgpt.com/docs/cli/reference). BCG request transformation is experimental, not an official Codex context-management extension.
