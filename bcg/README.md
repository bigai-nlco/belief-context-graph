# construct_beliefs

A streaming engine that turns a role-tagged conversation or research trajectory into a **belief graph**: a set of typed *belief* and *decision* nodes, connected by typed relations, each node carrying its confidence and exact evidence offsets back into the original text.

The engine is incremental. It reads a conversation **one turn at a time**, and on every turn makes a single LLM call that extracts new nodes plus the typed relations linking them to what came before. Deduplication ("merge") runs both incrementally after each turn and once more at the end. There are no scenarios and no sessions — any input is normalised into a flat stream of turns tagged `user` / `assistant` / `tool`.

You can drive it two ways:

- **Over HTTP** (recommended) — run a small server and POST turns to it as your agent produces them. This is the primary, maintained path.
- **On a single machine** — run a whole input file through `scripts/run.py` in one shot.

---

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration-model_configjson)
- [Quick start (HTTP server)](#quick-start-http-server)
- [Single-machine mode (run.py)](#single-machine-mode-runpy)
- [Replaying a stream offline (online_driver.py)](#replaying-a-stream-offline-online_driverpy)
- [Input data formats](#input-data-formats)
- [Output artifacts](#output-artifacts)
- [Options reference](#options-reference)
- [Project layout](#project-layout)
- [Python API](#python-api)

---

## How it works

Each turn is routed only by its role. `user` and `tool` messages are treated as authoritative; `assistant` messages are the model's own reasoning and answers. `system` turns are recorded but produce no nodes, and `function` is treated as `tool`.

For every non-skipped turn the engine makes **one** LLM call (`extract.update_graph`) that returns:

- **belief nodes** — self-contained reasoning/memory units;
- **decision nodes** — the assistant's final answers (especially anything wrapped in `\boxed{...}`);
- **typed relations** among the new nodes and the existing graph.

Relations use four types: `causal`, `depends_on`, `supplements`, `contradicts`. The model only ever proposes edges that touch at least one *new* node; the engine rejects edges between two pre-existing nodes so old structure is never re-emitted.

**Evidence** is tracked precisely. In the default *sentence* mode the turn is split into whole sentences with exact character offsets, and each node points at the sentences that support it. In *excerpt* mode the model quotes spans verbatim and a three-stage matcher (exact → whitespace-normalised → fuzzy) locates them in the original text.

**Confidence** is assigned in two stages: an initial score from a `(role, stance)` lookup the moment a node is created, then incremental updates driven by the relations discovered as the conversation proceeds. Every move is recorded in each node's `confidence_history`.

**Merge / dedup** removes duplicate nodes. It runs in two places:

- a per-turn *incremental* merge (embedding-only, no LLM verification) right after each turn's new nodes are added;
- a final merge at the end of the trajectory (by default embedding candidates **verified** by the LLM).

Merging is **gated**: two nodes may be merged only when their source **role** is identical **and** their **node_type** is identical. A belief is never absorbed into a decision, and a user claim is never absorbed into an assistant conclusion.

---

## Installation

Requires **Python 3.10+**.

```bash
# from the project root (the directory containing this README)
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` installs the two core dependencies: `openai` (the OpenAI-compatible chat/embedding client) and `numpy` (cosine similarity + clustering used by the default embedding merge).

Embeddings can be served two ways. The default configuration calls an **OpenAI-compatible `/v1/embeddings` HTTP endpoint** (e.g. vLLM, SGLang, TEI, or OpenAI itself), which needs no extra Python packages. Only if you want to load the embedding model **weights in-process** (`"provider": "local"`) do you also need `sentence-transformers` and `torch` — uncomment them in `requirements.txt`.

---

## Configuration (`model_config.json`)

All endpoints and keys live in one JSON file. Copy the template and edit it:

```bash
cp model_config.example.json model_config.json
```

The file is **nested by model name**. Each top-level key is a *model-key* you can select on the command line; the key's name is used as the model name unless the entry sets its own `"model"`. Any key beginning with `embedding` is **reserved** — it is never chosen as the default chat model, which lets several embedding entries coexist with chat entries.

A chat entry must provide `base_url` and `api_key`; `max_tokens` and `pricing` are optional (pricing only drives the token-cost report). A minimal example:

```json
{
  "gpt-5.5": {
    "api_key": "sk-...",
    "base_url": "https://your-endpoint/v1",
    "max_tokens": 100000,
    "pricing": { "input_per_1k": 0.005, "output_per_1k": 0.03 }
  },
  "embedding": {
    "api_key": "EMPTY",
    "base_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen3-Embedding-8B",
    "batch_size": 8
  }
}
```

The engine looks for an embedding entry named `embedding` by default (override with `--embedding-key`). Two providers are supported, selected by the entry's `"provider"` field:

- **`openai`** (the default when `provider` is omitted) — any OpenAI-compatible `/v1/embeddings` endpoint. Requires `base_url`, `api_key`, `model`.
- **`local`** — load the weights in-process with sentence-transformers; no server needed. Requires only `model` (a Hugging Face repo id or a local directory of weights), plus the optional `sentence-transformers` / `torch` dependencies.

If no embedding entry is found, clustering is disabled and the merge strategy silently falls back from `embedding` to `llm`.

> **Keep `model_config.json` out of version control** — it holds real API keys. Commit only `model_config.example.json`.

---

## Quick start (HTTP server)

This is the recommended way to use the engine: start the server once, then push turns to it as your agent generates them.

**1. Start the server.**

```bash
python scripts/online_server.py \
    --config model_config.json \
    --model-key gpt-5.5 \
    --host 127.0.0.1 --port 8848 \
    --output-dir outputs_stream
```

It prints the address it is listening on and the available endpoints.

**2. Push turns.** Each turn is one JSON object carrying a `problem_id` (which trajectory it belongs to), a `role`, and the message `content`. Mark the last turn of a trajectory with `"is_trajectory_end": true` to trigger the final merge.

```bash
curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \
     -d '{"problem_id":"p1","role":"user","content":"Which alloy resists seawater corrosion best?"}'

curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \
     -d '{"problem_id":"p1","role":"assistant","content":"Titanium grade 2 is the standard choice. \\boxed{Titanium grade 2}","is_trajectory_end":true}'
```

Each call returns the current belief-graph snapshot as JSON. When the turn carries `is_trajectory_end=true`, the returned snapshot is the **complete** graph.

### Endpoints

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | — | `{"status":"ok","active":[...],"all":[...]}` |
| `POST /turn` | one turn object | current snapshot for that `problem_id` |
| `POST /turns` | a JSON array of turns, **or** NDJSON (one per line) | `{"pushed":n,"finalized":[...],"latest":{...}}` |
| `POST /finalize` | `{"problem_id":"p1"}` | the final snapshot (use if you never sent `is_trajectory_end`) |
| `GET /graph?problem_id=p1` | — | latest snapshot for that trajectory (404 if unknown) |

### Streaming one message in fragments

The default contract is **one JSON object = one complete turn**. If you stream a single message token-by-token, send the fragments with `"is_message_end": false`; they are buffered and concatenated until a fragment arrives with `is_message_end` true (which is implied by `is_trajectory_end`). Only then is the assembled turn ingested.

### Concurrency

The engine keeps process-global state (the token ledger and log paths) that each session swaps in and out around every call, and that swap is **not** thread-safe. The server therefore serialises all engine work behind a single lock: it is safe to run under `ThreadingHTTPServer` (health checks never block), but only one push/finalize executes at a time — exactly right for a sequential generation stream. For true parallelism, run several server processes, each owning a disjoint set of `problem_id`s.

> **Binding note:** `--host` defaults to `0.0.0.0`, which exposes the server on **all** network interfaces. Use `--host 127.0.0.1` to keep it local to your machine.

---

## Single-machine mode (`run.py`)

When you already have a whole conversation in a file and just want to process it end-to-end, use `scripts/run.py`. It normalises the input into items, runs each item through the same streaming engine, and writes one output sub-directory per item.

```bash
# simplest: a trajectory or conversation file, default options
python scripts/run.py --input data.json

# pick the chat model and embedding entry from the config
python scripts/run.py --input data.json \
    --model-key gpt-5.5 --embedding-key embedding

# whole-sentence evidence + topic clustering + embedding-verified final merge
python scripts/run.py --input data.json \
    --evidence-mode sentence --use-clustering \
    --merge-strategy embedding --merge-threshold 0.86

# free-span evidence (model quotes excerpts; no sentence splitting)
python scripts/run.py --input data.json --evidence-mode excerpt

# process only one item out of a multi-item file (by id or 0-based index)
python scripts/run.py --input data.json --item 3
```

`run.py`'s defaults for the options it shares with the server (`--model-key`, `--context-chars`, `--incremental-merge-threshold`) are aligned with `online_server.py`, so a file run and an HTTP run behave the same way unless you override them.

> **Maintenance note:** the HTTP server is the primary, actively-used path. `run.py` shares the exact same engine (it calls `construct_beliefs.pipeline.run_input`, which builds the same `StreamingBeliefBuilder`), so it stays in lock-step with the engine; it simply sees less day-to-day use than the server.

---

## Replaying a stream offline (`online_driver.py`)

`scripts/online_driver.py` is a thin bridge for feeding a **recorded** stream (JSONL, one turn-dict per line) through the online `SessionManager` — useful for replaying a captured run or piping a generator's output straight in.

```bash
# replay a recorded stream file
python scripts/online_driver.py -i stream.jsonl \
    --config model_config.json --model-key gpt-5.5 --output-dir outputs_stream

# pipe a live generator straight in
my_agent --stream | python scripts/online_driver.py --config model_config.json
```

Any trajectory that never sent `is_trajectory_end` is finalized automatically when the input ends.

---

## Input data formats

The loader (`construct_beliefs/loaders.py`) accepts several shapes and normalises all of them into a flat list of role-tagged turns.

**A trajectory** — a bare list of messages, or an object with a `trajectory` / `messages` key. This becomes one item:

```json
{
  "trajectory": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think> ..."},
    {"role": "assistant", "content": "...", "is_trajectory_end": true}
  ]
}
```

**Multi-session QA data** — a list of items, each carrying a `sessions` array (each session a list of `{role, content, has_answer}` turns), plus parallel `session_ids` / `dates` arrays and question metadata. Each item's sessions are flattened — by default sorted chronologically by date — into one turn stream, and every turn keeps its session's date so time attribution still works. Use `--keep-order` to preserve input order instead of date-sorting.

For the **HTTP / streaming** path, each turn dict additionally carries a `problem_id` (the trajectory key) and may set `is_message_end` (default true) and `is_trajectory_end` (default false). See [Quick start](#quick-start-http-server).

---

## Output artifacts

Each trajectory or item gets its own sub-directory under the output root (`<output-dir>/<problem_id>/` for the server, `<output-dir>/<item_id>/` for `run.py`). You will find:

| File | What it is |
|---|---|
| `result.json` | the main result: full trajectory, all nodes, relations, merges, counts, options, timing, token usage |
| `final_graph.json` | the final belief-graph snapshot |
| `belief_graph_latest.json` | the latest snapshot, overwritten each turn (streaming path) |
| `belief_graph.jsonl` | one snapshot per turn, then a final one — the graph's evolution (streaming path) |
| `trajectory.json` | the reconstructed conversation, in a shape `run.py` can replay |
| `trajectory_stream.jsonl` | append-only raw log of every received turn dict (streaming path) |
| `events.jsonl` | per-turn engine events (new node ids, relations added, incremental merges) |
| `token_usage.json` / `.txt` | token accounting, by stage, with an estimated cost if `pricing` was set |
| `logs/prompts.jsonl` | every prompt sent to the LLM, for auditing |
| `logs/embedding_calls.jsonl` | every embedding call, with full input texts and cache hits |
| `logs/merge_final.json` / `.log` | the final merge pass: candidates, similarities, LLM verifications, applied merges, edge rewiring — both machine- and human-readable |

A `timing.csv` is also appended in the output **root**, one row per finished item.

### Node shape (in `result.json`)

Each node carries an integer `id`, a `node_type` (`belief` or `decision`), the statement text (`belief`, plus `decision` for decision nodes), a `stance` (`asserted` / `recalled` / `speculated` / `judged`), an `entities` list, optional `event_time` / `time_text`, the `source` descriptor (role, turn index, trajectory index), a `confidence` with full `confidence_history`, and an `evidence` list whose entries point back into the original turn by exact character offsets.

---

## Options reference

These flags are common to `run.py`, `online_server.py`, and `online_driver.py` (the server adds `--host`/`--port`/`--quiet`; `run.py` adds `--item`/`--keep-order`/`--min-content-len`).

| Flag | Default | Meaning |
|---|---|---|
| `--config`, `-c` | `model_config.json` | path to the config file |
| `--output-dir`, `-o` | `outputs_stream` | output root; one sub-dir per trajectory/item |
| `--model-key` | `gpt-5.5` | which chat-model entry of the config to use |
| `--embedding-key` | `embedding` | which config entry holds the embedding endpoint |
| `--evidence-mode` | `sentence` | `sentence` = evidence is whole sentences (split + offsets); `excerpt` = model quotes spans verbatim |
| `--use-clustering` | off | group a turn's sentences by topic before extraction (sentence mode; needs an embedder) |
| `--cluster-threshold` | `0.6` | cosine floor for merging sentence clusters |
| `--cluster-min-sentences` | `4` | turns with fewer sentences skip clustering |
| `--cluster-buffer` | `0` | neighbour window used only for sentence embedding |
| `--merge-strategy` | `embedding` | final-merge strategy: `embedding` (candidates + LLM verify), `llm` (LLM proposes directly), or `off` |
| `--merge-threshold` | `0.86` | cosine threshold for final-merge candidate pairs |
| `--incremental-merge` / `--no-incremental-merge` | on | per-turn embedding-only merge (no LLM verification) |
| `--incremental-merge-threshold` | `0.86` | cosine threshold for the per-turn incremental merge |
| `--context-chars` | `100000` | char budget of the existing-nodes context block shown to the model |

> **CLI defaults vs. library defaults.** The table above lists the **command-line** defaults (the same across `run.py` and `online_server.py`). Constructing `StreamOptions()` directly in Python uses two slightly different built-in defaults — `incremental_merge_threshold=0.8` and `context_chars=9000` — because the CLI layer sets its own. If you drive the engine via the Python API and want CLI-identical behaviour, pass these explicitly.

`run.py`-only:

| Flag | Default | Meaning |
|---|---|---|
| `--item` | all | process only this item (by id or 0-based index) |
| `--keep-order` | off | for multi-session inputs, keep input order instead of date-sorting |
| `--min-content-len` | `0` | skip turns shorter than this many characters |

---

## Project layout

```
.
├── construct_beliefs/        # the engine (importable package)
│   ├── __init__.py           # exports BeliefGraph, StreamingBeliefBuilder, StreamOptions
│   ├── stream.py             # StreamingBeliefBuilder — per-turn engine + finalize
│   ├── online.py             # SessionManager / StreamingTrajectorySession (streaming driver)
│   ├── pipeline.py           # run_item / run_input — batch drivers over the engine
│   ├── extract.py            # the single per-turn LLM call (new nodes + relations)
│   ├── prompts.py            # all LLM prompts
│   ├── merge.py              # session-end + incremental merge (role & node_type gated)
│   ├── confidence.py         # two-stage confidence assignment
│   ├── evidence.py           # evidence offsets / excerpt matching
│   ├── split.py              # sentence splitting + global clustering
│   ├── graph.py              # the in-memory belief graph
│   ├── loaders.py            # input normalisers (trajectory / multi-session)
│   ├── llm.py                # OpenAI-compatible chat + embedding clients, token ledger
│   └── link.py               # legacy no-op shim (kept for import compatibility)
│
├── scripts/
│   ├── online_server.py      # the HTTP server (recommended entry point)
│   ├── run.py                # single-machine batch driver
│   └── online_driver.py      # replay a recorded JSONL stream
│
├── model_config.example.json # template — copy to model_config.json
├── requirements.txt
└── README.md
```

---

## Python API

You can also drive the engine directly, without the CLI or HTTP layer.

**Multiplex many trajectories** with a `SessionManager` (this is what the server uses):

```python
from construct_beliefs.online import SessionManager

mgr = SessionManager(config_path="model_config.json", model_key="gpt-5.5")
for turn in incoming_stream:                 # dicts with problem_id / role / content
    snapshot = mgr.push(turn)                # returns the live graph
# a trajectory finalizes on is_trajectory_end, or call mgr.finalize(problem_id)
```

**Build one trajectory** directly with the engine:

```python
from construct_beliefs.stream import StreamingBeliefBuilder, StreamOptions
from construct_beliefs.llm import load_config, make_client

cfg = load_config("model_config.json", model_key="gpt-5.5")
builder = StreamingBeliefBuilder(
    client=make_client(cfg),
    model=cfg["model"],
    item_id="p1",
    out_dir="outputs_stream/p1",
    options=StreamOptions(),       # all the merge / evidence / clustering knobs
)
builder.ingest_turn("user", "Which alloy resists seawater corrosion best?")
builder.ingest_turn("assistant", "Titanium grade 2. \\boxed{Titanium grade 2}")
result = builder.finalize()
```

`StreamOptions` exposes the same knobs as the CLI flags (`evidence_mode`, `use_clustering`, `merge_strategy`, `merge_threshold`, `incremental_merge`, `incremental_merge_threshold`, `context_chars`, …).
