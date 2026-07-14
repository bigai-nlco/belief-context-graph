# construct_beliefs

A streaming engine that turns a role-tagged conversation or research trajectory into a **belief graph**: a set of typed *belief* and *decision* nodes, connected by typed relations, each node carrying its confidence and exact evidence offsets back into the original text.

The engine is incremental. It reads a conversation **one turn at a time**, and on every turn runs a short three-phase pipeline: extract new nodes, deduplicate them against the graph, then extract the typed relations linking them to what came before. Deduplication ("merge") runs incrementally after each turn, and optionally once more at the end of the trajectory. There are no scenarios and no sessions — any input is normalised into a flat stream of turns tagged `user` / `assistant` / `tool`.

You can drive it two ways:

- **Over HTTP** (recommended) — run a small server and POST turns to it as your agent produces them. This is the primary, maintained path.
- **On a single machine** — run a whole input file through `bcg/run.py` in one shot.

---

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration-env--model_configjson)
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

For every non-skipped turn the engine runs a **three-phase pipeline**:

1. **Node extraction** (`extract.extract_nodes`) — one LLM call that returns only the new **belief nodes** (self-contained reasoning/memory units) and **decision nodes** (the assistant's final answers, especially anything wrapped in `\boxed{...}`). No relations yet.
2. **Incremental merge** — the new belief nodes are deduplicated against the graph immediately, *before* any edges are drawn, so relations are always attached to a clean, deduplicated graph. Decision nodes are excluded from this per-turn merge; only the final decision is retained as a decision at trajectory end (see below).
3. **Relation extraction** (`extract.extract_relations`) — one or more LLM calls that add the typed relations. The call links the surviving new nodes against the immediately-previous turn's nodes (plus new↔new). If that adjacent turn yields no cross-turn edge, the window walks backward one turn at a time until a current-to-prior edge is added or no earlier turn remains — so a turn may issue several relation calls.

Relations use three types: `depends_on`, `supplements`, and `contradicts`. Every accepted edge must touch at least one *new* node; the engine rejects edges between two pre-existing nodes so old structure is never re-emitted.

**Evidence** is tracked precisely. In the default *sentence* mode the turn is split into whole sentences with exact character offsets, and each node points at the sentences that support it. In *excerpt* mode the model quotes spans verbatim and a three-stage matcher (exact → whitespace-normalised → fuzzy) locates them in the original text.

**Confidence** is assigned when a node is created — an initial score from a `(role, stance)` prior — and is recomputed as evidence accumulates (for example when a merge folds another node's evidence into it). Every move is recorded in each node's `confidence_history`.

**Merge / dedup** removes duplicate nodes and runs in two places:

- a per-turn **incremental** merge, right after each turn's new belief nodes are added. It embeds the candidate statements and flags pairs above the threshold. By default (`--verify-merge`, on) it then calls the LLM once per candidate group to *verify* the merge is reasonable and *rewrite* the surviving node's text to cover all merged meanings; with `--no-verify-merge` it is embedding-only (no LLM call). This is where most deduplication happens. Candidate groups are node-disjoint by construction (each comes from its own connected component), so whenever a turn produces more than one candidate group, their verify calls run **concurrently** on a thread pool (default up to 8 at once — see `run_merge_pass(..., max_verify_workers=...)` in `merge.py`) instead of one at a time; only the cheap, CPU-only parsing of each group's result stays sequential, so grouping and the `logs/merge_*.json` audit trail are unaffected and stay deterministic.
- an optional **final** merge at the end of the trajectory, controlled by `--merge-strategy` (default **`off`**). When enabled it runs a whole-graph pass (`embedding` = candidates + LLM verify, or `llm` = the model proposes groups directly). The `embedding` strategy's LLM-verify step is parallelized the same way as the incremental merge above. Before it runs, finalize keeps only the latest generated decision as a `decision` and demotes earlier decisions to ordinary beliefs so they can participate in the final belief merge.

Merging is **gated**: two nodes may be merged only when their source **role** is identical **and** their **node_type** is identical. A belief is never absorbed into a decision, and a user claim is never absorbed into an assistant conclusion.

**Timing** is measured for each of the four latency-relevant sub-steps of every built turn — `node_generation` (phase 1), `merging` (incremental-merge embedding), `llm_check` (incremental-merge LLM verify; `0` when `--no-verify-merge`), and `edge_generation` (phase 3, summed over any backward-walk calls) — plus the whole-turn wall time and the optional final merge. See [Output artifacts](#output-artifacts).

---

## Installation

The integrated project requires **Python 3.11+**.

```bash
# from the repository root
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -e .
```

The project metadata installs the core dependencies, including `openai` for
OpenAI-compatible chat/embedding calls and `numpy` for similarity and
clustering.

Embeddings can be served two ways. The default configuration calls an
**OpenAI-compatible `/v1/embeddings` HTTP endpoint** (e.g. vLLM, SGLang, TEI,
or OpenAI itself), which needs no extra Python packages. To load embedding
weights in-process (`"provider": "local"`), install `sentence-transformers`
and the `torch` build matching the target hardware separately.

---

## Configuration (`.env` + `model_config.json`)

All credentials live in the project-root `.env`. Model names, endpoints,
pricing, and the environment-variable name for each key remain in JSON:

```bash
cp .env.example .env
cp bcg/model_config.example.json bcg/model_config.json
```

The file is **nested by model name**. Each top-level key is a *model-key* you can select on the command line; the key's name is used as the model name unless the entry sets its own `"model"`. Any key beginning with `embedding` is **reserved** — it is never chosen as the default chat model, which lets several embedding entries coexist with chat entries.

A chat entry must provide `base_url` and `api_key_env`; `max_tokens` and
`pricing` are optional. `api_key_env` is a variable name, not a secret. A
minimal example:

```json
{
  "gpt-5.5": {
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "https://your-endpoint/v1",
    "max_tokens": 100000,
    "pricing": { "input_per_1k": 0.005, "output_per_1k": 0.03 }
  },
  "embedding": {
    "api_key_env": "EMBEDDING_API_KEY",
    "base_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen3-Embedding-8B",
    "batch_size": 8
  }
}
```

The engine looks for an embedding entry named `embedding` by default (override with `--embedding-key`). Two providers are supported, selected by the entry's `"provider"` field:

- **`openai`** (the default when `provider` is omitted) — any OpenAI-compatible `/v1/embeddings` endpoint. Requires `base_url`, `api_key_env`, `model`.
- **`local`** — load the weights in-process with sentence-transformers; no server needed. Requires only `model` (a Hugging Face repo id or a local directory of weights), plus the optional `sentence-transformers` / `torch` dependencies.

If no embedding entry is found, clustering is disabled and the merge strategy silently falls back from `embedding` to `llm`.

Real key values belong only in the ignored root `.env`. Do not put them in
`model_config.json` or command files.

---

## Quick start (HTTP server)

This is the recommended way to use the engine: start the server once, then push turns to it as your agent generates them.

**1. Start the server.**

```bash
bcg construct server \
    --config model_config.json \
    --model-key gpt-5.5 \
    --host 127.0.0.1 --port 8848 \
    --output-dir outputs_stream
```

It prints the address it is listening on and the available endpoints.

**2. Push turns.** Each turn is one JSON object carrying a `problem_id` (which trajectory it belongs to), a `role`, and the message `content`. Mark the last turn of a trajectory with `"is_trajectory_end": true` to trigger finalization (writes `result.json`, runs the optional final merge, and writes this trajectory's `logs/timing.csv`).

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

Each `problem_id` is backed by its own session (`StreamingTrajectorySession`), which owns its own lock, its own token-usage tracker, and its own audit-log paths — bound through context-local state in `llm.py` (Python `contextvars`) rather than the process-global variables earlier versions used. As a result:

- Turns for the **same** `problem_id` are always processed strictly in the order they arrive, even if two requests for it reach the server at almost the same instant (the session's own lock serializes them).
- Turns for **different** `problem_id`s run **fully concurrently**, each on its own request thread under `ThreadingHTTPServer`, with no shared mutable state between them.
- `POST /turns` and `POST /input` additionally fan the distinct `problem_id`s / items found in one batch out across a small thread pool (up to 8 at once), so a single batch request doesn't itself serialize otherwise-unrelated trajectories.

One consequence: the server no longer needs a single global lock, and one server process is enough to get real parallelism across `problem_id`s. Running several server processes (each owning a disjoint set of `problem_id`s) is still a reasonable way to scale across CPU cores or machines, but it is no longer required just to get concurrent `problem_id`s handled by one process.

> **Binding note:** `--host` defaults to `0.0.0.0`, which exposes the server on **all** network interfaces. Use `--host 127.0.0.1` to keep it local to your machine.

---

## Single-machine mode (`run.py`)

When you already have a whole conversation in a file and just want to process it end-to-end, use `scripts/run.py`. It normalises the input into items, runs each item through the same streaming engine, and writes one output sub-directory per item.

```bash
# simplest: a trajectory or conversation file, default options
bcg construct run --input data.json

# pick the chat model and embedding entry from the config
bcg construct run --input data.json \
    --model-key gpt-5.5 --embedding-key embedding

# whole-sentence evidence + topic clustering + embedding-verified final merge
bcg construct run --input data.json \
    --evidence-mode sentence --use-clustering \
    --merge-strategy embedding --merge-threshold 0.86

# free-span evidence (model quotes excerpts; no sentence splitting)
bcg construct run --input data.json --evidence-mode excerpt

# process only one item out of a multi-item file (by id or 0-based index)
bcg construct run --input data.json --item 3
```

`run.py`'s defaults for the options it shares with the server (`--model-key`, `--context-chars`, `--incremental-merge-threshold`) are aligned with `online_server.py`, so a file run and an HTTP run behave the same way unless you override them.

> **Maintenance note:** the HTTP server is the primary, actively-used path. `run.py` and the public `BCGRunner` share the same `bcg.construct` engine, so all entry points stay in lock-step.

---

## Replaying a stream offline (`online_driver.py`)

`bcg/online_driver.py` is a thin bridge for feeding a **recorded** stream (JSONL, one turn-dict per line) through the online `SessionManager` — useful for replaying a captured run or piping a generator's output straight in.

```bash
# replay a recorded stream file
bcg construct replay -i stream.jsonl \
    --config model_config.json --model-key gpt-5.5 --output-dir outputs_stream

# pipe a live generator straight in
my_agent --stream | bcg construct replay --config model_config.json
```

Any trajectory that never sent `is_trajectory_end` is finalized automatically when the input ends.

---

## Input data formats

The loader (`bcg/construct/loaders.py`) accepts several shapes and normalises all of them into a flat list of role-tagged turns.

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
| `events.jsonl` | per-turn engine events (new node ids, relations added, incremental merges, per-turn sub-step timing) |
| `token_usage.json` / `.txt` | token accounting, by stage, with an estimated cost if `pricing` was set |
| `logs/prompts.jsonl` | every prompt sent to the LLM, for auditing |
| `logs/embedding_calls.jsonl` | every embedding call, with full input texts and cache hits |
| `logs/merge_final.json` / `.log` | the final merge pass: candidates, similarities, LLM verifications, applied merges, edge rewiring — both machine- and human-readable |
| `logs/timing.csv` | this trajectory's per-turn + summary timing (wide table, seconds) — see [Timing](#timing) below |

Timing artifacts (including a per-trajectory `logs/timing.csv`) are described in [Timing](#timing) below.

### Timing

Every built turn is timed for the four latency-relevant sub-steps, all in **seconds**:

| Sub-step | What it measures |
|---|---|
| `node_generation` | phase 1 — the node-extraction LLM call |
| `merging` | incremental-merge embedding (candidate generation) |
| `llm_check` | incremental-merge LLM verify+rewrite (`0` under `--no-verify-merge`) |
| `edge_generation` | phase 3 — relation LLM call(s), summed over any backward-walk attempts |

When a turn's incremental merge has more than one candidate group, their verify calls run concurrently (see [How it works](#how-it-works)), so `llm_check` reports the **wall-clock time of that parallel batch**, not the sum of each group's call time — the same change applies to the final merge's `llm_check` contribution when `--merge-strategy embedding` is used.

Skipped turns (system / empty / too-short) are not timed. Timing surfaces in three places:

- **`result.json` → `timing`** — keeps `start` / `end` / `duration_seconds`, and adds `per_turn` (one record per built turn with the four sub-steps + `turn_total`), `by_step` (per-trajectory totals and turn counts per sub-step), and `final_merge` (the trajectory-end merge's `merging` / `llm_check` / `total`; all `0` when `--merge-strategy off`).
- **`events.jsonl`** — each per-turn event carries a `timing` block (the four sub-steps + `turn_total`).
- **`logs/timing.csv`** — one self-contained **wide** table per trajectory, in that trajectory's own `logs/` folder (rewritten from scratch each finalize), all seconds, one row per `row_type`:
  - `turn` — one row per built turn: `node_generation`, `merging`, `llm_check`, `edge_generation`, `turn_total`;
  - `final_merge` — the trajectory-end merge pass (`0` when off);
  - `item` — one summary row per trajectory: the four sub-step totals, the summed `turn_total`, plus `n_nodes` / `n_beliefs` / `n_decisions` / `n_relations` / `n_merges`, the full-trajectory `duration_seconds`, and `result_path`.

  Columns: `row_type, item_id, turn_index, role, node_generation, merging, llm_check, edge_generation, turn_total, n_nodes, n_beliefs, n_decisions, n_relations, n_merges, duration_seconds, result_path`. The count / duration / path columns are populated only on `item` rows. A viz script can split cleanly on `row_type` (e.g. pandas `df[df.row_type == "turn"]`); to aggregate across trajectories, read each item's `logs/timing.csv` and concatenate.

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
| `--merge-strategy` | `off` | final-merge strategy: `off` (no trajectory-end merge), `embedding` (candidates + LLM verify), or `llm` (LLM proposes directly) |
| `--merge-threshold` | `0.86` | cosine threshold for final-merge candidate pairs |
| `--incremental-merge` / `--no-incremental-merge` | on | per-turn embedding merge right after each turn's new nodes |
| `--incremental-merge-threshold` | `0.86` | cosine threshold for the per-turn incremental merge |
| `--verify-merge` / `--no-verify-merge` | on | add an LLM verify+rewrite step to the per-turn incremental merge (only affects the incremental merge; needs an embedder). Off → embedding-only |
| `--context-chars` | `100000` | char budget of the existing-nodes context block shown to the model |

> **CLI defaults vs. library defaults.** The table above lists the **command-line** defaults (the same across `run.py` and `online_server.py`). Constructing `StreamOptions()` directly in Python uses several different built-in defaults — `merge_strategy="embedding"` (CLI: `off`), `verify_merge=False` (CLI: on), `incremental_merge_threshold=0.8` (CLI: `0.86`), and `context_chars=9000` (CLI: `100000`) — because the CLI layer sets its own. If you drive the engine via the Python API and want CLI-identical behaviour, pass these explicitly. Note also that `run.py` currently exposes only `--verify-merge` (on); the `--no-verify-merge` off-switch is available on `online_server.py`.

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
├── bcg/
    ├── construct/           # the engine (importable as bcg.construct)
    │   ├── __init__.py           # exports BeliefGraph, StreamingBeliefBuilder, StreamOptions
    │   ├── stream.py             # StreamingBeliefBuilder — per-turn engine + finalize
    │   ├── online.py             # SessionManager / StreamingTrajectorySession (streaming driver; per-session lock, concurrent problem_ids)
    │   ├── pipeline.py           # run_item / run_input — batch drivers over the engine
    │   ├── extract.py            # per-turn extraction: nodes (phase 1) + relations (phase 3)
    │   ├── prompts.py            # all LLM prompts
    │   ├── merge.py              # incremental (per-turn) + optional final merge, role & node_type gated, sub-step timed, parallel LLM-verify
    │   ├── confidence.py         # two-stage confidence assignment
    │   ├── evidence.py           # evidence offsets / excerpt matching
    │   ├── split.py              # sentence splitting + global clustering
    │   ├── graph.py              # the in-memory belief graph
    │   ├── loaders.py            # input normalisers (trajectory / multi-session)
    │   ├── llm.py                # OpenAI-compatible chat + embedding clients, token ledger (context-local state, concurrency-safe)
    │   ├── constants.py          # shared constants
    │   └── link.py               # legacy no-op shim (kept for import compatibility)
    │
    ├── online_server.py      # the HTTP server (recommended entry point)
    ├── run.py                # single-machine batch driver
    │── online_driver.py      # replay a recorded JSONL stream
    │
    ├── model_config.example.json # template — copy to model_config.json
    └── README.md
```

---

## Python API

You can also drive the engine directly, without the CLI or HTTP layer.

**Multiplex many trajectories** with a `SessionManager` (this is what the server uses):

```python
from bcg.construct.online import SessionManager

mgr = SessionManager(config_path="model_config.json", model_key="gpt-5.5")
for turn in incoming_stream:                 # dicts with problem_id / role / content
    snapshot = mgr.push(turn)                # returns the live graph
# a trajectory finalizes on is_trajectory_end, or call mgr.finalize(problem_id)
```

**Build one trajectory** directly with the engine:

```python
from bcg.construct.stream import StreamingBeliefBuilder, StreamOptions
from bcg.construct.llm import load_config, make_client

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
