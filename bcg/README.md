# bcg — Belief Context Graph construction

`bcg` turns a role-tagged conversation or research trajectory (`user` /
`assistant` / `tool` / `function` turns) into a **belief graph**: typed
*belief* and *decision* nodes, connected by typed relations (`depends_on`,
`supplements`, `contradicts`), each node carrying a confidence score and
evidence pointing back into the original text.

Construction is **incremental** — turns are processed one at a time: extract
new nodes, deduplicate them against the graph, then link them to what came
before. There are no scenarios or sessions; any input is normalised into a
flat, role-tagged stream of turns.

Two independent backends implement this, side by side, under
`bcg/construct/`:

- **`hybrid`** — local models only: a small generative model (Qwen) extracts
  nodes, a local classifier assigns stance, local NER assigns entities, and a
  second small model draws relations.
- **`unified`** — one general-purpose graph LLM does node extraction and
  relation extraction (as two separate calls), returning stance and entities
  together with the nodes.

Pick **one backend per run**. They share the same *shape* of entry points
(`pipeline.run_input` / `pipeline.run_item`, `online.SessionManager`,
`StreamingBeliefBuilder` / `StreamOptions`) but are **not** interchangeable
at the class level — their `StreamOptions` fields and node/graph internals
differ. Always go through the unified `bcg/run.py` / `bcg/online_server.py`
entry points below, or import the specific backend you want directly.

You can drive either backend two ways:

- **Batch** (`bcg/run.py`) — run a whole input file through the pipeline in
  one shot.
- **Online, over HTTP** (`bcg/online_server.py`) — start a small server and
  push turns to it as your agent produces them.

---

## Table of contents

- [How the two backends build a graph](#how-the-two-backends-build-a-graph)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage: batch (`bcg/run.py`)](#usage-batch-bcgrunpy)
- [Usage: online server (`bcg/online_server.py`)](#usage-online-server-bcgonline_serverpy)
- [Output artifacts](#output-artifacts)
- [Project layout](#project-layout)

---

## How the two backends build a graph

| | `bcg.construct.hybrid` | `bcg.construct.unified` |
|---|---|---|
| Node extraction | Small generative model (Qwen), one concurrent call **per semantic chunk** | One general-purpose graph LLM, one call **per turn** (returns nodes + stance + entities together) |
| Chunking | Semantic breakpoint chunking (adjacent-window embedding distance) + `<think>`/`<tool_call>`/`<tool_response>` isolation | Whole-turn sentence splitting (no chunking) |
| Stance | Local DeBERTa zero-shot 4-class classifier, run on every extracted node's text | Returned directly by the node-extraction call |
| Entities | Local spaCy NER (or HF token-classification), run **after** that turn's merge is complete | Returned directly by the node-extraction call |
| Relations | Separate non-thinking Qwen model; backward window is either the immediately-previous turn only (`search_previous_turns: false`) or a full backward walk (`true`, the example config's default) | Same model, always a full backward walk: current turn vs. the immediately-previous turn's surviving nodes, walking further back one turn at a time until a cross-turn edge lands (or no turn remains) |
| Evidence granularity | Whole semantic chunk (exact offsets) | Whole sentence (exact offsets, default) or model-quoted excerpt (`--evidence-mode excerpt`, located by a 3-stage exact→normalised→fuzzy matcher) |
| Confidence policy | `unified`: hardcoded `(role, stance)` table; `hybrid`: config-driven prior — both plus evidence + relation factor confidence in `confidence.py` | Role/stance prior table plus the same evidence and relation factor confidence fields in `confidence.py` |
| Merge | Incremental, embedding-only (**no** LLM verify step) | Incremental embedding merge, **optionally** LLM-verified + rewritten (`--verify-merge`, default off) |
| Runtime tuning | Almost entirely via `model_config.json`'s `belief_graph` block — `bcg/run.py hybrid` exposes no backend-specific CLI flags | Via CLI flags on `bcg/run.py` / `bcg/online_server.py` (`--evidence-mode`, `--incremental-merge*`, `--verify-merge`, `--context-chars`, `--min-content-len`) |
| API key | Same mechanism as `unified` — see [Configuration](#configuration) | `api_key_env` resolved from a project-root `.env` via `bcg.core.env` |


### `unified`, in more detail

Each non-skipped turn (`extract.py` + `stream.py`) runs:

1. **Node extraction** (`extract.extract_nodes`, one LLM call) — returns new
   **belief nodes** and **decision nodes**, each already carrying its
   `stance` (`asserted` / `recalled` / `judged` / `speculated`) and
   `entities`. Decision nodes are only meaningful for the assistant's final
   answer (e.g. wrapped in `\boxed{...}`); no relations yet.
2. **Incremental merge** (`merge.run_merge_pass`, `strategy="embedding"`) —
   new belief nodes are deduplicated against the graph immediately, before
   any edges are drawn. Decision nodes are excluded from this merge
   entirely. Candidates are embedded and flagged above
   `--incremental-merge-threshold` (default `0.8`); if `--verify-merge` is
   enabled, the LLM is called once per candidate group to confirm the merge
   and rewrite the surviving node's text.
3. **Relation extraction** (`extract.extract_relations`, one or more LLM
   calls) — links the surviving new nodes to the immediately-previous
   turn's surviving nodes (plus new↔new). If that adjacent turn yields no
   cross-turn edge, the window walks backward one turn at a time until a
   current-to-prior edge lands or no earlier turn remains — one turn can
   therefore issue several relation calls. Every accepted edge must touch
   at least one *new* node, so old↔old edges are never re-emitted.

**Evidence** (`evidence.py`): in *sentence* mode (default) the turn is split
into whole sentences with exact offsets; in *excerpt* mode the model quotes
a span verbatim and it's located by exact → whitespace-normalised → fuzzy
matching.

**Confidence** (`confidence.py`) is a logistic posterior:
`confidence = sigmoid(logit(initial_confidence) + evidence_confidence + factor_confidence)`.
`initial_confidence` is the node prior, `evidence_confidence` accumulates
additional evidence folded in later by a merge, and `factor_confidence` is
recomputed from active relation edges. `depends_on` contributes positive support
from `to_id` to `from_id`; `contradicts` contributes negative support from
`from_id` to `to_id`; `supplements` is semantic-only and keeps `weight` and
`activated_condition` as `null`. Every change is recorded in
`confidence_history`.

**Merge gating**: nodes may only merge when their source `role` and
`node_type` are identical. The canonical survivor is always the smallest id
in the group; absorbed nodes are archived in `graph.merges` and every
relation endpoint pointing at them is rewired to the canonical id.

**Timing**, per built turn: `node_generation`, `merging` (embedding pass),
`llm_check` (LLM verify; `0` under `--no-verify-merge`), `edge_generation`
(summed over any backward-walk calls), plus `turn_total`.

### `hybrid`, in more detail

Each non-skipped turn (`stream.py`, four phases):

1. **Chunk + extract.** The turn is split into semantic chunks
   (`split.py`: sentence-splits, embeds a configurable neighbour window
   around each sentence, and breaks where adjacent-window cosine distance
   exceeds `chunking.breakpoint_percentile_threshold`; undersized adjacent
   groups are merged per `min_chunk_sentences`). `<think>` / `<tool_call>` /
   `<tool_response>` spans are isolated into their own chunks when
   `chunking.isolate_tool_calls` is on. Every chunk is then submitted
   **concurrently** (`extractor_config.max_concurrency`, default `16`) to a
   small Qwen model behind an OpenAI-compatible endpoint
   (`extractor.py: QwenChunkExtractor`), which returns zero, one, or several
   `belief`/`decision` node texts per chunk — nothing else. `decision` is
   only honoured on assistant turns. Each extracted node's text is then run
   through a **local DeBERTa zero-shot classifier** (`stance.py`) to assign
   one of the four stance labels — the generative model is never trusted
   for stance.
2. **Incremental merge** (belief-only, embedding-only, no LLM verify —
   `merge.py` always calls `run_merge_pass(strategy="embedding",
   verify=False)`). Decision nodes never participate in any merge.
3. **Entities**, only after that turn's merge has settled and node text is
   stable — local spaCy NER by default (`named_entities.py`; also supports
   spaCy `EntityRuler` patterns, a spaCy `Matcher` rules pass, or a
   Hugging Face token-classification pipeline; `regex`/`llm` are rejected).
4. **Edge generation** — a second, separate, non-thinking Qwen model
   (`edge_generation.py: QwenEdgeGenerator`) links the turn's surviving
   node identities to a prior turn's nodes. `edge_generation.
   search_previous_turns` controls the window: `true` (the example config's
   default) walks backward turn-by-turn exactly like `unified`, stopping
   at the first cross-turn edge; `false` restricts the attempt to only the
   immediately-previous turn (the "conservative two-turn window").

**Evidence** (`evidence.py`) is always the whole contiguous chunk that
produced a node (exact offsets) — there's no sentence/excerpt toggle; the
`belief_graph.runtime.evidence_mode` config value ("chunk") is recorded but
not branched on anywhere.

**Confidence** uses the same logit/sigmoid posterior form as `unified`.
For `hybrid`, source reliability per role, stance quality per label, aggregation
method, relation default weight, relation activation threshold, propagation
delta threshold, and maximum propagation iterations all come from
`belief_graph.confidence` in `model_config.json`.

**Timing**, per built turn: `node_generation`, `merging`,
`entity_extraction`, `edge_generation`, plus `turn_total` — no `llm_check`
step exists for this backend (there is no merge-verify LLM call).

**Runtime tuning is config-only.** `bcg/run.py hybrid` /
`bcg/online_server.py hybrid` expose no chunking/extractor/merge/entity/edge
flags — every one of those knobs lives in `model_config.json`'s
`belief_graph` block (see [Configuration](#configuration)) and is applied
via `StreamOptions.apply_belief_graph_config()`.

---

## Installation

- **Both backends need:** `openai` (chat + optional OpenAI-compatible
  embeddings), `sentence-transformers` (local embeddings, used for merge
  candidate scoring and, for `hybrid`, semantic chunking), `numpy`, and
  `rich` (used by `bcg/cli_help.py`'s `RichArgumentParser` for the
  Typer-style `--help` output on `bcg/run.py`, `bcg/online_server.py`, and
  each backend's `python -m bcg.construct.<backend>` entry).
- **`hybrid` additionally needs:** `torch`, `transformers`, `spacy` (plus
  `python -m spacy download en_core_web_sm` for the default NER model,
  and, if you set `entities.method: "huggingface"`, whatever token-
  classification checkpoint you point it at — `dslim/bert-base-NER` in the
  example config).

If you're serving `hybrid`'s small extractor / edge-generation models
yourself, serve them with vLLM (OpenAI-compatible):

```bash
vllm serve /path/to/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --port 8001 --max-model-len 8192 --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 --dtype auto
```

`--served-model-name` must match `belief_graph.extractor.model` /
`belief_graph.edge_generation.model` in `bcg/model_config.json`.

---

## Configuration

`bcg/model_config.json` is **shared by both backends** (copy
`model_config.example.json` to get started). Top-level keys are chat-model
entries (e.g. `"gpt-5.5"`), plus two reserved keys:

- **`"embedding"`** (any key starting with `embedding` is reserved and
  never chosen as the default chat model). Selected by its `"provider"`
  field:
  - **`"openai"`** (default if `provider` is omitted) — any
    OpenAI-compatible `/v1/embeddings` endpoint. Requires `base_url`,
    `api_key_env`, `model`.
  - **`"local"`** — load the weights **in-process** via
    `sentence-transformers`; no server needed. Requires only `model` (an HF
    repo id or local weights directory) — the example config uses this.
  Used by both backends for merge-candidate scoring, and by `hybrid` for
  semantic chunking. If this entry is absent, incremental merge (and, for
  `hybrid`, chunking) is skipped with a warning rather than an error.
- **`"belief_graph"`** — **`hybrid`-only.** `extractor` / `stance` /
  `edge_generation` / `entities` / `confidence` / `chunking` / `runtime` /
  `incremental_merge` settings, loaded by `load_belief_graph_config()` from
  (in merge order) the top-level `belief_graph` key, then the selected
  chat-model entry's own `belief_graph` override if present. `unified`
  never reads this section at all.

**API keys**: both backends resolve them identically, through the shared
`bcg/env.py`. Which `.env` file gets read is resolved in this priority
order (`find_project_env()`):

1. `$BCG_ENV_FILE`, if set (any path);
2. `.env` in the **current working directory**;
3. `.env` at the source checkout's project root (next to `bcg/`).

That file is parsed once at import time (and again defensively inside
`resolve_config_api_key`) and loaded into `os.environ` — values already
present in the real environment are **not** overwritten.

For a given config entry, `resolve_config_api_key()` then:

1. reads `api_key_env` off that entry (falling back to a caller-chosen
   default — `OPENAI_API_KEY` for chat entries, `EMBEDDING_API_KEY` for the
   embedding entry, `BELIEF_GRAPH_LOCAL_API_KEY` for `hybrid`'s `extractor`
   / `edge_generation` blocks — if `api_key_env` is missing);
2. looks up that variable in `os.environ`;
3. if it's unset or empty, **falls back to a literal `api_key` field already
   in that same config entry** (a legacy/manual escape hatch — templates
   and docs should still prefer `api_key_env` + `.env`, never a checked-in
   secret);
4. raises `ValueError` if both are empty.

```
OPENAI_API_KEY=sk-...
BELIEF_GRAPH_LOCAL_API_KEY=unused-or-your-vllm-key
```

`model_config.example.json`, trimmed to the parts most run configurations
will actually touch:

```json
{
  "gpt-5.5": {
    "api_key_env": "OPENAI_API_KEY",
    "base_url": "https://litellm.mybigai.ac.cn/v1",
    "max_tokens": 100000,
    "temperature": 1,
    "top_p": 0.95,
    "pricing": { "input_per_1k": 0.005, "output_per_1k": 0.03 }
  },
  "embedding": {
    "provider": "local",
    "model": "/path/to/your/all-MiniLM-L6-v2",
    "device": "auto",
    "batch_size": 8,
    "max_length": 8192
  },
  "belief_graph": {
    "extractor": {
      "enabled": true, "provider": "openai",
      "base_url": "http://localhost:8001/v1",
      "api_key_env": "BELIEF_GRAPH_LOCAL_API_KEY",
      "model": "Qwen3.5-4B", "temperature": 0, "max_tokens": 4096,
      "max_concurrency": 16, "context_scope": "graph",
      "enable_thinking": false, "dynamic_node_cap": true,
      "node_cap_unit": "char", "node_cap_ratio": 0.004
    },
    "stance": {
      "model_path": "/path/to/your/deberta-v3-large-zeroshot-v2.0",
      "labels": { "asserted": {...}, "recalled": {...}, "judged": {...}, "speculated": {...} }
    },
    "edge_generation": {
      "provider": "openai", "base_url": "http://localhost:8001/v1",
      "api_key_env": "BELIEF_GRAPH_LOCAL_API_KEY", "model": "Qwen3.5-4B",
      "enable_thinking": false, "search_previous_turns": true
    },
    "runtime": { "evidence_mode": "chunk", "context_chars": 100000, "min_content_len": 0 },
    "incremental_merge": { "enabled": true, "threshold": 0.76, "keep_newest_text": false },
    "entities": { "method": "ml", "spacy_model": "en_core_web_sm" },
    "confidence": { "initial_method": "weighted_average", "evidence_method": "product", "default_relation_weight": 0.5, "input_confidence_threshold": 0.8, "propagation_min_confidence_delta": 0.001, "max_propagation_iterations": 3, ... },
    "chunking": { "enabled": true, "breakpoint_percentile_threshold": 95.0, "buffer_size": 1, "isolate_tool_calls": true }
  }
}
```

Every field under `belief_graph`'s sub-sections is **required** — each
normaliser (`normalize_extractor_config`, `normalize_edge_config`, ...)
raises `ValueError` listing exactly which key is missing, so copy from
`model_config.example.json` rather than writing a section from scratch.
`unified` reads `belief_graph.confidence` for relation-confidence propagation so batch `run.py` and `online_server.py` use the same thresholds; the rest of the `belief_graph` block is hybrid-only.

---

## Usage: batch (`bcg/run.py`)

Both entry points take the **backend name** (`hybrid` or `unified`) as the
first positional argument. If you omit it — either no arguments at all, or
the first token starts with `-` — `bcg/construct/dispatch.py` silently
selects the default backend, **`unified`**, for compatibility with
command lines written before the two backends were combined. A positional
token that *isn't* a flag must be an exact backend name, or it errors with
`unknown backend '...'; choose one of: hybrid, unified` rather than being
swallowed as an argument.

```bash
python bcg/run.py hybrid     --input data.json --model-key gpt-5.5 --embedding-key embedding

python bcg/run.py unified --input data.json --model-key gpt-5.5 --embedding-key embedding

# unified: free-span evidence (model quotes excerpts verbatim, no sentence splitting)
python bcg/run.py unified --input data.json --evidence-mode excerpt

# unified: turn off the per-turn incremental merge entirely
python bcg/run.py unified --input data.json --no-incremental-merge

# process only one item out of a multi-item file (by id or 0-based index)
python bcg/run.py hybrid --input data.json --item 3
```

**Flags common to both backends** (`_add_common_args` in `run.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `-i` | *(required)* | Input JSON/TXT — a trajectory, or multi-session QA items |
| `--config`, `-c` | `bcg/model_config.json` | Model config path |
| `--output-dir`, `-o` | `outputs` | Output root; each item gets its own subdirectory |
| `--model-key` | `gpt-5.5` | Which chat-model entry of the config to use |
| `--embedding-key` | `embedding` | Which config entry holds the embedding endpoint |
| `--item` | all | Process only this item (id or 0-based index) |
| `--keep-order` | off | For multi-session inputs, keep input array order instead of date-sorting |

**`unified`-only flags** (`_run_unified` in `run.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--evidence-mode` | `sentence` | `sentence` = whole-sentence evidence with offsets; `excerpt` = model quotes spans verbatim |
| `--incremental-merge` / `--no-incremental-merge` | on | Per-turn embedding merge right after each turn's new nodes |
| `--incremental-merge-threshold` | `0.8` | Cosine threshold for that merge |
| `--verify-merge` / `--no-verify-merge` | off | Enable or disable LLM verification and rewriting for merge groups |
| `--context-chars` | `9000` | Char budget of the existing-nodes context block shown to the model |
| `--min-content-len` | `0` | Skip turns shorter than this many characters |

These CLI defaults and the SDK defaults share the same values. Layered YAML
settings can override them; an explicit CLI flag takes precedence.

**`hybrid`-only flags:** none. Everything beyond the common flags above
comes from `model_config.json`'s `belief_graph` block (see
[Configuration](#configuration)).

Outputs go to `<output-dir>/<item>/` — see [Output artifacts](#output-artifacts).

---

## Usage: online server (`bcg/online_server.py`)

Same pipelines, exposed over HTTP. Request routing/parsing/concurrency is
one shared handler; only the config wiring and backend-specific flags
differ.

```bash
python bcg/online_server.py hybrid     --config bcg/model_config.json --port 8848
python bcg/online_server.py unified --config bcg/model_config.json --port 8848
```

`--host` defaults to `127.0.0.1` (local access only); pass `--host 0.0.0.0`
to listen on all interfaces. `--output-dir` defaults to `outputs_stream` here (batch's
default is `outputs`).

`unified`'s server exposes the same merge/evidence flags as
`run.py`:
`--evidence-mode`, `--incremental-merge`/`--no-incremental-merge`,
`--incremental-merge-threshold`, `--verify-merge`/`--no-verify-merge`,
`--context-chars`, `--min-content-len`. It also supports a self-rolling
dated `--output-dir`: a value like `outputs_7_6` or `outputs_{Y}_{m}_{d}`
is re-resolved to *today's* date each time a new session starts (plain
values like `outputs_stream` are left as-is). `hybrid`'s server exposes no
extra flags, same as its batch driver.

### Endpoints (identical for both backends)

| Method & path | Body / query | Returns |
|---|---|---|
| `GET /health` | — | `{"status": "ok", "active": [...problem_ids...], "all": [...]}` |
| `POST /turn` | one turn dict: `{"problem_id", "role", "content", "is_trajectory_end"?}` | current snapshot for that `problem_id` |
| `POST /turns` | a JSON array of turn dicts, or NDJSON | `{"pushed": n, "finalized": [...], "latest": {problem_id: snapshot}}` |
| `POST /input` (alias `/run`) | any shape `loaders.py`/`run.py` accepts — `{"trajectory": [...]}`, `{"messages": [...]}`, a bare message list, or multi-session QA data — with query params `?item=`, `?keep_order=1`, `?finalize=0` | `{"items": n, "finalized": [...], "latest": {...}}` |
| `POST /finalize` | `{"problem_id": "p1"}` | the final snapshot (use if you never sent `is_trajectory_end`) |
| `GET /graph?problem_id=p1` | — | latest snapshot for that trajectory (404 if unknown) |

Mark the last turn of a trajectory with `"is_trajectory_end": true` to
trigger finalization. To stream one message in fragments, send them with
`"is_message_end": false`; they're buffered and concatenated until a
fragment arrives with `is_message_end` true (implied by
`is_trajectory_end`) — only then is the assembled turn ingested.

```bash
curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \
     -d '{"problem_id":"p1","role":"user","content":"Which alloy resists seawater corrosion best?"}'

curl -s -X POST localhost:8848/turn -H 'content-type: application/json' \
     -d '{"problem_id":"p1","role":"assistant","content":"Titanium grade 2 is the standard choice. \\boxed{Titanium grade 2}","is_trajectory_end":true}'
```

**Concurrency**: each `problem_id` is backed by its own session
(`StreamingTrajectorySession`), guarding its own belief graph, token-usage
tracker, and audit-log paths behind its own lock. Turns for the **same**
`problem_id` are always processed strictly in arrival order; turns for
**different** `problem_id`s run fully concurrently on
`ThreadingHTTPServer`'s per-request threads. `POST /turns` and `POST
/input` additionally fan distinct `problem_id`s / items in one batch out
across a thread pool (up to 8 at once), so one batch request doesn't
serialize otherwise-unrelated trajectories.

---

## Output artifacts

Each item/trajectory gets its own sub-directory
(`<output-dir>/<item_id>/` for `run.py`, `<output-dir>/<problem_id>/` for
the server):

| File | What it is |
|---|---|
| `result.json` | Full result: trajectory, all nodes, relations, merges, counts, options, timing, token usage |
| `final_graph.json` | Final belief-graph snapshot |
| `belief_graph_latest.json` / `belief_graph.jsonl` | Latest / per-turn snapshots (streaming path only) |
| `trajectory.json` / `trajectory_stream.jsonl` | Reconstructed conversation / raw received-turn log (streaming path) |
| `events.jsonl` | Per-turn engine events (new node ids, relations added, merges, sub-step timing) |
| `token_usage.json` / `.txt` | Token accounting by stage, with estimated cost if `pricing` was set |
| `logs/prompts.jsonl` | Every LLM prompt sent, for auditing |
| `logs/embedding_calls.jsonl` | Every embedding call (inputs + cache hits) |
| `logs/merge_*.json` / `.log` | Merge audit trail: candidates, similarities, LLM verifications (if any), applied merges, edge rewiring |
| `logs/timing.csv` | Per-turn + summary timing, wide table, seconds |


`hybrid`'s `result.json` additionally carries a `turn_chunks` array (each
turn's chunk boundaries and text) that `unified`'s does not.

---

## Project layout

Confirmed from the actual export:

```
bcg/
  __init__.py
  env.py                    # find_project_env / load_project_env / resolve_config_api_key
  utils.py                  # get_random_uuid / utc_now
  cli_help.py               # RichArgumentParser (Typer-style --help via `rich`)
  run.py                    # unified batch entry point (hybrid | unified subcommand)
  online_server.py          # unified HTTP entry point  (hybrid | unified subcommand)
  online_driver.py          # replay driver behind `python -m bcg.construct.<backend> replay` (not read in this pass)
  model_config.json          # shared config (copy from model_config.example.json)
  construct/
    __init__.py
    dispatch.py             # DEFAULT_BACKEND="unified", BACKENDS, split_backend_args(argv)
    _backend_cli.py         # backend_main(name, argv) — the run/server/replay forwarder below each backend's cli.py
    hybrid/
      __init__.py  __main__.py  cli.py
      pipeline.py  stream.py  online.py
      extractor.py  edge_generation.py  stance.py  named_entities.py
      merge.py  evidence.py  confidence.py  graph.py  constants.py
      split.py  loaders.py  llm.py  prompts.py
    unified/
      __init__.py  __main__.py  cli.py
      pipeline.py  stream.py  online.py
      extract.py  merge.py  evidence.py  confidence.py  graph.py  constants.py
      split.py  loaders.py  llm.py  prompts.py  utils.py
```
