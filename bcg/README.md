# construct_beliefs (v3.0)

Streaming belief-graph construction from role-tagged turns.

A trajectory (or a multi-session conversation) is fed in turn by turn. For each
turn, ONE LLM call extracts the new belief **nodes** and the new forward
(`informs`) **edges**, given the existing graph as read-only context. When the
trajectory ends, a single backward (evaluation) pass, a confidence update, and a
merge/dedup pass run over the whole graph. The result is a confidence-weighted,
deduplicated belief graph plus a complete audit trail.

## What changed from v2

- **No scenarios.** There is no `research` / `conversation` switch and no
  `--scenario` flag. Every input is normalised to items, each a flat list of
  role-tagged turns.
- **Role-only routing.** Turns are routed by role: `user`, `assistant`, `tool`
  (`function` is treated as `tool`; `system` is recorded but yields no beliefs).
  There is no tag-based segmentation — the whole turn content goes to the model,
  which decides how many beliefs to extract. The `layer` field is gone;
  `source.type` is the role.
- **No sessions.** The session concept is removed; a trajectory is one flat
  stream of turns.
- **One call per turn.** Extraction and forward linking happen together in a
  single LLM call (previously two). New nodes are referenced by temporary ids
  (`n0`, `n1`, …); forward-edge endpoints are an existing integer id or a temp id.
- **Backward + merge at the end.** The backward pass + confidence update + merge
  run once at trajectory end, over the full graph.

## 1. Pipeline

```
per turn (ingest_turn):
  (sentence mode) split content into complete sentences  ─┐
  (optional) group sentences by topic cluster            ─┤ still ONE call
  ONE LLM call: existing graph (nodes + edges) as context ┘
      → NEW nodes (temp ids) + NEW forward edges
  allocate ids, resolve temp ids, attach evidence, add nodes + validated edges

at trajectory end (finalize):
  1. ONE backward call over the full graph → confirms / contradicts / extends
  2. confidence update from those relations
  3. merge / dedup pass over the full graph
  4. write final_graph.json + result.json
```

## 2. Input formats

Any of these is accepted and auto-normalised (no scenario flag):

- **Trajectory** — `{"trajectory": [{role, content}, ...]}`, a bare list of
  messages, or `{"messages": [...]}`. Becomes ONE item.
- **Multi-session QA data** — a list of items, each with `sessions` (a list of
  sessions, each a list of `{role, content, has_answer}`), plus parallel
  `session_ids` / `dates` arrays and question metadata. Each item's sessions are
  **flattened** into one turn stream; by default they are ordered
  chronologically by date (pass `--keep-order` to keep array order). Each turn
  inherits its session's date so time attribution still works.

## 3. Two evidence hyperparameters

Evidence behaviour is controlled by two independent knobs.

### `--evidence-mode {sentence, excerpt}` (default `sentence`)

- `sentence` — the turn is split into complete sentences with exact offsets; the
  model returns `supporting_sentence_indices`, so **evidence is always a whole
  sentence**. (If the model omits indices, evidence falls back to all sentences.)
- `excerpt` — the whole content goes to the model, which returns verbatim
  excerpts; each is located by a three-stage matcher (exact → whitespace-
  normalized → fuzzy). Evidence may be a sentence fragment.

### `--use-clustering` (default off; sentence mode only)

Groups a turn's sentences by topic cluster (global agglomerative clustering over
sentence embeddings) and shows them **grouped inside the same single call** — it
remains one call per turn. Requires the `embedding` config entry. Turns with
fewer than `--cluster-min-sentences` sentences skip clustering.

```
--cluster-threshold      cosine floor for merging clusters (default 0.6)
--cluster-min-sentences  below this, skip clustering (default 4)
--cluster-buffer         neighbour window used only for embedding (default 0)
```

## 4. Merging / deduplication (`--merge-strategy`)

Runs once at trajectory end over the full graph.

- `embedding` (default) — embed every active belief, flag pairs with cosine
  similarity ≥ `--merge-threshold` (default 0.86), group via union-find, then the
  LLM verifies each candidate group. Only confirmed duplicates merge.
- `llm` — the LLM scans the whole belief list and proposes merge groups.
- `off` — no merging.

Merge semantics: canonical = smallest id; confidence adopted from the newest
member (recorded in `confidence_history`); evidence unioned; absorbed nodes
archived in `merges`; edges rewired to the canonical id.

## 5. Evidence & time attribution

Every belief carries an `evidence` list; each entry has exact `start`/`end`
offsets into the original turn content (so `trajectory[ti].content[start:end] ==
text`), a `match` kind, a `via` tag (`split_sentence` | `llm_excerpt`), and a
`source` descriptor (`type`/role, `item_id`, `turn_index`, `trajectory_index`,
optional `date`). When a turn states when a fact/event happened, the belief gets
`time_text` (verbatim phrase) and, when resolvable against the turn date,
`event_time` (ISO).

## 6. Configuration

`model_config.json` is nested by model name; the reserved key `embedding` holds
the embedding endpoint and is never picked as the default chat model. See
`model_config.example.json`.

```json
{
  "gpt-5.5": { "api_key": "...", "base_url": "...", "max_tokens": 16000 },
  "embedding": { "api_key": "...", "base_url": "...", "model": "Qwen/Qwen3-Embedding-8B" }
}
```

Embedding providers: `openai` (any OpenAI-compatible `/v1/embeddings` endpoint)
or `local` (in-process via sentence-transformers; set `"provider": "local"` and
a HF repo id or local weights dir).

## 7. CLI

```bash
# defaults: sentence evidence, embedding-verified merge
python scripts/run.py --input data.json

# whole-sentence evidence + topic clustering
python scripts/run.py --input data.json --evidence-mode sentence --use-clustering

# free-span evidence, llm merge
python scripts/run.py --input data.json --evidence-mode excerpt --merge-strategy llm

# multi-session input, keep array order, one item
python scripts/run.py --input qa.json --keep-order --item my_question_id

# pick chat + embedding entries
python scripts/run.py --input data.json --model-key deepseek-v4-flash-260425 --embedding-key embedding
```

## 8. Outputs (per item, under `<out_dir>/<item_id>/`)

```
result.json            final graph + trajectory + relations + merges + token usage
final_graph.json       snapshot of the active graph
events.jsonl           per-turn + finalize event log (replayable)
token_usage.json/.txt  per-call token accounting
logs/
  prompts.jsonl          every LLM input (audit)
  merge_final.json/.log  merge pass detail (candidates, verifications, rewiring)
  embedding_calls.jsonl  every embedding call (when an embedder is used)
```

`result.json` highlights: `trajectory`, `all_beliefs`, `forward_relations`,
`backward_relations`, `merges`, `source_counts` (by role), `stance_counts`,
`token_usage`, `options`, `meta`.

## 9. Confidence rules

Initial confidence is a lookup on `(role, stance)`: `user` and `tool` sources
start high (asserted ≈ 0.92–0.95), `assistant` lower (asserted ≈ 0.85);
hedged stances (`recalled` / `judged` / `speculated`) are penalised. Backward
relations then move confidence: `confirms` raises, `contradicts` lowers, with
every change recorded in `confidence_history`. See `confidence.py`.

## 10. Visualizing

```bash
python scripts/visualize_beliefs_graph.py path/to/result.json -o graph.html
```

Nodes are colored by role and positioned by `trajectory_index`; edges show
`informs` (forward) and `confirms` / `contradicts` / `extends` (backward).
Clicking a node highlights its evidence in the trajectory (directly from
offsets) and shows its confidence history.

## 11. Offline smoke tests

No network / API keys (a fake model + fake embedder are monkey-patched):

```bash
python scripts/smoke_test.py          # batch pipeline end-to-end
python scripts/online_smoke_test.py   # streaming interface end-to-end
```

## 12. Streaming / online use

See `STREAMING.md` for the turn-by-turn streaming contract, the JSONL driver,
the HTTP service, and the in-process `SessionManager` API.
