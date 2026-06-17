# Streaming / Online Interface (v3)

Feed a trajectory **turn by turn** and get the live belief graph back after each
turn. The backward (evaluation) pass and merge/dedup run once, automatically, on
the turn that carries `is_trajectory_end: true` (or via an explicit
`finalize()`). No scenarios, no sessions — turns are routed by role only.

## 1. At a glance

```python
from construct_beliefs.online import SessionManager

mgr = SessionManager(config_path="model_config.json")     # one shared client + embedder
for turn in my_stream:                                     # dicts (see contract below)
    snapshot = mgr.push(turn)                              # live graph after each turn
# the turn with is_trajectory_end=True triggers finalize automatically
```

## 2. The streaming contract

Each pushed object is a dict:

```json
{
  "problem_id": "research_demo_0001",   // REQUIRED: routes the turn to its trajectory
  "role": "user",                        // user | assistant | tool (function == tool; system recorded only)
  "content": "…",                        // the turn text (tags inside are fine — not split)
  "date": "2023/04/10",                  // OPTIONAL: used for time attribution
  "is_message_end": true,                 // OPTIONAL: default true (one dict == one turn)
  "is_trajectory_end": false              // OPTIONAL: true on the LAST turn → finalize
}
```

Rules:
- `problem_id` is mandatory and groups turns; turns for different `problem_id`s
  may interleave freely — each trajectory keeps its own graph and token ledger.
- The whole `content` is handed to the model; `<think>` / `<tool_call>` /
  `<tool_response>` tags are NOT used to split it.
- `is_trajectory_end: true` finalizes the trajectory and returns the complete
  graph (with backward relations + merges). Pushing more turns afterwards raises
  `TrajectoryClosedError`.

### Tags inside `content`

Tags are passed through verbatim and may appear in evidence. The model is
instructed to read through them and extract substantive beliefs (claims,
tool-query intents, conclusions) without treating tags as separate documents.

## 3. Optional: token-level fragment streaming

If a producer emits a turn in pieces, send `is_message_end: false` on every
fragment except the last. Fragments with the same `problem_id` and `role` are
buffered and concatenated into ONE turn when `is_message_end: true` arrives. A
trajectory cannot end mid-message.

## 4. What gets written (per trajectory, under `<output_root>/<problem_id>/`)

```
trajectory_stream.jsonl   every received dict, in arrival order (raw log)
trajectory.json           assembled turns; run.py-compatible; complete at the end
belief_graph.jsonl        one snapshot per ingested turn (stage "turn") + one "final"
belief_graph_latest.json  the most recent snapshot
result.json               final result (same schema as the batch pipeline)
final_graph.json          final active-graph snapshot
events.jsonl              per-turn + finalize events
token_usage.json/.txt     per-trajectory token accounting (fully isolated)
logs/                      prompts.jsonl, merge_final.*, embedding_calls.jsonl
```

Per-turn snapshots are forward-only (`backward_relations` / `merges` empty); the
final snapshot carries the backward relations and merges.

## 5. Running the JSONL driver

```bash
# replay a recorded stream file:
python scripts/online_driver.py --stream examples/research_stream_example.jsonl

# pipe a live JSONL stream from your generator:
my_generator | python scripts/online_driver.py --stream -

# whole-sentence evidence + clustering + embedding merge (needs the embedding entry):
python scripts/online_driver.py --stream stream.jsonl \
    --evidence-mode sentence --use-clustering --merge-strategy embedding
```

`drive()` finalizes each trajectory on its `is_trajectory_end` turn and
finalizes any still-open trajectories at end of stream. Blank, non-JSON, and
non-object lines are skipped.

## 6. Running as an HTTP service

```bash
python scripts/online_server.py --config model_config.json --port 8077
```

`POST /push` a turn dict, get the live snapshot back; the `is_trajectory_end`
turn returns the finalized graph. See the script header for the route list and
the new flags (`--evidence-mode`, `--use-clustering`, `--cluster-*`,
`--merge-strategy`).

## 7. Calling it in-process

```python
from construct_beliefs.online import SessionManager
from construct_beliefs.stream import StreamOptions

mgr = SessionManager(
    config_path="model_config.json",
    options=StreamOptions(evidence_mode="sentence", use_clustering=True,
                          merge_strategy="embedding", merge_threshold=0.86),
)
for turn in turns:
    mgr.push(turn)
# push finalizes automatically on is_trajectory_end=True; otherwise:
# final_graph = mgr.finalize("research_demo_0001")
```

## 8. Design notes

- **One call per turn.** Each turn does extraction + forward linking in a single
  LLM call. Clustering (when on) only regroups the sentences shown in that one
  call — it does not add calls.
- **Deferred evaluation.** Backward linking, confidence updates, and merging are
  global operations, so they run once at the end over the whole graph rather
  than per turn.
- **Isolation.** Process-global state (token tracker, prompt-log path, embedder
  log path) is swapped per trajectory inside a context manager, so interleaved
  trajectories never cross-contaminate their accounting or logs.
