# construct_beliefs

Layered belief-graph construction from a multi-turn LLM trajectory.

The goal is *not* to dump every fact mentioned. The goal is to recover the
**belief evolution path** that connects user input → tool calls → final
answer, with the model's `<think>` reasoning and `<tool_response>` retrievals
serving as **key reasoning nodes** on that path.

Each belief is tagged with `source`, `stance`, `confidence`, and supporting
excerpts. Confidence is *dynamic*: when a later belief confirms an earlier
speculation, the earlier one's confidence is lifted; when a later belief
contradicts it, the earlier one is demoted.

## Project layout

```
construct_beliefs/
├── README.md
├── model_config.json
├── src/                  ← the package
│   ├── __init__.py
│   ├── segment.py        # split trajectory into typed segments
│   ├── prompts.py        # 5 extraction prompts + 1 linking prompt
│   ├── llm.py            # OpenAI client wrapper
│   ├── extract.py        # per-segment extraction
│   ├── link.py           # belief-relation linking
│   ├── confidence.py     # rule-based base + relation-driven dynamic update
│   └── pipeline.py       # stage orchestration
└── scripts/
    └── run.py            # CLI: --stage all | segment | io | reasoning | link | finalize
```

## Two-layer belief model

### Layer 1 — I/O beliefs (comprehensive)
Extracted from segments that mark the **inputs and outputs** of the agent:
- `user_input`  — user's stated facts, preferences, question (reformulated).
- `tool_call`   — what the assistant is querying for + any implicit hypothesis.
- `assistant_other` — final answer text (the assistant's direct output to the user).

Extraction is **comprehensive** — every meaningful claim is captured.

### Layer 2 — Reasoning beliefs (selective)
Extracted from segments that explain **why the agent went from input to output**:
- `think`         — model's internal deliberation.
- `tool_response` — facts retrieved by tool calls.

Extraction is **selective** — only **key reasoning nodes** that
(a) bridge between I/O beliefs, (b) are falsifiable hypotheses that later
steps can confirm / refute, or (c) are stand-alone facts the model commits to.

The reasoning-layer prompts receive the I/O beliefs as context so the model
can prioritise extractions that matter for the I/O path.

## Confidence

### Base rules (in `confidence.py`)
A `(source_type, stance)` table assigns the starting confidence. High-trust
sources (user input, tool responses) start near 0.9 for asserted claims.
Hedged stances are penalised proportionally.

### Dynamic update
Only the **backward** link layer (next section) moves confidence. Each
relation discovered there shifts the **earlier belief's** confidence:

- `confirms`     → boost the earlier belief toward the source's level,
                   capped at +0.20 per relation.
- `contradicts`  → drop the earlier belief proportionally to source confidence,
                   capped at −0.25 per relation.
- `extends`      → small bump (+0.05) reflecting weak corroboration.

Multiple updates compound and are clamped to `[0.05, 0.97]`. Every move is
recorded in the belief's `confidence_history` so the trail is auditable.

Forward relations (informs) do NOT touch confidence — they're descriptive.

## Belief relations — two passes

The linking pipeline runs two independent passes over the full belief list:

### Forward — **derivation flow** (`informs`)
Edge `A → B` means *"A is a piece of context, premise, or input that B was
reasoned from"*. `from_id` is the EARLIER informant, `to_id` is the LATER
derived belief. Typical patterns:
- `user_input → think`         (the input motivates the reasoning)
- `think → tool_call`          (the reasoning concludes in a search query)
- `tool_response → next think` (the retrieval feeds the next reasoning step)
- `think / tool_response → assistant_other` (the final answer concludes the chain)
- `user_input → tool_call`     (direct restatement of the question, no intermediate reasoning)

Forward edges DO NOT modify confidence. They exist to make the belief
evolution path explicit and visualisable.

### Backward — **evaluation** (`confirms` / `contradicts` / `extends`)
Edge `A → B` means *"the LATER belief A confirms / contradicts / extends the
EARLIER belief B"*. `from_id` is the LATER evidence, `to_id` is the
EARLIER target. These drive `confidence.py`'s dynamic update.

### Chronological renumbering
Belief ids are reassigned in chronological order (sorted by
`(trajectory_index, segment_index)`) before the two link passes run. This is
necessary because the per-layer extraction order makes `<think>` beliefs
(segment 0 of an assistant message) end up with HIGHER ids than `<tool_call>`
beliefs (segment 1 of the same message). After renumbering, `from_id < to_id`
genuinely means "earlier" — which is what the forward-direction validator
relies on. The renumbering is deterministic, so single-stage re-runs stay
consistent.

## Stage outputs (layered)

```
outputs/
├── 01_segments.json
├── 02_io_beliefs.json
├── 03_reasoning_beliefs.json
├── 04_forward_relations.json    ← derivation edges (informs)
├── 05_backward_relations.json   ← evaluation edges (confirms / contradicts / extends)
└── result.json                  ← FINAL: all beliefs (chronologically renumbered)
                                   + forward_relations + backward_relations
                                   + confidence (initial + adjusted + history)
```

Note: as of this version the `relations` field in `result.json` has been
split into `forward_relations` (informs) and `backward_relations` (confirms /
contradicts / extends). `visualize_beliefs_graph.py` reads both. Older
visualizers that look for `relations` still work via a fallback.

## Setup

```bash
pip install openai
cp model_config.example.json model_config.json
# edit model_config.json with your base_url / api_key / model
```

### Config formats — both supported

**Flat** (one model per file):
```json
{ "base_url": "https://...", "api_key": "sk-...", "model": "gpt-4o-mini" }
```

**Nested by model name** (LiteLLM-style — model name is the top-level key):
```json
{
  "gpt-5.5": {
    "api_key": "sk-...",
    "base_url": "https://litellm.example.com/",
    "max_tokens": 16000,
    "temperature": 1,
    "top_p": 0.95
  },
  "claude-3": { "api_key": "...", "base_url": "..." }
}
```

For the nested form: `--model-key` picks which entry to use. Without it, the
first key is used and its name becomes the model name. The pipeline reads
`max_tokens` from the chosen entry (forwarded to the chat completion call).
Note: the pipeline always uses `temperature=0` regardless of what the config
says, because deterministic JSON output is what makes extraction reliable.

## Running

End to end:
```bash
python scripts/run.py --input path/to/trajectory.json
# nested config with multiple models:
python scripts/run.py --input path/to/trajectory.json --model-key gpt-5.5
```

Stage by stage (useful while iterating on prompts — each stage saves its own
output, so you can re-run any single stage without redoing the rest):
```bash
python scripts/run.py --stage segment   --input path/to/trajectory.json
python scripts/run.py --stage io
python scripts/run.py --stage reasoning
python scripts/run.py --stage forward
python scripts/run.py --stage backward
python scripts/run.py --stage finalize
```

## Visualizing

The final `outputs/result.json` is **schema-compatible** with
`visualize_beliefs_graph.py` from the parent project. Just run it on the file:
```bash
python ../visualize_beliefs_graph.py outputs/result.json
```
Beliefs will show their layer (I/O vs reasoning), source, stance, and final
confidence. The extra fields (`relations`, `confidence_history`,
`initial_confidence`) are preserved in the JSON for any downstream tool that
wants to render the belief graph or confidence trail.
