"""
prompts.py
==========
All LLM prompts used in the project, organised by stage.

Stage 1 — I/O belief extraction (comprehensive)
    PROMPT_IO_USER_INPUT
    PROMPT_IO_TOOL_CALL
    PROMPT_IO_ASSISTANT_OTHER     (final answer text)

Stage 2 — reasoning belief extraction (selective, conditioned on I/O context)
    PROMPT_REASONING_THINK
    PROMPT_REASONING_TOOL_RESPONSE

Stage 3 — belief relation linking
    PROMPT_LINK_BELIEFS

Each prompt has placeholders that callers fill via str.replace:
    <<<CONTENT>>>             — the raw segment text
    <<<IO_CONTEXT>>>          — JSON-formatted I/O beliefs (stage 2 only)
    <<<BELIEFS_LIST>>>        — JSON-formatted full belief list (linking only)
"""

CONTENT_PLACEHOLDER     = "<<<CONTENT>>>"
IO_CONTEXT_PLACEHOLDER  = "<<<IO_CONTEXT>>>"
BELIEFS_LIST_PLACEHOLDER = "<<<BELIEFS_LIST>>>"


# =====================================================================
# Shared definitions (kept in one place so all prompts stay consistent)
# =====================================================================

_BELIEF_DEFINITION = """\
## What is a belief
A belief is a self-contained atomic memory unit, ideally shaped like:
    <subject, predicate, object/value, scope, time, source>

Each belief must satisfy BOTH:
1. **Single central semantics** — ONE thing (a single fact, preference, event, or decision).
2. **Context self-contained** — taken out of context, a reader still knows who / what / when / scope.

## Granularity rules — split nested or compound content

### A. Conjunctive content (split when topics are independent)
INPUT:   "User is researching Hindsight and prefers concise explanations."
SPLIT INTO:
  - "User is researching Hindsight."
  - "User prefers concise explanations."

### B. List of independent sub-topics
INPUT:   "User is studying Hindsight's graph retrieval, temporal retrieval, confidence, and evidence mechanism."
SPLIT INTO four atomic beliefs, one per sub-topic.

### C. State change across time
INPUT:   "User previously thought Hindsight used confidence scores, but now understands that observations use evidence tracking."
SPLIT INTO two beliefs (previously / now), preserving the transition.

### D. Fact + inference in one sentence (different epistemic status)
INPUT:   "User asks many questions about Hindsight, so user is likely designing a belief memory system."
SPLIT INTO:
  - "User asked multiple questions about Hindsight's memory architecture."   [stance: asserted]
  - "User may be designing or evaluating a belief memory system."             [stance: speculated]
Source-backed fact and inferred observation MUST be separate beliefs.
"""

_STANCE_DEFINITION = """\
## Stance (choose ONE per belief)
- **asserted**   — stated as a plain fact ("X is Y", "released in 1980").
- **recalled**   — based on memory ("I recall X", "I remember X").
- **speculated** — hedged ("might", "maybe", "perhaps", "could", "possibly").
- **judged**     — evaluative conclusion ("most likely", "best answer is", "I think X").

NOTE: do NOT output a confidence number — confidence is assigned downstream by code rules based on source × stance.
"""

_HARD_CONSTRAINTS = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (including unusual punctuation like "!Kung" or "Nǃai").
2. Do NOT add information not present in the segment. No outside knowledge.
3. Every belief MUST have at least one supporting excerpt (close-to-verbatim from the segment). No excerpt → drop the belief.
4. Deduplicate within this segment.
5. Empty list is OK if the segment expresses none.
"""

_OUTPUT_FORMAT = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "belief": "<self-contained atomic belief>",
      "stance": "asserted | recalled | speculated | judged",
      "supporting_excerpts": ["<excerpt 1>", "<excerpt 2>"]
    }
  ]
}
"""


# =====================================================================
# Stage 1 prompts — I/O layer (BE COMPREHENSIVE)
# =====================================================================

PROMPT_IO_USER_INPUT = f"""\
# Task
Extract every BELIEF expressed by the user in the input segment below. This is the I/O LAYER, so extract COMPREHENSIVELY — don't drop something just because it seems trivial.

{_BELIEF_DEFINITION}

## Source for this segment: USER INPUT
This is the user's raw input. Things to extract:
- **The user's question reformulated as a fact about the user**, e.g. "User is asking about <topic>", "User wants to know <X>".
- **Facts the user states** about themselves, their task, their domain, or the world.
- **Constraints the user imposes** ("answer in Chinese", "no code", "use tool X").
- **Background information** the user provides as given.

Skip:
- Instructions to the assistant that are pure procedure ("Use the web_search tool", "respond in JSON").
- Generic boilerplate ("Please answer thoroughly").

{_STANCE_DEFINITION}
For user input the typical stance is "asserted" (the user is committing to a claim) — use other stances only when the user explicitly hedges.

{_HARD_CONSTRAINTS}

{_OUTPUT_FORMAT}

## Input segment
{CONTENT_PLACEHOLDER}
"""


PROMPT_IO_TOOL_CALL = f"""\
# Task
Extract BELIEFS from the tool-call segment below. This is the I/O LAYER.

{_BELIEF_DEFINITION}

## Source for this segment: TOOL CALL
This segment is a JSON tool invocation made by the assistant. The belief content lies in:
- **What is being queried** — restate the query as a belief about the assistant's information need.
  Example: arguments.query = "John Marshall !Kung 1980"
       →   "Assistant is querying for information about John Marshall, !Kung, and the year 1980."
- **Any implicit hypothesis** the query commits to. A search query like "Marshall AND Adrienne LaFrance collaboration" implicitly asserts "the answer likely involves Marshall + LaFrance collaboration".
  Output such hypotheses as separate beliefs with stance "judged" or "speculated".
- **Tool name & purpose** if non-obvious — e.g. for a `finish` tool, extract the submitted answer as an asserted belief.

Skip:
- The literal JSON syntax, key names, schema details.

{_STANCE_DEFINITION}
For a tool_call query, the default stance is "asserted" for what the assistant is asking, and "judged" / "speculated" for any implicit hypothesis embedded in the query.

{_HARD_CONSTRAINTS}
Note: for tool_call segments, the supporting_excerpt may be a slice of the JSON (e.g. the query value) — that is fine.

{_OUTPUT_FORMAT}

## Input segment
{CONTENT_PLACEHOLDER}
"""


PROMPT_IO_ASSISTANT_OTHER = f"""\
# Task
Extract every BELIEF from the assistant's direct (non-think, non-tool_call) output segment below. This is typically the FINAL ANSWER given back to the user. Extract COMPREHENSIVELY — every factual claim, every conclusion.

{_BELIEF_DEFINITION}

## Source for this segment: FINAL ANSWER / DIRECT ASSISTANT OUTPUT
Things to extract:
- **The headline / boxed answer** as an asserted belief.
- **Every supporting fact** stated in the answer (entity attributes, dates, relationships).
- **Caveats** stated by the assistant ("but note that..."), as their own beliefs.

Skip:
- Formatting wrappers like `\\boxed{{}}` syntax — extract the content, not the wrapper.
- Pure transitions ("In conclusion,", "Therefore,").

{_STANCE_DEFINITION}
Final answer claims usually carry stance "asserted" or "judged" — be faithful to how the assistant phrased it.

{_HARD_CONSTRAINTS}

{_OUTPUT_FORMAT}

## Input segment
{CONTENT_PLACEHOLDER}
"""


# =====================================================================
# Stage 2 prompts — reasoning layer (SELECTIVE, given I/O context)
# =====================================================================

_SELECTIVITY_RULES = """\
## Selectivity rules — this is the REASONING LAYER, NOT the I/O layer

The I/O beliefs (user question, tool queries, final answer) have already been
extracted and are provided as context below. From THIS segment, extract ONLY
the **key reasoning nodes** that one of the following is true for:

1. **Bridges I/O**: the claim links an input belief to (a step toward) an output belief.
2. **Falsifiable hypothesis**: the model commits to a factual claim that could be confirmed
   or refuted by later steps. (e.g. "I recall that X collaborated with Y" — this is a
   memory the next tool call might verify.)
3. **Reusable fact**: the claim is a stand-alone fact that the model commits to and could
   be cited again. (Excludes trivial restatements.)

DO NOT extract:
- Pure procedural / planning text ("Let me search for X next", "First I need to ...").
- Self-questions and verification prompts ("But should I double-check?").
- Restatements of user input or earlier I/O beliefs already on file.
- Filler ("Okay,", "Hmm,", "let me see").
- Tool-call JSON or other syntax wrappers.

If nothing in this segment is a key reasoning node, return `{"beliefs": []}`.
"""


PROMPT_REASONING_THINK = f"""\
# Task
Extract KEY REASONING BELIEFS from the model's <think> segment below. This is the REASONING LAYER — be SELECTIVE, not comprehensive.

{_BELIEF_DEFINITION}

## Source for this segment: LLM REASONING (<think> block)
This is the model's internal deliberation. The model is working out HOW to get from the user's question to an answer.

{_SELECTIVITY_RULES}

{_STANCE_DEFINITION}
For reasoning content, all four stances are common — use whichever matches the model's actual wording. (A speculative claim in a <think> block is exactly what later steps might confirm.)

## I/O belief context (already extracted — DO NOT re-extract these)
{IO_CONTEXT_PLACEHOLDER}

{_HARD_CONSTRAINTS}

{_OUTPUT_FORMAT}

## Input segment (<think> content)
{CONTENT_PLACEHOLDER}
"""


PROMPT_REASONING_TOOL_RESPONSE = f"""\
# Task
Extract KEY REASONING BELIEFS from the tool-response segment below. This is the REASONING LAYER — be SELECTIVE.

{_BELIEF_DEFINITION}

## Source for this segment: TOOL RESPONSE
This is the output of a tool/function call (search results, retrieval, etc.) returned to the assistant. It is typically a large dump of text.

Extract ONLY facts that:
1. Directly address (or partially address) an I/O belief — esp. the user's question or a tool_call hypothesis.
2. Confirm or contradict an earlier reasoning hypothesis from a <think> block.
3. Are specific, citeable data points (named entities, dates, relationships, quantities).

DO NOT extract:
- Generic background biography unrelated to the user's question.
- Site navigation, related links, "see also" footers.
- Boilerplate wrappers ("Execution output of [...]:", "Your answer has been submitted").
- Tangential trivia.

{_SELECTIVITY_RULES}

## I/O belief context (already extracted — DO NOT re-extract these)
{IO_CONTEXT_PLACEHOLDER}

{_STANCE_DEFINITION}
Tool responses default to "asserted" — they are reporting facts. Use "speculated" or "judged" only if the tool itself hedges (e.g. "according to some sources...").

{_HARD_CONSTRAINTS}

{_OUTPUT_FORMAT}

## Input segment (<tool_response> content)
{CONTENT_PLACEHOLDER}
"""


# =====================================================================
# Stage 3 prompts — belief relation linking
# =====================================================================
#
# Two complementary passes over the full belief list:
#
#   FORWARD  (informs)        — derivation flow.  Edge A → B means "A is a
#                               premise / input / context that B was reasoned
#                               from".  A must be EARLIER than B.  Used to
#                               build the belief evolution path.  Does NOT
#                               affect confidence.
#
#   BACKWARD (confirms /      — evaluation.  Edge A → B means "the later
#             contradicts /     belief A confirms / contradicts / extends the
#             extends)          earlier belief B".  Drives the dynamic
#                               confidence update in confidence.py.

PROMPT_LINK_BACKWARD = f"""\
# Task
Given a full belief list extracted from a multi-turn conversation, identify
**backward (evaluation) relations** between beliefs that explain how later
beliefs change the *epistemic status* of earlier ones.

Each belief is labeled with an `id`, `belief` text, `stance`, and `source`
information. The list is ordered chronologically (earlier ids → earlier in
the conversation).

## Relation types

- **confirms**: a later belief provides evidence supporting an earlier
  speculation, recall, or judgment — basically saying "yes, the earlier
  belief was right". The target (to_id) MUST be earlier and ideally non-asserted.

- **contradicts**: a later belief refutes an earlier one with specific
  evidence. The target must be earlier.

- **extends**: a later belief elaborates on or makes more specific an
  earlier one without contradicting it. Use sparingly — only when the
  later belief truly builds on the earlier one (not when they are merely
  about the same topic).

## What to AVOID
- Linking beliefs that just happen to mention the same entity. Two beliefs
  about "John Marshall" are not automatically linked.
- Linking two beliefs that are restatements / paraphrases of the same thing.
- Linking forward (from earlier → later) — that is a separate pass.
- Producing weak / speculative relations. When in doubt, leave it out.

## Output (JSON only — no markdown fences, no commentary)
{{
  "relations": [
    {{
      "from_id": <int, the LATER belief — the evidence>,
      "to_id":   <int, the EARLIER belief — the target being confirmed/contradicted/extended>,
      "type":    "confirms" | "contradicts" | "extends",
      "note":    "<one short sentence explaining the link>"
    }}
  ]
}}

Empty list is fine: `{{"relations": []}}`.

## Full belief list
{BELIEFS_LIST_PLACEHOLDER}
"""


PROMPT_LINK_FORWARD = f"""\
# Task
Given a chronologically-ordered list of beliefs from a multi-turn
conversation, identify **forward (derivation) relations** of the form:

    "A informs B"  —  belief A is a piece of context, premise, or input that B was reasoned from.

Forward relations capture the **derivation flow** that explains how
downstream beliefs were produced from earlier ones. They are DIFFERENT from
confirms / contradicts / extends (those evaluate beliefs); forward relations
describe causation / dependency, NOT evidence quality.

## Typical patterns to look for

- A `user_input` belief informs a `think` (llm_reasoning) belief that
  responds to that input.
- A `think` belief informs a `tool_call` belief whose query embodies the
  reasoning conclusion.
- A `tool_response` belief informs a later `think` belief that processes
  the retrieved facts.
- A `think` and/or `tool_response` belief informs an `assistant_other`
  (final-answer) belief that concludes the work.
- A `user_input` belief may inform a `tool_call` directly, when the call is
  a literal restatement of the question and no intermediate reasoning was
  needed.

## What to AVOID

- Linking beliefs just because they mention the same entity (topical
  co-occurrence is NOT derivation).
- Linking beliefs that are paraphrases / restatements (that belongs to the
  separate backward `extends` relation).
- Linking forward to a later belief that is NOT actually derived from the
  candidate (e.g. two unrelated facts in the same think block).
- Producing every possible earlier predecessor — be selective. Each
  downstream belief typically has 1–3 actual informants, not 10.
- Linking BACKWARDS in time. `from_id` MUST be strictly less than `to_id`.

## Output (JSON only — no markdown fences, no commentary)
{{
  "forward_relations": [
    {{
      "from_id": <int, the EARLIER informant belief>,
      "to_id":   <int, the LATER informed belief>,
      "type":    "informs",
      "note":    "<one short sentence explaining the connection>"
    }}
  ]
}}

Empty list is fine: `{{"forward_relations": []}}`.

## Full belief list
{BELIEFS_LIST_PLACEHOLDER}
"""


# =====================================================================
# Registry: segment type → prompt template + whether it needs I/O context
# =====================================================================

EXTRACTION_PROMPTS = {
    # type:                (prompt_template,                needs_io_context)
    "user_input":           (PROMPT_IO_USER_INPUT,           False),
    "tool_call":            (PROMPT_IO_TOOL_CALL,            False),
    "assistant_other":      (PROMPT_IO_ASSISTANT_OTHER,      False),
    "think":                (PROMPT_REASONING_THINK,         True),
    "tool_response":        (PROMPT_REASONING_TOOL_RESPONSE, True),
}
