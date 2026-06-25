"""Prompt assembly for belief extraction, linking, and merge verification."""

from __future__ import annotations

CONTENT_PLACEHOLDER = "<<<CONTENT>>>"
SENTENCES_PLACEHOLDER = "<<<SENTENCES>>>"
GRAPH_CONTEXT_PLACEHOLDER = "<<<GRAPH_CONTEXT>>>"
IO_CONTEXT_PLACEHOLDER = "<<<IO_CONTEXT>>>"
SESSION_DATE_PLACEHOLDER = "<<<SESSION_DATE>>>"
EXISTING_BELIEFS_PLACEHOLDER = "<<<EXISTING_BELIEFS>>>"
NEW_BELIEFS_PLACEHOLDER = "<<<NEW_BELIEFS>>>"
EARLIER_BELIEFS_PLACEHOLDER = "<<<EARLIER_BELIEFS>>>"
SESSION_BELIEFS_PLACEHOLDER = "<<<SESSION_BELIEFS>>>"
CANDIDATE_GROUP_PLACEHOLDER = "<<<CANDIDATE_GROUP>>>"
BELIEFS_LIST_PLACEHOLDER = "<<<BELIEFS_LIST>>>"

_BELIEF_DEFINITION = """\
## Belief shape
A belief is a self-contained atomic memory unit:
<subject, predicate, object/value, scope, time, source>.

Rules:
1. Extract one central semantic claim per belief.
2. Preserve enough context for the claim to stand alone.
3. Split compound facts, lists, state changes, and fact-plus-inference statements.
4. Preserve names, dates, numbers, and quantities exactly as written.
5. Do not add outside knowledge.
"""

_STANCE_DEFINITION = """\
## Stance
Choose one:
- asserted: stated as a plain fact.
- recalled: based on memory.
- speculated: hedged or uncertain.
- judged: evaluative conclusion.
Do not output confidence; confidence is computed downstream.
"""

_TIME_FIELDS = f"""\
## Time attribution
The session containing this segment is dated: {SESSION_DATE_PLACEHOLDER}
For each belief, if the content states when the believed fact/event happened or
will happen, fill:
- "time_text": the verbatim temporal phrase from the content.
- "event_time": resolved ISO time only when it can be resolved from the phrase
  plus the session date. Use null when not confidently resolvable.
If the belief has no explicit time, set both fields to null. Never invent dates.
"""

_GRAPH_CONTEXT_BLOCK = f"""\
## Existing belief graph (context; read-only)
Use earlier beliefs only to resolve references and keep entity wording
consistent. Do not copy them into the output, and do not skip a current
claim just because a similar belief already exists.
{GRAPH_CONTEXT_PLACEHOLDER}
"""

_IO_CONTEXT_BLOCK = f"""\
## I/O belief context (already extracted; do not re-extract)
{IO_CONTEXT_PLACEHOLDER}
"""

_HARD_CONSTRAINTS_EXCERPT = """\
## Hard constraints
Every belief must include at least one supporting excerpt copied as a
verbatim contiguous substring from the segment. Deduplicate within the
segment. Return an empty list if there are no beliefs.
"""

_HARD_CONSTRAINTS_SENTENCES = """\
## Hard constraints
Every belief must list the sentence indices that support it in
"supporting_sentence_indices". If the whole group supports it, list all
indices. Deduplicate within the sentence group. Return an empty list if there
are no beliefs.
"""

_OUTPUT_FORMAT_EXCERPT = """\
## Output JSON only
{
  "beliefs": [
    {
      "belief": "<self-contained atomic belief>",
      "stance": "asserted | recalled | speculated | judged",
      "supporting_excerpts": ["<verbatim excerpt>"],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ]
}
"""

_OUTPUT_FORMAT_SENTENCES = """\
## Output JSON only
{
  "beliefs": [
    {
      "belief": "<self-contained atomic belief>",
      "stance": "asserted | recalled | speculated | judged",
      "supporting_sentence_indices": [0, 2],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ]
}
"""

_SELECTIVITY_RULES = """\
## Reasoning-layer selectivity
Extract only key reasoning nodes that bridge I/O, are falsifiable hypotheses,
or are reusable facts the model/tool commits to. Skip pure procedure,
self-questions, filler, and restatements already captured by I/O beliefs.
"""

_SENTENCE_MODE_NOTE = """\
## Sentence-group input
The source segment was sentence-split and semantically clustered. Below is one
cluster. Indices refer to this list. Extract only beliefs supported by these
sentences.
"""

_GUIDANCE = {
    ("research", "user_input"): (
        "Extract every belief expressed by the user in this input segment.",
        "Capture the user's information need as a fact about user intent, "
        "facts the user states, constraints, and background information. Skip "
        "pure assistant-procedure instructions.",
        False,
    ),
    ("research", "tool_call"): (
        "Extract beliefs from this tool-call segment.",
        "Capture what is being queried and any implicit hypothesis embedded in "
        "the query or tool arguments. Skip literal JSON syntax and schema details.",
        False,
    ),
    ("research", "assistant_other"): (
        "Extract every belief from the assistant's direct output segment.",
        "Capture headline answers, supporting facts, caveats, and conclusions. "
        "Skip formatting wrappers and pure transitions.",
        False,
    ),
    ("research", "think"): (
        "Extract key reasoning beliefs from this <think> segment.",
        _SELECTIVITY_RULES,
        True,
    ),
    ("research", "tool_response"): (
        "Extract key reasoning beliefs from this tool-response segment.",
        "Capture facts that address I/O beliefs, confirm or contradict earlier "
        "reasoning, or provide specific citeable data. Skip boilerplate and "
        "tangential background.\n\n" + _SELECTIVITY_RULES,
        True,
    ),
    ("conversation", "user_input"): (
        "Extract every belief expressed by the user in this conversation turn.",
        "Capture facts about the user/world, events, preferences, feelings, "
        "plans, questions, corrections, and updates. Write self-contained third "
        "person beliefs where useful. Skip greetings and formatting meta-instructions.",
        False,
    ),
    ("conversation", "assistant_other"): (
        "Extract every belief from this assistant conversation turn.",
        "Capture factual claims, recommendations, assessments, conclusions, "
        "commitments, and substantive acknowledgements of user facts. Skip "
        "politeness and pure scaffolding.",
        False,
    ),
}


def build_extraction_prompt(
    scenario: str,
    segment_type: str,
    *,
    mode: str = "excerpt",
    content: str | None = None,
    sentences: list[str] | None = None,
    graph_context: str = "[]",
    io_context: str | None = None,
    session_date: str | None = None,
) -> str | None:
    """Assemble a scenario-aware extraction prompt."""

    key = (scenario, segment_type)
    if key not in _GUIDANCE:
        key = ("research", segment_type)
        if key not in _GUIDANCE:
            return None
    task, guidance, needs_io_context = _GUIDANCE[key]
    parts = [
        f"# Task\n{task}\n",
        _BELIEF_DEFINITION,
        f"## Source guidance\n{guidance}\n",
        _STANCE_DEFINITION,
        _TIME_FIELDS,
        _GRAPH_CONTEXT_BLOCK,
    ]
    if needs_io_context:
        parts.append(_IO_CONTEXT_BLOCK)
    if mode == "sentences":
        parts.extend(
            [
                _SENTENCE_MODE_NOTE,
                _HARD_CONSTRAINTS_SENTENCES,
                _OUTPUT_FORMAT_SENTENCES,
                f"## Input sentences\n{SENTENCES_PLACEHOLDER}\n",
            ]
        )
    else:
        parts.extend(
            [
                _HARD_CONSTRAINTS_EXCERPT,
                _OUTPUT_FORMAT_EXCERPT,
                f"## Input segment\n{CONTENT_PLACEHOLDER}\n",
            ]
        )

    prompt = "\n".join(parts)
    prompt = prompt.replace(SESSION_DATE_PLACEHOLDER, session_date or "unknown")
    prompt = prompt.replace(GRAPH_CONTEXT_PLACEHOLDER, graph_context or "[]")
    prompt = prompt.replace(IO_CONTEXT_PLACEHOLDER, io_context or "[]")
    if mode == "sentences":
        prompt = prompt.replace(
            SENTENCES_PLACEHOLDER,
            "\n".join(
                f"[{i}] {sentence}" for i, sentence in enumerate(sentences or [])
            ),
        )
    else:
        prompt = prompt.replace(CONTENT_PLACEHOLDER, content or "")
    return prompt


PROMPT_LINK_FORWARD_INC = f"""\
# Task
A belief graph is being built incrementally. Existing beliefs were extracted
from earlier turns; new beliefs were just extracted from the current turn.
Identify forward derivation relations where A informs B.

Hard rules:
- to_id must be one of the new belief ids.
- from_id may be existing or new, and must be strictly less than to_id.
- type must be "informs".

Output JSON only:
{{
  "forward_relations": [
    {{"from_id": 0, "to_id": 1, "type": "informs", "note": "short reason"}}
  ]
}}

## Existing beliefs
{EXISTING_BELIEFS_PLACEHOLDER}

## New beliefs
{NEW_BELIEFS_PLACEHOLDER}
"""

PROMPT_LINK_BACKWARD_INC = f"""\
# Task
A session just ended. Earlier beliefs come from previous material; this-session
beliefs were extracted during the session that just ended. Identify backward
evaluation relations where later evidence confirms, contradicts, or extends
earlier beliefs.

Hard rules:
- from_id must be one of the this-session belief ids.
- to_id must be strictly less than from_id.
- type must be confirms, contradicts, or extends.

Output JSON only:
{{
  "relations": [
    {{"from_id": 2, "to_id": 0, "type": "confirms", "note": "short reason"}}
  ]
}}

## Earlier beliefs
{EARLIER_BELIEFS_PLACEHOLDER}

## This-session beliefs
{SESSION_BELIEFS_PLACEHOLDER}
"""

_MERGE_RULES = """\
Merge beliefs only when they express the same proposition about the same
entity, value/state, and time scope. Do not merge different values, different
time scopes, corrections, generalization/detail pairs, or different subjects.
"""

PROMPT_MERGE_VERIFY = f"""\
# Task
The candidate beliefs below may be duplicates. Identify groups that express
exactly the same meaning and should be merged.

{_MERGE_RULES}

Output JSON only:
{{
  "merge_groups": [
    {{
      "ids": [0, 1],
      "canonical_belief": "single best wording",
      "reason": "short reason"
    }}
  ]
}}

## Candidate beliefs
{CANDIDATE_GROUP_PLACEHOLDER}
"""

PROMPT_MERGE_FULL = f"""\
# Task
Scan the complete belief list and identify groups of exact duplicate beliefs.

{_MERGE_RULES}

Output JSON only:
{{"merge_groups": []}}

## Full belief list
{BELIEFS_LIST_PLACEHOLDER}
"""

EXTRACTION_PROMPTS = {
    segment_type: (
        build_extraction_prompt("research", segment_type) or "",
        needs_io_context,
    )
    for (scenario, segment_type), (_, _, needs_io_context) in _GUIDANCE.items()
    if scenario == "research"
}


# Legacy streaming prompt helpers retained for bcg.belief_graph.stream.
CONTENT_PLACEHOLDER = "<<<CONTENT>>>"
SENTENCES_PLACEHOLDER = "<<<SENTENCES>>>"
GRAPH_NODES_PLACEHOLDER = "<<<GRAPH_NODES>>>"
GRAPH_EDGES_PLACEHOLDER = "<<<GRAPH_EDGES>>>"
CURRENT_DATE_PLACEHOLDER = "<<<CURRENT_DATE>>>"
CANDIDATE_GROUP_PLACEHOLDER = "<<<CANDIDATE_GROUP>>>"
BELIEFS_LIST_PLACEHOLDER = "<<<BELIEFS_LIST>>>"


# =====================================================================
# Shared definitions
# =====================================================================

_BELIEF_DEFINITION = """\
## What is a belief
A belief is a self-contained memory / reasoning unit, usually shaped like:
    <subject, predicate, object/value, scope, time, source>

A belief should be understandable outside the original turn and should preserve
the causal or dependency role it may play in later reasoning.

Preserve the most specific supported wording in the belief itself.
Do NOT generalize a specific claim into a vague claim.

## Granularity — coherent units, not tiny shards
Do NOT shred one coherent point into many trivial beliefs. A belief is not forced
to be a single clause or sentence. It may include tightly coupled qualifiers,
conditions, reasons, results, or parameters when separating them would destroy the
meaning or the reasoning dependency.

A belief must satisfy BOTH:
1. **Single central semantics** — one coherent fact, state, event, constraint,
   hypothesis, reasoning step, recommendation, or intermediate conclusion.
2. **Context self-contained** — a reader still knows who / what / when / scope.

Split ONLY when the content contains independent propositions that can be reused,
confirmed, contradicted, or linked separately.

Examples:
- Too fine-grained:
    "The dataset is COCO." + "The task is instance segmentation." + "The model is Mask R-CNN."
  Better when these jointly define one experimental setup:
    "The experiment uses Mask R-CNN for COCO instance segmentation."

- Split independent facts:
    "The user is modifying belief granularity and also wants to replace the final evaluation pass."
      → "The user wants to modify belief granularity."
      +  "The user wants to replace the final evaluation pass."

- Keep coupled condition/result together:
    "If the parser still only accepts informs edges, three-type relations emitted by the prompt will be filtered out."
      → one belief, because the condition and result form one reusable dependency.

- Separate different epistemic status:
    "The logs show no CUDA error, so the failure is probably caused by a missing checkpoint."
      → "The logs show no CUDA error."
      +  "The failure is probably caused by a missing checkpoint."

## Entities
For every belief, output `entities`: specific, salient entities explicitly involved in the belief.

Include:
- named people, organizations, places, products, datasets, files, functions, classes, tools, models, APIs, variables, and concrete named concepts;
- specific qualified objects or concepts when the qualifier makes them distinguishable, e.g. "silver Honda Civic", "incremental merge", "format_graph_nodes", "COCO instance segmentation";
- possessive relatives or objects only when qualified, e.g. "Nisha's dad", "Jordan's dog", "the user's silver Honda Civic".

Do NOT include:
- pronouns: I, you, he, she, they, it, this, that;
- generic filler: "the content", "this turn", "the answer", "the issue", "the thing", "the result";
- bare generic nouns: car, file, code, graph, node, edge, model, prompt, time, data, message, unless they are qualified enough to be identifiable;
- abstract feelings or vague concepts unless the belief is specifically about that concept;
- temporal expressions as entities; put temporal information in event_time/time_text;
- duplicate surface forms referring to the same entity in one belief.

Use the most specific form supported by the CURRENT turn and existing graph context.
When in doubt, omit the entity rather than inventing one.
Use [] only if no meaningful entity exists.
"""

_DECISION_DEFINITION = """\
## What is a decision
A decision is the assistant's final answer, selected result, final option, or
explicit final conclusion, especially when it is wrapped in ``\\boxed{...}``.

Rules:
- Extract ``\\boxed{...}`` final answers as ``decisions`` instead of ordinary beliefs.
- A decision may depend on multiple beliefs or tool results; capture those links in relations.
- If the assistant gives a final answer without ``\\boxed{...}``, extract it as a decision only
  when it is clearly the final selected answer rather than an intermediate claim.
- Do not duplicate the same final answer as both a belief and a decision.
- Decisions also need ``entities``, stance, evidence, and time fields.
"""

_STANCE_DEFINITION = """\
## Stance (choose ONE per belief or decision)
- **asserted**   — stated as a plain fact or final answer ("X is Y", "released in 1980").
- **recalled**   — based on memory ("I recall X", "I remember X").
- **speculated** — hedged ("might", "maybe", "perhaps", "could", "possibly").
- **judged**     — evaluative conclusion, recommendation, ranking, diagnosis, or selected option
                  ("most likely", "best answer is", "I recommend X").

NOTE: do NOT output a confidence number — confidence is assigned downstream by code
rules based on (role, stance).
"""

_TIME_FIELDS = f"""\
## Time attribution
The current turn is dated: {CURRENT_DATE_PLACEHOLDER}
For each belief, if the content states WHEN the believed fact/event happened or will happen, fill:
  - "time_text":  the verbatim temporal phrase from the content (e.g. "March 15th", "two weeks ago").
  - "event_time": the resolved calendar time in ISO form, ONLY when it can be resolved from the
    phrase plus the current date (e.g. "March 15th" with a turn dated 2023/04/10 → "2023-03-15";
    use "2023-03" if only the month is known). Use null when not confidently resolvable.
If the belief has no explicit time attached, set both fields to null. NEVER invent dates.
"""

_GRAPH_CONTEXT_BLOCK = f"""\
## Existing belief graph (context — READ ONLY)
These NODES and EDGES were already extracted from EARLIER turns. Use them only to:
  - resolve pronouns or vague references in the current turn ("it", "the shop", "that issue"),
  - keep entity names and wording consistent with prior beliefs,
  - decide how NEW nodes from the CURRENT turn relate to existing nodes.

CRITICAL — do NOT copy old graph content as new output merely because it appears in the graph:
  - Extract beliefs ONLY from the CURRENT turn.
  - Existing graph context is read-only. Use it only to:
    - resolve pronouns or vague references,
    - keep entity names consistent,
    - decide how new nodes relate to existing nodes.
  - Do not copy existing beliefs from the graph.
  - However, if the CURRENT turn explicitly restates, confirms, corrects, or updates an existing belief, extract a NEW evidence-bearing belief for the CURRENT turn. Downstream merge/dedup may combine it later.


### Existing nodes
{GRAPH_NODES_PLACEHOLDER}

### Existing relations
{GRAPH_EDGES_PLACEHOLDER}
"""

_FORWARD_EDGE_RULES = """\
## Relations between nodes
After creating the NEW beliefs/decisions for this turn, emit relations that connect:
  - existing node → new node,
  - new node → existing node,
  - new node → new node.

Do NOT emit relations between two existing nodes. The code will reject existing node → existing node relations.

Endpoint rules:
- Relation endpoints must be either:
  - an existing integer id shown in Existing nodes, or
  - a new temporary id from this output ("nK" or "dK").
- Never invent ids.
- Never connect a node to itself.

Use ONLY these three relation types in the ``relations`` field:

1. **depends_on**
   A relies on B as a premise, input, assumption, tool result, user constraint, or required context.
   Example: "The proposed prompt-only change" depends_on → "The parser currently accepts only informs edges".

2. **supplements**
   A adds detail, scope, parameters, examples, evidence, or elaboration to B without changing or refuting it.
   Example: "Entities must include functions and files" supplements → "Belief nodes need an entities field".

3. **contradicts**
   A conflicts with, corrects, negates, or replaces B.
   Example: "The model output should use four typed relations" contradicts → "The graph uses generic informs edges".

Direction rule:
- Use the direction that makes the relation sentence natural:
  {"from": A, "to": B, "type": "depends_on"} means A depends on B.
  {"from": A, "to": B, "type": "supplements"} means A supplements B.
  {"from": A, "to": B, "type": "contradicts"} means A contradicts B.

## What to read when judging a relation
Judge the relation between two nodes ONLY from each node's semantic content field, NOT from any auxiliary field (``supporting_excerpts``, ``entities``, ``stance``, ``event_time``, ``time_text``, ``conf``, etc.):
- For an EXISTING node, read its ``content`` field.
- For a NEW belief created this turn, read its ``belief`` field.
- For a NEW decision created this turn, read its ``decision`` field.
Auxiliary fields such as ``supporting_excerpts`` and ``entities`` are evidence/metadata only. Do NOT decide that two nodes are related (or pick a relation type) because they share an excerpt or an entity; the relation must be justified by the meaning expressed in the content/belief/decision text itself.

Selection rules:
- Prefer relations directly supported by the CURRENT turn plus the read-only graph context.
- Be selective but complete enough to preserve the reasoning chain between turns.
- Do NOT link nodes merely because they share an entity.
- Prefer 0–4 high-value relations per new node; empty list is fine.
"""

_OUTPUT_FORMAT_EXCERPT = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "tmp_id": "n0",
      "belief": "<self-contained coherent belief>",
      "entities": ["<entity mentioned in this belief>", "<another entity>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "entities": ["<entity mentioned in this decision>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ],
  "relations": [
    { "from": <existing int id or "nK" or "dK">, "to": <existing int id or "nK" or "dK">, "type": "depends_on | supplements | contradicts", "note": "<one short sentence>" }
  ]
}
"""

_OUTPUT_FORMAT_SENTENCES = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "tmp_id": "n0",
      "belief": "<self-contained coherent belief>",
      "entities": ["<entity mentioned in this belief>", "<another entity>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_sentence_indices": [0, 2],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "entities": ["<entity mentioned in this decision>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_sentence_indices": [3],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ],
  "relations": [
    { "from": <existing int id or "nK" or "dK">, "to": <existing int id or "nK" or "dK">, "type": "depends_on | supplements | contradicts", "note": "<one short sentence>" }
  ]
}
"""

_HARD_CONSTRAINTS_EXCERPT = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the content. No outside knowledge.
3. Every belief and decision MUST have at least one supporting excerpt — a VERBATIM, CONTIGUOUS substring copied
   character-for-character from the content. No excerpt → drop that node.
4. Each new belief needs a unique "tmp_id": n0, n1, n2, … in output order.
5. Each new decision needs a unique "tmp_id": d0, d1, d2, … in output order.
6. Every belief and decision MUST include "entities" as a list, even if empty.
7. Empty beliefs / decisions / relations lists are OK when the content expresses none.
"""

_HARD_CONSTRAINTS_SENTENCES = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the sentences. No outside knowledge.
3. Every belief and decision MUST list the indices of the COMPLETE sentence(s) that support it in
   "supporting_sentence_indices" (use the [k] indices shown). Evidence is always a whole sentence.
   If the whole group supports it, list all its indices.
4. Each new belief needs a unique "tmp_id": n0, n1, n2, … in output order.
5. Each new decision needs a unique "tmp_id": d0, d1, d2, … in output order.
6. Every belief and decision MUST include "entities" as a list, even if empty.
7. Empty beliefs / decisions / relations lists are OK when the sentences express none.
"""


# =====================================================================
# Per-role guidance
# =====================================================================
# Each entry: (task_line, guidance_block, stance_hint)

_LEGACY_GUIDANCE = {
    "user": (
        "Extract every coherent BELIEF expressed by the USER in the turn below. "
        "Extract comprehensively, but avoid fragmenting one coherent user intent or constraint "
        "into many tiny nodes.",
        """\
## Source role: USER
Things to extract:
- **Facts about the user and their world** — possessions, attributes, relationships, locations
  ("The user drives a silver Honda Civic.").
- **Events** the user reports, with their time when stated ("The user had their car serviced on March 15th.").
- **Preferences, opinions, feelings** ("The user prefers synthetic oil.").
- **Plans and intentions** ("The user plans to rotate the tires next month.").
- **Constraints the user imposes** ("answer in Chinese", "no code", "use tool X") — keep tightly related constraints together when they form one instruction.
- **Questions / information needs**, reformulated as a fact about the user
  ("The user is asking how often to change the oil."). Keep the object and purpose of the question together.
- **Corrections or updates** to things said earlier — extract the NEW state as its own belief.

Do NOT infer habits, preferences, recurring patterns, intent, causality, or importance from a single mention.
Extract preferences, plans, or intentions only when the CURRENT turn explicitly states them.
A single event supports only that event unless recurrence or preference is explicitly stated.

Write each belief in the third person about "The user" (or the named person/entity) so it is
self-contained. Resolve pronouns using the existing graph context when unambiguous.

Skip: pure greetings / pleasantries with no factual content; purely cosmetic formatting instructions unless they affect the task semantics.""",
        'User statements are typically "asserted"; memories are "recalled" when phrased that way; '
        'hedged guesses are "speculated".',
    ),
    "assistant": (
        "Extract coherent BELIEFS and DECISIONS from the ASSISTANT turn below. The content may "
        "contain reasoning, tool-call syntax, and a final answer all together — read through ALL "
        "of it and preserve the reasoning chain without creating tiny redundant nodes.",
        """\
## Source role: ASSISTANT
The turn may mix internal reasoning, tool invocations, and the final answer. Extract:
- **Factual claims and intermediate conclusions** the assistant commits to (domain facts, numbers, diagnoses, derived states).
- **Recommendations / advice** given to the user.
- **Assessments** of the user's situation.
- **Final decisions**: when the assistant gives a final answer, especially inside ``\\boxed{...}``, put it in ``decisions`` instead of ``beliefs``.
- **Information needs expressed by a tool call** ONLY when the query or constraint is needed to explain a later tool result or final answer.
  Any hypothesis a query commits to is its own belief with stance "judged" or "speculated".
- **Key reasoning steps that are falsifiable, reusable, or needed by later turns** — keep enough detail to reconstruct causal/dependency chains between user request, tool result, reasoning, and final answer.
- **Tool-use commitments** ONLY when they carry semantic content that should be remembered or linked, not routine procedure.

Do NOT extract: pure procedure / planning filler ("Let me search next", "First I need to…") unless it encodes a substantive dependency; self-questions; raw tool-call JSON syntax / key names; or politeness.

Write each belief in the third person ("The assistant…", "The user…") so it is self-contained.
Resolve pronouns using the existing graph context when unambiguous.""",
        'Assistant claims and final answers are typically "asserted"; recommendations and evaluations '
        'are "judged"; hedged possibilities ("might be", "could be") are "speculated".',
    ),
    "tool": (
        "Extract coherent KEY BELIEFS from the TOOL output below (search results, retrieval, function return). "
        "Be selective about boilerplate, but preserve enough facts to support later reasoning and conclusions.",
        """\
## Source role: TOOL
This is the output returned by a tool / function call. Extract ONLY facts that:
1. Directly address (or partially address) the user's question or an assistant query/hypothesis.
2. Confirm, contradict, explain, or supplement an earlier assistant hypothesis.
3. Are specific, citeable data points (named entities, dates, relationships, quantities).
4. Are needed to preserve the dependency chain from tool result to later assistant conclusion.

Do NOT extract: generic background unrelated to the task; navigation / "see also" / related links;
boilerplate wrappers ("Execution output of […]:", "Your answer has been submitted"); tangential trivia.""",
        'Tool outputs default to "asserted" — they report facts. Use "speculated"/"judged" only if the '
        'tool itself hedges ("according to some sources…").',
    ),
}

# function-role turns are tool outputs.
_ROLE_ALIASES = {"function": "tool"}


def _resolve_role(role: str) -> str | None:
    role = (role or "").strip().lower()
    role = _ROLE_ALIASES.get(role, role)
    return role if role in _LEGACY_GUIDANCE else None


def format_sentences_for_prompt(sentences: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def format_clustered_sentences_for_prompt(
    sentences: list[str], clusters: list[list[int]]
) -> str:
    """Render the SAME indexed sentence list, but grouped by topic cluster.
    Indices [k] are GLOBAL across the whole list (unchanged), so the model still
    returns global supporting_sentence_indices. Grouping is presentation only —
    it remains ONE call for the whole content."""
    lines: list[str] = []
    for ci, idxs in enumerate(clusters):
        lines.append(f"--- topic group {ci} ---")
        for i in idxs:
            if 0 <= i < len(sentences):
                lines.append(f"[{i}] {sentences[i]}")
    return "\n".join(lines)


def build_update_prompt(
    role: str,
    *,
    mode: str = "sentences",  # "sentences" | "excerpt"
    content: str | None = None,
    sentences_block: str | None = None,  # pre-rendered indexed sentence list
    graph_nodes: str = "[]",
    graph_edges: str = "[]",
    current_date: str | None = None,
) -> str | None:
    """Assemble the single-call update prompt. Returns None for an unknown role."""
    key = _resolve_role(role)
    if key is None:
        return None
    task_line, guidance, stance_hint = _LEGACY_GUIDANCE[key]

    parts: list[str] = [
        "# Task",
        task_line,
        "\nYou maintain a belief graph INCREMENTALLY. From the CURRENT turn, output only the NEW "
        "belief/decision nodes and NEW typed relations. Existing nodes/relations (shown below) must not be repeated.\n",
        _BELIEF_DEFINITION,
        guidance + "\n",
        _STANCE_DEFINITION + stance_hint + "\n",
        _TIME_FIELDS,
        _GRAPH_CONTEXT_BLOCK,
        _FORWARD_EDGE_RULES,
    ]
    if mode == "excerpt":
        parts.append(_HARD_CONSTRAINTS_EXCERPT)
        parts.append(_OUTPUT_FORMAT_EXCERPT)
        parts.append(f"## Current turn content\n{CONTENT_PLACEHOLDER}\n")
    else:
        parts.append(
            "## Sentence input\n"
            "The current turn's content was split into COMPLETE sentences with stable indices [k]. "
            "Reference them in supporting_sentence_indices; evidence is always a whole sentence.\n"
        )
        parts.append(_HARD_CONSTRAINTS_SENTENCES)
        parts.append(_OUTPUT_FORMAT_SENTENCES)
        parts.append(f"## Current turn sentences\n{SENTENCES_PLACEHOLDER}\n")

    prompt = "\n".join(parts)
    prompt = prompt.replace(CURRENT_DATE_PLACEHOLDER, current_date or "unknown")
    prompt = prompt.replace(GRAPH_NODES_PLACEHOLDER, graph_nodes or "[]")
    prompt = prompt.replace(GRAPH_EDGES_PLACEHOLDER, graph_edges or "[]")
    if mode == "excerpt":
        prompt = prompt.replace(CONTENT_PLACEHOLDER, content or "")
    else:
        prompt = prompt.replace(SENTENCES_PLACEHOLDER, sentences_block or "")
    return prompt


# =====================================================================
# Merge / dedup prompts (unchanged contract)
# =====================================================================

_MERGE_RULES = """\
## When do two nodes express the SAME meaning (mergeable)?
- Same proposition or same final decision about the same entity with the same value/state and the same time scope,
  merely re-worded ("The user drives a silver Honda Civic." == "The user's car is a silver Honda Civic.").
- Pronoun vs explicit-name variants of the same statement.
- One node differs only by minor wording while preserving the same entities, scope, and claim.

## Hard role constraint
Nodes are mergeable ONLY when they have the SAME source role. Never merge across roles, even
when the text is semantically identical or highly similar. For example:
- A user claim must not merge with an assistant conclusion.
- An assistant conclusion must not merge with a tool result.
- A tool result must not merge with a user restatement.

## When are they NOT the same (do NOT merge)?
- Different source roles (user vs assistant vs tool/function), regardless of semantic similarity.
- Different values or quantities (32 mpg city vs 38-40 mpg highway).
- Different time scopes or different events.
- Different aspects of the same entity (owning the car vs the car's color — unless both nodes state both).
- One is a generalisation of the other, or one adds substantive new info (that is supplements/depends_on, not duplicate).
- Different subjects (a user's claim vs the assistant's recommendation about the same topic).
- A belief and its negation / correction.
- An intermediate belief and the final decision that depends on it.

Stance differences alone do NOT block a merge when the proposition is identical, but source ROLE
differences always block merging. Entities should help judge sameness, but identical entity lists
alone are NOT enough to merge.
"""

PROMPT_MERGE_VERIFY = f"""\
# Task
The candidate beliefs below were flagged as POTENTIAL duplicates by embedding
similarity. Decide which of them (if any) express EXACTLY the same meaning and
should be merged into one node of a belief graph.

{_MERGE_RULES}

## Output (JSON only — no markdown fences, no commentary)
{{
  "merge_groups": [
    {{ "ids": [<int>, <int>, ...],
       "canonical_belief": "<the single best self-contained wording for the merged belief>",
       "reason": "<one short sentence>" }}
  ]
}}

Rules:
- Each group needs at least 2 ids; every id must come from the candidates below.
- Every id in a group MUST have the same source role. Mixed-role groups are invalid.
- A belief id may appear in at most ONE group.
- Beliefs that are not exact duplicates of anything stay unmentioned.
- Empty list is fine: {{"merge_groups": []}}.

## Candidate beliefs
{CANDIDATE_GROUP_PLACEHOLDER}
"""

PROMPT_MERGE_FULL = f"""\
# Task
Below is the COMPLETE belief list of a belief graph built from one trajectory.
Scan it GLOBALLY and identify groups of beliefs that express EXACTLY the same
meaning and should be merged into one node (duplicates arise when the same fact
is restated across turns).

{_MERGE_RULES}

## Output (JSON only — no markdown fences, no commentary)
{{
  "merge_groups": [
    {{ "ids": [<int>, <int>, ...],
       "canonical_belief": "<the single best self-contained wording for the merged belief>",
       "reason": "<one short sentence>" }}
  ]
}}

Rules:
- Each group needs at least 2 ids; every id must exist in the list below.
- Every id in a group MUST have the same source role. Mixed-role groups are invalid.
- A belief id may appear in at most ONE group.
- Be conservative: when in doubt, do NOT merge.
- Empty list is fine: {{"merge_groups": []}}.

## Full belief list
{BELIEFS_LIST_PLACEHOLDER}
"""
