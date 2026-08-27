"""
prompts.py  (v4 relation-schema)
==========================================
All LLM prompts, organised by function.

This version updates the prompt contract toward the new belief-graph design:
* belief granularity is relaxed from sentence-level atomic shards to coherent,
  self-contained reasoning/memory units;
* every belief/decision is asked to include ``stance`` and ``entities``;
* assistant final answers wrapped in ``\\boxed{...}`` are extracted as
  separate ``decisions`` rather than ordinary beliefs;
* relation semantics are expressed with the three target edge types:
  ``depends_on`` | ``supplements`` | ``contradicts``.

Placeholders are filled via str.replace:
    <<<CONTENT>>> <<<SENTENCES>>> <<<GRAPH_NODES>>> <<<GRAPH_EDGES>>>
    <<<CANDIDATE_GROUP>>> <<<BELIEFS_LIST>>>
"""

from __future__ import annotations

import json

from .._shared.roles import normalize_role

CONTENT_PLACEHOLDER = "<<<CONTENT>>>"
SENTENCES_PLACEHOLDER = "<<<SENTENCES>>>"
GRAPH_NODES_PLACEHOLDER = "<<<GRAPH_NODES>>>"
GRAPH_EDGES_PLACEHOLDER = "<<<GRAPH_EDGES>>>"
CANDIDATE_GROUP_PLACEHOLDER = "<<<CANDIDATE_GROUP>>>"
BELIEFS_LIST_PLACEHOLDER = "<<<BELIEFS_LIST>>>"
NEW_NODE_IDS_PLACEHOLDER = "<<<NEW_NODE_IDS>>>"


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
- temporal expressions as entities; preserve temporal information in the belief or decision text instead;
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
- Decisions also need ``stance`` and ``entities``. Event time is assigned
  deterministically by the graph builder and must not be generated here.
"""

_STANCE_DEFINITION = """\
## Stance (choose ONE per belief or decision)
- **asserted**   — stated as a plain fact or final answer ("X is Y", "released in 1980").
- **recalled**   — based on memory ("I recall X", "I remember X").
- **speculated** — hedged ("might", "maybe", "perhaps", "could", "possibly").
- **judged**     — evaluative conclusion, recommendation, ranking, diagnosis, or selected option
                  ("most likely", "best answer is", "I recommend X").

"""

_USER_BELIEF_DEFINITION = """\
## What is a belief
A belief is a self-contained memory or reasoning unit, usually shaped like:
    <subject, predicate, object/value, scope, time, source>

Preserve the most specific supported wording. Each belief must express one
reusable semantic unit and remain understandable outside the original turn.

## Granularity
Merge clauses that jointly define one setup, condition, event, or causal step.
Split only propositions that can be independently confirmed, contradicted, or
reused. Do not merge independently searchable numbered clues solely because
they describe the same target. Keep claims with different epistemic status
separate.

## Entities
For each belief, list specific named or uniquely qualified people,
organizations, places, products, files, tools, models, APIs, datasets, and
concepts explicitly present in it. Exclude pronouns, temporal expressions,
bare generic nouns, vague concepts, and duplicates. Include task-defining
qualified roles when they distinguish reusable constraints, such as "winning
team", "first poet", or "target ODI match". Use [] when none exists.
"""

_USER_STANCE_DEFINITION = """\
## Stance
Choose one per belief: asserted = direct statement; recalled = explicit memory;
speculated = uncertain possibility; judged = assessment, recommendation, or
conclusion.
"""

_USER_GUIDANCE = """\
## Source role: USER
Extract the user's substantive request, facts, events, preferences, plans,
constraints, questions, corrections, and updates. Rewrite questions as
self-contained statements about what the user wants to know. Preserve related
constraints together. Write in the third person about "The user" or the named
subject. Skip greetings and purely cosmetic instructions.
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

_NODE_GRAPH_CONTEXT_BLOCK = f"""\
## Existing belief nodes (context — READ ONLY)
These NODES were already extracted from EARLIER turns. Use them only to:
  - resolve pronouns or vague references in the current turn ("it", "the shop", "that issue"),
  - keep entity names and wording consistent with prior beliefs.

CRITICAL — do NOT copy old node content as new output merely because it appears here:
  - Extract beliefs ONLY from the CURRENT turn.
  - Existing nodes are read-only context.
  - Do not copy existing beliefs from the graph.
  - However, if the CURRENT turn explicitly restates, confirms, corrects, or updates an existing belief, extract a NEW evidence-bearing belief for the CURRENT turn. Downstream merge/dedup may combine it later.

### Existing nodes
{GRAPH_NODES_PLACEHOLDER}
"""


def _has_existing_nodes(graph_nodes: str | None) -> bool:
    """Return whether a rendered graph-node block contains any nodes."""

    return (graph_nodes or "").strip() not in {"", "[]", "null", "None"}


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
   Examples:
   - "The user can be charged a late fee" depends_on → "The user's payment is past the due date".
   - "We will retrain the model on the new dataset" depends_on → "The new dataset has finished downloading".
   - "The bridge can carry the truck" depends_on → "The bridge's load limit is 40 tons".

2. **supplements**
   A adds detail, scope, parameters, examples, evidence, or elaboration to B without changing or refuting it.
   Examples:
   - "The meeting is in Room 302 on the third floor" supplements → "The meeting is scheduled for 3 PM".
   - "Two of the three servers are located in Frankfurt" supplements → "The service runs on three servers".
   - "The discount also applies to returning customers" supplements → "New customers get 20% off".

3. **contradicts**
   A conflicts with, corrects, negates, or replaces B.
   Examples:
   - "The store is closed on Sundays" contradicts → "The store is open every day of the week".
   - "The experiment failed to reproduce the result" contradicts → "The drug significantly lowers blood pressure".
   - "The deadline has been moved to Friday" contradicts → "The deadline is Wednesday".

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

Read the ENTIRE content/belief/decision text before judging, and base the relation on what the node ACTUALLY asserts as a whole — never on just a fragment of it. A node often describes a request, a question, a task, or a condition that merely MENTIONS a claim rather than asserting that claim itself. Treat "what the node is about" and "what the node states as true" as different things.
- Example: Node A = "The user asks us to verify whether the claim X is true." Node B = "Claim X is false."
  These two are NOT contradicts. Node A does not assert X; it only requests a check of X. Asserting X-is-false does not conflict with a request to verify X. (If anything, B could be a `supplements`/answer to A's request — but it is not a contradiction.)
- Only mark `contradicts` when both nodes, read in full, make assertions that cannot both be true.

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
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"]
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"]
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
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_sentence_indices": [0, 2]
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_sentence_indices": [3]
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
6. Empty beliefs / decisions / relations lists are OK when the content expresses none.
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
6. Empty beliefs / decisions / relations lists are OK when the sentences express none.
"""


# =====================================================================
# Per-role guidance
# =====================================================================
# Each entry: (task_line, guidance_block, stance_hint)

_GUIDANCE = {
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
        "Extract coherent BELIEFS and DECISIONS from the ASSISTANT turn below. Preserve the "
        "reasoning chain without creating tiny redundant nodes.",
        """\
## Source role: ASSISTANT
Extract:
- **Factual claims and intermediate conclusions** the assistant commits to (domain facts, numbers, diagnoses, derived states).
- **Recommendations / advice** given to the user.
- **Assessments** of the user's situation.
- **Final decisions**: when the assistant gives a final answer, especially inside ``\\boxed{...}``, put it in ``decisions`` instead of ``beliefs``.
- **Key reasoning steps that are falsifiable, reusable, or needed by later turns** — keep enough detail to reconstruct causal/dependency chains between user request, tool result, reasoning, and final answer.

Do NOT extract: pure procedure / planning filler ("Let me search next", "First I need to…") unless it encodes a substantive dependency; self-questions; or politeness.

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


def _resolve_role(role: str) -> str | None:
    role = (role or "").strip().lower()
    role = normalize_role(role)
    return role if role in _GUIDANCE else None


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
    task_line, guidance, _stance_hint = _GUIDANCE[key]

    parts: list[str] = [
        "# Task",
        task_line,
        "\nYou maintain a belief graph INCREMENTALLY. From the CURRENT turn, output only the NEW "
        "belief/decision nodes and NEW typed relations. Existing nodes/relations (shown below) must not be repeated.\n",
        _BELIEF_DEFINITION,
        _STANCE_DEFINITION,
        guidance + "\n",
    ]
    if _has_existing_nodes(graph_nodes):
        parts.append(_GRAPH_CONTEXT_BLOCK)
    parts.append(_FORWARD_EDGE_RULES)
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
    # Kept in the function signature for caller compatibility. Event metadata is
    # assigned deterministically when the graph node is created, not by the LLM.
    _ = current_date
    prompt = prompt.replace(GRAPH_NODES_PLACEHOLDER, graph_nodes or "[]")
    prompt = prompt.replace(GRAPH_EDGES_PLACEHOLDER, graph_edges or "[]")
    if mode == "excerpt":
        prompt = prompt.replace(CONTENT_PLACEHOLDER, content or "")
    else:
        prompt = prompt.replace(SENTENCES_PLACEHOLDER, sentences_block or "")
    return prompt


# =====================================================================
# Phase 1: Node extraction only (beliefs + decisions, no relations)
# =====================================================================

_OUTPUT_FORMAT_EXCERPT_NODES = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "belief": "<self-contained coherent belief>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"]
    }
  ],
  "decisions": [
    {
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"]
    }
  ]
}
"""

_OUTPUT_FORMAT_SENTENCES_NODES = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "belief": "<self-contained coherent belief>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_sentence_indices": [0, 2]
    }
  ],
  "decisions": [
    {
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_sentence_indices": [3]
    }
  ]
}
"""

_OUTPUT_FORMAT_EXCERPT_USER_NODES = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "belief": "<self-contained coherent belief>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"]
    }
  ]
}
"""

_OUTPUT_FORMAT_SENTENCES_USER_NODES = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "belief": "<self-contained coherent belief>",
      "stance": "asserted | recalled | speculated | judged",
      "entities": ["<specific entity>", "..."],
      "supporting_sentence_indices": [0, 2]
    }
  ]
}
"""

_HARD_CONSTRAINTS_EXCERPT_NODES = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the content. No outside knowledge.
3. Every belief and decision MUST have at least one supporting excerpt — a VERBATIM, CONTIGUOUS substring copied
   character-for-character from the content. No excerpt → drop that node.
4. Empty beliefs / decisions lists are OK when the content expresses none.
"""

_HARD_CONSTRAINTS_SENTENCES_NODES = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the sentences. No outside knowledge.
3. Every belief and decision MUST list the indices of the COMPLETE sentence(s) that support it in
   "supporting_sentence_indices" (use the [k] indices shown). Evidence is always a whole sentence.
   If the whole group supports it, list all its indices.
4. Empty beliefs / decisions lists are OK when the sentences express none.
"""

_HARD_CONSTRAINTS_EXCERPT_USER_NODES = """\
## Hard constraints
Preserve names, numbers, dates, quantities, and unusual punctuation exactly.
Use only the current content; do not add outside knowledge. Every belief must
include at least one VERBATIM, CONTIGUOUS "supporting_excerpts" substring copied
from the content. Drop beliefs without an excerpt. An empty beliefs list is valid
when the input expresses none.
"""

_HARD_CONSTRAINTS_SENTENCES_USER_NODES = """\
## Hard constraints
Preserve names, numbers, dates, quantities, and unusual punctuation exactly.
Use only the current indexed sentences; do not add outside knowledge. For every
belief, return all COMPLETE sentence indices that directly support it in
"supporting_sentence_indices". Drop beliefs without supporting sentences. An
empty beliefs list is valid when the input expresses none.
"""


def build_node_extraction_prompt(
    role: str,
    *,
    mode: str = "sentences",
    content: str | None = None,
    sentences_block: str | None = None,
    graph_nodes: str = "[]",
    graph_edges: str = "[]",
    current_date: str | None = None,
) -> str | None:
    """Phase 1 prompt: extract beliefs + decisions only (no relations).
    Returns None for an unknown role."""
    key = _resolve_role(role)
    if key is None:
        return None
    task_line, guidance, _stance_hint = _GUIDANCE[key]

    has_existing_nodes = _has_existing_nodes(graph_nodes)
    if key == "user" and not has_existing_nodes:
        task_intro = (
            "Extract coherent, self-contained beliefs from the current USER turn only. "
            "Preserve distinct constraints without fragmenting one coherent request."
        )
    else:
        task_intro = task_line

    parts: list[str] = ["# Task", task_intro]
    if key != "user" or has_existing_nodes:
        node_kinds = "belief" if key == "user" else "belief/decision"
        parts.append(
            "\nMaintain the belief graph incrementally. Output only NEW "
            f"{node_kinds} nodes from the CURRENT turn; relation extraction is separate. "
            "Do not repeat existing nodes.\n"
        )
    belief_definition = _BELIEF_DEFINITION
    stance_definition = _STANCE_DEFINITION
    role_guidance = guidance + "\n"
    if key == "user":
        belief_definition = _USER_BELIEF_DEFINITION
        stance_definition = _USER_STANCE_DEFINITION
        role_guidance = _USER_GUIDANCE
    parts.extend([belief_definition, stance_definition, role_guidance])
    if has_existing_nodes:
        parts.append(_NODE_GRAPH_CONTEXT_BLOCK)
    if mode == "excerpt":
        parts.append(
            _HARD_CONSTRAINTS_EXCERPT_USER_NODES
            if key == "user"
            else _HARD_CONSTRAINTS_EXCERPT_NODES
        )
        parts.append(
            _OUTPUT_FORMAT_EXCERPT_USER_NODES
            if key == "user"
            else _OUTPUT_FORMAT_EXCERPT_NODES
        )
        parts.append(f"## Current turn content\n{CONTENT_PLACEHOLDER}\n")
    else:
        if key != "user":
            parts.append(
                "## Sentence input\n"
                "The current turn's content was split into COMPLETE sentences with stable indices [k]. "
                "Reference them in supporting_sentence_indices; evidence is always a whole sentence.\n"
            )
        parts.append(
            _HARD_CONSTRAINTS_SENTENCES_USER_NODES
            if key == "user"
            else _HARD_CONSTRAINTS_SENTENCES_NODES
        )
        parts.append(
            _OUTPUT_FORMAT_SENTENCES_USER_NODES
            if key == "user"
            else _OUTPUT_FORMAT_SENTENCES_NODES
        )
        parts.append(f"## Current turn sentences\n{SENTENCES_PLACEHOLDER}\n")

    prompt = "\n".join(parts)
    # Kept in the function signature for caller compatibility. Event metadata is
    # assigned deterministically when the graph node is created, not by the LLM.
    _ = current_date
    prompt = prompt.replace(GRAPH_NODES_PLACEHOLDER, graph_nodes or "[]")
    # Node extraction intentionally receives historical nodes but no historical
    # relations. Relations are handled by the separate edge-extraction phase.
    _ = graph_edges
    if mode == "excerpt":
        prompt = prompt.replace(CONTENT_PLACEHOLDER, content or "")
    else:
        prompt = prompt.replace(SENTENCES_PLACEHOLDER, sentences_block or "")
    return prompt


def build_assistant_tool_result_extraction_prompt(
    *,
    mode: str,
    assistant_content: str | None = None,
    assistant_sentences_block: str | None = None,
    tool_items: str = "[]",
    graph_nodes: str = "[]",
) -> str:
    """Extract an Assistant turn and its Tool Results in one model call.

    The response keeps two explicit source partitions.  The caller validates
    and allocates each partition independently, so batching never collapses the
    original Assistant and Tool turns into one graph layer.
    """

    _task_line, assistant_guidance, _stance_hint = _GUIDANCE["assistant"]
    evidence_field = (
        '"supporting_excerpts": ["<verbatim Assistant excerpt>"]'
        if mode == "excerpt"
        else '"supporting_sentence_indices": [0, 2]'
    )
    assistant_input = (
        f"## Assistant turn content\n{assistant_content or ''}"
        if mode == "excerpt"
        else (
            "## Assistant turn sentences\n"
            "Indices are local to this Assistant turn.\n"
            f"{assistant_sentences_block or ''}"
        )
    )
    graph_context = ""
    if _has_existing_nodes(graph_nodes):
        graph_context = f"""## Existing belief nodes (read only)
{graph_nodes}

"""
    return f"""# Task
Extract belief/decision nodes from TWO ORDERED SOURCE LAYERS in one response:
1. the Assistant turn;
2. the Tool Result items produced by that Assistant turn.

The layers are batched only to save a model call. NEVER merge evidence or a
belief across the Assistant and Tool partitions. Relations are extracted later.
Existing nodes are read-only and must not be repeated.

{_BELIEF_DEFINITION}
{_DECISION_DEFINITION}
{_STANCE_DEFINITION}
{assistant_guidance}

Assistant tool-call JSON is handled deterministically by code. Extract the
Assistant's substantive reasoning, hypotheses, conclusions, and decisions, but
do not copy raw tool-call syntax as semantic beliefs.

{graph_context}
{assistant_input}

## Tool Result items
Each item is independent and belongs to the following Tool Result layer. Extract
at most the requested fact_limit facts per item. Use only that item's query and
results. Preserve exact names, dates, quantities, and versions. Prefer relevant,
self-contained facts that prevent repeating the search. Never transfer evidence
between item_index values.
{tool_items or "[]"}

## Output (JSON only)
{{
  "assistant": {{
    "beliefs": [
      {{
        "belief": "<self-contained Assistant belief>",
        "stance": "asserted | recalled | speculated | judged",
        "entities": ["<specific entity>", "..."],
        {evidence_field}
      }}
    ],
    "decisions": [
      {{
        "decision": "<final selected answer only>",
        "stance": "asserted | recalled | speculated | judged",
        "entities": ["<specific entity>", "..."],
        {evidence_field}
      }}
    ]
  }},
  "tool_items": [
    {{
      "item_index": 0,
      "beliefs": [{{
        "belief": "<self-contained fact from this item only>",
        "stance": "asserted | recalled | speculated | judged",
        "entities": ["<specific entity>", "..."]
      }}]
    }}
  ]
}}

Hard constraints:
- Preserve named entities, numbers, dates, and quantities exactly.
- Do not add outside knowledge.
- Assistant evidence indices/excerpts may reference only the Assistant input.
- Tool facts may use only their own item.
- Return every Tool Result item_index exactly once, even with an empty beliefs list.
- Empty Assistant beliefs/decisions are allowed.
"""


# =====================================================================
# Phase 2: Relation extraction only (on post-merge graph)
# =====================================================================

_RELATION_EDGE_RULES = """\
## Relations between nodes
Emit relations only inside the candidate node window shown below. Connect new nodes
(from this turn) to the candidate prior-turn nodes, or to each other:
  - new node → candidate prior-turn node,
  - candidate prior-turn node → new node,
  - new node → new node.

Do NOT emit relations where BOTH endpoints are absent from the "Nodes from this turn" list.
Do NOT infer or emit candidate prior-turn node → candidate prior-turn node relations.

Endpoint rules:
- Every endpoint must be an integer node id from the graph.
- At least one endpoint must be from the "Nodes from this turn" list.
- Never invent ids not shown in the graph.
- Never connect a node to itself.

Use ONLY these three relation types:

1. **depends_on**
   A relies on B as a premise, input, assumption, tool result, user constraint, or required context.
   Examples:
   - "The user can be charged a late fee" depends_on → "The user's payment is past the due date".
   - "We will retrain the model on the new dataset" depends_on → "The new dataset has finished downloading".
   - "The bridge can carry the truck" depends_on → "The bridge's load limit is 40 tons".

2. **supplements**
   A adds detail, scope, parameters, examples, evidence, or elaboration to B without changing or refuting it.
   Examples:
   - "The meeting is in Room 302 on the third floor" supplements → "The meeting is scheduled for 3 PM".
   - "Two of the three servers are located in Frankfurt" supplements → "The service runs on three servers".
   - "The discount also applies to returning customers" supplements → "New customers get 20% off".

3. **contradicts**
   A conflicts with, corrects, negates, or replaces B.
   Examples:
   - "The store is closed on Sundays" contradicts → "The store is open every day of the week".
   - "The experiment failed to reproduce the result" contradicts → "The drug significantly lowers blood pressure".
   - "The deadline has been moved to Friday" contradicts → "The deadline is Wednesday".

Direction rule:
- Use the direction that makes the relation sentence natural:
  {"from": A, "to": B, "type": "depends_on"} means A depends on B.
  {"from": A, "to": B, "type": "supplements"} means A supplements B.
  {"from": A, "to": B, "type": "contradicts"} means A contradicts B.


## What to read when judging a relation
Judge the relation between two nodes ONLY from each node's semantic content field, NOT from any auxiliary field (``entities``, ``stance``, ``event_time``, ``time_text``, ``conf``, etc.):
- For each node, read its ``content`` field (shown in the graph above).

Read the ENTIRE content text before judging, and base the relation on what the node ACTUALLY asserts as a whole — never on just a fragment of it. A node often describes a request, a question, a task, or a condition that merely MENTIONS a claim rather than asserting that claim itself. Treat "what the node is about" and "what the node states as true" as different things.
- Example: Node A = "The user asks us to verify whether the claim X is true." Node B = "Claim X is false."
  These two are NOT contradicts. Node A does not assert X; it only requests a check of X.
- Only mark `contradicts` when both nodes, read in full, make assertions that cannot both be true.

Auxiliary fields such as ``entities`` are metadata only. Do NOT decide that two nodes are related (or pick a relation type) because they share an entity; the relation must be justified by the meaning expressed in the content text itself.

Selection rules:
- Prefer relations directly supported by the turn content plus the graph context.
- Be selective but complete enough to preserve the reasoning chain between turns.
- Do NOT link nodes merely because they share an entity.
- Prefer 0–4 high-value relations per new node; empty list is fine.
"""

_RELATION_OUTPUT_FORMAT = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "relations": [
    { "from": <int node id>, "to": <int node id>, "type": "depends_on | supplements | contradicts", "note": "<one short sentence>" }
  ]
}

Rules:
- Every endpoint must be an integer node id from the graph above.
- At least one endpoint of each relation must be from the "Nodes from this turn" list.
- Empty list is fine when no meaningful relations exist: {"relations": []}.
"""


def build_relation_extraction_prompt(
    *,
    role: str,
    content: str,
    graph_nodes: str = "[]",
    graph_edges: str = "[]",
    new_node_ids: str = "[]",
    current_date: str | None = None,
) -> str:
    """Phase 2 prompt: extract relations only (on post-merge graph)."""
    parts: list[str] = [
        "# Task",
        "Extract typed relations for the belief graph based on the current turn.",
        "\nBelief and decision nodes have already been extracted from the current turn and merged "
        "into the graph. Your job is to identify meaningful semantic relations inside the "
        "candidate node window below: current-turn surviving new nodes plus at most one "
        "candidate prior turn, or current-turn nodes only.\n",
    ]
    if content.strip():
        parts.append(f"## Current turn ({role})\n" + CONTENT_PLACEHOLDER + "\n")
    parts.extend(
        [
            "## Candidate node window\n"
            "The graph below is deliberately limited to the current turn's surviving new nodes "
            "and one candidate prior turn's surviving nodes. It is not the full graph.\n\n"
            "### Candidate nodes\n" + GRAPH_NODES_PLACEHOLDER + "\n\n"
            "### Existing relations\n" + GRAPH_EDGES_PLACEHOLDER + "\n",
            "## Nodes from this turn\n" + NEW_NODE_IDS_PLACEHOLDER + "\n",
            _RELATION_EDGE_RULES,
            _RELATION_OUTPUT_FORMAT,
        ]
    )
    prompt = "\n".join(parts)
    prompt = prompt.replace(CONTENT_PLACEHOLDER, content or "")
    prompt = prompt.replace(GRAPH_NODES_PLACEHOLDER, graph_nodes or "[]")
    prompt = prompt.replace(GRAPH_EDGES_PLACEHOLDER, graph_edges or "[]")
    prompt = prompt.replace(NEW_NODE_IDS_PLACEHOLDER, new_node_ids or "[]")
    return prompt


_LAYERED_RELATION_OUTPUT_FORMAT = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "selected_previous_layer": <integer layer number or null>,
  "relations": [
    { "from": <int node id>, "to": <int node id>, "type": "depends_on | supplements | contradicts", "note": "<one short sentence>" }
  ]
}

Hard output constraints:
- You may connect current-turn nodes to nodes from ZERO OR ONE previous layer.
- If any current-to-previous relation is emitted, every such relation MUST use the
  same previous layer and ``selected_previous_layer`` MUST equal that layer number.
- If no current-to-previous relation is emitted, set ``selected_previous_layer`` to null.
- Current-turn to current-turn relations are allowed and do not select a previous layer.
- Never connect nodes from two different previous layers in the same response.
- Every endpoint must be an integer node id shown in the candidate graph.
- At least one endpoint of each relation must be from the current-turn node list.
- Empty relations are valid: {"selected_previous_layer": null, "relations": []}.
"""


_LAYERED_RELATION_EDGE_RULES = """\
## Relation rules
Judge meaningful semantic links using node ``content`` alone:
- ``depends_on``: A requires B as a premise, input, evidence, constraint, or context.
- ``supplements``: A adds useful detail or evidence to B without changing it.
- ``contradicts``: A conflicts with, corrects, negates, or replaces B.

Direction is literal: ``A -> B`` means A depends on, supplements, or contradicts B.
Read each complete content field; a request to verify a claim does not assert it.
Shared entities alone do not justify a relation.

Constraints:
- Every relation must contain at least one current-turn node.
- Current-to-current relations are allowed; previous-to-previous relations are not.
- Use only shown integer ids; no self-links or invented ids.
- Prefer 0-4 high-value relations per current node. An empty result is valid.
"""


def _annotate_layered_relation_nodes(
    graph_nodes: str,
    new_node_ids: str,
    candidate_layers: str,
) -> str:
    """Add code-owned layer labels without changing the flat candidate window."""
    try:
        nodes = json.loads(graph_nodes or "[]")
        current_ids = {
            int(value) for value in json.loads(new_node_ids or "[]") if isinstance(value, int)
        }
        layers = json.loads(candidate_layers or "[]")
        layer_by_id = {
            int(node_id): int(layer["layer"])
            for layer in layers
            if isinstance(layer, dict) and isinstance(layer.get("layer"), int)
            for node_id in layer.get("node_ids", [])
            if isinstance(node_id, int)
        }
        annotated: list[dict[str, object]] = []
        for original in nodes:
            if not isinstance(original, dict) or not isinstance(original.get("id"), int):
                continue
            node = dict(original)
            node_id = int(node["id"])
            node["candidate_layer"] = (
                "current" if node_id in current_ids else layer_by_id.get(node_id)
            )
            annotated.append(node)
        return json.dumps(annotated, ensure_ascii=False, indent=2)
    except (TypeError, ValueError, json.JSONDecodeError):
        return graph_nodes or "[]"


def build_layered_relation_extraction_prompt(
    *,
    role: str,
    content: str,
    graph_nodes: str = "[]",
    graph_edges: str = "[]",
    new_node_ids: str = "[]",
    candidate_layers: str = "[]",
    validation_feedback: str | None = None,
) -> str:
    """Build one relation prompt over several candidate previous-turn layers.

    The model sees all bounded candidate layers at once, but must select at most
    one of them for cross-turn relations.  This replaces sequential relation
    calls over one previous turn at a time for Assistant turns.
    """
    parts: list[str] = [
        "# Task",
        "Link the current Assistant belief nodes to the most relevant prior layer.",
    ]
    if content.strip():
        parts.append(
            f"## Current Assistant reasoning ({role})\n" + CONTENT_PLACEHOLDER + "\n"
        )
    annotated_nodes = _annotate_layered_relation_nodes(
        graph_nodes,
        new_node_ids,
        candidate_layers,
    )
    parts.extend(
        [
            "## Candidate previous layers\n"
            "Layer 1 is the nearest non-empty Graph turn; larger numbers are older. "
            "Layer membership is authoritative. Each candidate node repeats this mapping "
            "in ``candidate_layer``.\n"
            + candidate_layers
            + "\n",
            "## Candidate nodes\n"
            + annotated_nodes
            + "\n\n## Existing relations\n"
            + GRAPH_EDGES_PLACEHOLDER
            + "\n",
            "## Current-turn node ids\n" + NEW_NODE_IDS_PLACEHOLDER + "\n",
            _LAYERED_RELATION_EDGE_RULES,
            "## Layer selection\n"
            "Compare all candidates in one pass. Cross-turn relations may use ZERO OR "
            "ONE previous layer, never a mixture. Select null when none is meaningful.\n",
        ]
    )
    if validation_feedback:
        parts.append(
            "## Validation feedback from the previous attempt\n"
            + validation_feedback
            + "\nReturn a corrected complete JSON response.\n"
        )
    parts.append(_LAYERED_RELATION_OUTPUT_FORMAT)
    prompt = "\n".join(parts)
    prompt = prompt.replace(CONTENT_PLACEHOLDER, content or "")
    prompt = prompt.replace(GRAPH_EDGES_PLACEHOLDER, graph_edges or "[]")
    prompt = prompt.replace(NEW_NODE_IDS_PLACEHOLDER, new_node_ids or "[]")
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

PROMPT_MERGE_VERIFY_REWRITE = f"""\
# Task
The candidate nodes below were flagged as POTENTIAL duplicates by embedding
similarity (they already share the SAME source role and SAME node_type). Do BOTH
of the following in a single pass:

1. VERIFY (is the merge reasonable?): decide whether the candidates really express
   the SAME underlying proposition / decision and therefore should become ONE node.
   If they are not truly the same — different value, quantity, scope, time, or
   aspect, or one only supplements / contradicts / generalises the other — then do
   NOT merge them. Only group nodes whose meaning is genuinely the same.
2. REWRITE (consolidate the content): for every group you DO merge, write a single
   self-contained statement (`canonical_belief`) that PRESERVES THE UNION of meaning
   of ALL members. It must not drop any distinct, supported detail that any member
   carried; if members differ only in wording, give the clearest phrasing; if each
   carries a compatible extra qualifier, the merged statement must include all of
   them. The merged statement is the new content of the surviving node.

{_MERGE_RULES}

## Output (JSON only — no markdown fences, no commentary)
{{
  "merge_groups": [
    {{ "ids": [<int>, <int>, ...],
       "canonical_belief": "<one self-contained statement covering the FULL meaning of all members>"}}
  ]
}}

Rules:
- Each group needs at least 2 ids; every id must come from the candidates below.
- Every id in a group MUST have the same source role. Mixed-role groups are invalid.
- A node id may appear in at most ONE group.
- `canonical_belief` is REQUIRED for every group and must lose NO information.
- Candidates that are NOT truly the same stay unmentioned (they will NOT be merged).
- Empty list is fine when none should be merged: {{"merge_groups": []}}.

## Candidate nodes
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
