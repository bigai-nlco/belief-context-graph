"""
prompts.py  (v4 relation-schema)
==========================================
All LLM prompts, organised by function.

This version updates the prompt contract toward the new belief-graph design:
* belief granularity is relaxed from sentence-level atomic shards to coherent,
  self-contained reasoning/memory units;
* every belief/decision is asked to include ``entities``;
* assistant final answers wrapped in ``\\boxed{...}`` are extracted as
  separate ``decisions`` rather than ordinary beliefs;
* relation semantics are expressed with the three target edge types:
  ``depends_on`` | ``supplements`` | ``contradicts``.

Placeholders are filled via str.replace:
    <<<CONTENT>>> <<<SENTENCES>>> <<<GRAPH_NODES>>> <<<GRAPH_EDGES>>>
    <<<CANDIDATE_GROUP>>> <<<BELIEFS_LIST>>>
"""

from __future__ import annotations

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
- temporal expressions as entities; preserve temporal information in the belief
  or decision text instead (event metadata is assigned by the graph builder);
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
      "entities": ["<entity mentioned in this belief>", "<another entity>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"]
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "entities": ["<entity mentioned in this decision>"],
      "stance": "asserted | recalled | speculated | judged",
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
      "entities": ["<entity mentioned in this belief>", "<another entity>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_sentence_indices": [0, 2]
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "entities": ["<entity mentioned in this decision>"],
      "stance": "asserted | recalled | speculated | judged",
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
- **Tool calls**: extract every tool call as a belief — describe the assistant's information-seeking intent in natural language, capturing the tool name, the key parameters or constraints issued, and any hypothesis the call presupposes or commits to.
- **Key reasoning steps that are falsifiable, reusable, or needed by later turns** — keep enough detail to reconstruct causal/dependency chains between user request, tool result, reasoning, and final answer.

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
    task_line, guidance, stance_hint = _GUIDANCE[key]

    parts: list[str] = [
        "# Task",
        task_line,
        "\nYou maintain a belief graph INCREMENTALLY. From the CURRENT turn, output only the NEW "
        "belief/decision nodes and NEW typed relations. Existing nodes/relations (shown below) must not be repeated.\n",
        _BELIEF_DEFINITION,
        guidance + "\n",
        _STANCE_DEFINITION + stance_hint + "\n",
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
      "tmp_id": "n0",
      "belief": "<self-contained coherent belief>",
      "entities": ["<entity mentioned in this belief>", "<another entity>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"]
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "entities": ["<entity mentioned in this decision>"],
      "stance": "asserted | recalled | speculated | judged",
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
      "tmp_id": "n0",
      "belief": "<self-contained coherent belief>",
      "entities": ["<entity mentioned in this belief>", "<another entity>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_sentence_indices": [0, 2]
    }
  ],
  "decisions": [
    {
      "tmp_id": "d0",
      "decision": "<final selected answer, especially content inside \\boxed{...}>",
      "entities": ["<entity mentioned in this decision>"],
      "stance": "asserted | recalled | speculated | judged",
      "supporting_sentence_indices": [3]
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
4. Each new belief needs a unique "tmp_id": n0, n1, n2, … in output order.
5. Each new decision needs a unique "tmp_id": d0, d1, d2, … in output order.
6. Every belief and decision MUST include "entities" as a list, even if empty.
7. Empty beliefs / decisions lists are OK when the content expresses none.
"""

_HARD_CONSTRAINTS_SENTENCES_NODES = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the sentences. No outside knowledge.
3. Every belief and decision MUST list the indices of the COMPLETE sentence(s) that support it in
   "supporting_sentence_indices" (use the [k] indices shown). Evidence is always a whole sentence.
   If the whole group supports it, list all its indices.
4. Each new belief needs a unique "tmp_id": n0, n1, n2, … in output order.
5. Each new decision needs a unique "tmp_id": d0, d1, d2, … in output order.
6. Every belief and decision MUST include "entities" as a list, even if empty.
7. Empty beliefs / decisions lists are OK when the sentences express none.
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
    task_line, guidance, stance_hint = _GUIDANCE[key]

    parts: list[str] = [
        "# Task",
        task_line,
        "\nYou maintain a belief graph INCREMENTALLY. From the CURRENT turn, output only the NEW "
        "belief/decision nodes. Relations will be extracted in a separate step. "
        "Existing nodes (shown below) must not be repeated.\n",
        _BELIEF_DEFINITION,
        guidance + "\n",
        _STANCE_DEFINITION + stance_hint + "\n",
        _GRAPH_CONTEXT_BLOCK,
    ]
    if mode == "excerpt":
        parts.append(_HARD_CONSTRAINTS_EXCERPT_NODES)
        parts.append(_OUTPUT_FORMAT_EXCERPT_NODES)
        parts.append(f"## Current turn content\n{CONTENT_PLACEHOLDER}\n")
    else:
        parts.append(
            "## Sentence input\n"
            "The current turn's content was split into COMPLETE sentences with stable indices [k]. "
            "Reference them in supporting_sentence_indices; evidence is always a whole sentence.\n"
        )
        parts.append(_HARD_CONSTRAINTS_SENTENCES_NODES)
        parts.append(_OUTPUT_FORMAT_SENTENCES_NODES)
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
        f"## Current turn ({role})\n" + CONTENT_PLACEHOLDER + "\n",
        "## Candidate node window\n"
        "The graph below is deliberately limited to the current turn's surviving new nodes "
        "and one candidate prior turn's surviving nodes. It is not the full graph.\n\n"
        "### Candidate nodes\n" + GRAPH_NODES_PLACEHOLDER + "\n\n"
        "### Existing relations\n" + GRAPH_EDGES_PLACEHOLDER + "\n",
        "## Nodes from this turn\n" + NEW_NODE_IDS_PLACEHOLDER + "\n",
        _RELATION_EDGE_RULES,
        _RELATION_OUTPUT_FORMAT,
    ]
    prompt = "\n".join(parts)
    prompt = prompt.replace(CONTENT_PLACEHOLDER, content or "")
    prompt = prompt.replace(GRAPH_NODES_PLACEHOLDER, graph_nodes or "[]")
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
