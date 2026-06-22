"""
prompts.py  (v3)
================
All LLM prompts, organised by function.

v3 changes
----------
* No scenarios. Turns are routed ONLY by role: ``user`` | ``assistant`` | ``tool``.
* No tag-based segmentation. The WHOLE message content of a turn is handed to
  the model at once; the model decides how many beliefs to extract.
* ONE call per content does BOTH jobs: it returns the NEW belief nodes AND the
  NEW forward (``informs``) edges, given the existing graph (nodes + edges) as
  read-only context. New nodes are referenced by temporary ids ("n0", "n1", …);
  forward-edge endpoints may be an existing integer id OR a temporary id.

Update prompt — ``build_update_prompt(role, mode, ...)``:
    modes:  excerpt   — the model quotes verbatim excerpts as evidence
                        (evidence may be a sentence fragment);
            sentences — the content arrives as an indexed sentence list and the
                        model returns supporting_sentence_indices, so evidence
                        is always a COMPLETE sentence. When clustering is on the
                        same single call shows the sentences grouped by topic
                        cluster (still one call).

Backward / merge prompts run ONCE at trajectory end over the FULL graph:
    PROMPT_LINK_BACKWARD_ALL  — full graph → confirms / contradicts / extends
                                (from_id > to_id).
    PROMPT_MERGE_VERIFY       — verify an embedding-candidate group.
    PROMPT_MERGE_FULL         — whole-graph dedup (merge-strategy=llm).

Placeholders are filled via str.replace:
    <<<CONTENT>>> <<<SENTENCES>>> <<<GRAPH_NODES>>> <<<GRAPH_EDGES>>>
    <<<CURRENT_DATE>>> <<<ALL_BELIEFS>>> <<<CANDIDATE_GROUP>>> <<<BELIEFS_LIST>>>
"""

from __future__ import annotations

from typing import List, Optional


CONTENT_PLACEHOLDER       = "<<<CONTENT>>>"
SENTENCES_PLACEHOLDER     = "<<<SENTENCES>>>"
GRAPH_NODES_PLACEHOLDER   = "<<<GRAPH_NODES>>>"
GRAPH_EDGES_PLACEHOLDER   = "<<<GRAPH_EDGES>>>"
CURRENT_DATE_PLACEHOLDER  = "<<<CURRENT_DATE>>>"
ALL_BELIEFS_PLACEHOLDER   = "<<<ALL_BELIEFS>>>"
CANDIDATE_GROUP_PLACEHOLDER = "<<<CANDIDATE_GROUP>>>"
BELIEFS_LIST_PLACEHOLDER  = "<<<BELIEFS_LIST>>>"


# =====================================================================
# Shared definitions
# =====================================================================

_BELIEF_DEFINITION = """\
## What is a belief
A belief is a self-contained memory unit that captures ONE coherent point, ideally shaped like:
    <subject, predicate, object/value, scope, time, source>

Each belief must satisfy BOTH:
1. **One coherent point** — it is about a single subject and a single matter concerning that
   subject (one fact, one preference, one event, one decision, one question). A belief may be
   RICH: it carries the qualifiers, conditions, reasons, and details that belong to that one
   point. "One point" does NOT mean "one clause" or "the shortest possible sentence".
2. **Context self-contained** — read on its own, a reader still knows who / what / when / scope.

## Granularity — DEFAULT TO CONSOLIDATION (avoid fragmentation)
Produce the FEWEST beliefs that still keep each one single-pointed and self-contained. When
several details concern the same subject and the same matter, state them as ONE complete belief
instead of scattering them across many thin nodes. Over-splitting one statement into several
co-dependent or near-identical shards is a WORSE error than letting a belief be a little rich.
Capture all the substantive information — but consolidate it, do not multiply nodes.
(Still do not cram genuinely independent facts into one overloaded belief — that breaks rule 1.)

### Split ONLY when a part passes this test
Split into separate beliefs only when the resulting parts concern a DIFFERENT subject, a
DIFFERENT matter/aspect, a DIFFERENT point in time/state, OR a DIFFERENT epistemic status —
AND each part is independently meaningful and could be retrieved or updated on its own. If a
part fails this test, KEEP it inside the same belief as a qualifier or detail.

DO split:
- **Two genuinely independent topics.**
    "User is researching Hindsight and prefers concise explanations."
      → "The user is researching Hindsight."  +  "The user prefers concise explanations."
- **Fact + inference of different epistemic status** (also mark stance separately).
    "User asks a lot about Hindsight, so user is likely building a memory system."
      → "The user asked multiple questions about Hindsight."        [stance: asserted]
      +  "The user may be building a belief-memory system."         [stance: speculated]
- **A state change across time** — keep the previous state and the new state as two beliefs.

Do NOT split — CONSOLIDATE instead (these are the common over-fragmentation traps):
- **A claim + its qualifier / condition / reason / deadline** stays together.
    BAD : "The user wants the report." + "The report should be in Chinese." + "It is due Friday."
    GOOD: "The user wants the report written in Chinese, due Friday."
- **Several attributes of the SAME entity in the same statement** → one belief.
    BAD : "The user's car is a Honda Civic." + "The car is silver." + "It is a 2018 model."
    GOOD: "The user drives a silver 2018 Honda Civic."
- **A list that elaborates or supports a single point** → one belief, not one-per-item.
    BAD : a separate belief for every example the user gives to back up one preference.
    GOOD: "The user prefers functional programming, citing easier testing and fewer side effects."
- **Re-wordings of the same claim** → emit only ONE; pick the single most complete phrasing.

Quick check: if two candidate beliefs would always be retrieved together, updated together, or
one is meaningless without the other, they should have been ONE belief.
"""

_STANCE_DEFINITION = """\
## Stance (choose ONE per belief)
- **asserted**   — stated as a plain fact ("X is Y", "released in 1980").
- **recalled**   — based on memory ("I recall X", "I remember X").
- **speculated** — hedged ("might", "maybe", "perhaps", "could", "possibly").
- **judged**     — evaluative conclusion ("most likely", "best answer is", "I think X").

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
These NODES and EDGES were already extracted from EARLIER turns. They are shown so you can:
  - resolve pronouns / vague references in the current turn ("it", "the shop", "that issue"),
  - keep entity names and wording consistent with prior beliefs,
  - decide which existing node a NEW forward edge should point FROM.

CRITICAL — do NOT re-emit anything already in the graph:
  - Do NOT output a node that already exists below (no duplicates of existing nodes).
  - Do NOT output an edge that already exists below.
  - BUT still extract every claim the CURRENT turn makes, even if it restates an existing
    fact — a restatement becomes a NEW node here (downstream linking + dedup handle it).
    Only the *existing* nodes/edges must not be repeated; new nodes for the current turn are expected.

### Existing nodes
{GRAPH_NODES_PLACEHOLDER}

### Existing forward edges (informs)
{GRAPH_EDGES_PLACEHOLDER}
"""

_FORWARD_EDGE_RULES = """\
## Forward edges (derivation flow) — emit the NEW ones for this turn
A forward edge "A informs B" means belief A is a piece of context, premise, or input that
B was reasoned/derived from. This is causation / dependency, NOT evidence quality and NOT
paraphrase.

Typical patterns:
- a user fact/question informs an assistant claim or a tool query that responds to it;
- a tool result informs a later assistant conclusion that uses it;
- an earlier assistant reasoning step informs a later assistant final answer.

Rules:
- Each edge has "from", "to", "type": "informs", and a short "note".
- "to" MUST be a NEW node of THIS turn (a "nK" temp id).
- "from" may be an EXISTING integer node id (shown above) OR a NEW "nK" temp id.
- For a new→new edge, "from" must be an EARLIER new node than "to" (n0 before n1, etc.).
- Be selective: each new belief usually has 0–3 real informants, not 10. Do NOT link
  beliefs merely because they share an entity, and do NOT link paraphrases.
- Empty list is fine.
"""

_OUTPUT_FORMAT_EXCERPT = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "tmp_id": "n0",
      "belief": "<self-contained atomic belief>",
      "stance": "asserted | recalled | speculated | judged",
      "supporting_excerpts": ["<verbatim excerpt copied character-for-character from the content>"],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ],
  "forward_relations": [
    { "from": <existing int id or "nK">, "to": "nK", "type": "informs", "note": "<one short sentence>" }
  ]
}
"""

_OUTPUT_FORMAT_SENTENCES = """\
## Output (JSON only — no markdown fences, no commentary)
{
  "beliefs": [
    {
      "tmp_id": "n0",
      "belief": "<self-contained atomic belief>",
      "stance": "asserted | recalled | speculated | judged",
      "supporting_sentence_indices": [0, 2],
      "event_time": "<ISO time or null>",
      "time_text": "<verbatim temporal phrase or null>"
    }
  ],
  "forward_relations": [
    { "from": <existing int id or "nK">, "to": "nK", "type": "informs", "note": "<one short sentence>" }
  ]
}
"""

_HARD_CONSTRAINTS_EXCERPT = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the content. No outside knowledge.
3. Every belief MUST have at least one supporting excerpt — a VERBATIM, CONTIGUOUS substring copied
   character-for-character from the content. No excerpt → drop the belief.
4. Each new belief needs a unique "tmp_id": n0, n1, n2, … in output order.
5. Empty beliefs list is OK if the content expresses none.
"""

_HARD_CONSTRAINTS_SENTENCES = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the sentences. No outside knowledge.
3. Every belief MUST list the indices of the COMPLETE sentence(s) that support it in
   "supporting_sentence_indices" (use the [k] indices shown). Evidence is always a whole sentence.
   If the whole group supports it, list all its indices.
4. Each new belief needs a unique "tmp_id": n0, n1, n2, … in output order.
5. Empty beliefs list is OK if the sentences express none.
"""


# =====================================================================
# Per-role guidance
# =====================================================================
# Each entry: (task_line, guidance_block, stance_hint)

_GUIDANCE = {
    "user": (
        "Extract the BELIEFS expressed by the USER in the turn below. "
        "Aim for full COVERAGE of the substantive information — facts, preferences, events, plans, "
        "constraints, and questions all matter for long-term memory — but coverage means capturing "
        "the information, NOT maximizing node count: fold details about the same point into ONE "
        "belief (see the granularity rules) instead of emitting many thin, near-identical nodes.",
        """\
## Source role: USER
Things to extract:
- **Facts about the user and their world** — possessions, attributes, relationships, locations
  ("The user drives a silver Honda Civic.").
- **Events** the user reports, with their time when stated ("The user had their car serviced on March 15th.").
- **Preferences, opinions, feelings** ("The user prefers synthetic oil.").
- **Plans and intentions** ("The user plans to rotate the tires next month.").
- **Constraints the user imposes** ("answer in Chinese", "no code", "use tool X") — only when substantive.
- **Questions / information needs**, reformulated as a fact about the user
  ("The user is asking how often to change the oil.").
- **Corrections or updates** to things said earlier — extract the NEW state as its own belief.

Write each belief in the third person about "The user" (or the named person/entity) so it is
self-contained. Resolve pronouns using the existing graph context when unambiguous.

Skip: pure greetings / pleasantries with no factual content; meta-instructions about answer formatting.""",
        'User statements are typically "asserted"; memories are "recalled" when phrased that way; '
        'hedged guesses are "speculated".',
    ),
    "assistant": (
        "Extract the BELIEFS from the ASSISTANT turn below. The content may contain reasoning, "
        "tool-call syntax, and a final answer all together — read through ALL of it and extract "
        "the substantive beliefs, consolidating related points into one belief rather than "
        "splitting a single conclusion across several near-identical nodes. Do NOT treat tags as "
        "separate documents; judge the whole turn.",
        """\
## Source role: ASSISTANT
The turn may mix internal reasoning, tool invocations, and the final answer. Extract:
- **Factual claims and conclusions** the assistant commits to (domain facts, numbers, diagnoses, the final answer).
- **Recommendations / advice** given to the user.
- **Assessments** of the user's situation.
- **Information needs expressed by a tool call** — restate the query as a belief about what the
  assistant is looking for ("The assistant is searching for the Burj Khalifa's floor count.").
  Any hypothesis a query commits to is its own belief with stance "judged" or "speculated".
- **Key reasoning steps that are falsifiable or reusable** — a committed factual claim or hypothesis
  that a later step could confirm or refute.

Do NOT extract: pure procedure / planning filler ("Let me search next", "First I need to…"),
self-questions ("should I double-check?"), raw tool-call JSON syntax / key names, or politeness.

Write each belief in the third person ("The assistant…", "The user…") so it is self-contained.
Resolve pronouns using the existing graph context when unambiguous.""",
        'Assistant claims and final answers are typically "asserted"; recommendations and evaluations '
        'are "judged"; hedged possibilities ("might be", "could be") are "speculated".',
    ),
    "tool": (
        "Extract KEY BELIEFS from the TOOL output below (search results, retrieval, function return). "
        "It is often a large dump — be selective and extract the citeable facts, not the boilerplate.",
        """\
## Source role: TOOL
This is the output returned by a tool / function call. Extract ONLY facts that:
1. Directly address (or partially address) the user's question or an assistant query/hypothesis.
2. Confirm or contradict an earlier assistant hypothesis.
3. Are specific, citeable data points (named entities, dates, relationships, quantities).

Do NOT extract: generic background unrelated to the task; navigation / "see also" / related links;
boilerplate wrappers ("Execution output of […]:", "Your answer has been submitted"); tangential trivia.

Consolidate: when several data points describe the SAME entity, combine them into one belief
("The Burj Khalifa has 163 floors and is 828 m tall.") rather than one node per attribute.""",
        'Tool outputs default to "asserted" — they report facts. Use "speculated"/"judged" only if the '
        'tool itself hedges ("according to some sources…").',
    ),
}

# function-role turns are tool outputs.
_ROLE_ALIASES = {"function": "tool"}


def _resolve_role(role: str) -> Optional[str]:
    role = (role or "").strip().lower()
    role = _ROLE_ALIASES.get(role, role)
    return role if role in _GUIDANCE else None


def format_sentences_for_prompt(sentences: List[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def format_clustered_sentences_for_prompt(
    sentences: List[str], clusters: List[List[int]]
) -> str:
    """Render the SAME indexed sentence list, but grouped by topic cluster.
    Indices [k] are GLOBAL across the whole list (unchanged), so the model still
    returns global supporting_sentence_indices. Grouping is presentation only —
    it remains ONE call for the whole content."""
    lines: List[str] = []
    for ci, idxs in enumerate(clusters):
        lines.append(f"--- topic group {ci} ---")
        for i in idxs:
            if 0 <= i < len(sentences):
                lines.append(f"[{i}] {sentences[i]}")
    return "\n".join(lines)


def build_update_prompt(
    role: str,
    *,
    mode: str = "sentences",                 # "sentences" | "excerpt"
    content: Optional[str] = None,
    sentences_block: Optional[str] = None,   # pre-rendered indexed sentence list
    graph_nodes: str = "[]",
    graph_edges: str = "[]",
    current_date: Optional[str] = None,
) -> Optional[str]:
    """Assemble the single-call update prompt. Returns None for an unknown role."""
    key = _resolve_role(role)
    if key is None:
        return None
    task_line, guidance, stance_hint = _GUIDANCE[key]

    parts: List[str] = [
        "# Task",
        task_line,
        "\nYou maintain a belief graph INCREMENTALLY. From the CURRENT turn, output only the NEW "
        "belief nodes and the NEW forward edges. Existing nodes/edges (shown below) must not be repeated.\n",
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
            "Reference them in supporting_sentence_indices; evidence is always a whole sentence. "
            "A single belief MAY cite several sentence indices (even from different topic groups) "
            "when they jointly express one coherent point — consolidate across sentences instead of "
            "forcing one belief per sentence.\n")
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
# Backward (evaluation) linking — full graph, once at trajectory end
# =====================================================================

PROMPT_LINK_BACKWARD_ALL = f"""\
# Task
Below is the COMPLETE belief graph built from one trajectory (all turns, in order).
Ids are chronological: a smaller id was extracted EARLIER. Identify **backward
(evaluation) relations** that explain how a LATER belief changes the *epistemic
status* of an EARLIER one.

## Relation types
- **confirms**: a later belief provides evidence supporting an earlier speculation, recall,
  judgment, or prediction — independent corroboration, or a later observation matching an
  earlier plan/guess.
- **contradicts**: a later belief refutes an earlier one with specific evidence (a changed
  value, a corrected fact, an opposite outcome).
- **extends**: a later belief elaborates / makes more specific an earlier one without
  contradicting it. Use sparingly — only when it truly builds on the earlier belief.

## What to AVOID
- Linking beliefs that merely mention the same entity.
- Linking a PURE restatement by the same source (verbatim repeats are handled by dedup).
  A repeat counts as confirms only when it arrives with NEW grounding (a different source,
  a later time, or an observed outcome matching an earlier plan).
- Weak / speculative relations. When in doubt, leave it out.

## Hard rules
- "from_id" is the LATER belief (the evidence); "to_id" is the EARLIER target.
- "to_id" MUST be strictly less than "from_id".

## Output (JSON only — no markdown fences, no commentary)
{{
  "relations": [
    {{ "from_id": <int later>, "to_id": <int earlier>,
       "type": "confirms" | "contradicts" | "extends",
       "note": "<one short sentence>" }}
  ]
}}

Empty list is fine: {{"relations": []}}.

## Full belief graph
{ALL_BELIEFS_PLACEHOLDER}
"""


# =====================================================================
# Merge / dedup prompts (unchanged contract)
# =====================================================================

_MERGE_RULES = """\
## When do two beliefs express the SAME meaning (mergeable)?
- Same proposition about the same entity with the same value/state and the same time scope,
  merely re-worded ("The user drives a silver Honda Civic." == "The user's car is a silver Honda Civic.").
- Pronoun vs explicit-name variants of the same statement.
- The SAME point expressed at trivially different wording or granularity where NEITHER side adds
  substantive information the other lacks ("The user is researching Hindsight." ==
  "The user has been looking into Hindsight."). If one side adds a real value, qualifier, scope,
  or detail the other lacks, they are NOT duplicates — that difference is "extends", keep both.

## When are they NOT the same (do NOT merge)?
- Different values or quantities (32 mpg city vs 38-40 mpg highway).
- Different time scopes or different events.
- Different aspects of the same entity (owning the car vs the car's color — unless both beliefs state both).
- One is a generalisation of the other, or one adds substantive new info (that is "extends", not duplicate).
- Different subjects (a user's claim vs the assistant's recommendation about the same topic).
- A belief and its negation / correction.

Stance or source differences alone do NOT block a merge when the proposition is identical.
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
- A belief id may appear in at most ONE group.
- Be conservative: when in doubt, do NOT merge.
- Empty list is fine: {{"merge_groups": []}}.

## Full belief list
{BELIEFS_LIST_PLACEHOLDER}
"""
