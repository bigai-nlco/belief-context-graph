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
* relation semantics are expressed with the four target edge types:
  ``causal`` | ``depends_on`` | ``supplements`` | ``contradicts``.

Placeholders are filled via str.replace:
    <<<CONTENT>>> <<<SENTENCES>>> <<<GRAPH_NODES>>> <<<GRAPH_EDGES>>>
    <<<CURRENT_DATE>>> <<<CANDIDATE_GROUP>>> <<<BELIEFS_LIST>>>
"""

from __future__ import annotations

from typing import List, Optional


CONTENT_PLACEHOLDER       = "<<<CONTENT>>>"
SENTENCES_PLACEHOLDER     = "<<<SENTENCES>>>"
GRAPH_NODES_PLACEHOLDER   = "<<<GRAPH_NODES>>>"
GRAPH_EDGES_PLACEHOLDER   = "<<<GRAPH_EDGES>>>"
CURRENT_DATE_PLACEHOLDER  = "<<<CURRENT_DATE>>>"
CANDIDATE_GROUP_PLACEHOLDER = "<<<CANDIDATE_GROUP>>>"
BELIEFS_LIST_PLACEHOLDER  = "<<<BELIEFS_LIST>>>"


# =====================================================================
# Shared definitions
# =====================================================================

_BELIEF_DEFINITION = """\
## What is a belief
A belief is a self-contained memory / reasoning unit, usually shaped like:
    <subject, predicate, object/value, scope, time, source>

A belief should be understandable outside the original turn and should preserve
the causal or dependency role it may play in later reasoning.

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
    "If the parser still only accepts informs edges, four-type relations emitted by the prompt will be filtered out."
      → one belief, because the condition and result form one reusable dependency.

- Separate different epistemic status:
    "The logs show no CUDA error, so the failure is probably caused by a missing checkpoint."
      → "The logs show no CUDA error."
      +  "The failure is probably caused by a missing checkpoint."

## Entities
For every belief, output ``entities``: a list of all salient entities explicitly
involved in the belief. Include people, systems, files, variables, models,
datasets, organizations, tools, functions, classes, concepts, and named values
that the node is about. Preserve entity wording exactly when possible.

- Use [] only if no meaningful entity exists.
- Do not include generic filler such as "the content", "this turn", or "the answer"
  unless it is the actual subject under discussion.
"""

_DECISION_DEFINITION = """\
## What is a decision
A decision is the assistant's final answer, selected result, final option, or
explicit final conclusion, especially when it is wrapped in ``\boxed{...}``.

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
These NODES and EDGES were already extracted from EARLIER turns. They are shown so you can:
  - resolve pronouns / vague references in the current turn ("it", "the shop", "that issue"),
  - keep entity names and wording consistent with prior beliefs,
  - decide how NEW nodes from this turn relate to existing nodes.

CRITICAL — do NOT re-emit anything already in the graph:
  - Do NOT output a node that already exists below as a duplicate of an existing node.
  - Do NOT output an edge/relation that already exists below.
  - BUT still extract every claim the CURRENT turn makes, even if it restates an existing
    fact — a restatement becomes a NEW node here; downstream merge/dedup can combine it.

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

Use ONLY these four relation types in the ``relations`` field:

1. **causal**
   A produces, triggers, changes, prevents, enables, or directly explains B.
   Example: "The checkpoint file is missing" causal → "Training cannot resume from that checkpoint".

2. **depends_on**
   A relies on B as a premise, input, assumption, tool result, user constraint, or required context.
   Example: "The proposed prompt-only change" depends_on → "The parser currently accepts only informs edges".

3. **supplements**
   A adds detail, scope, parameters, examples, evidence, or elaboration to B without changing or refuting it.
   Example: "Entities must include functions and files" supplements → "Belief nodes need an entities field".

4. **contradicts**
   A conflicts with, corrects, negates, or replaces B.
   Example: "The model output should use four typed relations" contradicts → "The graph uses generic informs edges".

Direction rule:
- Use the direction that makes the relation sentence natural:
  {"from": A, "to": B, "type": "depends_on"} means A depends on B.
  {"from": A, "to": B, "type": "causal"} means A causes/explains B.
  {"from": A, "to": B, "type": "supplements"} means A supplements B.
  {"from": A, "to": B, "type": "contradicts"} means A contradicts B.

Selection rules:
- Be selective but complete enough to preserve the reasoning chain between turns.
- Do NOT link nodes merely because they share an entity.
- Prefer 0–4 high-value relations per new node; empty list is fine.
- Relation endpoints may be an existing integer id or a new temporary id ("nK" / "dK").
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
    { "from": <existing int id or "nK" or "dK">, "to": <existing int id or "nK" or "dK">, "type": "causal | depends_on | supplements | contradicts", "note": "<one short sentence>" }
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
    { "from": <existing int id or "nK" or "dK">, "to": <existing int id or "nK" or "dK">, "type": "causal | depends_on | supplements | contradicts", "note": "<one short sentence>" }
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
- **Information needs expressed by a tool call** — restate the query as a belief about what the
  assistant is looking for ("The assistant is searching for the Burj Khalifa's floor count.").
  Any hypothesis a query commits to is its own belief with stance "judged" or "speculated".
- **Key reasoning steps that are falsifiable, reusable, or needed by later turns** — keep enough detail to reconstruct causal/dependency chains between user request, tool result, reasoning, and final answer.
- **Tool-use commitments** when they carry semantic content, e.g. what entity/query/constraint the assistant is using.

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
            "Reference them in supporting_sentence_indices; evidence is always a whole sentence.\n")
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
