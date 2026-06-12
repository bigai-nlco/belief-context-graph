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
