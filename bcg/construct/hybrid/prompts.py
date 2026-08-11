"""Prompts for node extraction, incremental relation generation, and merging."""

from __future__ import annotations

import json as _json
from typing import Any

from .._shared.roles import normalize_role

CANDIDATE_GROUP_PLACEHOLDER = "<<<CANDIDATE_GROUP>>>"
BELIEFS_LIST_PLACEHOLDER = "<<<BELIEFS_LIST>>>"

CHUNK_PLACEHOLDER = "<<<CHUNK>>>"
TURN_CONTENT_PLACEHOLDER = "<<<TURN_CONTENT>>>"
GRAPH_NODES_PLACEHOLDER = "<<<GRAPH_NODES>>>"
RELATION_NODES_PLACEHOLDER = "<<<RELATION_NODES>>>"


# =====================================================================
# Generative node-extraction prompt (per semantic chunk, Qwen)
# =====================================================================
# Adapted from the v1 belief-extraction prompt, trimmed to the streaming
# design: the model produces node TEXT and node_type only. Stance is inferred
# separately by the local classifier, entities are attached post-merge by the
# local NER, and event_time is stamped by the graph builder — so stance,
# entities, and time fields are intentionally NOT requested here.


_BELIEF_DEFINITION = """\
## What is a belief
A belief is a self-contained memory / reasoning unit, usually shaped like:
    <subject, predicate, object/value, scope, time, source>
 
A belief should be understandable outside the original turn and should preserve
the causal or dependency role it may play in later reasoning.
 
Preserve the most specific supported wording in the belief itself.
Do NOT generalize a specific claim into a vague claim.
 
## Granularity — coherent units, not tiny shards
Prefer FEWER, more complete beliefs. Do NOT shred one coherent point into many
trivial beliefs. A belief is not forced to be a single clause or sentence. It may
include tightly coupled qualifiers, conditions, reasons, results, or parameters
when separating them would destroy the meaning or the reasoning dependency.
 
Consolidate: when two or more adjacent sentences describe the SAME subject, event,
policy, or claim from different angles (e.g. a statement and an elaboration,
attribution, cause, consequence, or supporting detail of it), combine them into ONE
belief rather than emitting one node per sentence.
  Example — combine, do NOT split:
    "Following the transition to democratic rule, SAPS was intended to adopt
     'human rights policing'. Julia Hornberger's research on Johannesburg SAPS
     found officers struggled to reconcile those ideals with the job's practical
     demands."
  -> one belief: "After democratization, SAPS was meant to adopt 'human rights
     policing', but Julia Hornberger's research on Johannesburg SAPS found officers
     struggled to reconcile that ideal with practical job demands."
 
A belief must satisfy BOTH:
1. Single central semantics — one coherent fact, state, event, constraint,
   hypothesis, reasoning step, recommendation, or intermediate conclusion (it may
   carry its directly-attached elaboration/attribution/cause).
2. Context self-contained — a reader still knows who / what / when / scope.
 
Split ONLY when the content contains genuinely independent propositions that can
be reused, confirmed, contradicted, or linked separately — not merely two sentences
about the same topic.
"""

_DECISION_DEFINITION = """\
## What is a decision
A decision is the assistant's final answer, selected result, final option, or
explicit final conclusion, especially when it is wrapped in ``\\boxed{...}``.
 
Rules:
- Extract ``\\boxed{...}`` final answers as ``decisions`` instead of ordinary beliefs.
- If the assistant gives a final answer without ``\\boxed{...}``, extract it as a
  decision only when it is clearly the final selected answer rather than an
  intermediate claim.
- Do not duplicate the same final answer as both a belief and a decision.
- A decision must be SELF-CONTAINED: never output a bare label, verdict, option,
  or value on its own (e.g. "Supported", "Refuted", "Yes", "Option B", "42").
"""

# Each entry: (task_line, guidance_block). Stance hints removed on purpose.
_GUIDANCE: dict[str, tuple] = {
    "user": (
        "Extract every coherent BELIEF expressed by the USER in the chunk below. "
        "Extract comprehensively, but avoid fragmenting one coherent user intent "
        "or constraint into many tiny nodes.",
        """\
## Source role: USER
Things to extract:
- The user's core request or TASK, and any claim, statement, question, or item
  the user presents for the assistant to verify, answer, classify, summarize, or
  act on — reformulated as a fact about the user ("The user asks to verify the
  claim that South African police recorded 217 torture cases and 3,661 assault
  cases in 2017/2018."). This applies even when the turn is a structured task
  spec (e.g. fields like Claim/Question/Task): capture the substantive task and
  its central claim(s), not the field labels.
- Facts about the user and their world — possessions, attributes, relationships,
  locations ("The user drives a silver Honda Civic.").
- Events the user reports, with their time when stated.
- Preferences, opinions, feelings ("The user prefers synthetic oil.").
- Plans and intentions ("The user plans to rotate the tires next month.").
- Constraints the user imposes ("answer in Chinese", "no code", "use tool X",
  "choose exactly one label from this set") — keep tightly related constraints
  together when they form one instruction.
- Questions / information needs, reformulated as a fact about the user
  ("The user is asking how often to change the oil.").
- Corrections or updates to things said earlier — extract the NEW state.
 
Every USER turn that carries a substantive request or claim should yield at
least one node. Only genuinely contentless turns (bare greetings/acknowledgements)
may yield none.
 
Do NOT infer habits, preferences, patterns, intent, causality, or importance
from a single mention. Extract preferences/plans/intentions only when the chunk
explicitly states them.
 
Write each belief in the third person about "The user" (or the named person /
entity) so it is self-contained. Resolve pronouns using the read-only context
when unambiguous.
 
Skip: pure greetings / pleasantries with no factual content; standalone field
labels or answer-format option lists with no substantive content.""",
    ),
    "assistant": (
        "Extract coherent BELIEFS and DECISIONS from the ASSISTANT chunk below. "
        "The chunk may contain reasoning, tool-call syntax, and/or a final answer "
        "— read all of it and preserve the reasoning chain without creating tiny "
        "redundant nodes.",
        """\
## Source role: ASSISTANT
The chunk may mix internal reasoning, tool invocations, and the final answer. Extract:
- Factual claims and intermediate conclusions the assistant commits to.
- Recommendations / advice given to the user.
- Assessments of the user's situation.
- Final decisions: when the assistant gives a final answer, especially inside
  ``\\boxed{...}``, put it in ``decisions`` instead of ``beliefs``.
- Tool calls: a chunk may be exactly one tool call, e.g.
  ``<tool_call>{"name": "serper_search", "arguments": {"query": "..."}}</tool_call>``.
  For such a chunk, emit EXACTLY ONE belief that renders the call in natural
  language — the tool name, the key argument(s)/query, and the hypothesis/goal it
  checks ("The assistant calls serper_search with the query 'annual rainfall in
  Lisbon 2024' to find an authoritative weather report."). Do NOT copy the raw
  JSON, and do NOT emit the field names
  (name/arguments/query) as separate nodes.
- Key reasoning steps that are falsifiable, reusable, or needed by later turns.
 
Do NOT extract: pure procedure / planning filler ("Let me search next") unless it
encodes a substantive dependency; self-questions; politeness.
 
Consolidation: do not restate the same action, claim, or decision as two
near-identical nodes (e.g. an intent to search AND the search itself, or the same
fact phrased two ways). Emit each distinct point once, in its most complete and
specific form. When the intent to call a tool and the tool call itself fall in the
same chunk, emit only the tool-call node.
 
Write each belief in the third person ("The assistant…", "The user…") so it is
self-contained. Resolve pronouns using the read-only context when unambiguous.""",
    ),
    "tool": (
        "Extract coherent KEY BELIEFS from the TOOL output chunk below (search "
        "results, retrieval, function return). Be selective about boilerplate, but "
        "preserve enough facts to support later reasoning and conclusions.",
        """\
## Source role: TOOL
This is output returned by a tool / function call. Extract ONLY facts that:
1. Directly (or partially) address the user's question or an assistant query.
2. Confirm, contradict, explain, or supplement an earlier assistant hypothesis.
3. Are specific, citeable data points (named entities, dates, relationships, quantities).
4. Are needed to preserve the dependency chain from tool result to later conclusion.
 
Do NOT extract: generic background unrelated to the task; navigation / "see also"
links; boilerplate wrappers; tangential trivia.
 
Write each belief in the third person so it is self-contained.""",
    ),
}


def _resolve_extraction_role(role: str) -> str | None:
    key = (role or "").strip().lower()
    key = normalize_role(key)
    return key if key in _GUIDANCE else None


def format_graph_nodes_context(
    nodes: list[dict[str, Any]],
    *,
    char_budget: int = 9000,
) -> str:
    """Compact existing-NODES context (no relations) for the extraction prompt."""
    if not nodes:
        return "[]"
    ordered = sorted(nodes, key=lambda node: int(node.get("id", 0)))
    lines: list[str] = []
    for node in ordered:
        node_type = node.get("node_type") or "belief"
        text = str(
            node.get("decision")
            if node_type == "decision"
            else node.get("belief") or node.get("belief") or node.get("decision") or ""
        )
        if len(text) > 240:
            text = text[:220] + " ..."
        source = node.get("source") or {}
        item = {
            "id": node.get("id"),
            "node_type": node_type,
            "role": node.get("role") or source.get("role"),
            "turn": source.get("turn_id"),
            "content": text,
        }
        lines.append(_json.dumps(item, ensure_ascii=False))
    total = sum(len(line) + 4 for line in lines)
    omitted = 0
    while lines and total > char_budget:
        total -= len(lines[0]) + 4
        lines.pop(0)
        omitted += 1
    items = [f"  (... {omitted} earlier node(s) omitted ...)"] if omitted else []
    items.extend("  " + line for line in lines)
    return "[\n" + ",\n".join(items) + "\n]"


_EXTRACTION_HARD_CONSTRAINTS = """\
## Hard constraints
1. Preserve named entities, numbers, dates, quantities EXACTLY as written
   (incl. unusual punctuation like "!Kung").
2. Do NOT add information not present in the CHUNK. No outside knowledge.
3. Extract ONLY from the CHUNK. The existing nodes are READ-ONLY context for
   resolving references — never copy them as output.
4. Write each belief/decision in the third person and make it SELF-CONTAINED:
   understandable on its own, without the chunk. Never output a bare label,
   verdict, option, number, or sentence fragment (e.g. "Supported", "Yes", "42",
   "Option B"); always restate the subject it refers to (the claim/question/item).
5. Do NOT emit multiple nodes that express the same fact, action, or decision in
   different words or at different specificity. Output each distinct point once,
   in its most complete and specific form.
6. Empty lists are OK when the chunk expresses none.
"""

_EXCERPT_CONSTRAINT = (
    "7. Every belief/decision MUST include at least one supporting excerpt: a "
    "verbatim, contiguous substring copied character-for-character from the CHUNK.\n"
)


def _output_schema(is_assistant: bool, require_excerpt: bool) -> str:
    exc = (
        ',\n      "supporting_excerpts": ["<verbatim excerpt copied from the chunk>"]'
        if require_excerpt
        else ""
    )
    beliefs = (
        '  "beliefs": [\n'
        '    { "belief": "<self-contained coherent belief>"' + exc + " }\n"
        "  ]"
    )
    head = "## Output (JSON only — no markdown fences, no commentary)\n{\n"
    if not is_assistant:
        return head + beliefs + "\n}\n"
    decisions = (
        '  "decisions": [\n'
        '    { "decision": "<final selected answer, especially content inside \\boxed{...}>"'
        + exc
        + " }\n"
        "  ]"
    )
    return head + beliefs + ",\n" + decisions + "\n}\n"


def build_chunk_extraction_prompt(
    role: str,
    *,
    chunk_text: str,
    graph_nodes: str = "[]",
    turn_content: str | None = None,
    require_excerpt: bool = False,
    max_nodes: int = 0,
) -> str | None:
    """Assemble the per-chunk extraction prompt. Returns None for unknown roles.

    ``graph_nodes`` is a pre-rendered existing-NODES context block (no edges).
    Extraction is chunk-local by default; only historical nodes are given as
    read-only context.

    Optional (all off by default):
    - ``turn_content``: when provided (and different from the chunk), the full
      turn is added as read-only context to help resolve references outside the
      chunk. Controlled by the extractor's ``include_turn_content``.
    - ``require_excerpt``: when true, the output schema requests a verbatim
      supporting excerpt per node and a matching hard constraint is added.
    - ``max_nodes``: when > 0, a soft cap on the number of nodes for this chunk is
      added to the prompt (the extractor also enforces it as a hard backstop).
    """
    key = _resolve_extraction_role(role)
    if key is None:
        return None
    task_line, guidance = _GUIDANCE[key]
    is_assistant = key == "assistant"

    cap_line = ""
    if max_nodes and max_nodes > 0:
        cap_line = (
            f"\nFrom this chunk, extract a total of AT LEAST 1 node and AT MOST {max_nodes} nodes. "
            "If more candidates exist, keep only the most important/complete ones "
            "and consolidate the rest.\n"
        )

    parts: list[str] = [
        "# Task",
        task_line,
        "\nYou maintain a belief graph INCREMENTALLY. From the CHUNK below, output "
        "only the NEW belief/decision nodes it expresses. Relations and duplicate "
        "detection are handled in separate steps — do not output relations."
        + cap_line
        + "\n",
        _BELIEF_DEFINITION,
        guidance + "\n",
    ]
    if is_assistant:
        parts.append(_DECISION_DEFINITION)

    parts.append(
        "## Existing belief nodes (context — READ ONLY, no relations)\n"
        "These nodes were extracted from EARLIER turns. Use them only to resolve "
        "pronouns/vague references and to keep entity names and wording consistent. "
        "Do NOT copy them as output.\n"
        f"{GRAPH_NODES_PLACEHOLDER}\n"
    )
    if (
        turn_content
        and turn_content.strip()
        and turn_content.strip() != chunk_text.strip()
    ):
        parts.append(
            "## Full turn (context — READ ONLY)\n"
            "The chunk below is one part of this turn. Use the full turn only to "
            "resolve references inside the chunk; extract ONLY what the chunk states.\n"
            f"{TURN_CONTENT_PLACEHOLDER}\n"
        )

    constraints = _EXTRACTION_HARD_CONSTRAINTS
    if require_excerpt:
        constraints = _EXTRACTION_HARD_CONSTRAINTS + _EXCERPT_CONSTRAINT
    parts.append(constraints)
    parts.append(_output_schema(is_assistant, require_excerpt))
    parts.append(f"## Chunk to extract from\n{CHUNK_PLACEHOLDER}\n")

    prompt = "\n".join(parts)
    prompt = prompt.replace(GRAPH_NODES_PLACEHOLDER, graph_nodes or "[]")
    if TURN_CONTENT_PLACEHOLDER in prompt:
        prompt = prompt.replace(TURN_CONTENT_PLACEHOLDER, turn_content or "")
    prompt = prompt.replace(CHUNK_PLACEHOLDER, chunk_text or "")
    return prompt


RELATION_PROMPT = f"""\
You are a conservative relation-edge annotator for a belief graph.

Use ONLY these three relation types in the `relations` field:

1. **depends_on**
   A relies on B as a premise, input, assumption, tool result, user constraint,
   or required context.
   Examples:
   - "The assistant calls serper_search with the query 'annual rainfall in Lisbon
     2024'" depends_on "The user asks for Lisbon's 2024 rainfall total".
   - "The assistant concludes that the reported total is 774 mm" depends_on "The
     weather service result lists a 2024 annual total of 774 mm".

2. **supplements**
   A adds detail, scope, parameters, examples, evidence, or elaboration to B
   without changing or refuting it.
   Examples:
   - "The user requests that the answer include both millimetres and inches"
     supplements "The user asks for Lisbon's 2024 rainfall total".
   - "Trump described houses of worship as essential places" supplements
     "President Trump ordered states to reopen houses of worship".

3. **contradicts**
   A conflicts with, corrects, negates, or replaces B.
   Examples:
   - "The store is closed on Sundays" contradicts "The store is open every day".
   - "The deadline has moved to Friday" contradicts "The deadline is Wednesday".

Direction rule:
- `{{"from": A, "to": B, "type": "depends_on"}}` means A depends on B.
- `{{"from": A, "to": B, "type": "supplements"}}` means A supplements B.
- `{{"from": A, "to": B, "type": "contradicts"}}` means A contradicts B.

Necessity gate:
- Do not force an edge merely because two nodes occur in adjacent turns or
  discuss the same topic.
- Create an edge only when one of the three relations is explicit and necessary
  to preserve the reasoning or factual relationship.
- If no pair passes this gate, return an empty `relations` list. A graph window
  with no edges is fully valid.
- Never invent, guess, or add an edge for graph connectivity.
- At least one endpoint of every edge must have `is_current_turn=true`.
- Use only ids from the supplied node list.

Required output schema (JSON only, no markdown or commentary):
{{
  "relations": [
    {{
      "from": 12,
      "to": 5,
      "type": "depends_on",
      "note": "A concise sentence explaining the necessary relation."
    }}
  ]
}}

If no pair passes the necessity gate, return:
{{"relations": []}}

## Nodes in the current two-turn window
{RELATION_NODES_PLACEHOLDER}
"""


def build_relation_prompt(nodes: list[dict[str, Any]], current_node_ids: set) -> str:
    """Render one complete two-turn relation window for the edge model."""
    payload = []
    for node in sorted(nodes, key=lambda item: int(item.get("id", 0))):
        source = node.get("source") or {}
        node_type = node.get("node_type") or "belief"
        payload.append(
            {
                "id": node.get("id"),
                "turn": source.get("turn_id"),
                "role": node.get("role") or source.get("role") or source.get("type"),
                "node_type": node_type,
                "content": node.get("decision")
                if node_type == "decision"
                else node.get("belief"),
                "is_current_turn": node.get("id") in current_node_ids,
            }
        )
    return RELATION_PROMPT.replace(
        RELATION_NODES_PLACEHOLDER,
        _json.dumps(payload, ensure_ascii=False, indent=2),
    )
