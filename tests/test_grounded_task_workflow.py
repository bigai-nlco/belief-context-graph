"""Source attribution regressions for the task-2 runs, independent of provider randomness."""
import json

import pytest

from bcg.construct.unified.prompts import build_node_extraction_prompt
from bcg.construct.unified.task_state import (
    build_task_state,
    classify_node,
    clean_annotation,
    restrict_to_shown_tasks,
    shown_workflow,
    task_history,
)

ROOT = "Research social protection programs in India, Pakistan, Bangladesh and Nepal and create a summary table."
GAP = "found mixed information with some positive effects reported, but I need to search more for failures"
ACTION = "focusing on self-exclusion mechanisms and invisible design barriers that perpetuate exclusion"


@pytest.mark.parametrize("mode", ["excerpt", "sentences"])
@pytest.mark.parametrize("parents,gaps", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_response_shape_requires_roles_and_exposes_only_grounded_reference_fields(mode, parents, gaps):
    rows = [{"ref": "ROOT", "content": "Compile the regional reports."}]
    if parents:
        rows.append({"id": 7, "content": "Check local coverage.", "task_role": "task"})
    if gaps:
        rows.append({"id": 8, "content": "The date is missing.", "gap_quote": "date is missing"})
    prompt = build_node_extraction_prompt("assistant", mode=mode, content="A current observation.",
        sentences_block="[0] A current observation.", graph_nodes=json.dumps(rows), task_state_enabled=True)
    example = json.loads(prompt.split("## Output (JSON only — no markdown fences, no commentary)\n", 1)[1]
                         .split("## Current turn", 1)[0].strip())
    belief, = example["beliefs"]
    assert {"task_role", "task_scope", "gap_quote", "next_action_quote", "root_refinement", "task_status"} <= belief.keys()
    assert ("task_ids" in belief) == parents and ("task_refinements" in belief) == parents
    assert ("closes_gaps" in belief) == gaps and "gap_quote" not in example["decisions"][0]


def test_list_introduction_is_not_a_displayable_next_step():
    for header in ("Items to investigate:", "下一步需要核实的是："):
        node = label(14, header, {"task_role": "plan", "task_scope": "root", "next_action_quote": header})
        assert node["task_role"] == "plan" and "next_action_quote" not in node


def label(i, source, raw, previous=(), root=ROOT):
    history = task_history(list(previous), list(previous), 12000, original_task=root)
    shown, gaps = shown_workflow(history)
    ann = clean_annotation(raw)
    restrict_to_shown_tasks([ann], {n["id"]: n["belief"] for n in previous if n["task_role"] == "task"},
                            root=shown, gaps=gaps)
    node = {"id": i, "belief": source, "role": "assistant", "supporting_excerpts": [source]}
    classify_node(node, ann, list(previous), original_task=root)
    return node


def test_opening_restatement_and_explicit_choice_have_different_outcomes():
    restated = label(9, "Analyze all four countries and make the requested table.", {"task_role": "task"})
    assert restated["task_role"] == "other" and restated["belief"]
    cherry_picked = label(11, ROOT, {"task_role": "task", "task_scope": "root", "root_refinement": {
        "kind": "narrower_scope", "parent_quote": "Pakistan", "child_quote": "Nepal"}})
    assert cherry_picked["task_role"] == "other"
    forged = label(24, "Write the requested report.", {"task_role": "task", "task_scope": "root",
        "root_refinement": {"kind": "narrower_scope", "parent_quote": "Bhutan", "child_quote": "report"}})
    assert forged["task_role"] == "other"
    child = label(40, "First, independently verify the evidence of BRAC exclusion in Bangladesh.",
                  {"task_role": "task", "task_scope": "root", "root_refinement": {
                      "kind": "narrower_scope", "parent_quote": "Pakistan",
                      "child_quote": "evidence of BRAC exclusion in Bangladesh"}})
    assert child["task_role"] == "task" and child["root_refinement"]["parent_quote"] == "Pakistan"


def test_root_gap_and_plan_are_independent_and_only_a_cited_same_scope_resolution_closes():
    fact = label(41, "Bangladesh's BRAC/CTFP program - " + GAP,
                 {"task_role": "evidence", "task_scope": "root", "gap_quote": GAP})
    plan = label(134, "I'll continue searching, " + ACTION,
                 {"task_role": "plan", "task_scope": "root", "next_action_quote": ACTION})
    orphan = label(135, "Search other sources.", {"task_role": "plan", "task_ids": [999], "task_scope": "root"})
    bad = label(136, "A new result.", {"task_role": "evidence", "task_scope": "root",
                "closes_gaps": [{"id": 41, "quote": "resolved the specific question"}]}, [fact])
    other = label(137, "Verified the failure example.", {"task_role": "evidence",
                  "closes_gaps": [{"id": 41, "quote": "Verified the failure example"}]}, [fact])
    nodes = [fact, plan, orphan, bad, other]
    state = build_task_state({"nodes": nodes})
    assert not state["nodes"] and state["root"]["open_gap_ids"] == [41]
    assert state["root"]["plan_ids"] == [134] and state["unassigned_plan_ids"] == [135]
    assert not state["gap_closures"] and not state["root"]["update_ids"]
    resolution = label(138, "The exclusion example is now documented.", {"task_role": "evidence",
        "task_scope": "root", "closes_gaps": [{"id": 41, "quote": "exclusion example is now documented"}]}, nodes)
    nodes = [*nodes, resolution]
    state = build_task_state({"nodes": nodes})
    assert not state["root"]["open_gap_ids"] and state["gap_closures"] == [{"gap_id": 41, "source_id": 138}]
    hist = task_history(nodes, nodes, 2500, original_task=ROOT)
    assert 41 not in shown_workflow(hist)[1] and json.loads(hist)[0]["ref"] == "ROOT"


def test_scope_does_not_override_explicit_task_ids_or_original_budget():
    h = task_history([], [], 160, original_task=ROOT * 100)
    assert len(h) <= 160 and shown_workflow(h)[0]
    assert "ROOT" not in task_history([], [], 160)
    n = label(1, "I still lack original records.", {"task_role": "evidence", "task_scope": "root",
              "gap_quote": "a completely different issue"})
    assert "gap_quote" not in n


def test_global_only_state_is_rendered_without_creating_a_task_or_a_factual_edge():
    from bcg.construct.unified.task_state import render_task_state
    first = label(41, "BRAC: " + GAP, {"task_role": "evidence", "task_scope": "root", "gap_quote": GAP})
    second = label(134, "Next: " + ACTION, {"task_role": "plan", "task_scope": "root", "next_action_quote": ACTION})
    graph = {"nodes": [first, second], "relations": []}
    graph["task_state"] = build_task_state(graph)
    text, ids, edges, tasks = render_task_state(graph, 1800)
    assert GAP in text and ACTION in text
    assert text.index("### Open gaps") < text.index("### Latest plan") and "Subtasks" not in text
    assert ids == [41, 134] and edges == tasks == []
    assert render_task_state(graph, 10) == ("", [], [], [])


def test_shared_progress_defined_once_and_parents_are_readable():
    from bcg.construct.unified.task_state import render_task_state
    a = {"id": 9, "belief": "Independently check coverage.", "task_role": "task"}
    b = {"id": 24, "belief": "Check rural coverage.", "task_role": "task", "task_ids": [9]}
    u = {"id": 104, "belief": "Two independent coverage records were reviewed.",
         "task_role": "progress", "task_ids": [9, 24], "task_status": "in_progress"}
    graph = {"nodes": [a, b, u], "relations": []}
    graph["task_state"] = build_task_state(graph)
    text, ids, _, tasks = render_task_state(graph, 1800)
    assert text.count(u["belief"]) == 1 and text.count("[B104]") >= 2
    assert "  - [T9] in progress (reported)" in text and "    - [T24] in progress (reported)" in text
    assert set(tasks) == {"T9", "T24"} and ids == [9, 24, 104]


def test_unassigned_gap_keeps_subject_sentence_without_inventing_subtask_or_blocker():
    from bcg.construct.unified.task_state import render_task_state

    existing_task = {"id": 38, "belief": "Check rural registry records.", "task_role": "task"}
    brac_sentence = ("3. Bangladesh's BRAC/CTFP program - found mixed information with some positive "
                     "effects reported, but I need to search more for failures")
    gap_node = label(55, brac_sentence, {"task_role": "evidence", "gap_quote": "I need to search more for failures"},
                     previous=[existing_task])
    bad_ref = label(56, "Another note - still missing records",
                    {"task_role": "evidence", "task_ids": [999], "gap_quote": "still missing records"},
                    previous=[existing_task])
    graph = {"nodes": [existing_task, gap_node, bad_ref], "relations": []}
    state = graph["task_state"] = build_task_state(graph)
    assert state["version"] == 4
    assert state["unassigned_gap_ids"] == [55]
    assert state["root"]["open_gap_ids"] == []
    assert state["nodes"][0]["open_gap_ids"] == [] and state["nodes"][0]["status"] == "open"
    text, ids, _, tasks = render_task_state(graph, 1800)
    assert "### Open gaps (evidence still missing)" in text
    assert "- [B55] Bangladesh's BRAC/CTFP program - found mixed information" in text  # list marker stripped
    assert "I need to search more for failures" in text and "blocked" not in text
    assert 55 in ids and 56 not in ids and tasks == ["T38"]

    # Subsequent extraction sees the unassigned gap and can resolve it when cited explicitly.
    hist = task_history([gap_node], [existing_task, gap_node], 12000, original_task=ROOT)
    assert 55 in shown_workflow(hist)[1]
    closed = label(60, "Documented BRAC/CTFP ultra-poor exclusion failures from field audits.",
                   {"task_role": "evidence", "closes_gaps": [
                       {"id": 55, "quote": "Documented BRAC/CTFP ultra-poor exclusion failures"}]},
                   previous=[existing_task, gap_node])
    after = build_task_state({"nodes": [existing_task, gap_node, closed], "relations": []})
    assert after["unassigned_gap_ids"] == []
    assert after["gap_closures"] == [{"gap_id": 55, "source_id": 60}]


def test_concrete_next_steps_survive_generic_transition_filler_in_english_and_cjk():
    from bcg.construct.unified.task_state import render_task_state

    plans = [
        label(10, "1. Search for more India programs (like PMAY, NFSA, etc.)",
              {"task_role": "plan", "task_scope": "root",
               "next_action_quote": "Search for more India programs (like PMAY, NFSA, etc.)"}),
        label(11, "2. Get more specific information about BRAC/CTFP failures",
              {"task_role": "plan", "task_scope": "root",
               "next_action_quote": "Get more specific information about BRAC/CTFP failures"}),
        label(12, "I'll continue searching, " + ACTION,
              {"task_role": "plan", "task_scope": "root", "next_action_quote": ACTION}),
        label(13, "Continue searching for more specific program details",
              {"task_role": "plan", "task_scope": "root",
               "next_action_quote": "Continue searching for more specific program details"}),
        label(14, "Let me read the report data file.",
              {"task_role": "plan", "task_scope": "root", "next_action_quote": "read the report data file"}),
    ]
    for p in plans:
        p["source"] = {"turn_id": 7}  # one assistant turn
    graph = {"nodes": plans, "relations": []}
    graph["task_state"] = build_task_state(graph)
    assert plans[0]["task_workflow_version"] == 4
    text, ids, _, _ = render_task_state(graph, 1800)
    assert ids == [10, 11, 12]
    assert "PMAY, NFSA" in text and "BRAC/CTFP failures" in text and "self-exclusion mechanisms" in text
    assert "read the report data file" not in text

    # CJK concrete plans are kept while pure CJK boilerplate is deprioritized.
    cjk_nodes = [
        label(20, "下一步调查尼泊尔社会保障项目的排斥机制。",
              {"task_role": "plan", "task_scope": "root", "next_action_quote": "调查尼泊尔社会保障项目的排斥机制"}),
        label(21, "接下来继续搜索更多相关信息。",
              {"task_role": "plan", "task_scope": "root", "next_action_quote": "继续搜索更多相关信息"}),
    ]
    for p in cjk_nodes:
        p["source"] = {"turn_id": 3}
    cjk_graph = {"nodes": cjk_nodes, "relations": []}
    cjk_graph["task_state"] = build_task_state(cjk_graph)
    cjk_text, cjk_ids, _, _ = render_task_state(cjk_graph, 1800)
    assert cjk_ids == [20] and "调查尼泊尔社会保障项目的排斥机制" in cjk_text

    # When the latest turn has only generic plans, show them rather than dropping the section.
    only_generic = {"nodes": plans[3:], "relations": []}
    only_generic["task_state"] = build_task_state(only_generic)
    fallback_text, fallback_ids, _, _ = render_task_state(only_generic, 1800)
    assert fallback_ids == [13, 14] and "read the report data file" in fallback_text

    # A newer turn's plan retires the older turn's intentions.
    newer = label(30, "Fetch the Nepal ILO scoping report.", {"task_role": "plan", "task_scope": "root",
                  "next_action_quote": "Fetch the Nepal ILO scoping report"})
    newer["source"] = {"turn_id": 9}
    later = {"nodes": [*plans, newer], "relations": []}
    later["task_state"] = build_task_state(later)
    assert render_task_state(later, None)[1] == [30]


def test_task_history_root_cap_covers_3800_char_prompts_within_12000_budget():
    long_prompt = ("Investigate regional social protection schemes. " * 80)[:3800]
    hist = json.loads(task_history([], [], 12000, original_task=long_prompt))
    assert hist[0]["ref"] == "ROOT" and hist[0]["content"] == long_prompt
    assert "[…]" not in hist[0]["content"]
    over_cap = ("Investigate regional social protection schemes. " * 100)[:4500]
    hist_over = json.loads(task_history([], [], 12000, original_task=over_cap))
    assert "[…]" in hist_over[0]["content"] and len(json.dumps(hist_over, ensure_ascii=False)) <= 12000


def test_grouped_sections_deduplicate_near_identical_items_and_drop_generic_gaps():
    from bcg.construct.unified.task_state import render_task_state

    items = [
        label(65, "3. I have some findings but need more detailed information on specific failures and exclusions",
              {"task_role": "progress", "task_scope": "root", "task_status": "in_progress",
               "gap_quote": "3. I have some findings but need more detailed information on specific failures and exclusions"}),
        label(125, "Search for Nepal programs (Bisketihar, NPSF, child protection programs)",
              {"task_role": "plan", "task_scope": "root",
               "next_action_quote": "Search for Nepal programs (Bisketihar, NPSF, child protection programs)"}),
        label(145, "Now I need to search for more specific information to fill gaps.",
              {"task_role": "progress", "task_scope": "root", "task_status": "in_progress",
               "gap_quote": "Now I need to search for more specific information to fill gaps."}),
        label(161, "I'm facing challenges accessing information on Pakistan's social protection programs.",
              {"task_role": "progress", "task_scope": "root", "task_status": "in_progress",
               "gap_quote": "I'm facing challenges accessing information on Pakistan's social protection programs."}),
        label(162, "deepening research on MGNREGA's exclusion patterns and examining BRAC's social dynamics",
              {"task_role": "plan", "task_scope": "root",
               "next_action_quote": "deepening research on MGNREGA's exclusion patterns and examining BRAC's social dynamics"}),
        label(173, "1. Specific exclusion cases for MGNREGA",
              {"task_role": "progress", "task_scope": "root", "task_status": "in_progress",
               "gap_quote": "1. Specific exclusion cases for MGNREGA"}),
        label(174, "Specific exclusion cases for MGNREGA",
              {"task_role": "plan", "task_scope": "root",
               "next_action_quote": "Specific exclusion cases for MGNREGA"}),
        label(185, "1. More specific MGNREGA exclusion cases",
              {"task_role": "progress", "task_scope": "root", "task_status": "in_progress",
               "gap_quote": "1. More specific MGNREGA exclusion cases"}),
        label(186, "1. More specific MGNREGA exclusion cases",
              {"task_role": "plan", "task_scope": "root",
               "next_action_quote": "1. More specific MGNREGA exclusion cases"}),
    ]
    for item in items:
        item["source"] = {"turn_id": item["id"] // 10}
    graph = {"nodes": items, "relations": []}
    graph["task_state"] = build_task_state(graph)
    text, ids, _, _ = render_task_state(graph, None)
    status_pos, gap_pos, plan_pos = (text.index(h) for h in (
        "### Task status and constraints", "### Open gaps (evidence still missing)", "### Latest plan"))
    assert status_pos < gap_pos < plan_pos
    # Latest root progress is the overall status; its leading "1. " is stripped.
    assert "- Overall task (the original user request): in progress (reported) [B185] — " \
           "More specific MGNREGA exclusion cases" in text
    # Generic filler gaps (B65, B145) are dropped; concrete ones keep full sentences.
    assert 65 not in ids and 145 not in ids
    assert "- [B161] I'm facing challenges accessing information on Pakistan's social protection programs." in text
    assert "- [B173] Specific exclusion cases for MGNREGA" in text[gap_pos:plan_pos]
    # Only the most recent turn's plan is current; older intentions retire.
    assert "- [B186] More specific MGNREGA exclusion cases" in text[plan_pos:]
    assert not {125, 162, 174} & set(ids)
