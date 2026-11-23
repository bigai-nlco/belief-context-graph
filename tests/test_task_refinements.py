"""Task hierarchy is a sourced distinction, not an automatic edge to a referenced ID."""
import copy
import json

import pytest

from bcg.construct.unified import extract, stream
from bcg.construct.unified.prompts import build_node_extraction_prompt
from bcg.construct.unified.task_state import build_task_state, classify_node, task_history


def anchor(i, text, **kwargs):
    return {"id": i, "belief": text, "task_role": "task", "role": "user",
            "supporting_excerpts": [text], **kwargs}


# Actual extraction and cited assistant sentence from RUN-0028 / drb2:2.
PARENT = anchor(2, "The user wants a report identifying specific social protection programs in India, Pakistan, Bangladesh, and Nepal and describing their failures in practice, covering India's 'Mahatma Gandhi National Rural Employment Guarantee Scheme,' Pakistan's 'Zakat program,' Bangladesh's 'Challenging the Frontiers of Poverty: Targeting the Ultra-Poor' project, and as many other significant programs as possible.")
PARENT["supporting_excerpts"] = ["Part One: I would like you to identify specific social protection programs in India, Pakistan, Bangladesh, and Nepal, and describe in detail their failures in practice."]
REPEAT = "The assistant's report must identify programs in India, Pakistan, Bangladesh, and Nepal and describe their failures."
SOURCE = "1. Part One: Identify specific programs in India, Pakistan, Bangladesh, and Nepal and describe their failures"


def proof(parent=2, kind="narrower_scope", wider="Pakistan", focus="Nepal only"):
    return {"parent_id": parent, "kind": kind, "parent_quote": wider, "child_quote": focus}


def replay(monkeypatch, claim, sentences, annotations, parents, shown=None, mode="sentence"):
    """Real parser + source-scoped classifier + TSG projection; only the LLM is stubbed."""
    response = {"belief": claim, "stance": "asserted", "entities": [], **annotations}
    response.update({"supporting_excerpts": [sentences[0]]} if mode == "excerpt" else {"supporting_sentence_indices": [0]})
    calls = []
    def respond(_client, _model, prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({"beliefs": [response]}, ensure_ascii=False)
    monkeypatch.setattr(extract.llm, "call_model", respond)
    out = extract.extract_nodes(None, "fixture", role="assistant", mode=mode,
                                sentences=sentences, content="\n".join(sentences),
                                graph_nodes_str=task_history([], parents, 12000) if shown is None else json.dumps(shown),
                                task_state_enabled=True)
    assert len(calls) == 1 and "task_refinements" in calls[0]
    item, = out["beliefs"]
    matched = item["supporting_excerpts"] if mode == "excerpt" else [sentences[i] for i in item["supporting_sentence_indices"]]
    child = {"id": 234, "node_type": "belief", "belief": item["belief"], "role": "assistant", "supporting_excerpts": matched}
    classify_node(child, item, parents)
    state = build_task_state({"nodes": [*parents, child], "relations": []})
    return child, state, [(r["from_id"], r["to_id"]) for r in state["relations"] if r["type"] == "part_of"]


@pytest.mark.parametrize("annotations", [
    {"task_role": "other", "task_ids": [2]},  # A correctly recognized restatement.
    {"task_role": "task", "task_ids": [2]},   # The original model output, unchanged.
    {"task_role": "task", "task_ids": [2], "task_refinements": [proof(focus="specific programs in India")]},
    {"task_role": "task", "task_ids": [2], "task_refinements": [proof(kind="new_deliverable", focus="Nepal")]},
    {"task_role": "task", "task_ids": [2], "task_refinements": [proof(wider=PARENT["belief"], focus=SOURCE)]},
    {"task_role": "task", "task_ids": [2], "task_refinements": [proof(kind="new_deliverable", wider="social protection programs", focus="describe their failures")]},
])
def test_recorded_restatement_is_a_belief_addressing_T2_not_another_task(monkeypatch, annotations):
    original = copy.deepcopy(PARENT)
    child, state, links = replay(monkeypatch, REPEAT, [SOURCE], annotations, [PARENT])
    assert original == PARENT
    assert child["belief"] == REPEAT and child["supporting_excerpts"] == [SOURCE]
    assert child["task_role"] == "other" and child["task_ids"] == [2]
    assert not links and [n["id"] for n in state["nodes"]] == ["T2"]
    assert {"from_id": "B234", "to_id": "T2", "type": "addresses"} in state["relations"]
    assert state["nodes"][0]["status_source_id"] is None


@pytest.mark.parametrize("mode", ["sentence", "excerpt"])
@pytest.mark.parametrize("source,witness", [
    ("Investigate program failures in Nepal only.", proof()),
    ("Investigate failures in Nepal during 2018–2020.", proof(wider="Nepal", focus="2018–2020")),
    ("为报告制作各地区延迟发放工资的对照表。", proof(kind="new_deliverable", wider="social protection programs", focus="各地区延迟发放工资的对照表")),
])
def test_real_scope_or_new_output_keeps_child_and_auditable_quotes(monkeypatch, mode, source, witness):
    child, state, links = replay(monkeypatch, source, [source],
        {"task_role": "task", "task_ids": [2], "task_refinements": [witness]}, [PARENT], mode=mode)
    assert links == [("T234", "T2")] and child["task_refinements"] == [witness]
    assert state["nodes"][-1]["parent_ids"] == ["T2"]
    edge, = [e for e in state["relations"] if e["type"] == "part_of"]
    assert edge["basis"] == {k: v for k, v in witness.items() if k != "parent_id"}


def test_each_parent_needs_its_own_grounds_in_the_shown_text_and_the_actual_cited_sentence(monkeypatch):
    parents = [PARENT, anchor(4, "Investigate health records."), anchor(5, "Study enrollment.")]
    shown = [{"id": n["id"], "task_role": "task", "content": n["belief"]} for n in parents]
    shown[1]["content"] = "Investigate"  # 'health' exists in storage but was not shown to the model.
    claim = "Measure failures in Nepal only."
    child, _, links = replay(monkeypatch, claim, [claim, "Build an unrelated map."],
        {"task_role":"task", "task_ids":[2,4,5,999], "task_refinements":[proof(), proof(4,wider="health"),
            proof(5,"new_deliverable","enrollment","unrelated map"), proof(999)]}, parents, shown)
    assert child["task_ids"] == [2] and links == [("T234","T2")]
    assert child["task_rejected_ids"] == [999] and child["task_unproven_parent_ids"] == [4,5]


@pytest.mark.parametrize("refs", [[999], ["2"], [True]])
def test_rejected_parent_claim_does_not_become_a_root(monkeypatch, refs):
    child, state, links = replay(monkeypatch, REPEAT, [SOURCE], {"task_role":"task","task_ids":refs}, [PARENT])
    assert not links and len(state["nodes"]) == 1 and child["task_role"] == "other" and child["task_ids"] == []


def test_identical_unlinked_outcome_is_not_a_second_root_but_an_independent_task_is(monkeypatch):
    claim = PARENT["belief"]
    child, _, links = replay(monkeypatch, claim, [claim], {"task_role":"task","task_ids":[]}, [PARENT])
    assert not links and child["task_role"] == "other" and child["task_ids"] == [2]
    claim = "Prepare a separate report on rural transport."
    child, state, links = replay(monkeypatch, claim, [claim], {"task_role":"task","task_ids":[]}, [PARENT])
    assert not links and child["task_role"] == "task" and len(state["nodes"]) == 2


@pytest.mark.parametrize("history", ["[]", "not JSON", '[{"id":2,"task_role":"plan"}]'])
def test_initial_or_unaddressable_history_needs_no_parent_proof_prompt(history):
    prompt = build_node_extraction_prompt("user", graph_nodes=history, task_state_enabled=True)
    assert "Task-state annotations" in prompt and "task_refinements" not in prompt


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("mode", ["sentences", "excerpt"])
def test_output_schema_places_witness_inside_belief_not_in_decisions_or_the_top_level(role, mode):
    hist = task_history([], [PARENT], 12000)
    for enabled in (False, True):
        prompt = build_node_extraction_prompt(role, mode=mode, graph_nodes=hist, task_state_enabled=enabled)
        block = prompt.split("## Output (JSON only", 1)[1].split("\n", 1)[1].split("## Current turn", 1)[0]
        schema = json.loads(block)
        assert ("task_refinements" in schema["beliefs"][0]) is enabled
        assert "task_refinements" not in schema and not any("task_refinements" in d for d in schema.get("decisions", []))
        assert "{TASK_STATE_FIELDS}" not in prompt


def test_actual_stream_persists_only_grounded_children_and_preserves_status_updates(monkeypatch, tmp_path):
    sources = ["Write a report about regional programs.", "Document failures in India and Nepal.", "The report must document failures in India and Nepal.",
               "Document failures in Nepal only.", "Work on the Nepal section is blocked."]
    labels = [
        {"task_role": "task"}, {"task_role": "task", "task_scope": "root", "root_refinement": {
            "kind": "new_deliverable", "parent_quote": "report about regional programs",
            "child_quote": "Document failures in India and Nepal"}}, {"task_role": "task", "task_ids": [1]},
        {"task_role": "task", "task_ids": [1], "task_refinements": [proof(1, wider="India")]},
        {"task_role": "progress", "task_status": "blocked", "task_ids": [3]}]
    calls = []
    def respond(_client, _model, prompt, **kwargs):
        i = len(calls)
        calls.append(prompt)
        return json.dumps({"beliefs": [
            {"belief": sources[i], "stance":"asserted", "entities":[], "supporting_sentence_indices":[0], **labels[i]}]})
    monkeypatch.setattr(extract.llm, "call_model", respond)
    monkeypatch.setattr(stream, "extract_layered_relations", lambda *a, **kw: {"relations": []})
    monkeypatch.setattr(stream, "extract_relations", lambda *a, **kw: {"relations": []})
    builder = stream.StreamingBeliefBuilder(client=None, model="fixture", item_id="refinements", out_dir=tmp_path,
        options=stream.StreamOptions(task_state_enabled=True, incremental_merge=False))
    for i, source in enumerate(sources):
        builder.ingest_turn("assistant" if i else "user", source)
    assert len(calls) == len(sources) and "task_refinements" not in calls[0]
    builder.finalize()
    saved = json.loads((tmp_path / "final_graph.json").read_text())
    assert [n["belief"] for n in saved["nodes"]] == sources and len(saved["nodes"][1]["evidence_ids"]) == 1
    state = saved["task_state"]
    assert [(t["id"], t["status"]) for t in state["nodes"]] == [("T1","open"),("T3","blocked")]
    edge, = [e for e in state["relations"] if e["type"] == "part_of"]
    assert (edge["from_id"], edge["to_id"], edge["basis"]["parent_quote"]) == ("T3","T1","India")
    assert {"from_id":"B2","to_id":"T1","type":"addresses"} in state["relations"]


@pytest.mark.parametrize("focus,claim", [
    ("assess wheelchair accessibility at Amber Library only", "Assess wheelchair accessibility at Amber Library only as one subtask of the comparison."),
    ("produce a measured floor plan of the entrance ramp at Amber Library", "The assistant is tasked with producing a measured floor plan of the entrance ramp at Amber Library as a component of the wheelchair-accessibility comparison."),
])
def test_replay_gpt6_luna_positive_refinement_witnesses(monkeypatch, focus, claim):
    # Witnesses actually returned by GPT-6-Luna on the invented library tasks.
    parent = anchor(7, "Compare wheelchair accessibility at the fictional Amber Library and Cobalt Library.")
    witness = proof(7, wider=parent["belief"], focus=focus)
    _, _, links = replay(monkeypatch, claim, [focus], {"task_role":"task","task_ids":[7],"task_refinements":[witness]}, [parent])
    assert links == [("T234","T7")]
