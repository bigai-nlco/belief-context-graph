import json
from types import SimpleNamespace
from .native_stages import tool_stage
from .replay import change,select_unit
from .score import quality

def test_production_tool_prompt_does_not_inherit_obsolete_stances():
    class Rec:
        graph_model="gpt-6-luna"
        calls=[]
        def chat(self,p,limit):
            assert "almost_certain" in p and "asserted" not in p and "source_rank" in p
            assert '"item_index": 1' in p and "real input snippet" in p
            return {"items":[{"item_index":1,"beliefs":[{"belief":"Fact as seen.","stance":"possible","entities":[],"source_rank":1}]}]}
    ctx={"items":[{"item_index":1,"query":"original query","tool_name":"web_search","results":[{"rank":1,"title":"real input snippet"}]}]}
    result=tool_stage(ctx,Rec(),None)
    assert result["items"][0]["beliefs"][0]["source_rank"]==1
    assert result["items"][0]["beliefs"][0]["stance"]=="possible"

def test_gold_independent_alignment_and_missing_denominator():
    n={"belief":"A identifies a significant difference","supporting_sentence_indices":[1]}
    d={"text":n["belief"],"indices":[1]}
    assert select_unit([n],d)[0]==n
    assert select_unit([{**n,"supporting_sentence_indices":[4]}],d)[0] is None
    score=quality([{"group":"node","gold":{"stance":"possible","task_role":"evidence"},"ambiguous":["task_role"],"prediction":{}},
                   {"group":"edge","gold":{"type":"none"},"ambiguous":[],"prediction":{"type":"none"}}])
    assert score["n"]==2 and score["correct"]==1 and score["group_balanced_macro_f1"]==.5

def test_versions_change_only_choice_explanations():
    payload={"model":"m","input":"frozen","questions":[{"name":"item_0_kind","instructions":"old","choices":[{"value":"none","description":"old desc"}]}]}
    edited=change(payload,{"kind":{"instruction":"more","choices":{"none":"new desc"}}})
    assert edited["input"]==payload["input"] and edited["questions"][0]["choices"][0]["value"]=="none"
    assert payload["questions"][0]["choices"][0]["description"]=="old desc"
