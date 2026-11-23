import json
from . import run
from .data import DS,BASE,old,rd,sha,key
from .labels import truth
def test_blind_split_and_provenance():
    rows=old.read_rows(DS/"cases.jsonl"); gold=truth()
    assert len(rows)==len(gold)==500
    tasks=lambda group:{r["task"] for r in rows if r["split"]==group}
    assert not tasks("dev")&tasks("test")
    assert len(tasks("test"))==12
    assert all(sha(BASE/k)==v for k,v in rd(DS/"blind.lock.json").items())
def test_weights_do_not_modify_request_or_graph_confidence():
    req={"model":"gpt-6-luna-decision","input":'[{"item":0,"A":"left","B":"right"}]',"questions":[{"name":"item_0_type","instructions":"pick","choices":[{"value":"none","description":"no edge"}]}]}
    assert run.transformed(req,"v6",{})==run.transformed(req,"v4",{})
    a={"choice":"none","probabilities":[{"value":"none","probability":.61},{"value":"supplements","probability":.29},{"value":"depends_on","probability":.08},{"value":"contradicts","probability":.02}]}
    assert run.choose(a,{"none":.35})=="supplements"
    assert run.choose({**a,"choice":"contradicts"},{"none":.2})=="contradicts"
    assert run.choose({"choice":"none"},{"none":.2})=="none"
    assert a["choice"]=="none"
def test_probes_are_scoped_by_item_without_gold():
    req={"input":"[]","questions":[dict(name=f"item_{i}_type") for i in range(2)]}
    answers=[{"name":f"item_{i}_{q}","choice":choice} for i,v in enumerate([["no","yes","yes"],["no","no","yes"]]) for q,choice in zip(("clash","uses","adds"),v)]
    ans=run.combine(req,{"answers":list(reversed(answers))})
    assert [a["choice"] for a in ans["answers"]]==["depends_on","supplements"]
def test_replay_no_holdout_in_version_selection():
    s=rd(BASE/"selection.lock.json")
    assert s["holdout_unopened"] and sha(DS/"blind.lock.json")==s["dataset"]
    assert s["version"]==max(s["dev_metric"],key=s["dev_metric"].get).split("_")[1]
    assert run.spec(s["version"])==s["spec"] and sha(DS/"challenge.jsonl")==s["challenge_hash"]
def test_local_response_drops_stale_encoding_but_preserves_provider_charge():
    import gzip,httpx
    orig=httpx.Response(200,request=httpx.Request("POST","https://localhost/decisions"),headers={"content-encoding":"gzip","x-litellm-response-cost":"0.1"},content=gzip.compress(b'{"answers":[]}'))
    body={"answers":[{"choice":"none"}]};out=run.local_json(orig,body)
    assert out.json()==body and "content-encoding" not in out.headers and out.headers["x-litellm-response-cost"]=="0.1"
