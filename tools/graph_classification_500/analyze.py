"""Deterministic secondary measures; no endpoint use, no fitting on test."""
from collections import Counter,defaultdict
from .grade import *
from .run import replay,llm
from .repair import lane
def propose(arm,c,o):
    if o["status"]!="ok" or c["group"]=="node":return None
    chats=[a for a in o["calls"] if a["provider"]=="chat" and a.get("response",{}).get("choices")]
    if not chats:return None
    try:body=llm.parse_json_response(chats[-1]["response"]["choices"][0]["message"]["content"])
    except (ValueError,TypeError,KeyError):return None
    d=c["payload"]
    if c["group"]=="edge":
        return any((e.get("from"),e.get("to"))==(d["from_id"],d["to_id"]) for e in body.get("relations",[]) if isinstance(e,dict))
    if arm=="native_current":nodes=[*body.get("beliefs",[]),*body.get("decisions",[])]
    else:nodes=[{**n,"belief":n.get("text") or n.get("belief") or n.get("decision")} for n in body.get("nodes",[]) if isinstance(n,dict)]
    n,_=replay.select_unit(nodes,d)
    return None if n is None else any(z.get("parent_id")==d["parent_id"] for z in (n.get("task_refinements") or []) if isinstance(z,dict))
def analyze():
    out=BASE/"analysis";out.mkdir(exist_ok=True);g=truth()
    allrows=old.read_rows(DS/"cases.jsonl");cases={c["id"]:c for c in allrows}
    armset={"dev":["native_current","chat_fixed",*(f"decision_v{x}" for x in (1,4,5,6,7))],
            "test":["native_current","chat_fixed","chat_v6","decision_v1","decision_v4","decision_v6","pipeline_v6"],
            "regression":["native_current","chat_fixed","decision_v1","decision_v4","decision_v6"],
            "challenge":["chat_fixed","chat_v6","decision_v1","decision_v4","decision_v6"]}
    summaries={s:{a:score(a,s) for a in arms} for s,arms in armset.items()}
    assert all(k["observed"]==k["expected"] for sets in summaries.values() for k in sets.values())
    old.dump(out/"all_scores.json",summaries)
    rows={a:read(a,"test")[2] for a in armset["test"]};sens={}
    for a,rr in rows.items():sens[a]=scoring.quality([{**r,"ambiguous":[]} for r in rr])
    old.dump(out/"ambiguity_sensitivity.json",sens)
    cis={a+"__"+b:bootstrap(rows[a],rows[b]) for a,b in [
        ("decision_v6","decision_v4"),("decision_v6","decision_v1"),("decision_v6","chat_v6"),("decision_v6","chat_fixed"),("pipeline_v6","native_current")]}
    old.dump(out/"paired_task_cluster_ci.json",cis)
    stage=[];counts=defaultdict(Counter)
    for arm in ("native_current","pipeline_v6"):
        for r in rows[arm]:
            c=cases[r["id"]];f={"edge":"type","parent":"kind"}.get(c["group"])
            if f is None or f in r["ambiguous"]:continue
            o=rd(lane(arm,"test")/(r["id"]+".json"));p=propose(arm,c,o)
            rg=r["gold"][f];pr=r["prediction"].get(f,"<missing>");positive=rg!="none"
            stage.append(dict(arm=arm,id=r["id"],group=r["group"],positive=positive,raw_proposed=p,reference=rg,prediction=pr))
            if positive:
                z=counts[arm+":"+r["group"]];z["positive"]+=1;z["true_candidate_proposed"]+=p is True;z["missing_alignment"]+=p is None;z["returned_positive"]+=pr not in ("none","<missing>");z["correct_type"]+=pr==rg
    old.dump(out/"stage_proposals.json",counts);scoring.csvwrite(out/"stage_proposals.csv",stage,["arm","id","group","positive","raw_proposed","reference","prediction"])
    diags={};dcalls=[]
    for arm in ("chat_v6","decision_v6"):
        which="direct-status-fixed" if arm=="decision_v6" else "direct-status"
        obs=[rd(p) for p in (BASE/"diagnostics"/which/arm).glob("N*.json")]
        assert len(obs)==sum(c["group"]=="node" and c["payload"].get("speaker")!="tool" and c["split"]=="test" for c in allrows)
        valid=[o for o in obs if "task_status" in g[o["id"]]["labels"] and "task_status" not in g[o["id"]]["ambiguous"]]
        diags[arm]={"queried":len(obs),"errors":sum(o["status"]!="ok" for o in obs),"direct":scoring.labels_stats([(g[o["id"]]["labels"]["task_status"],o["prediction"].get("task_status","<missing>")) for o in valid])}
        for o in obs:dcalls.extend(o["calls"])
    old.dump(out/"direct_status.json",diags)
    usage=Counter();providers=defaultdict(Counter)
    batches=[c for sets in summaries.values() for s in sets.values() for c in [s["physical_usage"]]]
    for m in batches:usage.update({k:m[k] for k in ("input","output","total","cost","calls","known_cost","known_total")})
    for cal in dcalls:
        u=cal.get("usage") or {};cc=u.get("total_cost");usage.update(dict(input=u.get("prompt_tokens") or 0,output=u.get("completion_tokens") or 0,total=u.get("total_tokens") or 0,cost=float(cc) if cc is not None else 0,calls=1,known_cost=cc is not None,known_total=u.get("total_tokens") is not None))
    # Unsuccessful initial sandbox probes are kept outside r1. No provider response or usage.
    pilots=[rd(p) for p in (BASE/"experiments").glob("*/dev/pilot/[NEP]*.json")]
    old.dump(out/"physical_usage.json",{"v1_v4_controls_and_all_physical_development_and_test_calls_including_diagnostics":usage,"unconnected_pilot_attempts":len(pilots),"pilot_known_tokens":sum(1 for p in pilots for a in p["calls"] if a.get("usage")),"unknown_cost_is_not_zero":True})
    chosen=rd(BASE/"selection.lock.json");recomputed={k:summaries["dev"][k]["quality"]["group_balanced_macro_f1"] for k in chosen["dev_metric"]}
    assert max(recomputed,key=recomputed.get)=="decision_"+chosen["version"]
    old.dump(BASE/"selection.transport-audit.json",{"original_lock_sha256":sha(BASE/"selection.lock.json"),"development_only_after_transport_repair":recomputed,
             "selected_unchanged":chosen["version"],"before_source_sha256":sha(BASE/"configs/run_before_transport_fix.py"),"after_source_sha256":sha(__import__("pathlib").Path(__file__).with_name("run.py"))})
    flat=[]
    for split,arms in summaries.items():
        for a,s in arms.items():
            q=s["quality"];u=s["usage"]
            flat.append(dict(split=split,arm=a,cases=s["observed"],aligned=s["aligned"],errors=s["errors"],labels=q["n"],accuracy=q["field_accuracy"],macro=q["group_balanced_macro_f1"],**q["group_macro_f1"],calls=u["calls"],input=u["input"],output=u["output"],total=u["total"],price_known_calls=u["known_cost"],known_usd=u["cost"] if u["known_cost"] else None,physical_calls=s["physical_usage"]["calls"]))
    scoring.csvwrite(out/"comparisons.csv",flat,list(flat[0]))
    print("test",[(a,round(s["quality"]["group_balanced_macro_f1"],4),s["aligned"],s["errors"]) for a,s in summaries["test"].items()])
if __name__=="__main__":analyze()
