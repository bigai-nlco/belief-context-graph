"""Re-score archived responses without network. JSON, complete item CSVs, usage, confusions and paired CI."""
import argparse
import math
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np
from tools.graph_classification_bench import score as scoring, preflight as oldgold
from .data import BASE,DS,rd,sha,old
from .labels import truth,once
from .run import data_for,spec
def read(arm,split,trial="r1"):
    from .repair import lane
    where=lane(arm,split) if trial=="r1" else BASE/"experiments"/arm/split/trial;all_cases,_=data_for(split);cs={c["id"]:c for c in all_cases}
    gold=(oldgold.truth() if split=="regression" else rd(DS/"challenge_gold.json") if split=="challenge" else truth())
    rows=[];calls=[]
    for id_,c in cs.items():
        p=where/(id_+".json")
        if not p.exists():continue
        o=rd(p);assert o["uid"]==c["uid"] and o["case_hash"]==old.digest(c)
        g=gold[id_];diag=o["diagnostics"]
        rows.append(dict(id=id_,task=c["task"],benchmark="drbench" if c["task"]>=10000 else "drb2",group=c["group"],band=c["band"],status=o["status"],aligned=diag.get("aligned",True) and o["status"]=="ok",pair_proposed=diag.get("pair_proposed"),
                         turn=c["turn"],source=f"{c['run']}/{c['graph']}:{c['prompt_line']}",gold=g["labels"],ambiguous=g["ambiguous"],prediction=o["prediction"],reason=g["reason"]))
        for i,call in enumerate(o["calls"],1):
            u=call.get("usage") or {}; cost=u.get("total_cost");raw=call.get("response",{})
            if cost is None:cost=(raw.get("usage") or {}).get("cost")
            calls.append(dict(id=id_,call=i,stage=c["group"],provider=call["provider"],model=call["model"],input=u.get("prompt_tokens"),output=u.get("completion_tokens"),reasoning=u.get("reasoning_tokens"),total=u.get("total_tokens"),
                  cost=float(cost) if cost is not None else None,seconds=call["seconds"],reused=bool(o.get("reused_call_artifact") or call.get("reused_from")),http=call.get("http")))
    return where,len(cs),rows,calls
def totals(calls):
    t={f:sum(v[f] for v in calls if v[f] is not None) for f in ("input","output","reasoning","total","cost","seconds")}
    t.update(calls=len(calls),known_cost=sum(v["cost"] is not None for v in calls),known_total=sum(v["total"] is not None for v in calls),known_reasoning=sum(v["reasoning"] is not None for v in calls))
    return t
def score(arm,split,trial="r1"):
    where,expected,rows,calls=read(arm,split,trial);assert expected and rows
    summ={"arm":arm,"split":split,"expected":expected,"observed":len(rows),"errors":sum(r["status"]!="ok" for r in rows),"n_tasks":len({r["task"] for r in rows}),"aligned":sum(r["aligned"] for r in rows),"quality":scoring.quality(rows),
          "aligned_quality":scoring.quality([r for r in rows if r["aligned"]]),"usage":totals(calls),"physical_usage":totals([c for c in calls if not c["reused"]]),
          "aligned_by_group":{g:[sum(r["aligned"] for r in rows if r["group"]==g),sum(r["group"]==g for r in rows)] for g in ("node","parent","edge")},
          "by_benchmark":{b:scoring.quality([r for r in rows if r["benchmark"]==b]) for b in sorted({r["benchmark"] for r in rows})}}
    if where.name=="r2":
        prior=[]
        for p in (where.parent/"r1").glob("[NEP]*.json"):
            for a in rd(p)["calls"]:
                u=a.get("usage") or {};cc=u.get("total_cost")
                prior.append(dict(input=u.get("prompt_tokens"),output=u.get("completion_tokens"),reasoning=u.get("reasoning_tokens"),total=u.get("total_tokens"),cost=float(cc) if cc is not None else None,seconds=a["seconds"]))
        new=summ["physical_usage"].copy();summ["transport_repair"]={"initial_physical":totals(prior),"new_physical":new,"original_dir":"r1","corrected_dir":"r2"}
        summ["physical_usage"]={k:new[k]+v for k,v in totals(prior).items()}
    old.dump(where/"score.json",summ)
    scoring.csvwrite(where/"case_scores.csv",rows,["id","group","task","benchmark","turn","source","band","status","aligned","pair_proposed","gold","ambiguous","prediction","reason"])
    scoring.csvwrite(where/"usage.csv",calls,["id","call","stage","provider","model","input","output","reasoning","total","cost","seconds","http","reused"])
    confusion=[{"field":f,"gold":pair.split(" -> ")[0],"prediction":pair.split(" -> ")[1],"count":n} for f,s in summ["quality"]["fields"].items() for pair,n in s["confusion"].items()]
    scoring.csvwrite(where/"confusions.csv",confusion,["field","gold","prediction","count"])
    return summ
def bootstrap(rows,other,n=3000):
    """Resample complete original-task clusters identically for both arms, preserving multilabel rows."""
    ts=sorted({r["task"] for r in rows});assert {r["id"] for r in rows}=={r["id"] for r in other}
    idx=np.random.default_rng(20261119).integers(0,len(ts),(n,len(ts)));fs=sorted({f for r in rows for f in r["gold"] if f not in r["ambiguous"]})
    fields={r["group"]:{f for z in rows if z["group"]==r["group"] for f in z["gold"] if f in fs} for r in rows}
    def sample(side):
        out={}
        for f in fs:
            triples=[(r["task"],r["gold"][f],r["prediction"].get(f) or "<missing>") for r in side if f in r["gold"] and f not in r["ambiguous"]]
            labs=sorted({k for _,g,p in triples for k in (g,p)});t=np.zeros((len(ts),len(labs),len(labs)),dtype=int)
            for task,g,p in triples:t[ts.index(task),labs.index(g),labs.index(p)]+=1
            z=t[idx].sum(axis=1);sup=z.sum(axis=2);den=sup+z.sum(axis=1);tp=np.diagonal(z,axis1=1,axis2=2)
            vals=np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den>0)
            out[f]=np.divide(np.where(sup>0,vals,0).sum(axis=1),(sup>0).sum(axis=1),out=np.full(n,np.nan),where=(sup>0).sum(axis=1)>0)
        out["balanced"]=np.nanmean(np.array([np.nanmean(np.array([out[f] for f in fr]),axis=0) for fr in fields.values()]),axis=0)
        return out
    a,b=sample(rows),sample(other);qs=[scoring.quality(s) for s in (rows,other)]
    def current(q,f):return q["group_balanced_macro_f1"] if f=="balanced" else q["fields"][f]["macro_f1"]
    return {f:{"difference":current(qs[0],f)-current(qs[1],f),"lo":float(np.nanpercentile(a[f]-b[f],2.5)),"hi":float(np.nanpercentile(a[f]-b[f],97.5)),"clusters":len(ts),"replicates":n} for f in a}
def select():
    arms=["decision_v"+v for v in ("1","4","5","6","7")]
    by={a:score(a,"dev") for a in arms}
    assert all(x["observed"]==x["expected"]==250 for x in by.values())
    chosen=min(arms,key=lambda a:(-by[a]["quality"]["group_balanced_macro_f1"],by[a]["usage"]["total"],a))
    v=chosen.split("_")[1];scores={a:by[a]["quality"]["group_balanced_macro_f1"] for a in arms}
    import json
    lock={"version":v,"spec":spec(v),"dev_metric":scores,"dataset":sha(DS/"blind.lock.json"),"challenge_hash":sha(DS/"challenge.jsonl"),"challenge_gold":sha(DS/"challenge_gold.json"),"runner_hash":sha(__import__("pathlib").Path(__file__).with_name("run.py")),"criterion":"dev group-balanced macro-F1; v6 grouped out-of-fold; token tiebreak","holdout_unopened":True}
    p=BASE/"selection.lock.json"
    if not p.exists():assert not list((BASE/"experiments").glob("*/test/*/[NEP]*.json"))
    once(p,(json.dumps(lock,indent=2,sort_keys=True)+"\n").encode());print("Selection",v,scores)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["select","score"]);p.add_argument("split",nargs="?");p.add_argument("arms",nargs="*");a=p.parse_args()
    if a.action=="select":select()
    else:
        for arm in a.arms:
            s=score(arm,a.split);print(arm,a.split,s["observed"],s["errors"],round(s["quality"]["group_balanced_macro_f1"],4),s["aligned"],s["usage"])
