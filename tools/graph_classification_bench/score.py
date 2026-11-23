"""Deterministic, field-level scoring against the sealed independent references."""
import argparse
import csv
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
from .freeze import BASE,dump,read_rows
from .preflight import LABELS,truth

def add(pred,gold):
    return int(pred==gold)

def labels_stats(values):
    labs=sorted({g for g,p in values});by={}
    for k in labs:
        tp=sum(g==p==k for g,p in values);fp=sum(p==k and g!=k for g,p in values);fn=sum(g==k and p!=k for g,p in values)
        by[k]={"support":tp+fn,"precision":tp/(tp+fp) if tp+fp else 0.,"recall":tp/(tp+fn) if tp+fn else 0.,
               "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {"n":len(values),"correct":sum(g==p for g,p in values),"accuracy":sum(g==p for g,p in values)/len(values) if values else None,
            "macro_f1":sum(s["f1"] for s in by.values())/len(by) if by else None,"by_label":by,
            "confusion":{f"{g} -> {p}":v for (g,p),v in sorted(Counter(values).items())}}

def quality(rows):
    fs=defaultdict(list);gs=defaultdict(list);amb=defaultdict(list);ncase=exact=0
    for r in rows:
        local=[]
        for f,g in r["gold"].items():
            p=r["prediction"].get(f) or "<missing>"
            if f in r["ambiguous"]:
                amb[f].append((g,p));continue
            fs[f].append((g,p));gs[r["group"]].append(f);local.append(p==g)
        if local:ncase+=1;exact+=all(local)
    field={f:labels_stats(v) for f,v in fs.items()}
    groups={key:sum(field[f]["macro_f1"] for f in set(fields))/len(set(fields)) for key,fields in gs.items()}
    n=sum(v["n"] for v in field.values());correct=sum(v["correct"] for v in field.values())
    return {"group_balanced_macro_f1":sum(groups.values())/len(groups) if groups else None,"group_macro_f1":groups,
            "fields":field,"field_accuracy":correct/n if n else None,"correct":correct,"n":n,
            "fully_correct_cases":exact,"scored_cases":ncase,"ambiguous_fields":sum(map(len,amb.values())),
            "ambiguous_primary_label_agreement_diagnostic":{f:labels_stats(v) for f,v in amb.items()}}

def csvwrite(path,rows,columns):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in columns})

def collect(arm,split,trial):
    out=BASE/"experiments"/arm/split/trial; gt=truth()
    data={r["id"]:r for r in read_rows(BASE/"dataset/cases.jsonl") if r["split"]==split};rows=[];calls=[]
    for p in sorted(out.glob("[NEP][0-9][0-9][0-9].json")):
        o=json.loads(p.read_text());c=data[o["id"]];g=gt[o["id"]];assert c["uid"]==o["uid"]
        rows.append({"id":c["id"],"group":c["group"],"task":c["task"],"turn":c["turn"],"run":c["run"],"band":c["band"],
                     "status":o["status"],"aligned":o["diagnostics"].get("aligned",True) if o["status"]=="ok" else False,
                     "pair_proposed":o["diagnostics"].get("pair_proposed"),"gold":g["labels"],"ambiguous":g["ambiguous"],"prediction":o["prediction"],"reason":g["reason"]})
        for i,call in enumerate(o["calls"]):
            u=call.get("usage") or {};ct=u.get("completion_tokens");rt=u.get("reasoning_tokens")
            calls.append({"id":c["id"],"i":i+1,"group":c["group"],"provider":call["provider"],"model":call["model"],
                          "input":u.get("prompt_tokens"),"output_incl_reasoning":ct,"reasoning":rt,
                          "output_excl_reasoning":None if rt is None or ct is None else ct-rt,
                          "total":u.get("total_tokens"),"cost":u.get("total_cost"),"seconds":call["seconds"]})
    return out,data,rows,calls

def score(arm,split,trial,write=True):
    out,data,rows,calls=collect(arm,split,trial)
    if not rows:raise ValueError(f"No observations for {arm}/{split}/{trial}")
    totals={k:sum(float(v[k]) for v in calls if v[k] is not None) for k in
            ("input","output_incl_reasoning","output_excl_reasoning","reasoning","total","cost","seconds")}
    totals.update(calls=len(calls),cost_known_calls=sum(c["cost"] is not None for c in calls),
                  reasoning_known_calls=sum(c["reasoning"] is not None for c in calls),total_known_calls=sum(c["total"] is not None for c in calls))
    summary={"arm":arm,"split":split,"trial":trial,"requested":len(data),"finished":len(rows),"failed":sum(r["status"]!="ok" for r in rows),
             "aligned":sum(r["aligned"] for r in rows),"quality":quality(rows),"quality_aligned_only":quality([r for r in rows if r["aligned"]]),
             "usage":totals,"alignment_by_group":{group:dict(aligned=sum(r["aligned"] for r in rows if r["group"]==group),n=sum(r["group"]==group for r in rows)) for group in ("node","parent","edge")}}
    if write:
        dump(out/"score.json",summary)
        csvwrite(out/"case_scores.csv",rows,["id","group","run","task","turn","band","status","aligned","pair_proposed","gold","ambiguous","prediction","reason"])
        csvwrite(out/"usage.csv",calls,["id","i","group","provider","model","input","output_incl_reasoning","reasoning","output_excl_reasoning","total","cost","seconds"])
        cs=[dict(field=f,groundtruth=gp.split(" -> ")[0],prediction=gp.split(" -> ")[1],n=n) for f,stat in summary["quality"]["fields"].items() for gp,n in stat["confusion"].items()]
        csvwrite(out/"confusions.csv",cs,["field","groundtruth","prediction","n"])
    return summary

def paired_ci(arm1,arm2,split="test",n=1500,seed=817,trial="r1"):
    lhs=collect(arm1,split,trial)[2];rhs=collect(arm2,split,trial)[2];assert {r["id"] for r in lhs}=={r["id"] for r in rhs}
    tasks=sorted({r["task"] for r in lhs});rnd=random.Random(seed);d=[]
    by=[{t:[r for r in side if r["task"]==t] for t in tasks} for side in (lhs,rhs)]
    for _ in range(n):
        sample=rnd.choices(tasks,k=len(tasks))
        s=[quality([r for t in sample for r in side[t]])["group_balanced_macro_f1"] for side in by];d.append(s[0]-s[1])
    d.sort()
    return {"difference":quality(lhs)["group_balanced_macro_f1"]-quality(rhs)["group_balanced_macro_f1"],"lo":d[int(n*.025)],"hi":d[int(n*.975)],"task_clusters":len(tasks),"replicates":n}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm");p.add_argument("split");p.add_argument("--trial",default="r1");a=p.parse_args()
    s=score(a.arm,a.split,a.trial);q=s["quality"]
    print(f'{a.arm} {a.split} n={s["finished"]}/{s["requested"]} failures={s["failed"]} matched={s["aligned"]} macro={q["group_balanced_macro_f1"]:.3f} {q["correct"]}/{q["n"]}',q["group_macro_f1"])
