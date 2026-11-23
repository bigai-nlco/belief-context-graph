"""Separate, real rare-edge audit. Historical flag selects leads only; no type/note on annotation sheets."""
import argparse
import gzip
import json
import re
from collections import defaultdict,Counter
from .data import *
from .labels import once
def collect():
    out=DS/"challenge_pool.json"
    if out.exists():return rd(out)
    main=old.read_rows(DS/"cases.jsonl");prev=old.read_rows(old.BASE/"dataset/cases.jsonl")
    excluded={objkey(c) for c in main+prev if c["group"]=="edge"}|{("edge",*sorted((txt(c["A"]["text"]),txt(c["B"]["text"])))) for c in rd(old.BASE/"supplement/sealed.json")}
    pairs=[];numbers=defaultdict(list)
    for run in RUNS:
        for t,g in idsafe(runroot(run)):
            raw=rd(g/"belief_graph_latest.json")
            flagged={(e.get("from_id"),e.get("to_id")) for e in raw["relations"] if e.get("type")=="contradicts"}
            arr,ctx=json.loads(gzip.decompress((CACHE/(run+"-"+g.name+".json.gz")).read_bytes()))
            endpoints={}
            for c in arr:
                if c["group"]!="edge":continue
                for l,k in (("A","from_id"),("B","to_id")):
                    id_=c["payload"][k];text=c["payload"][l]
                    if id_ not in endpoints:
                        endpoints[id_]={"run":run,"graph":g.name,"task":t,"prompt_line":c["prompt_line"],"node_id":id_,"text":text,**c["endpoint_evidence"][l]}
            for a,b in sorted(flagged):
                if a in endpoints and b in endpoints:pairs.append((endpoints[a],endpoints[b],"flag"))
            for e in endpoints.values():
                if t==120 and re.search("2020|2019",e["text"]) and re.search("Spain|France|Italy|Croatia|Albania|Greece|Montenegro|Monaco",e["text"],re.I) and re.search(r"\d[.,]\d",e["text"]):
                    numbers[t].append(e)
    def pairkey(a,b):return ("edge",*sorted((txt(a["text"]),txt(b["text"]))))
    def frac(x):return {n for n in NUM.findall(x) if len(n)>1 and n not in {"2010","2019","2020"}}
    extra=[]
    for t,es in numbers.items():
        for i,a in enumerate(es):
            for b in es[:i]:
                if a["run"]==b["run"] and a["node_id"]==b["node_id"]:continue
                na,nb=frac(a["text"]),frac(b["text"])
                if not na or not nb or na==nb:continue
                aa={z.lower() for z in re.findall("Spain|France|Italy|Croatia|Albania|Greece|Montenegro|Monaco",a["text"],re.I)}
                bb={z.lower() for z in re.findall("Spain|France|Italy|Croatia|Albania|Greece|Montenegro|Monaco",b["text"],re.I)}
                if len(aa)!=1 or aa!=bb:continue
                s=old.sim(a["text"],b["text"])
                if s<.28:continue
                extra.append((a,b,"numeric:"+sorted(aa)[0]))
    def stable(z):return h([z[0]["run"],z[0]["node_id"],z[1]["run"],z[1]["node_id"]])
    picked=[];counts=Counter()
    for a,b,band in sorted(pairs,key=stable)+sorted(extra,key=stable):
        k=pairkey(a,b)
        if k in excluded or a["text"]==b["text"]:continue
        if band=="flag" and counts[a["task"],band]>=5:continue
        if band!="flag" and counts[band]>=5:continue
        if len(picked)>=80:break
        counts[a["task"],band]+=1;counts[band]+=1;excluded.add(k);picked.append({"A":a,"B":b})
    picked.sort(key=lambda c:h([c["A"]["text"],c["B"]["text"],"audit"]));old.dump(out,picked)
    return picked
def publish():
    pool=collect();rows=[];gt={}
    a=BASE/"annotations/challenge.txt"
    for line in a.read_text().splitlines():
        if not line.strip():continue
        ix,lab,why=line.split(maxsplit=2)
        if ":" in ix:
            from .catalog import index
            z,x,y=ix.split(":");cat=index();c={"A":cat[x],"B":cat[y]};ix=z
        else:c=pool[int(ix)-1]
        a0=c["A"];b0=c["B"];id_="X"+ix.zfill(3)
        assert a0["task"]==b0["task"] and id_ not in gt
        labs={"c":"contradicts","d":"depends_on","s":"supplements","n":"none"};gt[id_]={"labels":{"type":labs[lab]},"ambiguous":[],"reason":why}
        rows.append({"id":id_,"uid":h(c)[:20],"group":"edge","band":"rare_audit","task":a0["task"],"run":a0["run"],"graph":a0["graph"],"turn":a0["turn"],"prompt_line":a0["prompt_line"],"split":"challenge",
                     "payload":{"speaker":"assistant","from_id":1,"to_id":2,"A":a0["text"],"B":b0["text"],"current_ids":[1]},
                     "endpoint_evidence":{k:{f:c[k][f] for f in ("role","turn","quote")} for k in ("A","B")},"pointers":c})
    known={objkey(r) for r in old.read_rows(DS/"cases.jsonl")+old.read_rows(old.BASE/"dataset/cases.jsonl") if r["group"]=="edge"}
    known|={("edge",*sorted((txt(c["A"]["text"]),txt(c["B"]["text"])))) for c in rd(old.BASE/"supplement/sealed.json")}
    assert len({objkey(r) for r in rows})==len(rows) and not {objkey(r) for r in rows}&known
    once(DS/"challenge.jsonl","".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode())
    once(DS/"challenge_gold.json",json.dumps(gt,ensure_ascii=False,indent=2).encode());print("challenge sealed",len(rows),Counter(x["labels"]["type"] for x in gt.values()))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["read","freeze"]);p.add_argument("lo",nargs="?",type=int,default=1);p.add_argument("hi",nargs="?",type=int,default=30);a=p.parse_args()
    if a.action=="freeze":publish()
    else:
        c=collect();print("candidate pairs",len(c))
        for i,z in enumerate(c,1):
            if a.lo<=i<=a.hi:print(f"{i} A: "+z["A"]["text"]+"\n   B: "+z["B"]["text"])
