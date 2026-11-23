"""Deterministic side-blind reading packet. Do not show model identities or gold during review."""
from .grade import *
from .data import h
def make(stage):
    cases=data_for("test")[0];by={c["id"]:c for c in cases}
    arms=["chat_v6","decision_v6"] if stage=="fixed" else ["native_current","pipeline_v6"]
    raw=[{r["id"]:r for r in read(a,"test")[2]} for a in arms];assert all(len(b)==250 for b in raw)
    out=BASE/"diagnostics"/"blind";out.mkdir(parents=True,exist_ok=True);rows=[];ids=set();key_={}
    for f in ("stance","node_type","task_role","task_status","kind","type"):
        elig=[id_ for id_,r in raw[0].items() if f in r["gold"] and f not in r["ambiguous"]]
        if stage=="stage" and f not in ("node_type","kind","type"):continue
        for disagreement in (True,False) if stage=="fixed" else (True,):
            ev=[i for i in elig if i not in ids and (raw[0][i]["prediction"].get(f)!=raw[1][i]["prediction"].get(f))==disagreement]
            if not ev:continue
            i=min(ev,key=lambda i:h([stage,f,disagreement,i]));ids.add(i);c=by[i];d=c["payload"]
            order=[0,1] if int(h([stage,i])[:2],16)%2 else [1,0];a,b=[raw[j][i]["prediction"].get(f,"<missing>") for j in order]
            context={k:d[k] for k in ("A","B","child","parent","text","speaker","source_evidence","thinking_only") if k in d}
            rows.append({"id":i,"field":f,"content":context,"A":a,"B":b})
            key_[i]={"A":arms[order[0]],"B":arms[order[1]]}
    old.jsonl(out/(stage+"-packet.jsonl"),rows);old.dump(out/(stage+"-key.json"),key_)
    for r in rows:
        d=r["content"];e=d.get("source_evidence",[])
        print(r["id"],r["field"],"A:",r["A"],"B:",r["B"],"\n  ",str(d.get("text",d.get("A",d.get("parent",""))))[:250],"|",str(d.get("B",d.get("child",e)))[:250])
if __name__=="__main__":
    import sys
    make(sys.argv[1])
