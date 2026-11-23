"""Seal independently authored source-based reference labels; never send them to a provider."""
import csv
import io
import json
from collections import Counter
from pathlib import Path
from tools.graph_classification_bench import freeze as old, replay
from tools.graph_classification_bench.preflight import LABELS
from .data import DS, BASE, objkey, key, sha, rd

ST=dict(c="confirmed",a="almost_certain",p="probable",m="possible",u="unknown",l="unlikely",d="doubtful")
TY=dict(b="belief",d="decision")
RO=dict(t="task",p="plan",e="evidence",i="intermediate_conclusion",c="constraint",g="progress",o="other")
SU=dict(b="blocked",i="in_progress",c="reported_complete",o="open",u="unspecified",x="cancelled")
AM=dict(s="stance",y="node_type",r="task_role",z="task_status")
def once(path,body):
    if path.exists():
        if path.read_bytes()!=body:raise ValueError(f"Sealed artifact changed: {path.name}")
    else:
        with path.open("xb") as h:h.write(body)
def seal():
    rows=old.read_rows(DS/"cases.jsonl");by={r["id"]:r for r in rows};ctx=rd(DS/"native_context.json")
    lock=rd(DS/"source_lock.json");assert lock["cases"]==sha(DS/"cases.jsonl") and lock["context"]==sha(DS/"native_context.json")
    prev=old.read_rows(old.BASE/"dataset/cases.jsonl")
    assert not {objkey(r) for r in prev}&{objkey(r) for r in rows} and not {key(r) for r in prev}&{key(r) for r in rows}
    assert len(rows)==len(by)==len({objkey(r) for r in rows})==len({r["uid"] for r in rows})==500
    splits=[{r["task"] for r in rows if r["split"]==s} for s in ("dev","test")]
    assert not splits[0]&splits[1] and not splits[1]&{r["task"] for r in prev}
    gold={};sources=[]
    for filename in ("nodes","parents","edges"):
        path=BASE/"annotations"/(filename+".txt");sources.append(path)
        for line in path.read_text().splitlines():
            if not line.strip():continue
            if filename=="nodes":
                id_,s,t,r,z,a,reason=line.split(maxsplit=6)
                labs={k:codes[v] for k,codes,v in (("stance",ST,s),("node_type",TY,t),("task_role",RO,r),("task_status",SU,z)) if v!="-"}
                amb=[AM[ch] for ch in a if ch!="-"]
            else:
                id_,code,a,reason=line.split(maxsplit=3);f="kind" if filename=="parents" else "type"
                labs={f:({"s":"narrower_scope","a":"new_deliverable","n":"none"} if f=="kind" else {"s":"supplements","d":"depends_on","c":"contradicts","n":"none"})[code]}
                assert a in {"?","-"};amb=[f] if a=="?" else []
            assert id_ not in gold and reason and set(amb)<=labs.keys() and all(v in LABELS[k] for k,v in labs.items()),id_
            c=by[id_];d=c["payload"];w=ctx[key(c)];replay.guard(d)
            if filename=="nodes":
                assert ("node_type" in labs)==(d["speaker"]=="assistant"),id_
                assert ("task_role" in labs)==(d["speaker"]!="tool" and labs.get("node_type")!="decision"),id_
                assert ("task_status" in labs)==(labs.get("task_role")=="progress"),id_
            if filename=="edges":
                nodes={n["id"]:n["content"] for n in w["previous"]+w["current"]}
                assert [nodes[d["from_id"]],nodes[d["to_id"]]]==[d["A"],d["B"]]
                assert not any({e.get("from",e.get("from_id")),e.get("to",e.get("to_id"))}=={d["from_id"],d["to_id"]} for e in w["existing"])
                assert any(e["quote"] for e in c["endpoint_evidence"].values()),id_
            elif d.get("speaker")!="tool":
                sentences=old.sentences(w["sentences"]);assert [sentences[i] for i in d["indices"]]==d["source_evidence"],id_
                if filename=="parents":assert any(n["id"]==d["parent_id"] and n["content"]==d["parent"] and n.get("task_role")=="task" for n in w["history"])
            else:assert any(i["item_index"]==d["item_index"] and any(n["rank"]==d["source_rank"] for n in i["results"]) for i in w["items"])
            gold[id_]={"labels":labs,"ambiguous":amb,"reason":reason}
    assert gold.keys()==by.keys()
    out=io.StringIO();tab=csv.writer(out,delimiter="\t",lineterminator="\n");tab.writerow(["id","labels","ambiguous_fields","reason"])
    for k,g in sorted(gold.items()):tab.writerow([k," ".join(f"{f}={v}" for f,v in g["labels"].items()),",".join(g["ambiguous"]) or "-",g["reason"]])
    once(DS/"groundtruth.tsv",out.getvalue().encode());protocol=Path(__file__).with_name("protocol.md")
    once(DS/"protocol.md",protocol.read_bytes())
    files=[*sources,*(DS/n for n in ("cases.jsonl","native_context.json","source_lock.json","groundtruth.tsv","protocol.md"))]
    once(DS/"blind.lock.json",(json.dumps({str(p.relative_to(BASE)):sha(p) for p in files},indent=2,sort_keys=True)+"\n").encode())
    print("Locked",len(gold),"cases;",len(ctx),"source windows;",*[len(s) for s in splits],"tasks; fields",Counter(f for g in gold.values() for f in g["labels"]),"ambiguous",Counter(f for g in gold.values() for f in g["ambiguous"]))
def truth():
    with (DS/"groundtruth.tsv").open() as f:
        return {r["id"]:{"labels":dict(x.split("=",1) for x in r["labels"].split()),"ambiguous":[] if r["ambiguous_fields"]=="-" else r["ambiguous_fields"].split(","),"reason":r["reason"]} for r in csv.DictReader(f,delimiter="\t")}
if __name__=="__main__":seal()
