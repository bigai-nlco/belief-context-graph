"""Freeze provenance/labels and check the invariants before any paid evaluation."""
import csv
import json
from collections import Counter
from . import replay
from .freeze import BASE, dump, read_rows, sentences
from bcg.construct.unified import decisions

LABELS=dict(stance=decisions.STANCES,node_type=decisions.TYPES,task_role=decisions.ROLES,task_status=decisions.STATUSES,kind=decisions.REFINEMENTS,type=decisions.RELATIONS)
def truth():
    with (BASE/"dataset/groundtruth.tsv").open() as f:rows=list(csv.DictReader(f,delimiter="\t"))
    return {r["id"]:{"labels":dict(w.split("=",1) for w in r["labels"].split()),"ambiguous":[] if r["ambiguous_fields"]=="-" else r["ambiguous_fields"].split(","),"reason":r["reason"]} for r in rows}

def main():
    ds=BASE/"dataset";rows=read_rows(ds/"cases.jsonl");g=truth()
    replay.freeze_native_context(ds);ctx=json.loads((ds/"native_context.json").read_text())
    assert len(rows)==len(g)==len({r["uid"] for r in rows})==100
    assert set(g)=={r["id"] for r in rows}
    split={s:{r["task"] for r in rows if r["split"]==s} for s in ("dev","test")}
    assert not split["dev"]&split["test"]
    for r in rows:
        d=r["payload"];k=f"{r['run']}/{r['graph']}:{r['prompt_line']}";c=ctx[k];ans=g[r["id"]]
        assert all(v in LABELS[f] for f,v in ans["labels"].items()) and set(ans["ambiguous"])<=ans["labels"].keys() and ans["reason"]
        replay.guard(d)
        if r["group"]=="edge":
            data={n["id"]:n["content"] for n in c["previous"]+c["current"]}
            assert [data[d["from_id"]],data[d["to_id"]]]==[d["A"],d["B"]]
            # Prior relation between these same two endpoints would leak the target.
            assert not any({e.get("from",e.get("from_id")),e.get("to",e.get("to_id"))}=={d["from_id"],d["to_id"]} for e in c["existing"])
        elif d.get("speaker")!="tool":
            idx=sentences(c["sentences"])
            assert [idx[i] for i in d["indices"]]==d["source_evidence"]
            if r["group"]=="parent":
                assert any(n["id"]==d["parent_id"] and n["content"]==d["parent"] and n.get("task_role")=="task" for n in c["history"])
        else:
            item=next(i for i in c["items"] if i["item_index"]==d["item_index"])
            assert any(n["rank"]==d["source_rank"] for n in item["results"])
    lock={p.name:replay.byteshash(p) for p in (ds/"cases.jsonl",ds/"runtime_context.json",ds/"native_context.json",ds/"groundtruth.tsv")}
    if (ds/"blind.lock.json").exists():assert json.loads((ds/"blind.lock.json").read_text())==lock
    else:dump(ds/"blind.lock.json",lock)
    print("Locked:",len(rows),"cases,",len(split["dev"]),"dev tasks,",len(split["test"]),"held-out tasks.",Counter(r["group"] for r in rows))
if __name__=="__main__":main()
