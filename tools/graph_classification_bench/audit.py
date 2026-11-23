"""Blinded post-lock audit; independent of numeric scoring and model selection."""
import json
from .freeze import BASE,digest,dump,read_rows
from .score import collect

def main():
    dest=BASE/"audit";dest.mkdir(exist_ok=True)
    cs={c["id"]:c for c in read_rows(BASE/"dataset/cases.jsonl") if c["split"]=="test"}
    arms=("chat_fixed","decision_v4")
    ans={a:{x["id"]:x for x in collect(a,"test","r1")[2]} for a in arms}
    keys={};packets=[]
    for group in ("node","parent","edge"):
        ids=sorted((i for i,c in cs.items() if c["group"]==group and ans[arms[0]][i]["prediction"]!=ans[arms[1]][i]["prediction"]),key=lambda i:digest(["blindaudit-v1",i]))
        for i in ids[:3]:
            order=arms if int(digest(["presentation-v1",i])[-1],16)%2 else arms[::-1]
            keys[i]=dict(zip("AB",order))
            packets.append({"id":i,"source":cs[i]["payload"],**{k:ans[a][i]["prediction"] for k,a in keys[i].items()}})
    for name,data in [("paired_blind.json",packets),("presentation_key.json",keys)]:
        f=dest/name
        if f.exists():assert json.loads(f.read_text())==data
        else:dump(f,data)
    # For disagreements between the two generating pipelines, don't publish model
    # identity with candidate texts displayed to the manual alignment reviewer.
    matches=[];key={}
    for arm in ("native_current","pipeline_decision_v4"):
        path=BASE/"experiments"/arm/"test/r1"
        for c in cs.values():
            o=json.loads((path/(c["id"]+".json")).read_text())
            a=o["diagnostics"]
            if a.get("aligned") is False and a.get("candidates"):
                code=digest(["alignment-anonymous",arm,c["id"]])[:7];key[code]=[arm,c["id"]]
                matches.append({"code":code,"id":c["id"],"unit":c["payload"].get("text") or c["payload"].get("child"),
                                "candidates":[n["text"] for n in a["candidates"]]})
    for name,data in [("alignment_blind.json",sorted(matches,key=lambda r:r["code"])),("alignment_key.json",key)]:
        f=dest/name
        if f.exists():assert json.loads(f.read_text())==data
        else:dump(f,data)
    print("Blind pairs:",len(packets),"ambiguous alignment packets:",len(matches))
if __name__=="__main__":main()
