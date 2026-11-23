"""Share a physically identical provider call across uncalibrated and calibrated decoders."""
import argparse
from .run import *
from .data import old
def estimate(c,o,ver):
    pred=copy.deepcopy(o["prediction"]);g=c["group"]
    if o["status"]=="ok" and g in ("edge","parent"):
        field="type" if g=="edge" else "kind";call=o["calls"][0];body=call["response"]
        ans=(decoded(body,spec(ver)["calibration"]) if ver=="v6" else body)["answers"][0]
        assert ans["name"]=="item_0_"+field;pred[field]=ans["choice"]
        if ver=="v6":call["applied_answers"]=[{"choice":ans["choice"],"name":ans["name"]}]
        else:call.pop("applied_answers",None)
    return pred
def build(split):
    locked=rd(BASE/"selection.lock.json");assert locked["version"]=="v6" and locked["spec"]==spec("v6")
    rows,ctx=data_for(split)
    arms=["decision_v4"] if split!="regression" else ["native_current","chat_fixed","decision_v1","decision_v4","decision_v6"]
    for arm in arms:
        out=BASE/"experiments"/arm/split/"r1";out.mkdir(parents=True,exist_ok=True)
        for c in rows:
            dst=out/(c["id"]+".json")
            if dst.exists():continue
            if split=="regression":
                source=old.BASE/"experiments"/("decision_v4" if arm=="decision_v6" else arm)/c["split"]/"r1"/dst.name
            else:
                from .repair import lane
                source=lane("decision_v6",split)/dst.name
            obj=rd(source);assert obj["case_hash"]==old.digest(c)
            if arm in ("decision_v4","decision_v6"):
                assert spec("v4")["questions"]==spec("v6")["questions"] and not spec("v6").get("provenance")
                obj["prediction"]=estimate(c,obj,armver(arm))
            obj.update(arm=arm,reused_call_artifact=str(source.relative_to(ROOT)))
            writenew(dst,obj)
        new_or_same(out/"manifest.json",{"arm":arm,"split":split,"physical_network_calls":0,"selection_lock":sha(BASE/"selection.lock.json"),"method":"reuse identical raw recorded responses; preserve originals and provenance; only decode differently for v6"})
        print(split,arm,len(rows))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["regression","test","challenge"]);build(p.parse_args().split)
