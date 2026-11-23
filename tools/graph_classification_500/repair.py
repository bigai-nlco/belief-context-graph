"""Transport-only repair. Reuse a raw response iff the complete wire body matches exactly."""
import argparse
import copy
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from openai.types.chat import ChatCompletion
from .run import *
def lane(arm,split):
    top=BASE/"experiments"/arm/split
    return top/("r2" if (top/"r2/complete.json").is_file() else "r1")
class Replayer(Live):
    def __init__(self,cfg,arm,meta,oldfile,offline):
        super().__init__(cfg,arm,armver(arm),meta)
        self.offline=offline;self.cache=rd(oldfile)["calls"];self.used=set();self.ref=str(oldfile.relative_to(BASE));real=self.client;self.hit=None
        def create(**kw):
            found=self.take("chat",kw)
            if found:
                i,s=found;self.hit=i
                return ChatCompletion.model_validate(s["response"])
            if offline:raise RuntimeError("Missing offline Chat response")
            return real.chat.completions.create(**kw)
        self.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),close=real.close)
    def take(self,provider,wire):
        for i,entry in enumerate(self.cache):
            if i not in self.used and entry["provider"]==provider and entry["request"]==wire and "response" in entry:
                self.used.add(i);return i,entry
        return None
    def chat(self,prompt,limit):
        self.hit=None;r=super().chat(prompt,limit)
        if self.hit is not None:
            self.calls[-1]["reused_from"]=self.ref+"#"+str(self.hit);self.calls[-1]["seconds"]=self.cache[self.hit]["seconds"]
        return r
    def post(self,url,*,headers,json,timeout,follow_redirects):
        wire=transformed(json,self.ver,self.meta);found=self.take("decisions",wire)
        if found:
            i,e=found;entry=copy.deepcopy(e);entry["reused_from"]=self.ref+"#"+str(i);self.calls.append(entry);data=entry["response"]
            u=entry.get("usage",{});cost=u.get("total_cost");raw=httpx.Response(200,request=httpx.Request("POST",url),json=data,headers={"x-litellm-response-cost":str(cost)} if cost is not None else {})
        else:
            if self.offline:raise RuntimeError("Missing offline Decisions response")
            raw=replay.Recorder.post(self,url,headers=headers,json=wire,timeout=timeout,follow_redirects=False)
            if not raw.is_success:return raw
            data=raw.json();entry=self.calls[-1]
        if self.ver=="v6":data=decoded(data,spec("v6")["calibration"])
        if self.ver=="v7":data=combine(json,data)
        entry["applied_answers"]=[{"name":a["name"],"choice":a["choice"]} for a in data["answers"]]
        return local_json(raw,data)
def repair(arm,split,offline,workers):
    top=BASE/"experiments"/arm/split;src=top/"r1";dst=top/"r2";dst.mkdir(parents=True,exist_ok=True)
    cases,ctx=data_for(split);extras=rd(DS/"endpoint_provenance.json") if spec(armver(arm)).get("provenance") else {}
    cfg=llm.load_config(str(replay.PROFILE),model_key="graph");assert cfg["model"]=="gpt-6-luna"
    stop=threading.Event()
    def job(c):
        p=src/(c["id"]+".json");q=dst/p.name
        if q.exists() or stop.is_set():return
        obj=rd(p)
        if obj["status"]=="ok" or "DecodingError" not in obj["diagnostics"].get("message",""):
            obj["reused_call_artifact"]=str(p.relative_to(BASE))
        else:
            rec=Replayer(cfg,arm,evidence(c,extras),p,offline);tok=llm.bind_usage_tracker(llm.TokenUsageTracker())
            try:
                pred,diag=pipeline(c,ctx[key(c)],rec,True) if arm.startswith("pipeline_") else replay.fixed(rec,c)
                obj.update(status="ok",prediction=pred,diagnostics=diag)
            except Exception as e:
                if offline:raise
                if any(a.get("http") in (401,403,429) or "error" in a for a in rec.calls):stop.set()
                obj.update(status="error",prediction={},diagnostics={"message":str(e)[:180] if not old.SECRET.search(str(e)) else "[redacted]","error_class":type(e).__name__})
            finally:llm.unbind_usage_tracker(tok);rec.client.close()
            obj["calls"]=rec.calls;obj["repair_of"]=str(p.relative_to(BASE))
        writenew(q,obj)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(job,cases))
    if stop.is_set():raise SystemExit("Provider issue; first recorded attempts retained")
    assert all((dst/(r["id"]+".json")).is_file() for r in cases)
    m={"arm":arm,"split":split,"records":len(cases),"original_request_check":"exact decoded request body, endpoint and model","change":"drop stale encoding/length only","source":"r1","offline":offline}
    new_or_same(dst/"manifest.json",m);new_or_same(dst/"complete.json",m)
    print(arm,split,"complete",len(cases),flush=True)
def statuses():
    p=BASE/"diagnostics/direct-status/decision_v6";dst=BASE/"diagnostics/direct-status-fixed/decision_v6";dst.mkdir(parents=True,exist_ok=True)
    for f in p.glob("N*.json"):
        o=rd(f);j=o["calls"][0];a=(j.get("response") or {}).get("answers",[])
        if o["status"]=="error" and len(a)==1 and a[0].get("type")=="choice" and a[0].get("name")=="item_0_task_status" and a[0].get("choice") in replay.decisions.STATUSES:
            o.update(status="ok",prediction={"task_status":a[0]["choice"]},repair_of=str(f.relative_to(BASE)));o.pop("error",None)
        new_or_same(dst/f.name,o)
    print("direct-status transport audit:",len(list(dst.glob("N*.json"))))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm");p.add_argument("split",nargs="?");p.add_argument("--online",action="store_true");p.add_argument("--workers",type=int,choices=range(1,11),default=4);a=p.parse_args()
    statuses() if a.arm=="status" else repair(a.arm,a.split,not a.online,a.workers)
