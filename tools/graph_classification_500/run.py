"""Fixed-input Decisions/Chat and actual graph proposal-stage replay. Never loads reference answers."""
import argparse
import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FunctionType
import httpx
from bcg.construct.unified import llm
from tools.graph_classification_bench import freeze as old, replay, native_stages
from .data import BASE, DS, ROOT, EXT, key, runroot, rd, sha, txt

HERE=Path(__file__).parent
def spec(v):
    if v in ("v1","v4"):return {"questions":replay.versions("decision_"+v)}
    if v=="v6":return rd(BASE/"configs/v6.json")
    p=HERE/"variants"/(v+".json");cfg=rd(p)
    if "inherit" in cfg:return {**spec(cfg["inherit"]),**cfg}
    return cfg
def choose(answer,weight):
    """Only Decision probabilities for this candidate, never propagated graph confidence."""
    initial=answer.get("choice");ps={p["value"]:p["probability"] for p in answer.get("probabilities",[]) if isinstance(p.get("probability"),(int,float))}
    if not ps or initial not in ps or initial=="contradicts":return initial
    # Do not promote unseen conflict classes or very weak alternatives.
    opts=[k for k,p in ps.items() if k==initial or (k!="contradicts" and p>=.12)]
    return max(opts,key=lambda k:(ps[k]*weight.get(k,1.),k==initial))
def decoded(body,config):
    body=copy.deepcopy(body)
    for ans in body.get("answers",[]):
        field=ans["name"].split("_",2)[2];weight=config.get(field)
        if weight:ans["choice"]=choose(ans,weight)
    return body
def transformed(body,version,meta):
    spec_=spec(version);p=replay.change(body,spec_["questions"])
    if spec_.get("provenance") and any(q["name"].endswith("_type") and not q["name"].endswith("_node_type") for q in p["questions"]):
        rows=json.loads(p["input"])
        for row in rows:
            for e in ("A","B"):
                prov=meta.get(txt(row[e])) or {}
                if prov.get("role") or prov.get("quote"):
                    row[e+"_source"]={k:v for k,v in prov.items() if k in ("role","quote") and v}
        p["input"]=json.dumps(rows,ensure_ascii=False)
    if spec_.get("edge_probes"):
        qs=[]
        for q in p["questions"]:
            if q["name"].split("_",2)[2]!="type":qs.append(q);continue
            item=q["name"].split("_")[1]
            for channel,instruction in spec_["edge_probes"].items():
                qs.append({"name":f"item_{item}_{channel}","type":"choice",
                   "instructions":f"Evaluate only item {item}; source quotations clarify only the endpoints. "+instruction,
                   "choices":[{"value":"yes","description":"The pair satisfies this specific test."},{"value":"no","description":"The pair does not establish this test."}]})
        p["questions"]=qs
    return p
def combine(original,body):
    """Reconstitute one semantic label per original pair, including for the same-questions Chat control."""
    answers=body.get("answers") or [];by={a["name"]:a for a in answers};out=[]
    if len(by)!=len(answers):raise ValueError("Duplicate probe answer")
    for q in original["questions"]:
        name=q["name"]
        if name.split("_",2)[2]!="type":
            out.append(by.pop(name));continue
        p=name.rsplit("_",1)[0]+"_";bits=[by.pop(p+k)["choice"] for k in ("clash","uses","adds")]
        if not all(b in ("yes","no") for b in bits):raise ValueError("Invalid binary decision")
        value=next((v for b,v in zip(bits,("contradicts","depends_on","supplements")) if b=="yes"),"none")
        out.append({"name":name,"type":"choice","choice":value})
    if by:raise ValueError("Unexpected probe")
    return {**body,"answers":out}
class Live(replay.Recorder):
    def __init__(self,cfg,arm,ver,meta):
        super().__init__(cfg,"chat_fixed" if arm.startswith("chat") else "decision",{})
        self.ver=ver;self.meta=meta
    def post(self,url,*,headers,json,timeout,follow_redirects):
        resp=super().post(url,headers=headers,json=transformed(json,self.ver,self.meta),timeout=timeout,follow_redirects=False)
        if self.ver=="v6" and self.arm!="chat_fixed" and resp.is_success:
            data=decoded(resp.json(),spec(self.ver)["calibration"])
            self.calls[-1]["applied_answers"]=[{"name":a["name"],"choice":a["choice"]} for a in data["answers"]]
            return local_json(resp,data)
        if self.ver=="v7" and any(q["name"].split("_",2)[2]=="type" for q in json["questions"]) and resp.is_success:
            data=combine(json,resp.json());self.calls[-1]["applied_answers"]=data["answers"]
            return local_json(resp,data)
        return resp

def local_json(original,body):
    # The original HTTP entity may have been gzip-encoded. Our new in-memory JSON
    # entity is not. Never copy the old Content-Encoding or old Content-Length.
    price=original.headers.get("x-litellm-response-cost")
    return httpx.Response(original.status_code,headers={"x-litellm-response-cost":price} if price is not None else {},request=original.request,json=body)

class _SourcePaths:
    def __truediv__(self,part):
        if part!="results/studio":raise ValueError(part)
        return _Runs()
class _Runs:
    def __truediv__(self,run):return runroot(run)
f=native_stages.paired_arm
# Only the storage location of historical trajectories differs for the older archive.
pipeline=FunctionType(f.__code__,{**f.__globals__,"ROOT":_SourcePaths()},f.__name__,f.__defaults__,f.__closure__)
def armver(arm):
    return arm.rsplit("_",1)[1] if "_" in arm and arm.rsplit("_",1)[1].startswith("v") else "v1"
def writenew(p,data):
    with p.open("x",encoding="utf8") as f:json.dump(data,f,ensure_ascii=False,indent=1)
def new_or_same(p,o):
    if p.exists():
        if rd(p)!=o:raise RuntimeError(f"Different fingerprint in {p}; new trial required")
    else:writenew(p,o)
def data_for(split):
    if split=="regression":
        return old.read_rows(old.BASE/"dataset/cases.jsonl"),rd(old.BASE/"dataset/native_context.json")
    if split=="challenge":
        return old.read_rows(DS/"challenge.jsonl"),{}
    return [c for c in old.read_rows(DS/"cases.jsonl") if c["split"]==split],rd(DS/"native_context.json")
def evidence(c,extras):
    meta=extras.get(key(c),{}).copy()
    if c.get("endpoint_evidence"):
        for k,v in c["endpoint_evidence"].items():meta[txt(c["payload"][k])]=v
    return meta
def run(arms,split,workers,trial,limit,ids):
    sealed=rd(DS/"blind.lock.json")
    assert all(sha(BASE/p)==v for p,v in sealed.items())
    lock=BASE/"selection.lock.json"
    if split not in ("dev",) and not lock.exists():raise SystemExit("No model selection lock")
    if split not in ("dev",):
        sl=rd(lock)
        if any(armver(a) not in {"v1","v4",sl["version"]} for a in arms):raise ValueError("Holdout: locked arms only")
        if sl["spec"]!=spec(sl["version"]):raise ValueError("Locked configuration changed")
    rows,contexts=data_for(split);cfg=llm.load_config(str(replay.PROFILE),model_key="graph")
    assert cfg["model"]=="gpt-6-luna" and cfg["base_url"].rstrip("/").removesuffix("/v1")=="https://litellm.mybigai.ac.cn"
    extras=rd(DS/"endpoint_provenance.json") if (DS/"endpoint_provenance.json").exists() and any(spec(armver(a)).get("provenance") for a in arms) else {}
    jobs=[];dirs={}
    for arm in arms:
        if arm not in ("native_current","chat_fixed") and not any(arm.startswith(p) for p in ("decision_v","chat_v","pipeline_v")):raise ValueError(arm)
        ver=armver(arm);where=BASE/"experiments"/arm/split/trial;where.mkdir(parents=True,exist_ok=True);dirs[arm]=where
        manifest={"arm":arm,"split":split,"version":spec(ver),"dataset_lock":sha(DS/"blind.lock.json"),"selection":sha(lock) if split!="dev" else None,
            "model":"gpt-6-luna","reasoning_effort":"none","decision_model":"gpt-6-luna-decision","provider":"https://litellm.mybigai.ac.cn/v1",
            "dependencies":{str(p.relative_to(ROOT)):sha(p) for p in [Path(replay.__file__),Path(native_stages.__file__),ROOT/"bcg/construct/unified/decisions.py",ROOT/"bcg/construct/unified/prompts.py",replay.PROFILE]},
            "source_provenance":sha(DS/"endpoint_provenance.json") if spec(ver).get("provenance") and extras else None}
        new_or_same(where/"manifest.json",manifest)
        requested=sorted((c for c in rows if not ids or c["id"] in ids),key=lambda c:old.digest(["order",c["uid"]]))
        for c in requested[:limit]:
            if not (where/(c["id"]+".json")).is_file():jobs.append((arm,c))
    jobs.sort(key=lambda ac:old.digest(["jobs",ac[1]["uid"],ac[0]]))
    lock_=threading.Lock();stop=threading.Event();done=errors=transport=0
    def one(pair):
        nonlocal done,errors,transport
        if stop.is_set():return
        arm,c=pair;ver=armver(arm);where=dirs[arm]
        rec=Live(cfg,arm,ver,evidence(c,extras));tracker=llm.bind_usage_tracker(llm.TokenUsageTracker())
        o={"arm":arm,"id":c["id"],"uid":c["uid"],"case_hash":old.digest(c)}
        try:
            if arm=="native_current" or arm.startswith("pipeline_"):
                pred,why=pipeline(c,contexts[key(c)],rec,arm.startswith("pipeline_"))
            else:pred,why=replay.fixed(rec,c)
            o.update(prediction=pred,diagnostics=why,status="ok")
        except Exception as e:
            msg=str(e);o.update(status="error",prediction={},diagnostics={"error_class":type(e).__name__,"message":msg[:240] if not old.SECRET.search(msg) else "[redacted]"})
        finally:rec.client.close();llm.unbind_usage_tracker(tracker)
        o["calls"]=rec.calls;writenew(where/(c["id"]+".json"),o)
        with lock_:
            done+=1;errors+=o["status"]!="ok"
            h=[a.get("http") or (a.get("error") or {}).get("http") for a in rec.calls]
            network=any("error" in a or (isinstance(status,int) and status>=400) for a,status in zip(rec.calls,h))
            transport=transport+1 if network else 0
            if any(code in (401,403,429) for code in h) or transport>=3:stop.set()
            if done%40==0 or o["status"]!="ok" or len(jobs)<=8:
                print(f"{done}/{len(jobs)} {arm} {c['id']} {o['status']} {o['diagnostics'] if o['status']!='ok' else ''}",flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(one,jobs))
    print("new",done,"errors",errors,"circuit",stop.is_set(),flush=True)
    if stop.is_set():raise SystemExit(2)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["dev","test","regression","challenge"]);p.add_argument("arms",nargs="+");p.add_argument("--workers",type=int,choices=range(1,13),default=5);p.add_argument("--trial",default="r1");p.add_argument("--limit",type=int);p.add_argument("--ids",nargs="+");a=p.parse_args()
    run(a.arms,a.split,a.workers,a.trial,a.limit,a.ids)
