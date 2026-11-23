"""One pre-bounded probability decoder. Development estimates are cross-fit by original task."""
import json
import math
from collections import Counter
from .labels import truth,once
from .data import BASE,DS,rd,h,sha,old
from .run import spec,choose,decoded,writenew
from tools.graph_classification_bench.score import labels_stats
GRID=[(p,d) for p in (1,.75,.5,.35,.25) for d in (1,1.65,2.5)]
def fit(items,kind):
    clean=[(a,g) for a,g,f in items if f==kind]
    best=(-1,{});diag=[]
    for non,de in GRID:
        if kind=="kind" and de!=1:continue
        weights={"none":non,**({"depends_on":de} if kind=="type" else {})}
        metric=labels_stats([(g,choose(a,weights)) for a,g in clean])["macro_f1"]
        val=metric-.018*(abs(math.log(non))+abs(math.log(de)))
        diag.append(dict(weights=weights,f1=metric,objective=val))
        if val>best[0]:best=(val,weights)
    return best[1],diag
def freeze():
    gt=truth();cs=old.read_rows(DS/"cases.jsonl");dev=[c for c in cs if c["split"]=="dev"]
    src=BASE/"experiments/decision_v4/dev/r1";dst=BASE/"experiments/decision_v6/dev/r1";dst.mkdir(parents=True,exist_ok=True)
    cfgdir=BASE/"configs";cfgdir.mkdir(exist_ok=True)
    items=[];orig={};labels_by_case={}
    tasks=sorted({c["task"] for c in dev},key=lambda t:h(["fold",t]));fold={t:i%5 for i,t in enumerate(tasks)}
    for c in dev:
        o=rd(src/(c["id"]+".json"));orig[c["id"]]=o
        if c["group"]=="node" or o["status"]!="ok":continue
        f="kind" if c["group"]=="parent" else "type";a=o["calls"][0]["response"]["answers"][0];labels_by_case[c["id"]]=(f,a)
        if f not in gt[c["id"]]["ambiguous"]:items.append((fold[c["task"]],a,gt[c["id"]]["labels"][f],f))
    params={};diags={}
    for held in [*range(5),-1]:
        results={f:fit([(a,g,k) for i,a,g,k in items if i!=held],f) for f in ("kind","type")}
        params[held]={f:v[0] for f,v in results.items()};diags[held]={f:v[1] for f,v in results.items()}
    cfg={"questions":spec("v4")["questions"],"calibration":params[-1],"decoder":"fixed multiplicative prior for none / depends_on, floor .12; protect raw contradicts"}
    once(cfgdir/"v6.json",(json.dumps(cfg,indent=2,sort_keys=True)+"\n").encode())
    old.dump(cfgdir/"v6-fit-audit.json",{"fold_by_task":fold,"grid":GRID,"fold_params":params,"fit_objectives":diags})
    for c in dev:
        o=orig[c["id"]];p=dst/(c["id"]+".json");o.update(arm="decision_v6",fold=fold[c["task"]],reused_call_artifact=str((src/p.name).relative_to(BASE)))
        if c["id"] in labels_by_case:
            f,a=labels_by_case[c["id"]];o["prediction"][f]=choose(a,params[fold[c["task"]]][f])
            o["calls"][0]["applied_answers"]=[{"name":a["name"],"choice":o["prediction"][f]}]
        if p.exists():assert rd(p)==o
        else:writenew(p,o)
    old.dump(dst/"manifest.json",{"arm":"decision_v6","split":"dev","calibration":"out of fold, grouped by task; all network calls reused byte-for-byte from decision_v4","config":cfg,"gold_lock":sha(DS/"blind.lock.json")})
    print("v6 fold weights",params)
if __name__=="__main__":freeze()
