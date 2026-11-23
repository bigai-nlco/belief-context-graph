"""Neutral candidates from real logs; original answer/type/note never appear on a labeler's sheet."""
import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from tools.graph_classification_bench import freeze as old

ROOT=old.ROOT
BASE=ROOT/"results/graph-classification-500-20261119"
DS=BASE/"dataset"
CACHE=Path("/tmp/bcg-classification500-window-cache-20261119-v1")
SEED="bcg500-source-stratified-20261119-v1"
EXT="report-cases-20261021"
OLD=old.TASKS
UNSEEN={30,76,77,81,87,88,128}
EXTERNAL={10001,10004,10023,10024,10097}
RUNS=["RUN-0008","RUN-0012","RUN-0014","RUN-0015","RUN-0018","RUN-0021","RUN-0022","RUN-0023","RUN-0024","RUN-0025","RUN-0026","RUN-0028","RUN-0029",EXT]
GZ=lambda b: gzip.compress(json.dumps(b,ensure_ascii=False).encode(),compresslevel=1,mtime=0)
def h(v):return old.digest(v)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):return json.loads(p.read_text())
def runroot(r):return ROOT/"results"/EXT if r==EXT else ROOT/"results/studio"/r
def key(c):return f"{c['run']}/{c['graph']}:{c['prompt_line']}"
def task_name(t):return f"drbench:DR{t-10000:04d}" if t>=10000 else f"drb2:{t}"
def txt(s):return re.sub(r"\s+"," ",s).strip().casefold().strip(" .,;:。…")
def objkey(c):
    d=c["payload"];s=c["group"]
    if s=="node":return s,txt(d["text"])
    if s=="parent":return s,txt(d["parent"]),txt(d["child"])
    return s,*sorted([txt(d["A"]),txt(d["B"])])
def idsafe(r):
    if not (r/"graphs").is_dir():return
    pat="drbench/trials/*/agent/bcg/state.json" if r.name==EXT else "drb2/trials/*/agent/bcg/state.json"
    for state in sorted(r.glob(pat)):
        code=state.parents[2].name
        t=10000+int(code[2:]) if r.name==EXT else int(code)
        if t not in OLD|UNSEEN|EXTERNAL:continue
        with state.open("rb") as f:m=re.search(rb'"problem_id"\s*:\s*"([^"]+)',f.read(2000))
        if not m:continue
        g=r/"graphs"/m[1].decode()
        if all((g/p).is_file() for p in ("trajectory.json","events.jsonl","logs/prompts.jsonl")):yield t,g

BLOCK=re.compile(r"unable|cannot|could not|cannot find|insufficient|still missing|did not (?:find|return)|blocked|budget|exhausted|缺少|未能|找不到|不足|用尽",re.I)
READY=re.compile(r"enough|sufficient|ready|completed|finished|already (?:found|covered)|can now|no longer|decided (?:against|not)|abandon|已完成|足够|已找到|取消|放弃",re.I)
FINAL=re.compile(r"concludes?|recommends?|therefore|in sum|overall|should|best|key finding|proposes?|the answer|most (?:effective|important)|结论|建议|综上|因此|总体|应当|优先",re.I)
NEW=re.compile(r"table|matrix|timeline|report|dataset|checklist|comparative|survey|ranking|catalog|图表|表格|时间线|清单|对照|报告|列表|评估",re.I)
NUM=re.compile(r"\d+(?:[.,]\d+)?%?")
DIFF=re.compile(r"contrary|rather than|disagree|inconsistent|differ|however|not |but |contrast|whereas|correction|instead|doesn't|并非|而非|不一致|但是|然而|相反",re.I)
DEP=re.compile(r"because|therefore|depends|based on|requires?|prohibit|without|only if|consequently|used to|so that|依据|必须|依赖|由于|因此|要求|禁止",re.I)
def nodeband(c):
    d=c["payload"];t=d["text"]
    if d["speaker"]=="tool":return "tool"
    if d["speaker"]=="user":return "user"
    if c["band"]=="terminal":return "terminal_choice" if FINAL.search(t) else "terminal_claim"
    if READY.search(t):return "progress_ready"
    if BLOCK.search(t):return "progress_gap"
    if re.search(r"suggest|infer|possibly|appears|uncertain|might|likely|estimate|may |推测|可能|似乎",t,re.I):return "tentative"
    return "mixed"
def eband(d,prop):
    a,b=d["A"],d["B"]; together=a+" "+b;over=old.sim(a,b)
    numbers=[{x for x in NUM.findall(t) if len(x)>1} for t in (a,b)]
    if all(numbers) and numbers[0]!=numbers[1] and over>.19:return "numeric"
    if DIFF.search(together) and over>.20:return "contrast"
    if prop and DEP.search(together):return "dependency"
    if prop:return "proposed"
    if over>.58:return "near"
    return "control"

def mine(run,t,g):
    fn=CACHE/(run+"-"+g.name+".json.gz")
    if fn.is_file():return fn
    base,cc=old.scan(run,t,g)
    ps=old.read_rows(g/"logs/prompts.jsonl");ev=old.read_rows(g/"events.jsonl")
    traj=rd(g/"trajectory.json")["trajectory"]
    byturn={e["trajectory_index"]:e for e in ev if isinstance(e.get("trajectory_index"),int)}
    node_orig={i:(k,e.get("role")) for k,e in byturn.items() for i in (e.get("new_node_ids") or [])}
    bytxt=defaultdict(list)
    for c,_ in base:
        if c["group"]=="node":bytxt[c["turn"]].append(c["payload"])
    def provenance(id_,text):
        k,role=node_orig.get(id_,(None,None)); a=txt(text).rstrip(" …")
        cand=[d for d in bytxt[k] if txt(d["text"])==a or (len(a)>=45 and txt(d["text"]).startswith(a))]
        src=cand[0]["source_evidence"] if len(cand)==1 else []
        quote=next((str(z if isinstance(z,str) else (z.get("snippet") or z.get("title") or "")) for z in src),"")
        return {"role":role,"turn":k,"quote":quote[:320]}
    safe={};extras=[]; times={}
    for (_,_,i),v in cc.items():
        pr=ps[i-1]["prompt"]
        if old.SECRET.search(pr):continue
        k=str(i);safe[k]=v
        if v["kind"]=="semantic_node":
            from bcg.construct.unified.task_state import canonical_task_role
            v["history"]=[{**n,**({"task_role":canonical_task_role(n["task_role"])} if n.get("task_role") else {})} for n in old.array(old.section(pr,"### Earlier nodes"))]
        if v["kind"]!="edge":continue
        v["original_content"]=old.section(pr,"## Current Assistant reasoning (assistant)")
        curr={n["id"]:n["content"] for n in v["current"]};prv={n["id"]:n["content"] for n in v["previous"]}; texts={**prv,**curr}
        turn=next((node_orig[n][0] for n in curr if n in node_orig),None)
        if turn is None:continue
        e=byturn[turn];seen=set()
        try: prop=json.loads(str(e.get("raw_relation_output") or "{}").strip().removeprefix("```json").removesuffix("```")).get("relations",[])
        except (ValueError,AttributeError):prop=[]
        # Ignore the historical type and note. Proposal-only sampling enriches strong candidates;
        # predictions are never references, and absence is never a negative gold label.
        pairset={(z.get("from"),z.get("to")) for z in prop if isinstance(z,dict)}
        for a in curr:
            peers=sorted(prv,key=lambda b:-old.sim(texts[a],texts[b]))[:3]
            for b in peers:seen.add((a,b))
        pairs=seen|pairset|{(b,a) for a,b in pairset if a in curr and b in prv and int(h([run,i,a,b])[:3],16)%4==0}
        for a,b in sorted((a,b) for a,b in pairs if type(a)is int and type(b)is int):
            if a==b or not {a,b}<=set(texts) or not {a,b}&set(curr) or (v["role"]=="tool" and len({a,b}&set(curr))!=1):continue
            if any({z.get("from",z.get("from_id")),z.get("to",z.get("to_id"))}=={a,b} for z in v["existing"]):continue
            if not all(old.valid_text(texts[n]) for n in (a,b)):continue
            d={"A":texts[a],"B":texts[b],"from_id":a,"to_id":b,"current_ids":sorted(curr),"speaker":v["role"]}
            bd=eband(d,(a,b) in pairset)
            if a in prv and b in curr and bd=="control":bd="reverse"
            meta={"A":provenance(a,texts[a]),"B":provenance(b,texts[b])}
            co={"group":"edge","task":t,"turn":turn,"run":run,"graph":g.name,"prompt_line":i,"payload":d,"band":bd,"endpoint_evidence":meta}
            co["uid"]=h([run,g.name,i,"edge",a,b])[:20]
            extras.append(co)
    arr=[]
    for c,_ in base:
        if str(c["prompt_line"]) not in safe or c["group"]=="edge":continue
        if c["group"]=="node":c["band"]=nodeband(c)
        else:
            d=c["payload"]; t0=d["child"]
            if d.get("candidate_origin")=="verbatim_work_clause":
                c["band"]="work_product" if NEW.search(t0) else "work_scope"
            elif old.ACTION.search(t0):c["band"]="method"
            elif old.sim(d["parent"],t0)>.58:c["band"]="restated"
            else:c["band"]="outcome" if NEW.search(t0) else "adjacent"
        arr.append(c)
    grouped=arr+extras
    fn.write_bytes(GZ([grouped,safe]))
    return fn

QUOTA={
 "node":{"user":10,"tool":14,"terminal_choice":16,"terminal_claim":6,"progress_ready":13,"progress_gap":13,"tentative":16,"mixed":12},
 "parent":{"work_product":14,"work_scope":16,"method":7,"restated":6,"outcome":10,"adjacent":7},
 "edge":{"dependency":18,"proposed":15,"numeric":18,"contrast":15,"near":9,"reverse":6,"control":9}}

def select():
    if (DS/"cases.jsonl").exists():raise SystemExit("Case set already frozen")
    CACHE.mkdir(exist_ok=True,mode=0o700);DS.mkdir(parents=True,exist_ok=True)
    legacy=old.read_rows(old.BASE/"dataset/cases.jsonl")
    used={objkey(r) for r in legacy};excluded={key(r) for r in legacy}
    for j in rd(old.BASE/"supplement/sealed.json"):
        used.add(("edge",*sorted([txt(j["A"]["text"]),txt(j["B"]["text"])])))
    pools=defaultdict(list);files={};inv=[]
    for run in RUNS:
        for t,g in idsafe(runroot(run)):
            f=mine(run,t,g);files[run,g.name]=f
            arr,_=json.loads(gzip.decompress(f.read_bytes())); n=Counter()
            for c in arr:
                if key(c) in excluded or objkey(c) in used:continue
                if old.SECRET.search(json.dumps(c,ensure_ascii=False)):continue
                s="dev" if t in OLD else "test";n[c["group"]]+=1
                pools[s,c["group"],c["band"]].append(c)
            inv.append([run,t,g.name,dict(n)])
        print("source scan",run,"candidates",sum(len(v) for v in pools.values()),flush=True)
    selected=[];count=Counter();near=defaultdict(list);usedturn=Counter()
    def take(s,group,band):
        pool=pools[s,group,band]
        # Round-robin tasks/runs within source stratum. Hash order never depends on predictions.
        for c in sorted(pool,key=lambda c:(count[s,group,c["task"]]*3+count[s,c["task"]]*.34+
                  count[s,group,c["run"]]*.13+usedturn[s,c["task"],c["turn"],group]*2.5,
                  h([SEED,c["uid"]]))):
            k=objkey(c)
            if k in used or usedturn[s,c["run"],c["graph"],c["turn"],group]>=2:continue
            if any(k[1:]==z[1:] for z in near[group,c["task"]]):continue
            body=" ".join(k[1:]);vocab=set(re.findall(r"\w+|[\u3400-\u9fff]",body))
            if len(vocab)>7 and any(len(vocab&vv)/max(1,len(vocab|vv))>.89 for vv in near["vocab",group,c["task"]]):continue
            used.add(k);near[group,c["task"]].append(k);near["vocab",group,c["task"]].append(vocab)
            t=c["task"];count[s,group,t]+=1;count[s,t]+=1;count[s,group,c["run"]]+=1
            usedturn[s,t,c["turn"],group]+=1;usedturn[s,c["run"],c["graph"],c["turn"],group]+=1
            co=dict(c,split=s,task_name=task_name(t))
            selected.append(co);count[s,group,band,"selected"]+=1
            return True
        return False
    for s in ("dev","test"):
        for group,quota in QUOTA.items():
            for i in range(max(quota.values())):
                for band,n in quota.items():
                    if i<n:take(s,group,band)
            while sum(count[s,group,b,"selected"] for b in quota)<sum(quota.values()):
                if not any(take(s,group,b) for b in sorted(quota,key=lambda b:count[s,group,b,"selected"]/quota[b])):
                    raise RuntimeError(f"Insufficient {s}/{group}")
    selected.sort(key=lambda c:(c["group"],h([SEED,c["uid"],"order"])))
    groups=Counter();ctx={};byfile=defaultdict(list)
    for c in selected:
        groups[c["group"]]+=1;c["id"]=c["group"][0].upper()+f"{groups[c['group']]:03d}"
        byfile[c["run"],c["graph"]].append(c)
    for source,rows in byfile.items():
        _,cc=json.loads(gzip.decompress(files[source].read_bytes()))
        for c in rows:ctx[key(c)]=cc[str(c["prompt_line"])]
    old.jsonl(DS/"cases.jsonl",selected);old.dump(DS/"native_context.json",ctx)
    old.dump(DS/"source_lock.json",{"seed":SEED,"cases":sha(DS/"cases.jsonl"),"context":sha(DS/"native_context.json"),
        "legacy_cases_hash":sha(old.BASE/"dataset/cases.jsonl"),"legacy_windows_excluded":len(excluded),"inventory":inv,
        "split_tasks":{s:sorted({c["task_name"] for c in selected if c["split"]==s}) for s in ("dev","test")},
        "counts":{s:{g:dict(Counter(c["band"] for c in selected if c["group"]==g and c["split"]==s)) for g in QUOTA} for s in ("dev","test")}})
    print("Frozen",len(selected),groups)

def blind(group,lo,hi,size):
    for c in old.read_rows(DS/"cases.jsonl"):
        if c["group"]!=group or not lo<=int(c["id"][1:])<=hi:continue
        d=c["payload"]; print(c["id"], d.get("speaker",""), "thinking" if d.get("thinking_only") else "")
        for k in ("text","A","B","parent","child","source_evidence"):
            if k in d:
                v=d[k] if isinstance(d[k],str) else " | ".join(x if isinstance(x,str) else (x.get("snippet") or x.get("title") or "") for x in d[k])
                print(k+":", " ".join(v.split())[:size])
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["freeze","read"]);p.add_argument("group",nargs="?");p.add_argument("lo",type=int,nargs="?",default=1);p.add_argument("hi",type=int,nargs="?",default=30);p.add_argument("--width",type=int,default=430);a=p.parse_args()
    select() if a.command=="freeze" else blind(a.group,a.lo,a.hi,a.width)
