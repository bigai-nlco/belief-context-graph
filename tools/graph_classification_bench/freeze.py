"""Freeze real, pre-verdict candidates. Never use historical label values as answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "graph-classification-bench-20261118"
SEED = "DRB2-graph-boundaries:20261118:v1"
TASKS = {2,5,8,9,23,27,37,38,49,52,58,93,96,98,100,101,103,105,108,110,119,120,121}
RUNS = ("RUN-0018","RUN-0021","RUN-0028","RUN-0029")
STOP = set("the and that from with this assistant user believes intends wants needs report research should must source sources which these about only their search results requested".split())
FILLER = re.compile(r"\[BLOCKED\]|no results|budget exhausted|is using .{0,30}to |tool_call_id|not accessible for this task|The web_search tool returned:", re.I)
SECRET = re.compile(r"(?i)(?:bearer\s+[A-Za-z0-9_.=-]{12,}|(?:sk-|ghp_|gho_|AIza)[A-Za-z0-9_-]{17,}|(?:api[_ -]?key|access[_ -]?token)\s*[:=]\s*[A-Za-z0-9_-]{12,}|BEGIN (?:RSA )?PRIVATE KEY)")
PROGRESS = re.compile(r"already|still|yet|unable|missing|found|enough|insufficient|remaining|lack|could not|so far|not (?:found|able)|blocked|已|还缺|仍|未能|不足|尚", re.I)
BOUNDARY = re.compile(r"however|although|instead|rather|unlike|only|not |may |might|uncertain|estimat|infer|compared|contradict|来源|可能|但|仅|推断|不一定|相比", re.I)
WORK = re.compile(r"compar|distinguish|determin|table|classif|verify|review|assess|investigat|specific|summari|identify|section|deliver|organize|extract|need|search|分析|表格|需要|查找|核对",re.I)
FINAL = re.compile(r"final (?:answer|report|conclusion)|in conclusion|concludes|最终|综上",re.I)
ACTION = re.compile(r"(?:plans?|intends?|wants?) to (?:continue |try |use |directly )?(?:search|fetch|retrieve|look for|recover|access|read)|continue searching",re.I)
ASSERTION = re.compile(r"found that|states? that|reported that|indicates?|according to|compared|whereas|\d+(?:\.\d+)?%|research (?:finds|shows|suggests)|specified that|evidence suggests|findings? (?:show|indicate)|认为|发现|证据表明",re.I)
DELIVER = re.compile(r"\b(?:table|calculate|calculation|comparison|classif(?:y|ication)|breakdown|synthesi[sz]e|sections?|matrix|timeline|divide|distinguish|summar(?:y|ize)|produce|estimate|measure|report should|user (?:wants|requires|asks))\b",re.I)
BOILER = re.compile(r"fetch unavailable|unavailable for this task|tool returned: (?:source:|error)|web search limit reached|search budget exhausted|continue (?:searching|retrieving|fetching)|search results (?:were|are) blocked",re.I)
BAN = re.compile(r"must not (?:view|use|consult)|prohibits viewing|(?:article|urls?).{0,35}not be (?:viewed|accessed)|must not be accessed|blocked source",re.I)
OUTCOME = re.compile(r"\b(?:explain|compare|assess|determine|classif\w*|analy[sz]\w*|prepare|develop|construct|quantif\w*|summari[sz]\w*|calculate|table|chart|dataset|timeline|list|catalog|matri[xc]|breakdown|typology|synthesi\w*|identif\w*)\b|表格|比较|对比|分类|汇总|阐述|分析|概述|解释|整理|区分|统计|计算|总结|列出|评估|说明",re.I)
ARTIFACT = re.compile(r"\b(?:table|matrix|chart|timeline|checklist|figure|dataset|worksheet|index|indices|ranking|map)\b|表格|列表|矩阵|图表|时间线|清单|指数|指标",re.I)
SEARCH_FIRST=re.compile(r"^(?:(?:I |we )?(?:need to |will |should |must )?)?(?:search|fetch|browse|retrieve|look for|read (?:more|the))\b|^(?:我需要|我要|要|先)?(?:搜索|寻找|检索|抓取|查找)",re.I)

def digest(x):
    return hashlib.sha256((x if isinstance(x, str) else json.dumps(x,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()

def tokens(x):
    return (set(re.findall(r"[a-z]{3,}|\d+(?:\.\d+)?",x.lower()))-STOP) | set(re.findall(r"(?=([\u3400-\u9fff]{2}))",x))

def sim(a,b):
    a,b=tokens(a),tokens(b)
    return len(a&b)/max(1,(len(a)*len(b))**.5)

def section(prompt, heading):
    m=re.search(r"(?m)^"+re.escape(heading)+r"\s*\n",prompt)
    if not m: return ""
    tail=prompt[m.end():]
    # Source is the final section. Its own embedded Markdown headings are data.
    return (tail if heading=="## Current turn sentences" else re.split(r"(?m)^#{1,3} (?!\d)",tail,maxsplit=1)[0]).strip()

def array(s):
    start=s.find("[")
    if start<0: return []
    try: return json.JSONDecoder().raw_decode(s[start:])[0]
    except ValueError: return []

def read_rows(p):
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def dump(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")

def jsonl(path, rows):
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))

def sentences(block):
    ms=list(re.finditer(r"(?m)^\[(\d+)\]\s*",block))
    return {int(m[1]):block[m.end():(ms[i+1].start() if i+1<len(ms) else len(block))].strip()
            for i,m in enumerate(ms)}

def neutral_earlier(prompt):
    # 'goal/subgoal/task' here identifies an originally visible PARENT candidate only,
    # never the target child's label. Do not import earlier predictions on the target.
    raw=array(section(prompt,"### Earlier nodes"))
    return [{"id":a["id"],"content":a["content"],**({"task_role":"task"} if a.get("task_role") in {"goal","subgoal","task"} else {})}
            for a in raw if type(a.get("id")) is int and isinstance(a.get("content"),str)]

def valid_text(x):
    return isinstance(x,str) and 24<=len(x)<=720 and not FILLER.search(x) and not SECRET.search(x)

def scan(run, tid, graph):
    prompts=read_rows(graph/"logs/prompts.jsonl")
    events=read_rows(graph/"events.jsonl")
    trajectory=json.loads((graph/"trajectory.json").read_text())["trajectory"]
    by_role=defaultdict(list)
    for ix,p in enumerate(prompts):
        text=p.get("prompt","")
        for role in ("user","assistant"):
            if f"## Source role: {role.upper()}" in text:
                by_role[role].append((ix,p))
    context={}
    cases=[]
    def record(group,pidx,turn,data,band,rank=0):
        common={"group":group,"run":run,"task":tid,"turn":turn,"graph":graph.name,"prompt_line":pidx+1}
        key=digest([SEED,common,data])[:18]
        cases.append((dict(common,uid=key,band=band,payload=data),rank))
    def put(i,kind,**kwargs):
        context[(run,graph.name,i+1)]={"kind":kind,"prompt_sha256":digest(prompts[i]["prompt"]),**kwargs}
    for event in events:
        turn=event.get("trajectory_index")
        role=event.get("role")
        if type(turn) is not int or not (0<=turn<len(trajectory)) or role not in ("user","assistant","tool"): continue
        try: result=json.loads(str(event["raw_output"]).strip().removeprefix("```json").removesuffix("```").strip())
        except (TypeError,ValueError): continue
        if not isinstance(result,dict):continue
        if role=="tool":
            for ix in reversed([i for i,p in enumerate(prompts) if p["ts"]<=event["ts"] and "\nItems:\n" in p["prompt"]][-2:]):
                items=array(prompts[ix]["prompt"].split("\nItems:\n")[-1])
                matched=[]
                for item in result.get("items",[]):
                    index=item.get("item_index")
                    source=next((y for y in items if y.get("item_index")==index),{})
                    for n in item.get("beliefs",[]):
                        text=n.get("belief"); rank=n.get("source_rank")
                        evidence=[{k:z[k] for k in ("rank","title","snippet") if k in z} for z in source.get("results",[]) if z.get("rank")==rank]
                        if not valid_text(text) or len(evidence)!=1:continue
                        raw=str(trajectory[turn].get("content",""))
                        fields=[v for k,v in evidence[0].items() if k!="rank" and isinstance(v,str)]
                        if not any(any(" ".join(s.split()[i:i+4]).casefold() in raw.casefold()
                                       for i in range(0,min(len(s.split())-3,12))) for s in fields):continue
                        if sim(text," ".join(fields))<.21:continue
                        matched.append((text,index,rank,evidence))
                if not matched:continue
                put(ix,"tool_facts",items=items,original_prompt=prompts[ix]["prompt"])
                for text,index,rank,evidence in matched:
                    record("node",ix,turn,{"speaker":"tool","text":text,"item_index":index,"source_rank":rank,
                                          "source_evidence":evidence},"tool",.3+bool(BOUNDARY.search(text)))
                break
            continue
        candidates=[(ix,p) for ix,p in by_role[role] if p["ts"]<=event["ts"]]
        if not candidates:continue
        ix,p=candidates[-1]; prompt=p["prompt"]; block=section(prompt,"## Current turn sentences")
        indexed=sentences(block); raw=str(trajectory[turn].get("content",""))
        if not indexed or not any(s.strip("<>/thinking ")[:50] in raw for s in indexed.values() if len(s)>20):continue
        history=neutral_earlier(prompt)
        # Real numbered work items can be covered by one broad historical node.
        # Keep the stated clause verbatim, even if the old extractor missed it.
        # Only earlier visible anchors are eligible, never the final tree.
        anchors=[h for h in history if h.get("task_role")=="task" and valid_text(h["content"])]
        if role=="assistant" and anchors and turn<=16:
            for i,s in indexed.items():
                if not re.match(r"^\s*(?:\d+[.):、]|[-*]\s|[A-F][.):])",s):continue
                t=re.sub(r"^\s*(?:\d+[.):、]|[-*]|[A-F][.):])\s*","",s).strip()
                if (not valid_text(t) or len(t)>370 or not OUTCOME.search(t) or SEARCH_FIRST.search(t)
                    or re.search(r"https?://|must not|不得|不要使用",t,re.I)):continue
                broad=min(anchors,key=lambda h:h["id"])
                put(ix,"semantic_node",role=role,sentences=block,history=history)
                record("parent",ix,turn,{"parent_id":broad["id"],"parent":broad["content"],"child":t,
                        "source_evidence":[s],"indices":[i],"candidate_origin":"verbatim_work_clause"},
                       "artifact_clause" if ARTIFACT.search(t) else "scope_clause",.8)
        # Only neutral text and supporting indices leave this step. Do NOT use the
        # containing list (belief vs decision), stance, task_role, task_ids or graph edges.
        for n in [*result.get("beliefs",[]),*result.get("decisions",[])]:
            if not isinstance(n,dict):continue
            text=n.get("belief") or n.get("decision"); refs=n.get("supporting_sentence_indices") or []
            if not valid_text(text) or not refs or not all(type(i) is int and i in indexed for i in refs):continue
            evidence=[indexed[i] for i in refs]
            if any(len(s)>1300 or SECRET.search(s) for s in evidence) or sum(map(len,evidence))>1900:continue
            if any(s.strip().strip("<>") in {"thinking","/thinking"} for s in evidence):continue
            put(ix,"semantic_node",role=role,sentences=block,history=history)
            payload={"speaker":role,"text":text,"indices":refs,"source_evidence":evidence,"thinking_only":raw.strip().startswith("<thinking>") and not re.sub(r"(?s)<tool_call>.*?</tool_call>","",raw[raw.rfind("</thinking>")+len("</thinking>"):]).strip()}
            terminal=role=="assistant" and turn==len(trajectory)-1 and any(i>=3 for i in refs)
            band=("user" if role=="user" else "terminal" if terminal else "fact" if ASSERTION.search(text) and not ACTION.search(text)
                  else "progress" if PROGRESS.search(text) else "reason")
            record("node",ix,turn,payload,band,.6*bool(BOUNDARY.search(text))+.23*bool(WORK.search(text)))
            if role!="assistant" or not (WORK.search(text) or DELIVER.search(text)):continue
            parents=[h for h in history if h.get("task_role")=="task" and valid_text(h["content"])]
            parents=sorted(parents,key=lambda h:-sim(text,h["content"]))
            # Include broader objectives, not only the closest lexical match; these
            # are genuine earlier anchors visible at this call, not invented examples.
            picked={h["id"]:h for h in parents[:3]}
            if parents:
                broad=min(parents,key=lambda h:h["id"]); picked[broad["id"]]=broad
            for order,parent in enumerate(picked.values()):
                overlap=sim(text,parent["content"])
                if overlap<.08:continue
                method=ACTION.search(text)
                band=("method" if method else "restated" if overlap>.66 else
                      "specific" if DELIVER.search(text) and int(parent["id"])<=1 else
                      "output" if DELIVER.search(text) else "adjacent")
                record("parent",ix,turn,{"parent_id":parent["id"],"parent":parent["content"],"child":text,
                        "source_evidence":evidence,"indices":refs},band, .3*bool(BOUNDARY.search(text)) + min(.48,overlap))
    for ix,p in enumerate(prompts):
        pr=p.get("prompt",""); assistant="## Candidate previous nodes" in pr; tool="## Candidate prior Thinking nodes" in pr
        if not assistant and not tool or "## Validation feedback" in pr:continue
        prev=array(section(pr,"## Candidate previous nodes" if assistant else "## Candidate prior Thinking nodes"))
        curr=array(section(pr,"## Current-turn nodes" if assistant else "## Current Tool Result nodes"))
        if not prev or not curr:continue
        # Match prompt to the actual arrival event using current-turn node ids, not selected edges.
        ids={n["id"] for n in curr}; related=[e for e in events if ids.intersection(e.get("new_node_ids") or [])]
        if not related:continue
        turn=related[0]["trajectory_index"]; existing=array(section(pr,"## Existing relations"))
        put(ix,"edge",role="assistant" if assistant else "tool",previous=prev,current=curr,existing=existing)
        for a in curr:
            if not valid_text(a["content"]) or BOILER.search(a["content"]):continue
            eligible=[b for b in prev if valid_text(b["content"]) and sim(a["content"],b["content"])>=.12]
            eligible.sort(key=lambda b:-sim(a["content"],b["content"]))
            for b in eligible[:3]:
                if a["id"]==b["id"] or BOILER.search(b["content"]):continue
                # The direction is part of the question, not copied from any output.
                reverse=int(digest([SEED,tid,ix,a["id"],b["id"]])[:4],16)%5==0
                fr,to=(b,a) if reverse else (a,b)
                if ACTION.search(a["content"]) and ACTION.search(b["content"]) and sim(a["content"],b["content"])>.76:continue
                band="reverse" if reverse else "tool" if tool else "fact" if ASSERTION.search(a["content"]+" "+b["content"]) else "qualified" if BOUNDARY.search(a["content"]+" "+b["content"]) else "related"
                record("edge",ix,turn,{"A":fr["content"],"B":to["content"],"from_id":fr["id"],"to_id":to["id"],
                       "current_ids":sorted(ids),"speaker":"assistant" if assistant else "tool"},band,
                       min(.65,sim(fr["content"],to["content"]))+.25*bool(BOUNDARY.search(fr["content"]+" "+to["content"])))
    # Sample attributable factual clauses from the actual compact tool-extraction input.
    # Require a contiguous phrase in one original tool turn (not a keyword match).
    tool_inputs=[(i,str(m.get("content","")).casefold()) for i,m in enumerate(trajectory) if m.get("role")=="tool"]
    for ix,p in enumerate(prompts):
        pr=p.get("prompt","")
        if "\nItems:\n" not in pr:continue
        items=array(pr.split("\nItems:\n")[-1])
        for item in items:
            for z in item.get("results",[]):
                raw=str(z.get("snippet") or z.get("title") or "")
                for part in re.split(r"(?<=[.!?。])\s+",re.sub(r"^\*\*.*?\*\*\s*","",raw)):
                    part=part.strip()
                    if not (48<=len(part)<=380) or not part.endswith((".","。")) or "..." in part or "…" in part or SECRET.search(part):continue
                    if not re.search(r"\b(?:is|was|were|found|suggests?|reported|increased|decreased|show\w*|study|analysis|evidence)\b",part,re.I):continue
                    ts=[t for t,v in tool_inputs if part[:55].casefold() in v]
                    if len(ts)!=1:continue
                    evidence=[{k:z[k] for k in ("rank","title","snippet") if k in z}]
                    put(ix,"tool_facts",items=items,original_prompt=pr)
                    record("node",ix,ts[0],{"speaker":"tool","text":part,"item_index":item["item_index"],"source_rank":z["rank"],"source_evidence":evidence,"candidate_origin":"verbatim_snippet"},
                           "tool",.5+bool(BOUNDARY.search(part))*.3)
                    break
    return cases,context

def freeze(dest):
    if dest.exists(): raise SystemExit(f"Refusing to overwrite frozen data: {dest}")
    tasks=sorted(TASKS,key=lambda t:digest([SEED,t]))
    dev=set(tasks[:9])
    stock=[]; ctx={}; inventory=[]
    for rn in RUNS:
        run=ROOT/"results/studio"/rn
        for tid in sorted(TASKS):
            state=run/f"drb2/trials/{tid}/agent/bcg/state.json"
            if not state.exists():continue
            with state.open("rb") as f:m=re.search(rb'"problem_id"\s*:\s*"([^"]+)',f.read(1500))
            if not m:continue
            g=run/"graphs"/m[1].decode()
            if not (g/"logs/prompts.jsonl").exists():continue
            found,c=scan(rn,tid,g);stock.extend(found);ctx.update(c)
            inventory.append([rn,tid,g.name,dict(Counter(x["group"] for x,_ in found))])
    output=[]; seen=set(); chosen_turns=set()
    counts=Counter(); groups=Counter()
    for split,quotas in [("dev",{"node":16,"parent":12,"edge":12}),("test",{"node":24,"parent":18,"edge":18})]:
        for group,number in quotas.items():
            bands={"node":{"user":.12,"tool":.15,"terminal":.21,"fact":.21,"progress":.19,"reason":.12},
                   "parent":{"scope_clause":.29,"artifact_clause":.22,"method":.13,"restated":.13,"specific":.09,"output":.08,"adjacent":.06},
                   "edge":{"reverse":.16,"tool":.26,"qualified":.22,"fact":.25,"related":.11}}[group]
            pool=[(r,rank) for r,rank in stock if r["group"]==group and (r["task"] in dev)==(split=="dev")]
            for _ in range(number):
                best=None
                for row,rank in pool:
                    d=row["payload"]; key=digest([group,d.get("text") or d.get("child") or d.get("A"), d.get("parent") or d.get("B")])
                    if key in seen or (row["task"],row["turn"],group) in chosen_turns:continue
                    if group=="node" and BAN.search(d["text"]) and counts[split,"ban"]>=1:continue
                    band=row["band"];task=row["task"]; jitter=int(row["uid"][:6],16)/0xffffff
                    score=(number*bands[band]-groups[split,group,band])/(number*bands[band]) - .48*counts[group,task] - .12*counts[split,task] + .38*rank + .15*jitter
                    if best is None or score>best[0]:best=(score,row,key)
                if best is None:raise RuntimeError((split,group,"insufficient candidates"))
                _,r,key=best;seen.add(key);chosen_turns.add((r["task"],r["turn"],group))
                counts[group,r["task"]]+=1;counts[split,r["task"]]+=1;groups[split,group,r["band"]]+=1
                if group=="node" and BAN.search(r["payload"]["text"]):counts[split,"ban"]+=1
                output.append(dict(r,id=f"{group[0].upper()}{1+sum(x['group']==group for x in output):03d}",split=split))
    dest.mkdir(parents=True)
    jsonl(dest/"cases.jsonl",output)
    selected={f"{r['run']}/{r['graph']}:{r['prompt_line']}":ctx[r["run"],r["graph"],r["prompt_line"]] for r in output}
    dump(dest/"runtime_context.json",selected)
    dump(dest/"manifest.json",{"seed":SEED,"dev_tasks":sorted(dev),"test_tasks":sorted(TASKS-dev),"source_inventory":inventory,
        "selection":"source-only cadence, topical overlap and boundary cues; no target labels, old topology, final reports or model-error mining",
        "cases_sha256":hashlib.sha256((dest/"cases.jsonl").read_bytes()).hexdigest(),
        "context_sha256":hashlib.sha256((dest/"runtime_context.json").read_bytes()).hexdigest(),
        "strata":{"/".join(key):n for key,n in sorted(groups.items())}})
    print("Frozen",dest,{s:dict(Counter(r["group"] for r in output if r["split"]==s)) for s in ("dev","test")})

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=BASE/"dataset")
    freeze(parser.parse_args().out)
