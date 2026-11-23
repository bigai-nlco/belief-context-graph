"""Offline wire payload check: candidate endpoints, directions, class sets and inference settings stay fixed."""
import argparse
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace
import httpx
from bcg.construct.unified import llm
from tools.graph_classification_bench import replay,freeze as old
from .data import BASE,DS,CACHE,key,rd,txt
from .run import transformed,evidence
def provenance():
    out=DS/"endpoint_provenance.json"
    if out.exists():return
    by=defaultdict(set)
    for c in old.read_rows(DS/"cases.jsonl"):
        if c["group"]=="edge":by[c["run"],c["graph"]].add(c["prompt_line"])
    result={}
    for (run,graph),lines in by.items():
        candidates,contexts=json.loads(gzip.decompress((CACHE/(run+"-"+graph+".json.gz")).read_bytes()))
        # Accumulate from the relevant prompt only, not from later graph states.
        for c in candidates:
            if c["group"]!="edge" or c["prompt_line"] not in lines:continue
            meta=result.setdefault(key(c),{})
            for k,v in c["endpoint_evidence"].items():
                t=txt(c["payload"][k]);prior=meta.get(t)
                if prior and prior!=v:meta[t]={}  # Ambiguous identical-text provenance: omit.
                elif prior is None:meta[t]=v
    old.dump(out,result)
def check(version):
    cfg=llm.load_config(str(replay.PROFILE),model_key="graph");assert cfg["model"]=="gpt-6-luna"
    out=BASE/"controls";out.mkdir(exist_ok=True);records=[];unchanged=0
    extras=rd(DS/"endpoint_provenance.json")
    for c in old.read_rows(DS/"cases.jsonl"):
        if c["split"]!="dev":continue
        meta=evidence(c,extras);i=0
        def mock(url,*,headers,json,timeout,follow_redirects):
            nonlocal i,unchanged
            i+=1;l=transformed(json,"v1",meta);r=transformed(json,version,meta)
            def signature(p):return [(q["name"],[v["value"] for v in q["choices"]]) for q in p["questions"]]
            if version=="v7" and any(q["name"].split("_",2)[2]=="type" for q in l["questions"]):
                assert len(r["questions"])==3*len(l["questions"])
                assert {q["name"].split("_",2)[2] for q in r["questions"]}=={"clash","uses","adds"}
                assert all([v["value"] for v in q["choices"]]==["yes","no"] for q in r["questions"])
            else:assert signature(l)==signature(r)
            left=__import__("json").loads(l["input"]);right=__import__("json").loads(r["input"])
            assert left==[{k:v for k,v in row.items() if k not in ("A_source","B_source")} for row in right]
            def canonical(p):return {"model":p["model"],"parameters":{},"endpoint":url,"prompt":{"input":p["input"],"questions":p["questions"]}}
            if l!=r:records.append({"case_id":f"{c['id']}-{i}","baseline_request":canonical(l),"candidate_request":canonical(r)})
            else:unchanged+=1
            answers=[{"name":q["name"],"type":"choice","choice":"progress" if q["name"].endswith("task_role") else q["choices"][0]["value"]} for q in l["questions"]]
            return httpx.Response(200,request=httpx.Request("POST",url),json={"answers":answers,"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}})
        replay.fixed(SimpleNamespace(post=mock),c)
    file=out/(version+"-paired.jsonl");old.jsonl(file,records)
    script=__import__("pathlib").Path.home()/".codex/skills/prompt-template-ablation/scripts/check_controlled_pairs.py"
    cp=subprocess.run([sys.executable,str(script),str(file)],capture_output=True,text=True,check=True)
    result=json.loads(cp.stdout);result["exactly_unchanged"]=unchanged;old.dump(out/(version+"-check.json"),result);print(result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("version");a=p.parse_args();provenance();check(a.version)
