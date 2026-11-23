"""Replay both production generation+classification paths on frozen preprocessed windows.

Use the actual compact Tool Result function with its preprocessing boundary seeded
from the historical wire items. This applies TODAYS schema, including source_rank,
even when the recorded run predates it. No rule-only fallback is counted as a
semantic proposal. No embedding, graph merge, new Agent turn or web search.
"""
import argparse
import copy
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FunctionType, SimpleNamespace
from bcg.construct.unified import decisions, extract, llm, prompts
from bcg.construct.unified.task_state import shown_task_texts
from . import replay
from .freeze import BASE, ROOT, digest, dump, read_rows, section, sentences

def tool_stage(ctx,rec,classifier):
    frozen={int(item["item_index"]):item for item in ctx["items"]}
    def bound_rule(**kwargs):
        item=frozen.get(int(kwargs["content"]))
        if not item:return None
        node={"tool_name":item["tool_name"],"tool_result_count":len(item["results"]),
              "tool_result_items":item["results"],"extraction_method":"rule_tool_result"}
        return {"nodes":[node]}
    def captured(_client,_model,text,**kw):
        return json.dumps(rec.chat(text,kw["max_tokens"]),ensure_ascii=False)
    fn=extract.extract_compact_tool_result_nodes_batch
    # Isolated copy of the function's globals for this replay only. Never patch a
    # production singleton across threads. The entire production function body is unchanged.
    isolated=FunctionType(fn.__code__,{**fn.__globals__,"llm":SimpleNamespace(call_model=captured,
                          parse_json_response=llm.parse_json_response),"extract_rule_tool_result_nodes":bound_rule},fn.__name__,fn.__defaults__,fn.__closure__)
    isolated.__kwdefaults__=fn.__kwdefaults__
    raw=isolated(None,rec.graph_model,items=[{"content":str(i),"sentences":[],"query":frozen[i]["query"] if i in frozen else ""}
             for i in range(max(frozen)+1)],mode="sentences",reasoning_effort="none",max_tokens=8192,classifier=classifier)
    if rec.calls and ("error" in rec.calls[-1] or rec.calls[-1].get("http",200)>=400):
        raise ValueError("compact tool extraction call failed")
    items=[]
    for i,v in enumerate(raw):
        if v and v.get("extraction_method")=="compact_llm_tool_result":
            items.append({"item_index":i,"beliefs":v["nodes"]})
    return {"items":items}

def paired_arm(c,ctx,rec,is_decision):
    cfg={"base_url":"https://litellm.mybigai.ac.cn/v1","api_key_env":"LITELLM_OPENAI_API_KEY","model":"gpt-6-luna-decision"}
    clf=decisions.DecisionClassifier(cfg,post=rec.post) if is_decision else None
    def capture(original,limit):
        if ctx["kind"]=="tool_facts":return tool_stage(ctx,rec,clf)
        if not clf:return rec.chat(original,limit)
        role=ctx["role"]
        if ctx["kind"]=="edge":
            kw={"role":role,"content":ctx.get("original_content",""),"previous_nodes":replay.enc(ctx["previous"]),
                "current_nodes":replay.enc(ctx["current"]),"graph_edges":replay.enc(ctx["existing"]),"decisions":True}
            if role=="assistant":
                fb=section(original,"## Validation feedback from the previous attempt")
                neutral=prompts.build_layered_relation_extraction_prompt(**kw,validation_feedback=fb or None)
            else:neutral=prompts.build_relation_extraction_prompt(**kw)
            output=rec.chat(neutral,limit)
            if not isinstance(output.get("relations"),list):raise ValueError("neutral generator omitted relations array")
            curr={n["id"] for n in ctx["current"]};text={n["id"]:n["content"] for n in ctx["current"]+ctx["previous"]}
            clean=clf.edges(output["relations"],text,curr,role=role)
            level=output.get("selected_previous_layer")
            if not any(r["from"] not in curr or r["to"] not in curr for r in clean):level=None
            return {"selected_previous_layer":level,"relations":clean}
        history=replay.enc(ctx["history"]);block=ctx["sentences"];s=sentences(block)
        neutral=prompts.build_node_extraction_prompt(role,mode="sentences",sentences_block=block,graph_nodes=history,task_state_enabled=True,decisions=True)
        obj=rec.chat(neutral,limit)
        if not isinstance(obj.get("nodes"),list):raise ValueError("neutral generator omitted nodes array")
        # Content itself is not sent in the classification row; it determines only
        # the protected thinking-only flag (same logic as DecisionClassifier.nodes).
        tfile=ROOT/"results/studio"/c["run"]/"graphs"/c["graph"]/"trajectory.json"
        turn=json.loads(tfile.read_text())["trajectory"][c["turn"]]
        content=extract.strip_valid_tool_calls(str(turn["content"])) if role=="assistant" else str(turn["content"])
        return clf.nodes(obj["nodes"],role=role,sentences=[s.get(i,"") for i in range(max(s)+1)],
                         content=content,shown_tasks=shown_task_texts(history),task_state=True)
    return replay.native(SimpleNamespace(chat=capture),c,ctx)

def run(which,split,workers):
    ds=BASE/"dataset";lockpath=BASE/"selection.lock.json"
    if not lockpath.is_file():raise SystemExit("Choose version exclusively on dev first")
    version=json.loads(lockpath.read_text())["selected"]
    is_decision=which=="decision";arm="pipeline_"+version if is_decision else "native_current"
    where=BASE/"experiments"/arm/split/"r1";where.mkdir(parents=True,exist_ok=True)
    spec=replay.versions(version) if is_decision else {}
    rows=[r for r in read_rows(ds/"cases.jsonl") if r["split"]==split]
    contexts=json.loads((ds/"native_context.json").read_text())
    cfg=llm.load_config(str(replay.PROFILE),model_key="graph")
    assert cfg["model"]=="gpt-6-luna" and cfg["base_url"].rstrip("/").removesuffix("/v1")=="https://litellm.mybigai.ac.cn"
    files=[Path(__file__).resolve(),Path(replay.__file__),Path(extract.__file__),Path(prompts.__file__),Path(decisions.__file__),replay.PROFILE]
    manifest={"arm":arm,"split":split,"selected_version":version,"source_hashes":{str(f.relative_to(ROOT)):replay.byteshash(f) for f in files},
              "dataset":json.loads((ds/"blind.lock.json").read_text()),"version_spec":spec,"selection_hash":replay.byteshash(lockpath)}
    if (where/"manifest.json").is_file():assert manifest==json.loads((where/"manifest.json").read_text())
    else:
        dump(where/"manifest.json",manifest)
        for f in files:
            target=where/"source"/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    from threading import Event
    stop=Event()
    def one(c):
        dst=where/(c["id"]+".json")
        if dst.exists() or stop.is_set():return
        ctx=contexts[f"{c['run']}/{c['graph']}:{c['prompt_line']}"]
        prev=BASE/"experiments/native"/split/"r1"/dst.name
        if not is_decision and ctx["kind"]!="tool_facts" and prev.exists():
            # No regeneration or double billing: preserve exact already-replayed
            # native response, window, fingerprint and provenance in the canonical view.
            r=json.loads(prev.read_text());r.update(arm=arm,reused_call_artifact=str(prev.relative_to(BASE)))
        else:
            rec=replay.Recorder(cfg,version,spec);token=llm.bind_usage_tracker(llm.TokenUsageTracker())
            r={"id":c["id"],"uid":c["uid"],"case_hash":digest(c),"arm":arm}
            try:
                pred,diag=paired_arm(c,ctx,rec,is_decision)
                r.update(status="ok",prediction=pred,diagnostics=diag)
            except Exception as e:
                msg=str(e)
                r.update(status="error",prediction={},diagnostics={"error_class":type(e).__name__,"message":msg[:200] if not replay.SECRET.search(msg) else "[redacted]"})
            finally:rec.client.close();llm.unbind_usage_tracker(token)
            r["calls"]=rec.calls
        with dst.open("x") as f:json.dump(r,f,ensure_ascii=False,indent=2)
        if r["status"]!="ok" and any(a.get("http") in (401,403,429) or (a.get("error") or {}).get("http") in (401,403,429) for a in r["calls"]):stop.set()
        print(arm,split,c["id"],r["status"],flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(one,rows))
    if stop.is_set():raise SystemExit("Provider error: stopped")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm",choices=["native","decision"]);p.add_argument("split",choices=["dev","test"]);p.add_argument("--workers",type=int,choices=range(1,7),default=2)
    a=p.parse_args();run(a.arm,a.split,a.workers)
