"""Supplement only: rare conflict/direction cases from real graph inputs (no synthetic rewrites)."""
import argparse
import json
import re
from bcg.construct.unified import llm
from . import replay
from .freeze import BASE, ROOT, array, digest, dump, read_rows, section

# Manually specified against raw endpoint texts, never against older edges. Each pointer
# is (run, task, real prompt line, original node id). Some pairs span calls or runs.
SP=(12,120,147,486); HIST_SP=(12,120,8,30); IT=(12,120,111,363)
CR=(14,120,16,37);CR_APX=(14,120,125,355)
DESIGN=[
    (SP,HIST_SP,"contradicts","同是西班牙 2020 年到访人次，以千计分别为 51,700 与 36,410，非单位换算造成；只判断两句是否相容。"),
    (IT,SP,"contradicts","同一运行先后把意大利 2020 年同一指标报为 20,427 千与 38,400 千。"),
    (CR_APX,CR,"contradicts","两句对 2020 一致，但前者把 2012–2019 概括为约三千万，后者明确 2019 超六千万。"),
    ((8,120,75,188),(8,120,75,189),"none","年份相同、数值不同，但分别是摩纳哥和黑山；不是对同一主体的两个结论。"),
    ((8,8,85,190),(8,8,85,191),"none","同篇论文、同类参数、结论方向不同，但分别在讨论金与银；不能用数值差异强造矛盾。"),
    ((8,120,75,187),HIST_SP,"none","36,410,000 人次与 36,410 千人次相等。A 没有比 B 多给信息，不能标反驳或补充。"),
    ((14,98,36,91),(8,98,13,36),"supplements","五月二十八日和 Durham 对原先只给出死亡年份的传记叙述增加了可用细节。"),
    ((14,120,12,25),(8,120,75,187),"supplements","2020 的数量经单位换算一致；A 补充了 2019 基数与下降趋势。"),
]
def endpoint(ref):
    rn,t,line,i=ref;run=f"RUN-{rn:04d}";s=ROOT/"results/studio"/run/f"drb2/trials/{t}/agent/bcg/state.json"
    with s.open("rb") as f:graph=re.search(rb'"problem_id"\s*:\s*"([^"]+)',f.read(1500))[1].decode()
    p=read_rows(s.parents[5]/"graphs"/graph/"logs/prompts.jsonl")[line-1]["prompt"]
    nodes=array(section(p,"## Current-turn nodes"))+array(section(p,"## Current Tool Result nodes"))
    n=next(n for n in nodes if n["id"]==i)
    return {"run":run,"task":t,"graph":graph,"prompt_line":line,"node_id":i,"text":n["content"],"prompt_sha256":digest(p)}
def freeze():
    out=BASE/"supplement";out.mkdir(exist_ok=True)
    records=[]
    for i,(a,b,g,r) in enumerate(DESIGN,1):
        A=endpoint(a);B=endpoint(b);records.append({"id":f"X{i:03d}","A":A,"B":B,"gold":g,"reason":r})
    f=out/"sealed.json"
    if f.is_file():assert json.loads(f.read_text())==records
    else:dump(f,records)
    print("sealed",replay.byteshash(f))
    return records
def run(arm):
    records=freeze()
    spec=replay.versions(arm);out=BASE/"supplement"/arm;out.mkdir(exist_ok=True)
    cfg=llm.load_config(str(replay.PROFILE),model_key="graph")
    for r in records:
        dst=out/(r["id"]+".json")
        if dst.exists():continue
        # Local distinct endpoint IDs avoid collisions between different graphs.
        c={"group":"edge","payload":{"speaker":"assistant","from_id":1,"to_id":2,"current_ids":[1],"A":r["A"]["text"],"B":r["B"]["text"]}}
        rec=replay.Recorder(cfg,arm,spec);tok=llm.bind_usage_tracker(llm.TokenUsageTracker())
        try: pred,_=replay.fixed(rec,c);result={"prediction":pred,"status":"ok"}
        except Exception as e:result={"status":"error","prediction":{},"error_class":type(e).__name__}
        finally:rec.client.close();llm.unbind_usage_tracker(tok)
        result.update(id=r["id"],arm=arm,source_sha256=replay.byteshash(BASE/"supplement/sealed.json"),calls=rec.calls)
        with dst.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
        if result["status"]!="ok":raise SystemExit("provider error; inspect recorded attempt")
    print(arm,"complete")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm",choices=["freeze","chat_fixed","decision_v1","decision_v4"]);a=p.parse_args()
    freeze() if a.arm=="freeze" else run(a.arm)
