"""Ask the status question irrespective of the preceding role gate; no reference labels in queries."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from bcg.construct.unified import decisions,llm
from .run import *
def main():
    assert rd(BASE/"selection.lock.json")["version"]=="v6"
    cfg=llm.load_config(str(replay.PROFILE),model_key="graph")
    rows=[r for r in data_for("test")[0] if r["group"]=="node" and r["payload"]["speaker"]!="tool"]
    arms=("chat_v6","decision_v6");stop=Event()
    def one(ac):
        arm,c=ac;out=BASE/"diagnostics/direct-status"/arm;out.mkdir(parents=True,exist_ok=True);p=out/(c["id"]+".json")
        if p.exists() or stop.is_set():return
        rec=Live(cfg,arm,"v6",{});token=llm.bind_usage_tracker(llm.TokenUsageTracker())
        d=c["payload"];row={"speaker":d["speaker"],"text":d["text"],"source_evidence":d["source_evidence"],"thinking_only":bool(d.get("thinking_only")),"referenced_tasks":{}}
        try:
            clf=decisions.DecisionClassifier({"model":"gpt-6-luna-decision","api_key_env":"LITELLM_OPENAI_API_KEY","base_url":"https://litellm.mybigai.ac.cn/v1"},post=rec.post)
            pred=clf.ask("status",[row],[decisions.Question("task_status","Which status did the speaker explicitly report for the identified work?",decisions.STATUSES)])[0]
            obj=dict(status="ok",prediction=pred)
        except Exception as e:
            obj=dict(status="error",prediction={},error=type(e).__name__)
            # A forced question on a non-progress item may yield a genuine refusal.
            # Retain that outcome; stop dispatch only for transport / account errors.
            if any("error" in a or a.get("http",200)>=400 for a in rec.calls):stop.set()
        finally:llm.unbind_usage_tracker(token);rec.client.close()
        obj.update(id=c["id"],arm=arm,calls=rec.calls);writenew(p,obj)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,[(a,c) for c in rows for a in arms]))
    if stop.is_set():raise SystemExit("Direct-status check stopped on error.")
    print("direct status",len(rows),"x",len(arms))
if __name__=="__main__":main()
