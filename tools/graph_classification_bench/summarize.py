"""Build a cross-arm table and reproducible uncertainty/stability estimates, offline."""
import json
from collections import Counter
from . import score
from .freeze import BASE, dump, read_rows
from .preflight import truth

ARMS = ("chat_fixed", "decision_v1", "decision_v2", "decision_v3", "decision_v4",
        "native_current", "pipeline_decision_v4")


def main():
    gold = truth()
    cases = read_rows(BASE/"dataset/cases.jsonl")
    hand = json.loads((BASE/"audit/summary.json").read_text())["sensitivity"]
    stats, per_id = [], {}
    for a in ARMS:
        per_id[a] = {}
        for s in ("dev","test"):
            _, requested, rows, calls = score.collect(a, s, "r1")
            assert len(rows) == len(requested)
            result = score.score(a, s, "r1", write=False)
            u, q = result["usage"], result["quality"]
            stats.append(dict(arm=a, split=s, n=len(rows), errors=result["failed"], macro_f1=q["group_balanced_macro_f1"],
                              **q["group_macro_f1"], correct=q["correct"], scored_fields=q["n"], ambiguous_fields=q["ambiguous_fields"],
                              aligned=result["aligned"], calls=u["calls"], input_tokens=int(u["input"]),
                              output_tokens_incl_reasoning=int(u["output_incl_reasoning"]), total_tokens=int(u["total"]),
                              cost_known_calls=u["cost_known_calls"], cost_known_usd=u["cost"] if u["cost_known_calls"] else None,
                              elapsed_provider_call_seconds=u["seconds"], providers=dict(Counter(c["provider"] for c in calls))))
            per_id[a].update({r["id"]:r for r in rows})
    score.csvwrite(BASE/"arm_summary.csv", stats, list(stats[0]))
    ci = {a+" minus "+b:score.paired_ci(a,b) for a,b in [
        ("decision_v4","chat_fixed"),("pipeline_decision_v4","native_current"),("decision_v4","decision_v1")]}
    stable_ids = set(json.loads((BASE/"stability15.lock.json").read_text())["ids"])
    rep = {}
    for a in ("chat_fixed","decision_v1","decision_v4"):
        _, _, sampled, _ = score.collect(a,"test","stability15")
        assert {r["id"] for r in sampled} == stable_ids
        n = eq = 0
        changes = []
        for r in sampled:
            prev = per_id[a][r["id"]]
            assert prev["status"] == r["status"] == "ok"
            if r["prediction"] != prev["prediction"]:
                changes.append({"id":r["id"], "first":prev["prediction"], "second":r["prediction"]})
            for f in set(r["gold"]) - set(r["ambiguous"]):
                n += 1
                eq += prev["prediction"].get(f) == r["prediction"].get(f)
        rep[a] = dict(n=len(sampled), exact_repeats=len(sampled)-len(changes), unchanged_clear_fields=eq, clear_fields=n, changes=changes)
    extra = json.loads((BASE/"supplement/sealed.json").read_text())
    xrows = []
    for r in extra:
        xo = {f:r[f] for f in ("id","A","B","gold","reason")}
        for a in rep:
            o = json.loads((BASE/f"supplement/{a}"/(r["id"]+".json")).read_text())
            assert o["status"] == "ok"
            xo[a] = o["prediction"]["type"]
        xrows.append(xo)
    score.csvwrite(BASE/"supplement/comparison.csv", xrows, list(xrows[0]))
    extra_accuracy = {a: sum(r[a]==r["gold"] for r in xrows) for a in rep}
    comparison = []
    for c in cases:
        i = c["id"]
        r = {f:c[f] for f in ("id","split","group","run","task","turn","graph","prompt_line","uid","band")}
        r.update(input=c["payload"],gold=gold[i]["labels"],ambiguous=gold[i]["ambiguous"],label_reason=gold[i]["reason"])
        for a in ARMS:
            pred = per_id[a][i]; r[a] = pred["prediction"]
            if a in hand:
                r[a+"_auto_aligned"] = pred["aligned"]
                recovered = {v["id"]:v for v in hand[a]["recovered_matches"]}
                r[a+"_manual_recovery"] = recovered.get(i, {}).get("decoded_prediction")
        comparison.append(r)
    score.csvwrite(BASE/"case_comparison.csv", comparison, list(comparison[0]))
    summary = dict(sources=dict(Counter(c["run"] for c in cases)), split_tasks={s:sorted({c["task"] for c in cases if c["split"]==s})
                   for s in ("dev","test")}, metrics=stats, differences_cluster_bootstrap=ci, repeatability=rep,
                   supplement={"total":len(extra),"correct":extra_accuracy}, audit=json.loads((BASE/"audit/summary.json").read_text()))
    dump(BASE/"overall.json", summary)
    print("Source cases:", summary["sources"], "paired CI:", ci, "stability:", rep, "supplement:", extra_accuracy)


if __name__ == "__main__":
    main()
