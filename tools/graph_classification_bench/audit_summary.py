"""Offline sensitivity checks. No model, network call, or change to primary scores."""
import csv
import json
import os
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from bcg.construct.unified import decisions, llm
from bcg.construct.unified.task_state import shown_task_texts
from . import replay
from .freeze import BASE, dump, read_rows, sentences
from .score import collect, quality


def table(file):
    with file.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def restored(case, ctx, raw, matched):
    """Decode the actual original replies. Validate every Decisions input before reusing it."""
    assert ctx["kind"] == "semantic_node"
    calls = iter(raw["calls"])
    chat = next(calls)
    assert chat["provider"] == "chat"
    obj = llm.parse_json_response(chat["response"]["choices"][0]["message"]["content"])
    assert "_parse_error" not in obj
    if raw["arm"] == "pipeline_decision_v4":
        def recorded(url, *, headers, json, timeout, follow_redirects):
            old = next(calls)
            assert old["provider"] == "decisions" and old["http"] == 200
            assert replay.change(json, replay.versions("decision_v4")) == old["request"]
            return httpx.Response(200, request=httpx.Request("POST", url), json=old["response"])

        firstrow = json.loads(raw["calls"][1]["request"]["input"])[0]
        src = sentences(ctx["sentences"])
        cfg = {"base_url": "https://litellm.mybigai.ac.cn/v1",
               "api_key_env": "LITELLM_OPENAI_API_KEY", "model": "gpt-6-luna-decision"}
        with patch.dict(os.environ, {"LITELLM_OPENAI_API_KEY": "offline-unused"}):
            clf = decisions.DecisionClassifier(cfg, post=recorded)
            obj = clf.nodes(obj["nodes"], role=ctx["role"],
                            sentences=[src.get(i, "") for i in range(max(src)+1)],
                            content="<thinking>archived flag</thinking>" if firstrow["thinking_only"] else "archived flag",
                            shown_tasks=shown_task_texts(replay.enc(ctx["history"])), task_state=True)
    assert next(calls, None) is None

    def exact(rows, d, **kwargs):
        found = [r for r in rows if (r.get("belief") or r.get("decision")) == matched]
        assert len(found) == 1
        return found[0], {"aligned": True, "alignment": "blind_manual"}

    with patch.object(replay, "select_unit", exact):
        return replay.native(SimpleNamespace(chat=lambda prompt, limit: obj), case, ctx)[0]


def main():
    directory = BASE / "audit"
    read = lambda f: json.loads((directory / f).read_text())
    akey = read("alignment_key.json")
    matches = {r["code"]: r for r in read("alignment_blind.json")}
    pkey = read("presentation_key.json")
    preferred = Counter("tie" if r["preferred"] == "tie" else pkey[r["id"]][r["preferred"]]
                        for r in table(directory / "paired_annotations.tsv"))
    cases = {r["id"]: r for r in read_rows(BASE / "dataset/cases.jsonl")}
    contexts = json.loads((BASE / "dataset/native_context.json").read_text())
    annotations = table(directory / "alignment_annotations.tsv")
    assert len(annotations) == len(akey) == 11 and len(preferred) > 0
    summary = {"blind_preferences_on_disagreements_only": dict(preferred),
               "alignment_review": dict(Counter(r["judgment"] for r in annotations)), "sensitivity": {}}
    for arm in ("native_current", "pipeline_decision_v4"):
        p, _, rows, _ = collect(arm, "test", "r1")
        amended = {r["id"]: dict(r) for r in rows}
        recovery = []
        for r in annotations:
            which, id_ = akey[r["code"]]
            if arm != which or r["judgment"] != "single":
                continue
            case = cases[id_]
            ctx = contexts[f"{case['run']}/{case['graph']}:{case['prompt_line']}"]
            matched = matches[r["code"]]["candidates"][int(r["candidate"])-1]
            assert amended[id_]["aligned"] is False
            pred = restored(case, ctx, json.loads((p/(id_+".json")).read_text()), matched)
            amended[id_].update(aligned=True, prediction=pred)
            recovery.append({"id": id_, "matched": matched, "decoded_prediction": pred})
        summary["sensitivity"][arm] = {"recovered_matches": recovery, "aligned": sum(r["aligned"] for r in amended.values()),
                                        "quality": quality(amended.values())}
    dump(directory/"summary.json", summary)
    for a, s in summary["sensitivity"].items():
        print(a, s["aligned"], s["quality"]["correct"], s["quality"]["group_balanced_macro_f1"])
    print("blind preferences:", preferred)


if __name__ == "__main__":
    main()
