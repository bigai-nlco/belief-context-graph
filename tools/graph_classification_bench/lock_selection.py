"""Persist version choice before any held-out model results exist."""
from datetime import datetime,timezone
from .freeze import BASE,dump
from .replay import byteshash
from .score import score
def main():
    lock=BASE/"selection.lock.json"
    if lock.exists():raise SystemExit("Already frozen; do not select again.")
    if list((BASE/"experiments").glob("*/test/*/[NEP][0-9]*.json")):raise SystemExit("Held-out data has already been run")
    dev={}
    for arm in ("decision_v1","decision_v2","decision_v3","decision_v4"):
        s=score(arm,"dev","r1")
        assert s["finished"]==s["requested"]==40
        dev[arm]={"macro_f1":s["quality"]["group_balanced_macro_f1"],"calls":s["usage"]["calls"],"score_sha256":byteshash(BASE/"experiments"/arm/"dev/r1/score.json")}
    best=max(r["macro_f1"] for r in dev.values())
    selected=min((a for a in dev if best-dev[a]["macro_f1"]<.005),key=lambda a:(dev[a]["calls"],a))
    dump(lock,{"selected":selected,"utc":datetime.now(timezone.utc).isoformat(),"selection_split":"dev_only","scores":dev,
               "dataset_lock_sha256":byteshash(BASE/"dataset/blind.lock.json"),"protocol_sha256":byteshash(BASE/"PROTOCOL.md"),
               "heldout_predictions_exist_at_selection":False})
    print("Selected",selected,"before opening test.")
if __name__=="__main__":main()
