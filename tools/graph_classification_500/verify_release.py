"""Standalone offline recomputation of the released CSVs. Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
def verify(root):
    with (root/"analysis/comparisons.csv").open(encoding="utf-8-sig") as f:table=list(csv.DictReader(f))
    for row in table:
        out=root/"experiments"/row["arm"]/row["split"]
        p=out/("r2" if (out/"r2/complete.json").exists() else "r1")/"case_scores.csv"
        fields=defaultdict(Counter);groups=defaultdict(set);seen=set()
        with p.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                assert r["id"] not in seen;seen.add(r["id"])
                gold,pred,amb=map(json.loads,(r["gold"],r["prediction"],r["ambiguous"]))
                for k,v in gold.items():
                    if k in amb:continue
                    fields[k][v,pred.get(k) or "<missing>"]+=1;groups[r["group"]].add(k)
        fieldf={};total=correct=0
        for k,m in fields.items():
            f=[]
            for c in {g for g,p in m}:
                n=sum(a for (g,p),a in m.items() if g==c);np=sum(a for (g,p),a in m.items() if p==c);hit=m[c,c]
                f.append(2*hit/(n+np));total+=n;correct+=hit
            fieldf[k]=sum(f)/len(f)
        macro=sum(sum(fieldf[k] for k in v)/len(v) for v in groups.values())/len(groups)
        assert len(seen)==int(row["cases"]) and total==int(row["labels"])
        assert math.isclose(correct/total,float(row["accuracy"])) and math.isclose(macro,float(row["macro"])),(row["arm"],row["split"])
    frozen=json.loads((root/"dataset/blind.lock.json").read_text())
    for name,h in frozen.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    print(f"OK: scores recomputed for {len(table)} arm/split comparisons; all {len(frozen)} sealed source/annotation artifacts verified.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("results",type=Path,nargs="?",default=Path.cwd());verify(p.parse_args().results)
