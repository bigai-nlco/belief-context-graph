"""Blind reader: show only neutral candidate texts and the evidence at this call."""
import argparse
from .freeze import BASE, read_rows

def main():
    p=argparse.ArgumentParser();p.add_argument("group",choices=["node","parent","edge"]);p.add_argument("first",type=int);p.add_argument("last",type=int)
    p.add_argument("--width",type=int,default=520);a=p.parse_args()
    for r in sorted(read_rows(BASE/"dataset/cases.jsonl"),key=lambda r:r["id"]):
        if r["group"]!=a.group or not a.first<=int(r["id"][1:])<=a.last:continue
        x=r["payload"]
        print(r["id"],r["task"],"turn",r["turn"],x.get("speaker",""),{k:v for k,v in x.items() if k in ("thinking_only","indices")})
        for k in ("text","parent","child","A","B","source_evidence"):
            if k in x:
                value=str(x[k]); print(k+":",value if len(value)<=a.width else value[:a.width]+" …")

if __name__=="__main__":main()
