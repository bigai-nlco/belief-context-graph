"""Real numeric assertions, independently locatable by stable ref. No latent or generated labels."""
import argparse
import json
import gzip
import re
from .data import *
def index():
    p=BASE/"annotations/claim-index.json"
    if p.is_file():return rd(p)
    rows={};unique=set()
    for run in RUNS:
        for task,g in idsafe(runroot(run)):
            arr,ctx=json.loads(gzip.decompress((CACHE/(run+"-"+g.name+".json.gz")).read_bytes()))
            for c in arr:
                if c["group"]!="edge":continue
                for k,ik in (("A","from_id"),("B","to_id")):
                    d=c["payload"];t=d[k];norm=txt(t)
                    if (task,norm) in unique or not re.search(r"\d",t) or t.lower().startswith(("the web_","the assistant plans","the user","the bcg_")):continue
                    unique.add((task,norm))
                    id_=h([run,g.name,d[ik],t])[:7];assert id_ not in rows
                    rows[id_]={"run":run,"graph":g.name,"task":task,"prompt_line":c["prompt_line"],"node_id":d[ik],"text":t,**c["endpoint_evidence"][k]}
    old.dump(p,rows);return rows
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("task",type=int);p.add_argument("pattern");p.add_argument("--n",type=int,default=48);a=p.parse_args()
    match=[(k,v) for k,v in index().items() if v["task"]==a.task and re.search(a.pattern,v["text"],re.I)]
    print("matches",len(match))
    for k,v in sorted(match,key=lambda z:(len(z[1]["text"]),z[0]))[:a.n]:print(k,v["text"])
