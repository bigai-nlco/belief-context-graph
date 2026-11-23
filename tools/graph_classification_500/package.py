"""Portable experiment artifacts, code fingerprints, no authentication files or headers."""
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from .data import BASE,DS,ROOT,sha,rd,old
from .verify_release import verify
def package():
    here=Path(__file__).parent.resolve();deploy=BASE/"reproduction";sources=deploy/"sources";sources.mkdir(parents=True,exist_ok=True)
    files=sorted({*here.glob("*.py"),*here.glob("*.md"),*here.glob("variants/*.json"),
                *(ROOT/"tools/graph_classification_bench").glob("*.py"),*(ROOT/"tools/graph_classification_bench/variants").glob("*.json"),
                *(ROOT/"bcg/construct/unified").glob("*.py"),ROOT/"bcg/construct/_shared/llm.py",ROOT/"bcg/core/errors.py",ROOT/"bcg/core/env.py",
                ROOT/"tools/case_study_studio/profiles/graph.yaml"})
    for f in files:
        target=sources/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if f.suffix in (".yaml",".json") and old.SECRET.search(f.read_text()):raise ValueError(f"Credential-like configuration in {f.name}; refused")
        shutil.copyfile(f,target)
    legacy=deploy/"legacy"/old.BASE.name/"dataset";legacy.mkdir(parents=True,exist_ok=True)
    for n in ("cases.jsonl","groundtruth.tsv","native_context.json","blind.lock.json"):
        shutil.copyfile(old.BASE/"dataset"/n,legacy/n)
    meta={"repository_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"exact_files_not_just_head":{str(f.relative_to(ROOT)):sha(f) for f in files},"net":"provider request bodies and response bodies only; no authorization headers / .env or tokens"}
    old.dump(deploy/"source_manifest.json",meta);shutil.copyfile(here/"verify_release.py",BASE/"verify_release.py");shutil.copyfile(here/"README.md",deploy/"README.md")
    verify(BASE)
    paths=sorted(p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="SHA256SUMS.json" and p.name!="claim-index.json")
    for p in paths:
        if p.suffix not in (".jsonl",".json",".tsv",".txt"):continue
        if "reproduction" in p.parts:continue
        if old.SECRET.search(p.read_text()):raise ValueError(f"Potential credential in {p.relative_to(BASE)}, not archived")
    contents={str(p.relative_to(BASE)):sha(p) for p in paths};old.dump(BASE/"SHA256SUMS.json",contents)
    archive=BASE.with_suffix(".zip");temp=archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as out:
        for p in paths+[BASE/"SHA256SUMS.json"]:out.write(p,BASE.name+"/"+str(p.relative_to(BASE)))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None and len(z.namelist())==len(contents)+1
        for k,v in contents.items():assert hashlib.sha256(z.read(BASE.name+"/"+k)).hexdigest()==v,k
    temp.replace(archive)
    print("Verified archive:",archive,"bytes",archive.stat().st_size,"files",len(contents)+1,"SHA256",sha(archive))
if __name__=="__main__":package()
