"""Produce a local, checksummed inspection bundle. Never copies .env or auth headers."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from .freeze import BASE, ROOT, SECRET, digest, read_rows
from .replay import byteshash
from .summarize import ARMS


def main():
    archive = BASE.with_suffix(".zip")
    if archive.exists():
        raise SystemExit(f"Archive already exists; refusing to overwrite: {archive}")
    code = BASE/"reproduce/tools/graph_classification_bench"
    code.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    for p in sorted(source_dir.rglob("*")):
        if p.suffix in {".py", ".json"} and "__pycache__" not in p.parts:
            target = code/p.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p,target)
    fileset = {r["id"]:r for r in read_rows(BASE/"dataset/cases.jsonl")}
    for a in ARMS:
        for s in ("dev", "test"):
            where = BASE/f"experiments/{a}/{s}/r1"
            ids = {i for i,r in fileset.items() if r["split"] == s}
            assert ids == {p.stem for p in where.glob("[NEP][0-9][0-9][0-9].json")}
            for i in ids:
                r = json.loads((where/(i+".json")).read_text())
                assert r["case_hash"] == digest(fileset[i]) and r["status"] == "ok" and r["calls"]
    # Include a minimal dependency snapshot for offline scoring on another machine.
    # Raw experiments additionally keep their exact prompt/decision code at run time.
    tracked = sorted({p for folder in ("bcg/config","bcg/core","bcg/construct/_shared","bcg/construct/unified","bcg/construct/hybrid")
                      for p in (ROOT/folder).rglob("*.py")})
    tracked.extend((ROOT/"bcg/construct").glob("*.py"))
    for name in ["bcg/__init__.py","bcg/LICENSE","bcg/config/defaults.yaml",
                 "tools/case_study_studio/profiles/graph.yaml","pyproject.toml"]:
        p = ROOT/name
        if p.is_file():
            tracked.append(p)
    for p in tracked:
        target = BASE/"reproduce"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    keys = json.loads((BASE/"dataset/blind.lock.json").read_text())
    assert all(byteshash(BASE/"dataset"/n) == digest_ for n,digest_ in keys.items())
    checksums = BASE/"SHA256SUMS.txt"
    members = sorted(p for p in BASE.rglob("*") if p.is_file() and p != checksums)
    forbids = {".env",".git",".venv","auth.json","__pycache__"}
    # One *discarded, unscored* pilot source contained a credential-shaped
    # string. Do not copy it verbatim; do not edit primary or local evidence.
    redacted = {}
    for p in members:
        assert not forbids.intersection(p.relative_to(BASE).parts)
        if p.suffix in {".json",".jsonl",".yaml",".tsv",".csv"}:
            content = p.read_text(encoding="utf-8-sig")
            if SECRET.search(content):
                if p.relative_to(BASE).as_posix() != "pilot_sampling_v1/runtime_context.json":
                    raise ValueError(f"Credential-like value in {p.relative_to(BASE)}; not sharing")
                safe, n = SECRET.subn("[REDACTED: unscored pilot source]",content)
                assert n == 1 and not SECRET.search(safe)
                json.loads(safe)
                redacted[p] = safe.encode()
    def transfer_hash(p):
        return hashlib.sha256(redacted[p]).hexdigest() if p in redacted else byteshash(p)
    checksums.write_text("".join(f"{transfer_hash(p)}  {p.relative_to(BASE).as_posix()}\n" for p in members))
    with tempfile.TemporaryDirectory(prefix=".decision-package-", dir=archive.parent) as staging:
        candidate = Path(staging)/"bundle.zip"
        with ZipFile(candidate, "x", compression=ZIP_DEFLATED, compresslevel=6) as f:
            for p in [*members, checksums]:
                entry=(Path(BASE.name)/p.relative_to(BASE)).as_posix()
                if p in redacted:
                    f.writestr(entry,redacted[p])
                else:
                    f.write(p,arcname=entry)
        with ZipFile(candidate) as f:
            assert f.testzip() is None and len(f.namelist()) == len(members)+1
            for p in [*members, checksums]:
                entry = (Path(BASE.name)/p.relative_to(BASE)).as_posix()
                assert hashlib.sha256(f.read(entry)).hexdigest() == transfer_hash(p)
            f.extractall(staging)
        verify = Path(staging)/BASE.name/"reproduce/tools/graph_classification_bench/reproduce_saved.py"
        subprocess.run([sys.executable,"-I","-B",str(verify)],cwd=staging,check=True)
        os.link(candidate,archive)  # atomic and refuses an existing target
    print(f"{archive} | {len(members)+1} files | {archive.stat().st_size:,} bytes | sha256 {byteshash(archive)}")


if __name__ == "__main__":
    main()
