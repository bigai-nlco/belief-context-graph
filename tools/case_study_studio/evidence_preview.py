"""Small, immutable projections of frozen evidence for the AI review reader."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path, PurePosixPath

from workflows._shared.case_study import MAX_BYTES, read_bundle

VERSION = 1
_lock = threading.Lock()


def keep_for_review(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (name in {"case.json", "result.json", "report.md", "final_answer.txt"}
            or (parts[-1] == "requests.jsonl")
            or ("trajectories" in parts and not {"bcg", "bcg_team"}.intersection(parts)))


def preview_bundle(source: Path, source_sha: str) -> Path:
    """Retain complete files, including their original hashes and line numbers.

    The full frozen bundle is never modified. Cache identity includes its digest
    and this projection's format version; partial writes are never published.
    """
    if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("Invalid frozen evidence digest")
    target = source.with_name(f"{source.stem}.preview-v{VERSION}-{source_sha}.json.gz")
    if target.is_file():
        return target
    with _lock:
        if target.is_file():
            return target
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
            raise ValueError("Frozen evidence checksum mismatch")
        bundle = read_bundle(source)
        if len(bundle["cases"]) != 1:
            raise ValueError("A review preview requires exactly one case")
        entry = bundle["cases"][0]
        kept, omitted, seen = [], [], set()
        for file in entry["files"]:
            name = file["path"]
            parts = PurePosixPath(name).parts
            if not parts or name.startswith("/") or ".." in parts or "\\" in name or name in seen:
                raise ValueError("Invalid evidence path")
            seen.add(name)
            if keep_for_review(name):
                raw = file["text"].encode("utf-8")
                if len(raw) != file["size"] or hashlib.sha256(raw).hexdigest() != file["digest"]:
                    raise ValueError("Evidence file checksum mismatch")
                kept.append(file)
            else:
                omitted.append({"path": name, "size": file["size"]})
        if not any(f["path"] == "case.json" for f in kept):
            raise ValueError("Missing case metadata")
        projection = {"kind": "review-preview", "version": VERSION, "source_sha256": source_sha,
                      "omitted_files": omitted}
        data = {"format": bundle["format"], "version": bundle["version"],
                "projection": projection, "cases": [{**entry, "files": kept}]}
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_BYTES:
            raise ValueError("Preview exceeds import size limit")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=source.parent, prefix=".preview-", delete=False) as handle:
                temporary = Path(handle.name)
                os.chmod(temporary, 0o600)
                with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as stream:
                    stream.write(raw)
            temporary.replace(target)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    return target
