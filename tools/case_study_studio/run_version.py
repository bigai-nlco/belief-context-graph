"""Capture local launch provenance once, without fetching Git or reading secrets.

The manifest fingerprints project sources, not installed packages or model weights.
It identifies dirty working copies but is not a backup of their uncommitted edits.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

AGENT_ROOTS = ("frontier_agent", "apodex", "workflows", "plugins", "benchmarks/public")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".jinja", ".jinja2", ".j2"}
FINGERPRINTS = ("source_sha256", "agent_sha256", "graph_sha256", "lock_sha256")


def github_repository(value: object) -> str | None:
    """Only expose a canonical GitHub URL, never credentials or a local remote path."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"(?:https://(?:[^/@\s]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)"
        r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?", value.strip())
    if not match or any(part in {".", ".."} for part in match.groups()):
        return None
    return f"https://github.com/{match[1]}/{match[2]}"


def git_metadata(root: Path) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(root), *args], env=env, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=5, check=False)
            return result.stdout.rstrip("\n") if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    # A copied folder nested inside another repository must not inherit its HEAD.
    top = git("rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != root.resolve():
        return {"commit": None, "branch": None, "dirty": None, "repository_url": None}
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=normal", "--ignore-submodules=none")
    return {
        "commit": git("rev-parse", "--verify", "HEAD"),
        "branch": git("symbolic-ref", "--quiet", "--short", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "repository_url": github_repository(git("remote", "get-url", "origin")),
    }


def _digest(manifest: dict[str, str]) -> str | None:
    if not manifest:
        return None
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def capture_version(root: Path, mode: str) -> tuple[dict, dict[str, str]]:
    """Return the public version stamp and a path/hash inventory for batch.json."""
    root = root.resolve()
    captured_at = datetime.now(UTC).isoformat()
    git = git_metadata(root)
    manifest: dict[str, str] = {}
    complete = True
    folders = (*AGENT_ROOTS, "tools/case_study_studio/profiles", *(("bcg",) if mode == "bcg" else ()))
    candidates = {root / name for name in ("pyproject.toml", "uv.lock", "tools/case_study_studio/run_job.py",
                                          "tools/case_study_studio/run_version.py") if (root / name).exists()}
    for folder in folders:
        candidates.update(file for file in (root / folder).rglob("*") if file.is_file() and (
            file.suffix in SOURCE_SUFFIXES or
            ("prompts" in file.parts and file.suffix in {".md", ".txt", ".json"}))
            and "__pycache__" not in file.parts)
    for file in sorted(candidates):
        # Do not follow source symlinks out of this checkout.
        if not file.resolve().is_relative_to(root):
            complete = False
            continue
        try:
            manifest[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        except OSError:
            complete = False
    agent = {name: digest for name, digest in manifest.items() if name.startswith(tuple(f"{p}/" for p in AGENT_ROOTS))}
    graph = {name: digest for name, digest in manifest.items() if name.startswith("bcg/")}
    version = {"schema_version": 1, "captured_at": captured_at, **git,
               "source_sha256": _digest(manifest), "agent_sha256": _digest(agent), "graph_sha256": _digest(graph),
               "lock_sha256": manifest.get("uv.lock"), "fingerprint_complete": complete}
    return version, manifest


def public_version(value: object) -> dict | None:
    """Whitelist persisted fields for the API. Never inspect current Git on a read."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None

    def digest(key: str, pattern: str = r"[0-9a-f]{64}") -> str | None:
        candidate = value.get(key)
        return candidate if isinstance(candidate, str) and re.fullmatch(pattern, candidate) else None

    commit = digest("commit", r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
    branch = value.get("branch")
    stamp = value.get("captured_at")
    return {"schema_version": 1, "commit": commit,
            "branch": branch[:240] if isinstance(branch, str) else None,
            "dirty": value.get("dirty") if type(value.get("dirty")) is bool else None,
            "repository_url": github_repository(value.get("repository_url")),
            "captured_at": stamp[:80] if isinstance(stamp, str) else None,
            "fingerprint_complete": value.get("fingerprint_complete") is True,
            **{key: digest(key) for key in FINGERPRINTS}}
