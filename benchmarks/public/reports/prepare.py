"""``python -m benchmarks.public.runner.run_reports prepare``: locate or fetch report-benchmark data.

For each benchmark, in order:

1. **Locate.** Existing data under ``--data-root`` is used as-is (both the
   documented layout and a full repository clone are recognized). Otherwise
   ``--from PATH`` imports a local clone, and otherwise the pinned GitHub
   revision is downloaded (unless ``--no-download``).
2. **Verify and record.** Task files are loaded and ``MANIFEST.json`` records
   the source, revision, commit, and checksums. Runs copy it into ``run.json``.
3. **Render (DRBench).** Text renderings are generated into ``_derived/``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from benchmarks.public.reports import drb2, drbench

MANIFEST_NAME = "MANIFEST.json"
SOURCES = {
    drb2.NAME: {"repo": "imlrz/DeepResearch-Bench-II", "default_revision": "main"},
    drbench.NAME: {"repo": "ServiceNow/drbench", "default_revision": "main"},
}
PREPARABLE = tuple(SOURCES)
_TIMEOUT = httpx.Timeout(60.0, read=300.0)


class PrepareError(RuntimeError):
    """Raised when benchmark data cannot be located, fetched, or verified."""


@dataclass
class PrepareResult:
    benchmark: str
    directory: Path
    action: str
    tasks: int
    manifest: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def prepare(
    benchmark: str,
    data_root: Path,
    *,
    revision: str | None = None,
    source: Path | None = None,
    download: bool = True,
    force: bool = False,
    render: bool = True,
    log: Any = print,
) -> PrepareResult:
    if benchmark not in SOURCES:
        raise PrepareError(
            f"`prepare` supports {', '.join(PREPARABLE)}; {benchmark} is prepared manually."
        )
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if benchmark == drb2.NAME:
        return _prepare_drb2(data_root, revision, source, download, force, log)
    return _prepare_drbench(data_root, revision, source, download, force, render, log)


# ---------------------------------------------------------------------------
# DeepResearch Bench II
# ---------------------------------------------------------------------------


def _prepare_drb2(
    data_root: Path,
    revision: str | None,
    source: Path | None,
    download: bool,
    force: bool,
    log: Any,
) -> PrepareResult:
    target_dir = data_root / drb2.DIRECTORY_CANDIDATES[0]
    existing = _existing_drb2(data_root)
    origin: dict[str, Any]
    if existing is not None and not force and source is None:
        tasks_file = existing
        action = "verified existing"
        origin = _previous_origin(tasks_file.parent) or {"type": "existing"}
    elif source is not None:
        source = Path(source).expanduser().resolve()
        candidate = source if source.is_file() else source / drb2.TASKS_FILENAME
        if not candidate.is_file():
            raise PrepareError(f"{drb2.TASKS_FILENAME} was not found at {source}.")
        target_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = target_dir / drb2.TASKS_FILENAME
        if candidate.resolve() != tasks_file.resolve():
            shutil.copyfile(candidate, tasks_file)
        license_file = candidate.parent / "DATA_LICENSE"
        if (
            license_file.is_file()
            and license_file.resolve() != (target_dir / "DATA_LICENSE").resolve()
        ):
            shutil.copyfile(license_file, target_dir / "DATA_LICENSE")
        action = f"imported from {source}"
        origin = {"type": "local", "path": str(source), **_git_origin(candidate.parent)}
    elif download:
        repo = SOURCES[drb2.NAME]["repo"]
        rev = revision or SOURCES[drb2.NAME]["default_revision"]
        commit = _resolve_commit(repo, rev, log)
        ref = commit or rev
        target_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = target_dir / drb2.TASKS_FILENAME
        log(f"Downloading {repo}@{ref} {drb2.TASKS_FILENAME} …")
        _download_file(_raw_url(repo, ref, drb2.TASKS_FILENAME), tasks_file)
        with contextlib.suppress(PrepareError):
            _download_file(_raw_url(repo, ref, "DATA_LICENSE"), target_dir / "DATA_LICENSE")
        action = "downloaded"
        origin = {"type": "github", "repo": repo, "revision": rev, "commit": commit}
    else:
        raise PrepareError(
            f"{drb2.NAME} data was not found under {data_root} and --no-download was set."
        )

    try:
        tasks = drb2.load_tasks(data_root, data_file=tasks_file)
    except drb2.DRB2DataError as exc:
        raise PrepareError(str(exc)) from exc
    languages: dict[str, int] = {}
    for task in tasks:
        languages[task.metadata["language"]] = languages.get(task.metadata["language"], 0) + 1
    manifest = {
        "benchmark": drb2.NAME,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": origin,
        "tasks_file": tasks_file.name,
        "tasks_sha256": _sha256(tasks_file),
        "tasks": len(tasks),
        "languages": languages,
        "rubric_items": sum(sum(task.metadata["rubric_counts"].values()) for task in tasks),
    }
    _write_manifest(tasks_file.parent, manifest)
    return PrepareResult(drb2.NAME, tasks_file.parent, action, len(tasks), manifest)


def _existing_drb2(data_root: Path) -> Path | None:
    try:
        return drb2.find_tasks_file(data_root)
    except drb2.DRB2DataError:
        return None


# ---------------------------------------------------------------------------
# DRBench
# ---------------------------------------------------------------------------


def _prepare_drbench(
    data_root: Path,
    revision: str | None,
    source: Path | None,
    download: bool,
    force: bool,
    render: bool,
    log: Any,
) -> PrepareResult:
    target_dir = data_root / drbench.DIRECTORY_CANDIDATES[0]
    existing = _existing_drbench(data_root)
    if existing is not None and not force and source is None:
        benchmark_dir = existing
        action = "verified existing"
        origin = _previous_origin(benchmark_dir) or {
            "type": "existing",
            **_git_origin(benchmark_dir),
        }
    elif source is not None:
        source = Path(source).expanduser().resolve()
        try:
            data_dir = drbench.resolve_data_dir(source)
        except drbench.DRBenchDataError as exc:
            raise PrepareError(str(exc)) from exc
        _replace_data(target_dir, data_dir)
        benchmark_dir = target_dir
        action = f"imported from {source}"
        origin = {"type": "local", "path": str(source), **_git_origin(source)}
    elif download:
        repo = SOURCES[drbench.NAME]["repo"]
        rev = revision or SOURCES[drbench.NAME]["default_revision"]
        commit = _resolve_commit(repo, rev, log)
        ref = commit or rev
        log(f"Downloading {repo}@{ref} (drbench/data only; about 100 MB) …")
        _download_drbench_data(repo, ref, target_dir)
        benchmark_dir = target_dir
        action = "downloaded"
        origin = {"type": "github", "repo": repo, "revision": rev, "commit": commit}
    else:
        raise PrepareError(
            f"{drbench.NAME} data was not found under {data_root} and --no-download was set."
        )

    notes: list[str] = []
    counts: dict[str, int] = {}
    tasks_by_id = {}
    for subset in drbench.SUBSETS:
        try:
            tasks = drbench.load_tasks(data_root, data_file=benchmark_dir, subset=subset)
        except drbench.DRBenchDataError as exc:
            notes.append(f"subset {subset}: {exc}")
            continue
        counts[subset] = len(tasks)
        tasks_by_id.update({task.task_id: task for task in tasks})
    if not tasks_by_id:
        raise PrepareError(f"No DRBench tasks could be loaded from {benchmark_dir}.")
    rendered = 0
    if render:
        log(f"Rendering {len(tasks_by_id)} DRBench task workspaces …")
        for task in tasks_by_id.values():
            try:
                drbench.ensure_rendered(task)
            except drbench.DRBenchDataError as exc:
                raise PrepareError(str(exc)) from exc
            rendered += 1
    data_dir = drbench.resolve_data_dir(benchmark_dir)
    manifest = {
        "benchmark": drbench.NAME,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": origin,
        "data_dir": str(data_dir.relative_to(benchmark_dir)) if data_dir != benchmark_dir else ".",
        "subsets": counts,
        "tasks": len(tasks_by_id),
        "subsets_sha256": {
            path.name: _sha256(path) for path in sorted((data_dir / "subsets").glob("*.jsonl"))
        },
        "renderer_version": drbench.RENDERER_VERSION,
        "rendered_tasks": rendered,
    }
    _write_manifest(benchmark_dir, manifest)
    return PrepareResult(drbench.NAME, benchmark_dir, action, len(tasks_by_id), manifest, notes)


def _existing_drbench(data_root: Path) -> Path | None:
    try:
        benchmark_dir = drbench.find_benchmark_dir(data_root)
        drbench.resolve_data_dir(benchmark_dir)
    except drbench.DRBenchDataError:
        return None
    return benchmark_dir


def _replace_data(target_dir: Path, data_dir: Path) -> None:
    if target_dir.resolve() == data_dir.resolve():
        return  # Re-importing the prepared dataset must never delete its source.
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("subsets", "tasks"):
        destination = target_dir / name
        if (data_dir / name).resolve().is_relative_to(destination.resolve()):
            raise PrepareError("Cannot replace a dataset directory containing the import source.")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(data_dir / name, destination)
    derived = target_dir / drbench.DERIVED_DIRNAME
    if derived.exists():
        shutil.rmtree(derived)


def _download_drbench_data(repo: str, ref: str, target_dir: Path) -> None:
    url = f"https://codeload.github.com/{repo}/tar.gz/{ref}"
    staging = target_dir.parent / f".{target_dir.name}.download"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    extracted = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=_TIMEOUT) as response:
            if response.status_code != 200:
                raise PrepareError(f"GET {url} failed with HTTP {response.status_code}.")
            stream = _ResponseReader(response)
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    parts = Path(member.name).parts
                    # <repo>-<ref>/drbench/data/{subsets,tasks}/...
                    if len(parts) < 4 or parts[1:3] != ("drbench", "data"):
                        continue
                    if parts[3] not in {"subsets", "tasks"} or not member.isfile():
                        continue
                    if any(part in {"..", ""} for part in parts):
                        continue
                    destination = staging.joinpath(*parts[3:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    with destination.open("wb") as output:
                        shutil.copyfileobj(handle, output)
                    extracted += 1
    except httpx.HTTPError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise PrepareError(f"Downloading {url} failed: {exc}") from exc
    if extracted == 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise PrepareError(f"No drbench/data files were found in {url}.")
    _replace_data(target_dir, staging)
    shutil.rmtree(staging, ignore_errors=True)


class _ResponseReader(io.RawIOBase):
    """File-like adapter so tarfile can stream an httpx response."""

    def __init__(self, response: httpx.Response) -> None:
        self._iterator = response.iter_bytes()
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, target: Any) -> int:
        while not self._buffer:
            try:
                self._buffer = next(self._iterator)
            except StopIteration:
                return 0
        size = min(len(target), len(self._buffer))
        target[:size] = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return size


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _raw_url(repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def _resolve_commit(repo: str, revision: str, log: Any) -> str | None:
    if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision.lower()):
        return revision.lower()
    commit = _ls_remote_commit(repo, revision)
    if commit:
        return commit
    try:
        response = httpx.get(
            f"https://api.github.com/repos/{repo}/commits/{revision}",
            headers={"Accept": "application/vnd.github+json"},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if response.status_code == 200:
            sha = response.json().get("sha")
            return str(sha) if sha else None
        log(f"Could not resolve {repo}@{revision} to a commit (HTTP {response.status_code}).")
    except (httpx.HTTPError, ValueError) as exc:
        log(f"Could not resolve {repo}@{revision} to a commit: {exc}")
    return None


def _ls_remote_commit(repo: str, revision: str) -> str | None:
    """Resolve a branch or tag with ``git ls-remote`` (no API rate limit)."""

    try:
        output = subprocess.run(
            ["git", "ls-remote", f"https://github.com/{repo}.git", revision],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        sha, _, ref = line.partition("\t")
        if ref in {f"refs/heads/{revision}", f"refs/tags/{revision}^{{}}"}:
            return sha.strip()
    first = output.split("\t", 1)[0].strip()
    return first if len(first) == 40 else None


def _download_file(url: str, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=_TIMEOUT) as response:
            if response.status_code != 200:
                raise PrepareError(f"GET {url} failed with HTTP {response.status_code}.")
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise PrepareError(f"Downloading {url} failed: {exc}") from exc
    temporary.replace(destination)


def _git_origin(path: Path) -> dict[str, Any]:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            try:
                commit = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                return {}
            return {"commit": commit, "git_root": str(candidate)}
    return {}


def _previous_origin(directory: Path) -> dict[str, Any] | None:
    manifest = read_manifest(directory)
    origin = manifest.get("source") if manifest else None
    return origin if isinstance(origin, dict) else None


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = Path(directory) / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def dataset_manifest(benchmark: str, data_root: Path) -> dict[str, Any] | None:
    """The MANIFEST.json of an already-prepared benchmark, if present."""

    root = Path(data_root).expanduser().resolve()
    try:
        if benchmark == drb2.NAME:
            return read_manifest(drb2.find_tasks_file(root).parent)
        if benchmark == drbench.NAME:
            return read_manifest(drbench.find_benchmark_dir(root))
    except (drb2.DRB2DataError, drbench.DRBenchDataError):
        return None
    return None


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "MANIFEST_NAME",
    "PREPARABLE",
    "PrepareError",
    "PrepareResult",
    "dataset_manifest",
    "prepare",
    "read_manifest",
]
