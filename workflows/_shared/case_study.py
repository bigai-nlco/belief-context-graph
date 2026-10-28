"""Portable Case Study Studio evidence bundles. No model calls or remote uploads.

Each task owns its bundle; the batch runner combines them only after its workers
finish. Evidence files are allowlisted, hash-checked, and stored as UTF-8 text in
gzip-compressed JSON. Configuration, environments and arbitrary outputs are not
included. Run this module to package existing runs without rerunning the Agent.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
FORMAT = "bcg-case-bundle"
CASE_NAME = "case-study.json.gz"
BATCH_NAME = "case-studies.json.gz"
MAX_BYTES = 250 * 1024 * 1024
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_FILES = 300
_SECRET = re.compile(r"api[_-]?key|authorization|password|secret|access[_-]?token", re.I)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_bundle(path: Path, cases: list[dict[str, Any]]) -> Path:
    data = _json({"format": FORMAT, "version": 1, "cases": cases}).encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError("Case bundle exceeds the 250 MiB uncompressed import limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".case-study-", delete=False) as f:
            name = f.name
            os.chmod(name, 0o600)
            with gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as packed:
                packed.write(data)
        os.replace(name, path)
    finally:
        if name:
            Path(name).unlink(missing_ok=True)
    return path


def read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as f:
        raw = f.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Case bundle exceeds the import limit")
    data = json.loads(raw)
    if data.get("format") != FORMAT or data.get("version") != 1:
        raise ValueError("Unsupported case bundle")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Empty case bundle")
    return data


def export_case(
    directory: Path,
    *,
    metadata: dict[str, Any] | None = None,
    output: Path | None = None,
    graph_dir: Path | None = None,
) -> Path:
    """Collect a single task, without assuming Graph service filesystem access."""
    root = directory.resolve()
    if not root.is_dir():
        raise ValueError(f"Task directory does not exist: {root}")
    result = _read_json(root / "result.json")
    supplied = metadata or {}
    # Prior generated metadata preserves the full question when a later runner
    # provides only scoring fields. Never include SDK objects or profile secrets.
    target = output or root / CASE_NAME
    previous: dict[str, Any] = {}
    if target.is_file():
        for item in read_bundle(target)["cases"][0].get("files", []):
            if item["path"] == "case.json":
                previous = json.loads(item["text"])
    allowed = (
        "task_id", "benchmark", "mode", "pipeline", "model", "question",
        "final_response", "status", "stopped_by", "duration_seconds",
        "usage", "graph_usage", "correct", "official_score",
    )
    info = {key: previous[key] for key in allowed if key in previous}
    info.update({key: result[key] for key in allowed if key in result})
    info.update({key: supplied[key] for key in allowed if key in supplied})
    info.setdefault("task_id", str(result.get("question_id") or root.name))
    info["task_id"] = str(info["task_id"])
    info.setdefault("benchmark", "")
    info.setdefault("mode", "bcg")
    if not info.get("final_response"):
        info["final_response"] = result.get("final_response") or result.get("predicted_answer") or ""
        for name in ("report.md", "agent/final_answer.txt"):
            file = root / name
            if file.is_file() and not file.is_symlink():
                info["final_response"] = file.read_text(encoding="utf-8")
                break

    paths: dict[str, Path] = {}
    warnings: list[str] = []

    def add(path: Path, base: Path, prefix: str = "") -> None:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
            return
        name = prefix + path.relative_to(base).as_posix()
        if name in paths:
            return
        if path.stat().st_size > MAX_FILE_BYTES:
            warnings.append(f"文件超过 200 MiB，未打包：{name}")
            return
        paths[name] = path

    # Explicit evidence subtrees, not rglob over a tool's working directory.
    for agent_root in (root, root / "agent"):
        trajectory = agent_root / "trajectories"
        prefix = ""
        if agent_root == root and root.parent.name == "trajectories":
            # Non-interactive SDK runs with APODEX_RUN_DIR write the task's
            # traces directly here, not in a second trajectories/ directory.
            trajectory, prefix = root, "trajectories/"
        if trajectory.is_dir() and not trajectory.is_symlink():
            for path in sorted(trajectory.rglob("*.json*")):
                if any(part in {"bcg", "bcg_team"} for part in path.relative_to(trajectory).parts):
                    continue
                if path.suffix not in {".json", ".jsonl"}:
                    continue
                if path.suffix == ".json" and path.with_suffix(".jsonl").is_file():
                    continue
                add(path, root, prefix)
        for memory in (agent_root / "bcg", agent_root / "bcg_team", trajectory / "bcg_team"):
            if memory.is_dir() and not memory.is_symlink():
                for pattern in ("state.json", "requests.jsonl", "graph_snapshots.jsonl", "index.json"):
                    for path in sorted(memory.rglob(pattern)):
                        add(path, root)

    # Older local runs can supply their service-side history without API calls.
    graph_ids = {
        str(_read_json(path).get("problem_id") or "")
        for path in paths.values() if path.name == "state.json"
    }
    for base in (root / "graphs", *([graph_dir.resolve()] if graph_dir else [])):
        for graph_id in sorted(graph_ids):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", graph_id):
                continue
            add(base / graph_id / "belief_graph.jsonl", base, "graphs/")
    if not any("trajectories/" in p for p in paths):
        warnings.append("缺少 Agent 运行轨迹。")
    if not any(p.endswith("state.json") for p in paths):
        warnings.append("缺少完整 Graph 记录。")
    if not any(p.endswith(("graph_snapshots.jsonl", "belief_graph.jsonl")) for p in paths):
        warnings.append("缺少历史 Graph 快照，未从最终图推造历史状态。")
    if len(paths) + 1 > MAX_FILES:
        raise ValueError("More than 300 evidence files in one case")

    secrets = sorted({
        value for key, value in os.environ.items()
        if _SECRET.search(key) and len(value) >= 12
    }, key=len, reverse=True)

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if _SECRET.search(key) else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
        return value

    files = []
    total_bytes = 0
    for name, path in [("case.json", None), *sorted(paths.items())]:
        original = _json(info) if path is None else path.read_text(encoding="utf-8")
        # Structured sanitization, not a regex over arbitrary evidence prose.
        if name.endswith(".jsonl"):
            lines = []
            for index, line in enumerate(original.splitlines(), 1):
                try:
                    lines.append(_json(redact(json.loads(line))) if line.strip() else "")
                except json.JSONDecodeError:
                    lines.append(redact(line))
                    warnings.append(f"{name} 第 {index} 行不完整，保留原文供导入检查。")
            text = "\n".join(lines) + "\n"
        else:
            text = _json(redact(json.loads(original)))
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > MAX_BYTES:
            raise ValueError("Case evidence exceeds the 250 MiB import limit")
        files.append({
            "path": name, "text": text, "size": len(text.encode("utf-8")),
            "digest": _sha(text), "originalDigest": _sha(original),
        })
    case_id = _sha(_json([str(root), info["task_id"], info["benchmark"], info["mode"]]))
    case = {
        "id": case_id, "task_id": info["task_id"], "benchmark": info["benchmark"],
        "mode": info["mode"], "files": files, "warnings": warnings,
    }
    return _write_bundle(target, [case])


def export_batch(directory: Path, *, task_ids: list[str] | None = None) -> list[Path]:
    """Combine completed per-task bundles; split large batches for browser limits."""
    root = directory.resolve()
    if task_ids is None:
        sources = sorted((root / "trials").glob(f"*/{CASE_NAME}"))
    else:
        sources = [root / "trials" / task_id / CASE_NAME for task_id in task_ids]
    groups: list[list[dict[str, Any]]] = []
    group: list[dict[str, Any]] = []
    size = 100
    seen: set[str] = set()
    for source in sources:
        if not source.is_file():
            continue
        for case in read_bundle(source)["cases"]:
            if case["id"] in seen:
                continue
            seen.add(case["id"])
            length = len(_json(case).encode("utf-8")) + 1
            if group and (size + length > MAX_BYTES - 1024 or len(group) >= 1000):
                groups.append(group)
                group, size = [], 100
            group.append(case)
            size += length
    if group:
        groups.append(group)
    outputs = [
        _write_bundle(root / (BATCH_NAME if len(groups) == 1 else f"case-studies-{i:03}.json.gz"), cases)
        for i, cases in enumerate(groups, 1)
    ]
    # Retire obsolete auto-generated parts only after the new set is complete.
    # Otherwise directory imports could pick up an older unpartitioned bundle.
    if outputs:
        for old in root.glob("case-studies*.json.gz"):
            if (old not in outputs
                    and re.fullmatch(r"case-studies(?:-\d{3,})?\.json\.gz", old.name)
                    and not old.is_symlink()):
                read_bundle(old)  # do not delete unrelated files with this name
                old.unlink()
    return outputs


def try_export_case(directory: Path, **kwargs: Any) -> Path | None:
    """Artifact failures must not invalidate a completed Agent run."""
    try:
        path = export_case(directory, **kwargs)
        logger.info("Case Study bundle: %s", path)
        return path
    except Exception:
        logger.exception("Case Study bundle could not be saved in %s", directory)
        return None


def try_export_batch(directory: Path, *, task_ids: list[str] | None = None) -> list[Path]:
    try:
        paths = export_batch(directory, task_ids=task_ids)
        for path in paths:
            logger.info("Case Study batch: %s", path)
        return paths
    except Exception:
        logger.exception("Case Study batch could not be saved in %s", directory)
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Single task or run containing trials/")
    parser.add_argument("--output", type=Path, help="Single-case output file (.json.gz)")
    parser.add_argument("--graph-dir", type=Path, help="Optional existing Graph service output root")
    args = parser.parse_args()
    if (args.directory / "trials").is_dir():
        if args.output:
            parser.error("--output applies to single tasks; batches use case-studies*.json.gz")
        for trial in sorted((args.directory / "trials").iterdir()):
            if (trial / "result.json").is_file():
                export_case(trial, graph_dir=args.graph_dir)
        for path in export_batch(args.directory):
            print(path)
    else:
        print(export_case(args.directory, output=args.output, graph_dir=args.graph_dir))


if __name__ == "__main__":
    main()
