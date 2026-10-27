"""Read one native ReAct case. Never infer past injections from a final Graph."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

MAX_BYTES = 250 * 1024 * 1024
SECRET = re.compile(r"api[_-]?key|authorization|password|secret|access[_-]?token", re.I)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if SECRET.search(k) else redact(v) for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def clipped(text: str, limit: int) -> dict[str, Any]:
    return {
        "text": text[:limit],
        "truncated": len(text) > limit,
        "original_chars": len(text),
    }


def read_text(path: Path) -> str:
    if path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Evidence must be a regular file under 250 MiB")
    return path.read_text(encoding="utf-8")


def load_files(source: Path) -> tuple[dict[str, str], list[str]]:
    if source.name == "result.json":
        source = source.parent
    if source.is_dir():
        files = {}
        for name in ("case.json", "result.json", "report.md"):
            path = source / name
            if path.is_file():
                files[name] = read_text(path)
        for agent in (source, source / "agent"):
            for folder, pattern in (
                ("trajectories", "*.jsonl"),
                ("bcg", "requests.jsonl"),
                ("bcg_team", "**/requests.jsonl"),
            ):
                for path in sorted((agent / folder).glob(pattern)):
                    if not path.resolve().is_relative_to(source.resolve()):
                        raise ValueError("Evidence path escapes the task directory")
                    files[path.relative_to(source).as_posix()] = read_text(path)
        if sum(len(v.encode()) for v in files.values()) > MAX_BYTES:
            raise ValueError("Evidence exceeds 250 MiB")
        return files, []
    if source.is_symlink():
        raise ValueError("Symlink evidence is not accepted")
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("Decompressed evidence exceeds 250 MiB")
        bundle = json.loads(raw)
    else:
        bundle = json.loads(read_text(source))
    if bundle.get("format") != "bcg-case-bundle" or bundle.get("version") != 1:
        raise ValueError("Expected a task directory or bcg-case-bundle v1")
    if len(bundle.get("cases", [])) != 1:
        raise ValueError("Supply exactly one case; batch bundles are not accepted")
    entry = bundle["cases"][0]
    files = {}
    for item in entry["files"]:
        name, text = item["path"], item["text"]
        if (
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or name in files
        ):
            raise ValueError("Invalid evidence path")
        if sha(text) != item["digest"] or len(text.encode()) != item["size"]:
            raise ValueError("Evidence digest mismatch")
        files[name] = text
    return files, entry.get("warnings", [])


def rows(text: str, name: str, warnings: list[str]) -> list[tuple[int, dict]]:
    result = []
    for line, value in enumerate(text.splitlines(), 1):
        if not value.strip():
            continue
        try:
            item = json.loads(value)
            if isinstance(item, dict):
                result.append((line, redact(item)))
        except ValueError:
            warnings.append(f"{name}:{line}: incomplete JSONL record omitted")
    return result


def owner(path: str) -> str:
    parts = path.split("/")
    boundary = next(
        (i for i, p in enumerate(parts) if p in {"trajectories", "bcg", "bcg_team"}), -1
    )
    return "/".join(parts[:boundary]) if boundary >= 0 else path.rsplit("/", 1)[0]


def content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(content(v) for v in value)
    if isinstance(value, dict):
        return content(value.get("text", value.get("content", "")))
    return ""


def prepare_case(source: Path) -> dict[str, Any]:
    files, warnings = load_files(source)
    meta = {
        **json.loads(files.get("result.json", "{}")),
        **json.loads(files.get("case.json", "{}")),
    }
    question = meta.get("question", "")
    report = files.get("report.md") or meta.get("final_response", "")
    injections: dict[tuple[str, str], list[dict]] = {}
    for name, text in files.items():
        if name.endswith("requests.jsonl"):
            for line, row in rows(text, name, warnings):
                if row.get("call") is not None:
                    key = (owner(name), str(row["call"]))
                    injections.setdefault(key, []).append(
                        {**row, "source_path": name, "line": line}
                    )
    steps = []
    for name, text in sorted(files.items()):
        if not name.endswith(".jsonl") or "trajectories" not in name.split("/"):
            continue
        records = rows(text, name, warnings)
        digest = sha(text)
        starts = [r for _, r in records if r.get("t") == "start"]
        question = question or (starts[0].get("user_message", "") if starts else "")
        previous = []
        for index, (line, row) in enumerate(records):
            if row.get("t") != "llm":
                continue
            call = row.get("turn")
            matches = injections.get((owner(name), str(call)), [])
            unique = {
                json.dumps(
                    {k: m.get(k) for k in ("system_memory", "graph_chars", "fallback")},
                    sort_keys=True,
                )
                for m in matches
            }
            match = matches[0] if len(unique) == 1 else None
            graph_text, status = "", "missing" if not matches else "ambiguous"
            if match:
                blocks = re.findall(
                    r"<belief_graph>([\s\S]*?)</belief_graph>",
                    match.get("system_memory", ""),
                )
                if match.get("fallback"):
                    status = "fallback"
                elif len(blocks) == 1:
                    graph_text = blocks[0].strip()
                    status = "matched" if graph_text else "empty"
                elif match.get("graph_chars") == 0:
                    status = "empty"
            thinking = content(
                row.get("reasoning_content")
                or row.get("thinking")
                or row.get("reasoning")
            )
            action = "\n\n".join(
                x
                for x in (
                    "Thinking:\n" + thinking if thinking else "",
                    content(row.get("content")),
                    "Tool calls:\n"
                    + json.dumps(row["tool_calls"], ensure_ascii=False, indent=2)
                    if row.get("tool_calls")
                    else "",
                )
                if x
            )
            results = []
            for _, following in records[index + 1 :]:
                if following.get("t") == "llm":
                    break
                if following.get("t") == "result" and str(following.get("turn")) == str(
                    call
                ):
                    results.append(
                        {
                            "tool": following.get("name"),
                            "tool_call_id": following.get("tool_call_id"),
                            "text": content(following.get("result")),
                        }
                    )
            step = {
                "id": "S" + sha(f"{name}:{digest}:{line}")[:16],
                "call": call,
                "agent": owner(name) or "Agent",
                "source": {"path": name, "digest": digest, "index": line, "call": call},
                "graph": {
                    **clipped(graph_text, 30000),
                    "status": status,
                    "source_path": match.get("source_path") if match else None,
                    "source_line": match.get("line") if match else None,
                },
                "action_text": clipped(action, 16000),
                "tool_results": clipped(
                    json.dumps(results, ensure_ascii=False, indent=2), 18000
                ),
                "previous_calls": [
                    {
                        "id": p["id"],
                        "call": p["call"],
                        "action_text": clipped(p["action_text"]["text"], 3000),
                        "tool_results": clipped(p["tool_results"]["text"], 3000),
                    }
                    for p in previous[-2:]
                ],
                "retained_indices": match.get("retained_indices") if match else None,
            }
            previous.append(step)
            steps.append(step)
    if not steps:
        raise ValueError("No native ReAct llm trajectory records found")
    if not report.strip():
        raise ValueError("No non-empty final report found")
    if not question.strip():
        raise ValueError("Task question is missing")
    identity = sorted(
        (name, sha(text))
        for name, text in files.items()
        if name in {"case.json", "report.md", "result.json"} or name.endswith(".jsonl")
    )
    return {
        "format": "bcg-ai-review-evidence",
        "version": 1,
        "evidence_sha256": sha(json.dumps(identity)),
        "task_id": str(meta.get("task_id") or source.name),
        "benchmark": meta.get("benchmark", ""),
        "question": clipped(question, 20000),
        "final_report": clipped(report, 50000),
        "task_status": meta.get("status", "unknown"),
        "steps": steps,
        "coverage": {
            "agent_calls": len(steps),
            "graph_matched_calls": sum(
                s["graph"]["status"] == "matched" for s in steps
            ),
            "truncated_calls": sum(
                any(s[k]["truncated"] for k in ("graph", "action_text", "tool_results"))
                for s in steps
            ),
            "warnings": warnings,
        },
    }


def batches(
    case: dict, *, chunk_chars: int = 60000, chunk_turns: int = 6
) -> list[dict]:
    groups: list[list[dict]] = []
    group: list[dict] = []
    size = 0
    for step in case["steps"]:
        n = len(json.dumps(step, ensure_ascii=False))
        if group and (size + n > chunk_chars or len(group) >= chunk_turns):
            groups.append(group)
            group, size = [], 0
        group.append(step)
        size += n
    if group:
        groups.append(group)
    return [
        {
            "task_id": case["task_id"],
            "question": case["question"],
            "final_report": case["final_report"],
            "eligible_ids": [s["id"] for s in g],
            "steps": g,
            "coverage": case["coverage"],
        }
        for g in groups
    ]
