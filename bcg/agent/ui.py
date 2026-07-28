"""Small web UI for monitoring BeliefTracer rollout artifacts."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bcg.cli_help import RichArgumentParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from bcg.agent.config import model_display_name, model_tag_for_path


DEFAULT_HOST = "172.25.10.2"
DEFAULT_PORT = 23456
DEFAULT_CONTAINER = "belief_tracer"
RUN_STATE_STALE_SECONDS = 120
AUTO_UI_LOG = Path(f"/tmp/belief_tracer-ui-{os.environ.get('USER', os.environ.get('LOGNAME', 'unknown'))}.log")
FAVICON_ASSET_NAME = "face-blowing-a-kiss-svgrepo-com.svg"
DEFAULT_MEMGRAPH_URI = "bolt://172.25.10.2:7687"
MEMGRAPH_NODE_LABELS = {"Claim", "BeliefVariable", "Evidence", "Decision", "Factor"}
MEMGRAPH_EDGE_TYPES = {
    "HAS_BELIEF",
    "OWNED_BY",
    "EVALUATED_BY",
    "INPUT_TO",
    "OUTPUT_TO",
    "REQUIRED_BY",
    "SUPPORTS",
}
EDGE_TYPE_PRIORITY = {
    "EVALUATED_BY": 100,
    "HAS_BELIEF": 90,
    "INPUT_TO": 80,
    "OUTPUT_TO": 70,
    "REQUIRED_BY": 60,
    "SUPPORTS": 50,
    "OWNED_BY": 10,
}


@dataclass(frozen=True)
class UiConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    artifacts_dir: Path = Path("artifacts/belief_tracer")
    poll_seconds: float = 2.0
    max_results: int = 100
    max_trajectories: int = 200
    container_name: str = DEFAULT_CONTAINER


def _json_read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iso(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _names_file(root: Path) -> Path:
    return root / ".run_names.json"


def _meta_file(root: Path) -> Path:
    return root / ".run_meta.json"


def _empty_meta_entry() -> dict[str, Any]:
    return {"display_name": "", "notes": "", "tags": []}


def _coerce_meta_entry(value: Any) -> dict[str, Any]:
    out = _empty_meta_entry()
    if isinstance(value, str):
        out["display_name"] = value.strip()
        return out
    if not isinstance(value, dict):
        return out
    name = value.get("display_name") or value.get("name") or ""
    if isinstance(name, str):
        out["display_name"] = name.strip()
    notes = value.get("notes")
    if isinstance(notes, str):
        out["notes"] = notes
    tags = value.get("tags")
    if isinstance(tags, list):
        out["tags"] = [str(t).strip() for t in tags if str(t).strip()]
    elif isinstance(tags, str):
        out["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    return out


def _meta_is_empty(entry: dict[str, Any]) -> bool:
    return (
        not entry.get("display_name")
        and not entry.get("notes")
        and not entry.get("tags")
    )


def _load_meta(root: Path) -> dict[str, dict[str, Any]]:
    meta_path = _meta_file(root)
    raw: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            raw.update({str(k): v for k, v in loaded.items()})
    legacy = _names_file(root)
    if legacy.exists() and not raw:
        try:
            with legacy.open("r", encoding="utf-8") as fh:
                old = json.load(fh)
        except Exception:
            old = None
        if isinstance(old, dict):
            for k, v in old.items():
                if isinstance(v, str) and v:
                    raw[str(k)] = {"display_name": v}
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        entry = _coerce_meta_entry(v)
        if not _meta_is_empty(entry):
            out[k] = entry
    return out


def _save_meta(root: Path, meta: dict[str, dict[str, Any]]) -> None:
    path = _meta_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: v for k, v in meta.items() if not _meta_is_empty(v)}
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp.replace(path)


def _strip_thinking_dir_suffix(name: str) -> str:
    for suffix in ("_no-thinking", "_thinking"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _infer_enable_thinking_from_dir(name: str) -> bool | None:
    if name.endswith("_thinking"):
        return True
    if name.endswith("_no-thinking"):
        return False
    return None


def _artifact_model_dir(path: Path) -> Path:
    if path.name == "run_state.json":
        return path.parent
    return path.parent.parent


def _artifact_config(path: Path) -> dict[str, Any]:
    model_dir = _artifact_model_dir(path)
    for candidate in (model_dir / "run_config.json", model_dir / "run_state.json", model_dir / "overall_summary.json"):
        if not candidate.exists():
            continue
        try:
            data = _json_read(candidate)
        except Exception:
            continue
        config = data.get("config") if isinstance(data, dict) else None
        if isinstance(config, dict):
            return config
    return {}


def _read_run_config_for_state(path: Path) -> dict[str, Any]:
    if path.name != "run_state.json":
        raise ValueError("run config is only valid for run_state.json entries")
    state: dict[str, Any] = {}
    try:
        raw_state = _json_read(path)
        if isinstance(raw_state, dict):
            state = raw_state
    except Exception:
        state = {}

    run_id = str(state.get("run_id") or "")
    candidates: list[Path] = []
    if run_id:
        candidates.append(path.parent / "run_configs" / f"{run_id}.json")
    candidates.append(path.parent / "run_config.json")
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = _json_read(candidate)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["_path"] = str(candidate)
            return payload
    return {
        "schema_version": 0,
        "source": "fallback_from_run_state",
        "run_id": run_id,
        "status": state.get("status") or "",
        "phase": state.get("phase") or "",
        "config": state.get("config") if isinstance(state.get("config"), dict) else {},
        "run_state_path": str(path),
    }


def _artifact_model_name(
    path: Path,
    *,
    config: dict[str, Any] | None = None,
    fallback_model: Any = "",
) -> str:
    config = config or {}
    model_dir = _artifact_model_dir(path).name
    raw_model = config.get("model") or fallback_model or _strip_thinking_dir_suffix(model_dir)
    model = model_tag_for_path(str(raw_model))
    enabled = config.get("enable_thinking")
    if not isinstance(enabled, bool):
        enabled = _infer_enable_thinking_from_dir(model_dir)
    if isinstance(enabled, bool):
        return model_display_name(model, enabled)
    return model


def _result_summary(path: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        data = _json_read(path)
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        config = _artifact_config(path)
        records = data.get("records", []) if isinstance(data, dict) else []
        sample_count = sum(len(r.get("samples", [])) for r in records if isinstance(r, dict))
        correct_count = sum(
            int(bool(s.get("is_correct")))
            for r in records
            if isinstance(r, dict)
            for s in r.get("samples", [])
            if isinstance(s, dict)
        )
        return {
            "path": str(path),
            "name": path.parent.name,
            "model": _artifact_model_name(
                path,
                config=config,
                fallback_model=summary.get("model") or path.parent.parent.name,
            ),
            "thinking_mode": (
                "thinking"
                if config.get("enable_thinking") is True
                else "no thinking"
                if config.get("enable_thinking") is False
                else ""
            ),
            "benchmark": summary.get("benchmark") or path.parent.name,
            "status": "complete",
            "mtime": stat.st_mtime,
            "mtime_iso": _iso(stat.st_mtime),
            "size_bytes": stat.st_size,
            "summary": summary,
            "num_records": len(records),
            "num_samples": sample_count,
            "num_correct": correct_count,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "name": path.name,
            "model": _artifact_model_name(path),
            "benchmark": path.parent.name,
            "status": "unreadable",
            "mtime": stat.st_mtime,
            "mtime_iso": _iso(stat.st_mtime),
            "size_bytes": stat.st_size,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _jsonl_summary(path: Path) -> dict[str, Any]:
    stat = path.stat()
    line_count = 0
    last_entry: dict[str, Any] = {}
    parse_errors = 0
    last_line = ""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                line_count += 1
                last_line = line
        if last_line:
            try:
                parsed = json.loads(last_line)
                if isinstance(parsed, dict):
                    last_entry = parsed
            except Exception:
                parse_errors += 1
    except Exception as exc:
        return {
            "path": str(path),
            "name": path.name,
            "model": path.parent.parent.name if path.parent.parent else "",
            "benchmark": path.parent.name,
            "status": "unreadable",
            "mtime": stat.st_mtime,
            "mtime_iso": _iso(stat.st_mtime),
            "size_bytes": stat.st_size,
            "error": f"{type(exc).__name__}: {exc}",
        }
    config = _artifact_config(path)
    return {
        "path": str(path),
        "name": path.name,
        "model": _artifact_model_name(
            path,
            config=config,
            fallback_model=last_entry.get("model") or path.parent.parent.name,
        ),
        "thinking_mode": (
            "thinking"
            if config.get("enable_thinking") is True
            else "no thinking"
            if config.get("enable_thinking") is False
            else ""
        ),
        "benchmark": str(last_entry.get("benchmark") or path.parent.name),
        "status": "live",
        "mtime": stat.st_mtime,
        "mtime_iso": _iso(stat.st_mtime),
        "size_bytes": stat.st_size,
        "num_samples": line_count,
        "parse_errors": parse_errors,
        "last_completed_iso": last_entry.get("completed_iso") or "",
        "run_id": last_entry.get("run_id") or "",
    }


def _run_state_summary(path: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        state = _json_read(path)
    except Exception as exc:
        return {
            "path": str(path),
            "status": "unreadable",
            "mtime": stat.st_mtime,
            "mtime_iso": _iso(stat.st_mtime),
            "error": f"{type(exc).__name__}: {exc}",
        }
    updated = float(state.get("updated_at") or stat.st_mtime)
    age = max(0.0, time.time() - updated)
    status = str(state.get("status") or "unknown")
    if status in {"starting", "running"} and age > RUN_STATE_STALE_SECONDS:
        status = "stale"
    completed = int(state.get("completed_samples") or 0)
    total = state.get("total_samples")
    total = int(total) if isinstance(total, (int, float)) else None
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    run_id = str(state.get("run_id") or "")
    history_config = path.parent / "run_configs" / f"{run_id}.json" if run_id else None
    latest_config = path.parent / "run_config.json"
    run_config_path = (
        str(history_config)
        if history_config is not None and history_config.exists()
        else str(latest_config)
        if latest_config.exists()
        else ""
    )
    return {
        "path": str(path),
        "run_id": run_id,
        "run_config_path": run_config_path,
        "model": _artifact_model_name(path, config=config),
        "thinking_mode": (
            "thinking"
            if config.get("enable_thinking") is True
            else "no thinking"
            if config.get("enable_thinking") is False
            else ""
        ),
        "status": status,
        "phase": state.get("phase"),
        "pid": state.get("pid"),
        "updated_at": updated,
        "updated_iso": _iso(updated),
        "age_seconds": age,
        "started_iso": _iso(state.get("started_at")),
        "elapsed_seconds": state.get("elapsed_seconds"),
        "completed_benchmarks": state.get("completed_benchmarks"),
        "total_benchmarks": state.get("total_benchmarks"),
        "completed_samples": completed,
        "total_samples": total,
        "progress": (completed / total) if total else None,
        "current_benchmark": state.get("current_benchmark"),
        "current_expected_samples": state.get("current_expected_samples"),
        "live_paths": state.get("live_paths") or [],
        "result_paths": state.get("result_paths") or [],
        "summaries": state.get("summaries") or [],
        "error": state.get("error") or "",
        "config": config,
    }


def _scan_artifacts(root: Path, max_results: int) -> dict[str, Any]:
    root = root.resolve()
    result_paths = sorted(
        root.glob("**/results*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )[:max_results]
    stream_paths = sorted(
        root.glob("**/trajectories*.jsonl"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )[:max_results]
    state_paths = sorted(
        root.glob("**/run_state.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    names = _load_meta(root)
    runs = [_run_state_summary(p) for p in state_paths]
    streams = [_jsonl_summary(p) for p in stream_paths]
    results = [_result_summary(p) for p in result_paths]
    for entry in (*runs, *streams, *results):
        custom = names.get(entry.get("path") or "")
        if not custom:
            continue
        if custom.get("display_name"):
            entry["display_name"] = custom["display_name"]
        if custom.get("notes"):
            entry["notes"] = custom["notes"]
        if custom.get("tags"):
            entry["tags"] = list(custom["tags"])
    return {
        "root": str(root),
        "generated_at": time.time(),
        "generated_iso": _iso(time.time()),
        "runs": runs,
        "streams": streams,
        "results": results,
    }


def _resolve_artifact_path(root: Path, raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("missing path")
    candidate = Path(unquote(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path is outside artifacts root: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _read_stream_payload(path: Path, limit: int) -> dict[str, Any]:
    raw_rows: deque[str] = deque(maxlen=max(1, limit))
    rows: deque[dict[str, Any]] = deque(maxlen=max(1, limit))
    total = 0
    parse_errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            raw_rows.append(line)
    for line in raw_rows:
        try:
            parsed = json.loads(line)
        except Exception:
            parse_errors += 1
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)

    records_by_problem: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    correct = 0
    for entry in rows:
        sample = entry.get("sample") if isinstance(entry.get("sample"), dict) else {}
        problem_id = str(entry.get("problem_id") or entry.get("task_id") or "")
        if problem_id not in records_by_problem:
            records_by_problem[problem_id] = {
                "problem_id": problem_id,
                "task_id": entry.get("task_id"),
                "data_source": entry.get("data_source"),
                "question": entry.get("question"),
                "ground_truth": entry.get("ground_truth"),
                "num_samples": 0,
                "num_correct": 0,
                "samples": [],
            }
            order.append(problem_id)
        sample = dict(sample)
        sample["sample_index"] = entry.get("sample_index")
        sample["completed_iso"] = entry.get("completed_iso")
        sample["elapsed_seconds"] = entry.get("elapsed_seconds")
        rec = records_by_problem[problem_id]
        rec["samples"].append(sample)
        rec["num_samples"] += 1
        rec["num_correct"] += int(bool(sample.get("is_correct")))
        correct += int(bool(sample.get("is_correct")))

    records = [records_by_problem[key] for key in order]
    last = rows[-1] if rows else {}
    summary = {
        "benchmark": last.get("benchmark") or path.parent.name,
        "data_source": last.get("data_source") or "",
        "model": last.get("model") or (path.parent.parent.name if path.parent.parent else ""),
        "num_tasks": len(records),
        "num_completed_trajectories": total,
        "num_displayed_trajectories": len(rows),
        "num_correct_displayed": correct,
        "parse_errors": parse_errors,
        "live": True,
    }
    return {
        "summary": summary,
        "records": records,
        "num_samples": len(rows),
        "num_correct": correct,
        "_path": str(path),
        "_mtime_iso": _iso(path.stat().st_mtime),
        "_total_stream_rows": total,
    }


def _stable_suffix(*parts: Any) -> str:
    raw = "::".join(str(p) for p in parts if p is not None)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _sample_message(sample: dict[str, Any], role: str) -> dict[str, Any]:
    for msg in sample.get("trajectory") or []:
        if isinstance(msg, dict) and msg.get("role") == role:
            return msg
    return {}


def _assistant_text_for_step(sample: dict[str, Any], step_index: int) -> str:
    assistant_seen = 0
    for msg in sample.get("trajectory") or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if assistant_seen == step_index:
            parts = [
                msg.get("reasoning_content") or msg.get("reasoning") or "",
                msg.get("content") or "",
                msg.get("raw_content") or "",
            ]
            return "\n\n".join(str(p) for p in parts if p)
        assistant_seen += 1
    steps = sample.get("model_steps") if isinstance(sample.get("model_steps"), list) else []
    if 0 <= step_index < len(steps) and isinstance(steps[step_index], dict):
        step = steps[step_index]
        return "\n\n".join(
            str(step.get(k) or "")
            for k in ("reasoning_preview", "text_preview")
            if step.get(k)
        )
    return ""


def _candidate_source_phrases(question: str, assistant_text: str) -> list[str]:
    candidates: list[str] = []
    preferred_patterns = [
        r"1/r\^2",
        r"radial direction",
        r"spherical coordinates",
        r"divergence theorem",
        r"Gauss'?s theorem",
        r"surface integral",
        r"sphere of radius R",
        r"4\s*π",
        r"4\s*\\pi",
        r"origin",
    ]
    combined = f"{question}\n{assistant_text}"
    for pattern in preferred_patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            phrase = match.group(0)
            if phrase not in candidates:
                candidates.append(phrase)

    question_body = re.split(r"\n\s*[A-D]\.", question, maxsplit=1)[0]
    for chunk in re.split(r"[.;?\n]", question_body):
        words = re.findall(r"[A-Za-z0-9_/$\\^π∇.-]+", chunk)
        if 3 <= len(words) <= 10:
            phrase = " ".join(words).strip(" .")
            if len(phrase) >= 14 and phrase not in candidates:
                candidates.append(phrase)
        if len(candidates) >= 8:
            break
    return candidates[:8]


def _phrase_present(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return phrase.lower() in text.lower()


def _favicon_asset_path() -> Path:
    repo_asset = Path(__file__).resolve().parents[2] / "assets" / FAVICON_ASSET_NAME
    if repo_asset.is_file():
        return repo_asset
    return Path("assets") / FAVICON_ASSET_NAME


def _edge_weight(edge: dict[str, Any]) -> float:
    weight = edge.get("weight")
    return float(abs(weight)) if isinstance(weight, (int, float)) else 0.0


def _dedupe_directed_belief_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], tuple[tuple[float, float, int], int, dict[str, Any]]] = {}
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict) or not edge.get("source") or not edge.get("target"):
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source == target:
            continue
        relation = str(edge.get("type") or edge.get("label") or edge.get("relationship") or "RELATED_TO")
        normalized = dict(edge)
        normalized["source"] = source
        normalized["target"] = target
        normalized["type"] = relation
        pair_key = tuple(sorted((source, target)))
        score = (float(EDGE_TYPE_PRIORITY.get(relation.upper(), 0)), _edge_weight(normalized), -idx)
        current = selected.get(pair_key)
        if current is None or score > current[0]:
            selected[pair_key] = (score, idx, normalized)
    return [item[2] for item in sorted(selected.values(), key=lambda item: item[1])]


def _normalize_belief_memory_graph(memory: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(memory)
    normalized["nodes"] = [
        node for node in memory.get("nodes", []) if isinstance(node, dict) and node.get("id")
    ]
    normalized["edges"] = _dedupe_directed_belief_edges(
        [edge for edge in memory.get("edges", []) if isinstance(edge, dict)]
    )
    return normalized


_GPQA_DIAMOND_FIRST10_BELIEF_ANNOTATIONS: dict[str, dict[str, Any]] = {
    "gpqa_diamond-37": {
        "domain": "physics.vector_calculus",
        "decision_label": "C: 4 pi",
        "confidence": 0.97,
        "status": "auto_executed",
        "posterior_delta": 0.42,
        "beliefs": [
            {
                "key": "gauss_theorem",
                "claim": "The divergence integral can be evaluated as boundary flux by the Divergence Theorem.",
                "source": "Divergence Theorem",
                "posterior": 0.96,
                "strength": 0.94,
                "factor_weight": 0.91,
            },
            {
                "key": "inverse_square_radial",
                "claim": "The field is radial with magnitude 1/r^2, so its surface value cancels the sphere area factor.",
                "source": "the value of the field on the surface is",
                "posterior": 0.93,
                "strength": 0.90,
                "factor_weight": 0.86,
            },
            {
                "key": "sphere_flux",
                "claim": "The surface flux over the sphere integrates to 4 pi.",
                "source": "2\\pi \\cdot 2 = 4\\pi",
                "posterior": 0.98,
                "strength": 0.96,
                "factor_weight": 0.94,
            },
            {
                "key": "origin_singularity",
                "claim": "The apparent zero divergence away from the origin is reconciled by the singularity at r=0.",
                "source": "singularity at the origin",
                "posterior": 0.89,
                "strength": 0.83,
                "factor_weight": 0.72,
            },
            {
                "key": "dimensionless_option",
                "claim": "The result should be dimensionless, which rules out the R-dependent option.",
                "source": "Dimensional analysis also supports this",
                "posterior": 0.78,
                "strength": 0.72,
                "factor_weight": 0.53,
            },
        ],
    },
    "gpqa_diamond-187": {
        "domain": "physics.quantum_uncertainty",
        "decision_label": "A: ~10^(-16) J",
        "confidence": 0.95,
        "status": "auto_executed",
        "posterior_delta": 0.39,
        "beliefs": [
            {
                "key": "heisenberg",
                "claim": "Minimum momentum uncertainty follows from Delta x Delta p >= hbar/2.",
                "source": "Heisenberg Uncertainty Principle",
                "posterior": 0.95,
                "strength": 0.93,
                "factor_weight": 0.90,
            },
            {
                "key": "momentum_bound",
                "claim": "The minimum momentum uncertainty is hbar/(2 Delta x).",
                "source": "\\Delta p_{\\text{min}} = \\frac{\\hbar}{2 \\Delta x}",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.88,
            },
            {
                "key": "energy_momentum",
                "claim": "Energy uncertainty can be propagated as Delta E = v Delta p for the given velocity.",
                "source": "\\Delta E = v \\Delta p",
                "posterior": 0.88,
                "strength": 0.84,
                "factor_weight": 0.75,
            },
            {
                "key": "position_units",
                "claim": "0.1 nm converts to 1e-10 m.",
                "source": "Position uncertainty $\\Delta x = 0.1",
                "posterior": 0.91,
                "strength": 0.86,
                "factor_weight": 0.69,
            },
            {
                "key": "numeric_scale",
                "claim": "Substitution gives Delta E around 1.055e-16 J, matching option A.",
                "source": "1.055 \\times 10^{-16}",
                "posterior": 0.96,
                "strength": 0.94,
                "factor_weight": 0.92,
            },
        ],
    },
    "gpqa_diamond-150": {
        "domain": "physics.quantum_spin",
        "decision_label": "D: Ay eigenfunction can share A^2, not Az",
        "confidence": 0.94,
        "status": "auto_executed",
        "posterior_delta": 0.37,
        "beliefs": [
            {
                "key": "pauli_y",
                "claim": "The provided S matrix is the Pauli sigma_y matrix.",
                "source": "specifically $\\sigma_y$",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.82,
            },
            {
                "key": "hermitian_real",
                "claim": "Hermitian angular momentum operators have real eigenvalues.",
                "source": "eigenvalues of $A_y$ must be **real numbers**",
                "posterior": 0.95,
                "strength": 0.93,
                "factor_weight": 0.88,
            },
            {
                "key": "eigenvalues",
                "claim": "The eigenvalues of Ay are plus or minus h/(4 pi).",
                "source": "The eigenvalues of $A_y$ are:",
                "posterior": 0.93,
                "strength": 0.89,
                "factor_weight": 0.80,
            },
            {
                "key": "not_standard_basis",
                "claim": "The standard z-basis vectors are not eigenvectors of Ay.",
                "source": "standard basis vectors are *not* the eigenvectors of $A_y$",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.73,
            },
            {
                "key": "commutation",
                "claim": "Ay commutes with A^2 but not with Az, so option D is consistent.",
                "source": "Different components of angular momentum do not commute",
                "posterior": 0.94,
                "strength": 0.92,
                "factor_weight": 0.89,
            },
        ],
    },
    "gpqa_diamond-23": {
        "domain": "chemistry.spectroscopy",
        "decision_label": "B: C6H10O2",
        "confidence": 0.93,
        "status": "auto_executed",
        "posterior_delta": 0.35,
        "beliefs": [
            {
                "key": "broad_oh",
                "claim": "A very broad 3000 cm-1 absorption supports a carboxylic acid O-H stretch.",
                "source": "Very broad absorption peak at 3000 wavenumbers",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.79,
            },
            {
                "key": "carbonyl",
                "claim": "The 1700 cm-1 strong absorption supports a carbonyl group.",
                "source": "Strong absorption peak at 1700 wavenumbers",
                "posterior": 0.88,
                "strength": 0.84,
                "factor_weight": 0.72,
            },
            {
                "key": "alkene",
                "claim": "The 1650 cm-1 absorption and vinyl hydrogens support an alkene.",
                "source": "Strong absorption peak at 1650 wavenumbers",
                "posterior": 0.87,
                "strength": 0.83,
                "factor_weight": 0.72,
            },
            {
                "key": "ms_45",
                "claim": "The m/z 45 fragment is diagnostic for a carboxylic acid fragment.",
                "source": "Fragment peak at m/z = 45",
                "posterior": 0.91,
                "strength": 0.88,
                "factor_weight": 0.81,
            },
            {
                "key": "formula_dbe",
                "claim": "C6H10O2 has the required two oxygens and two degrees of unsaturation.",
                "source": "Option B: $C_6H_{10}O_2$",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.87,
            },
        ],
    },
    "gpqa_diamond-186": {
        "domain": "physics.quantum_spin",
        "decision_label": "A: -12*hbar/25",
        "confidence": 0.96,
        "status": "auto_executed",
        "posterior_delta": 0.40,
        "beliefs": [
            {
                "key": "state_operator",
                "claim": "The state vector and Sy operator are identified correctly.",
                "source": "spin state is given as",
                "posterior": 0.93,
                "strength": 0.89,
                "factor_weight": 0.81,
            },
            {
                "key": "normalization",
                "claim": "The state norm is 25, so the expectation value must be normalized by 25.",
                "source": "|3i|^2 + |4|^2 = 9 + 16 = 25",
                "posterior": 0.96,
                "strength": 0.94,
                "factor_weight": 0.91,
            },
            {
                "key": "sigma_action",
                "claim": "Applying sigma_y to the spinor gives the vector (-4i, -3).",
                "source": "\\begin{pmatrix} -4i \\\\ -3 \\end{pmatrix}",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.79,
            },
            {
                "key": "inner_product",
                "claim": "The numerator of the expectation value is -12 hbar.",
                "source": "= -12\\hbar",
                "posterior": 0.94,
                "strength": 0.90,
                "factor_weight": 0.84,
            },
            {
                "key": "final_expectation",
                "claim": "Dividing by the norm gives -12 hbar / 25, matching option A.",
                "source": "\\langle S_y \\rangle = \\frac{-12\\hbar}{25}",
                "posterior": 0.97,
                "strength": 0.95,
                "factor_weight": 0.92,
            },
        ],
    },
    "gpqa_diamond-154": {
        "domain": "physics.black_hole_entropy",
        "decision_label": "A: 10^62 J/K",
        "confidence": 0.94,
        "status": "auto_executed",
        "posterior_delta": 0.36,
        "beliefs": [
            {
                "key": "distance_si",
                "claim": "The 10^10 parsec distance converts to about 3.1e26 m.",
                "source": "Distance $d = 10^{10}",
                "posterior": 0.91,
                "strength": 0.87,
                "factor_weight": 0.74,
            },
            {
                "key": "angle_rad",
                "claim": "The angular size converts to about 1.75e-19 radians.",
                "source": "$\\theta_{\\text{rad}} = 10^{-17}",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.73,
            },
            {
                "key": "small_angle",
                "claim": "Small-angle geometry gives an event-horizon diameter around 5.4e7 m.",
                "source": "small angle approximation $D = d \\cdot \\theta$",
                "posterior": 0.89,
                "strength": 0.84,
                "factor_weight": 0.74,
            },
            {
                "key": "bh_entropy_formula",
                "claim": "Bekenstein-Hawking entropy scales as k_B A/(4 l_P^2).",
                "source": "Bekenstein-Hawking entropy",
                "posterior": 0.94,
                "strength": 0.92,
                "factor_weight": 0.87,
            },
            {
                "key": "entropy_scale",
                "claim": "The computed entropy is order 10^62 J/K.",
                "source": "The order of magnitude is $10^{62}$",
                "posterior": 0.96,
                "strength": 0.94,
                "factor_weight": 0.91,
            },
        ],
    },
    "gpqa_diamond-165": {
        "domain": "physics.relativistic_mechanics",
        "decision_label": "B: relativistic vmax",
        "confidence": 0.96,
        "status": "auto_executed",
        "posterior_delta": 0.38,
        "beliefs": [
            {
                "key": "energy_conservation",
                "claim": "Maximum speed can be derived from conservation of total relativistic energy.",
                "source": "conservation of total energy",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.86,
            },
            {
                "key": "hooke_potential",
                "claim": "Hooke's law implies the potential energy U = kx^2/2.",
                "source": "potential energy function $U(x) = \\frac{1}{2}kx^2$",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.78,
            },
            {
                "key": "turning_energy",
                "claim": "At amplitude A the particle has gamma=1 and total energy mc^2 + kA^2/2.",
                "source": "E_A = (1)mc^2 + \\frac{1}{2}kA^2",
                "posterior": 0.93,
                "strength": 0.89,
                "factor_weight": 0.81,
            },
            {
                "key": "gamma_max",
                "claim": "At equilibrium the maximum-speed gamma is 1 + kA^2/(2mc^2).",
                "source": "\\gamma_{max} = 1 + \\frac{kA^2}{2mc^2}",
                "posterior": 0.94,
                "strength": 0.90,
                "factor_weight": 0.84,
            },
            {
                "key": "vmax_expression",
                "claim": "Solving for velocity gives the expression in option B.",
                "source": "v_{max} = c \\sqrt{1 - \\frac{1}{\\left(1 + \\frac{kA^2}{2mc^2}\\right)^2}}",
                "posterior": 0.97,
                "strength": 0.95,
                "factor_weight": 0.93,
            },
        ],
    },
    "gpqa_diamond-197": {
        "domain": "mathematics.differential_geometry",
        "decision_label": "C: +infinity",
        "confidence": 0.93,
        "status": "auto_executed",
        "posterior_delta": 0.34,
        "beliefs": [
            {
                "key": "conformal_metric",
                "claim": "The metric is conformal to the Euclidean disk metric.",
                "source": "conformal factor",
                "posterior": 0.89,
                "strength": 0.84,
                "factor_weight": 0.72,
            },
            {
                "key": "domain_boundary",
                "claim": "The denominator is positive only for r < 2.",
                "source": "denominator must be positive",
                "posterior": 0.91,
                "strength": 0.87,
                "factor_weight": 0.76,
            },
            {
                "key": "area_element",
                "claim": "The metric determinant gives area element 32/(4-r^2) dx dy.",
                "source": "\\sqrt{\\det(g)} = \\frac{32}{4-r^2}",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.80,
            },
            {
                "key": "log_divergence",
                "claim": "The radial area integral diverges logarithmically at r=2.",
                "source": "\\lim_{u \\to 0^+} \\ln(u)",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.86,
            },
            {
                "key": "infinite_area",
                "claim": "The metric blow-up at the boundary makes the total area infinite.",
                "source": "the total area is infinite",
                "posterior": 0.96,
                "strength": 0.94,
                "factor_weight": 0.91,
            },
        ],
    },
    "gpqa_diamond-119": {
        "domain": "chemistry.organic_reagents",
        "decision_label": "B: A = H3O+, B = HCl",
        "confidence": 0.44,
        "status": "ask_human",
        "posterior_delta": -0.18,
        "beliefs": [
            {
                "key": "cyanohydrin_product",
                "claim": "The first reaction product is a cyanohydrin.",
                "source": "2-hydroxy-2-methylbutanenitrile is a cyanohydrin",
                "posterior": 0.87,
                "strength": 0.83,
                "factor_weight": 0.71,
            },
            {
                "key": "acid_for_a",
                "claim": "The model infers that reagent A should be a proton source.",
                "source": "reagent A must be an acid",
                "posterior": 0.76,
                "strength": 0.72,
                "factor_weight": 0.63,
            },
            {
                "key": "h3o_choice",
                "claim": "The model chooses H3O+ as the proton source for reagent A.",
                "source": "A is most likely $H_3O^+$",
                "posterior": 0.70,
                "strength": 0.66,
                "factor_weight": 0.58,
            },
            {
                "key": "hcl_hydrolysis",
                "claim": "HCl is a strong acid suitable for nitrile hydrolysis to carboxylic acid.",
                "source": "HCl is the standard reagent",
                "posterior": 0.91,
                "strength": 0.88,
                "factor_weight": 0.82,
            },
            {
                "key": "ground_truth_conflict",
                "claim": "The listed correct option C conflicts with the model's chosen H3O+/HCl pair.",
                "source": "A = NaHSO3, B = HCl",
                "posterior": 0.92,
                "strength": 0.90,
                "factor_weight": 0.95,
                "direction": -1,
                "factor_type": "contradict",
                "source_type": "answer_key",
            },
        ],
    },
    "gpqa_diamond-82": {
        "domain": "chemistry.hydrocarbon_disproportionation",
        "decision_label": "C: 18",
        "confidence": 0.92,
        "status": "auto_executed",
        "posterior_delta": 0.33,
        "beliefs": [
            {
                "key": "hydrogen_fraction",
                "claim": "The 14.28 percent hydrogen mass fraction implies CnH2n.",
                "source": "mass fraction of hydrogen in Z is $14.28\\%$",
                "posterior": 0.88,
                "strength": 0.83,
                "factor_weight": 0.72,
            },
            {
                "key": "cyclohexane_z",
                "claim": "The saturated solvent Z is identified as cyclohexane.",
                "source": "Cyclohexane",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.77,
            },
            {
                "key": "mixture_y",
                "claim": "Mixture Y consists of benzene and cyclohexane.",
                "source": "Mixture Y consists of **Benzene** and **Cyclohexane**",
                "posterior": 0.88,
                "strength": 0.84,
                "factor_weight": 0.73,
            },
            {
                "key": "stoich",
                "claim": "Disproportionation is balanced by C6H8 + C6H10 -> C6H6 + C6H12.",
                "source": "C_6H_8 + C_6H_{10} \\rightarrow C_6H_6 + C_6H_{12}",
                "posterior": 0.93,
                "strength": 0.90,
                "factor_weight": 0.85,
            },
            {
                "key": "hydrogen_total",
                "claim": "Cyclohexene and 1,4-cyclohexadiene contain 10 + 8 = 18 hydrogens.",
                "source": "Total = $10 + 8 = 18$",
                "posterior": 0.95,
                "strength": 0.92,
                "factor_weight": 0.89,
            },
        ],
    },
    "gpqa_diamond-92": {
        "domain": "chemistry.chromatography",
        "decision_label": "B: similar polarities",
        "confidence": 0.94,
        "status": "auto_executed",
        "posterior_delta": 0.36,
        "beliefs": [
            {
                "key": "lab_context",
                "claim": "The phrase is interpreted in the context of purification and isolation after synthesis.",
                "source": "purification and isolation",
                "posterior": 0.86,
                "strength": 0.81,
                "factor_weight": 0.68,
            },
            {
                "key": "tlc_column",
                "claim": "TLC and column chromatography are the likely techniques behind the visual metaphor.",
                "source": "Thin Layer Chromatography (TLC)",
                "posterior": 0.88,
                "strength": 0.84,
                "factor_weight": 0.72,
            },
            {
                "key": "coelution",
                "claim": "Being on top of each other maps to overlapping chromatographic spots or peaks.",
                "source": "co-elution",
                "posterior": 0.92,
                "strength": 0.89,
                "factor_weight": 0.82,
            },
            {
                "key": "polarity_cause",
                "claim": "Similar polarity makes compounds move at similar rates through the stationary phase.",
                "source": "similar polarities",
                "posterior": 0.95,
                "strength": 0.92,
                "factor_weight": 0.89,
            },
            {
                "key": "distillation_reject",
                "claim": "Similar boiling points are less likely because the phrase is visual and chromatography-focused.",
                "source": "Similar boiling points would prevent separation via **distillation**",
                "posterior": 0.76,
                "strength": 0.70,
                "factor_weight": 0.51,
            },
        ],
    },
    "gpqa_diamond-59": {
        "domain": "biology.sars_cov_2",
        "decision_label": "C: incorrect statement",
        "confidence": 0.91,
        "status": "auto_executed",
        "posterior_delta": 0.31,
        "beliefs": [
            {
                "key": "answer_c_false",
                "claim": "The model identifies statement C as the exception among mostly correct statements.",
                "source": "incorrect statement is **C**",
                "posterior": 0.89,
                "strength": 0.84,
                "factor_weight": 0.73,
            },
            {
                "key": "prf_valid",
                "claim": "The -1 programmed ribosomal frameshifting description is treated as correct background.",
                "source": "-1 Programmed Ribosomal Frameshifting",
                "posterior": 0.84,
                "strength": 0.79,
                "factor_weight": 0.61,
            },
            {
                "key": "mismatch_repair_bad",
                "claim": "Calling nsp14 proofreading a mismatch repair mechanism is technically inaccurate.",
                "source": "Mismatch repair mechanism",
                "posterior": 0.90,
                "strength": 0.87,
                "factor_weight": 0.81,
            },
            {
                "key": "exonuclease_breakdown",
                "claim": "nsp14 ExoN is an exoribonuclease that breaks terminal phosphodiester bonds.",
                "source": "exoribonuclease",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.86,
            },
            {
                "key": "dsrna_contradiction",
                "claim": "Preventing dsRNA breakdown contradicts the described enzymatic function.",
                "source": "Prevents the breakdown of dsRNA",
                "posterior": 0.93,
                "strength": 0.90,
                "factor_weight": 0.84,
            },
        ],
    },
    "gpqa_diamond-172": {
        "domain": "physics.atomic_selection_rules",
        "decision_label": "A: route via |2,1,0>, probability 1/3",
        "confidence": 0.92,
        "status": "auto_executed",
        "posterior_delta": 0.32,
        "beliefs": [
            {
                "key": "delta_l",
                "claim": "Electric dipole transitions require Delta l = plus or minus one.",
                "source": "\\Delta l = \\pm 1",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.79,
            },
            {
                "key": "delta_m",
                "claim": "The magnetic quantum number selection rule permits m changes of 0 or plus/minus one.",
                "source": "\\Delta m = 0, \\pm 1",
                "posterior": 0.88,
                "strength": 0.83,
                "factor_weight": 0.68,
            },
            {
                "key": "intermediate_2p",
                "claim": "The only intermediate shell for the two-step path is the 2p shell.",
                "source": "the transition route passes through the $2p$ shell",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.75,
            },
            {
                "key": "three_sublevels",
                "claim": "The l=1 intermediate contains three degenerate m sublevels.",
                "source": "$2l+1 = 3$",
                "posterior": 0.91,
                "strength": 0.87,
                "factor_weight": 0.77,
            },
            {
                "key": "one_third_route",
                "claim": "Equal branching among three m sublevels makes the route through m=0 have probability 1/3.",
                "source": "probability of taking the specific route passing through $|2,1,0\\rangle$ is $1/3$",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.86,
            },
        ],
    },
    "gpqa_diamond-137": {
        "domain": "chemistry.michael_addition",
        "decision_label": "C: selected, conflicts with answer key A",
        "confidence": 0.42,
        "status": "ask_human",
        "posterior_delta": -0.20,
        "beliefs": [
            {
                "key": "product_a",
                "claim": "The first Michael addition gives trimethyl 2-(p-tolyl)propane-1,1,3-tricarboxylate.",
                "source": "trimethyl 2-(p-tolyl)propane-1,1,3-tricarboxylate",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.73,
            },
            {
                "key": "stork_enamine",
                "claim": "The second reaction is interpreted as a Stork enamine synthesis followed by hydrolysis.",
                "source": "Stork Enamine Synthesis",
                "posterior": 0.86,
                "strength": 0.81,
                "factor_weight": 0.66,
            },
            {
                "key": "product_b",
                "claim": "The second product is identified as 3-(2-oxocyclohexyl)butanenitrile.",
                "source": "3-(2-oxocyclohexyl)butanenitrile",
                "posterior": 0.88,
                "strength": 0.84,
                "factor_weight": 0.70,
            },
            {
                "key": "reactant_c",
                "claim": "The model explicitly derives cyclohexane-1,3-dione as reactant C.",
                "source": "cyclohexane-1,3-dione",
                "posterior": 0.92,
                "strength": 0.89,
                "factor_weight": 0.93,
                "direction": -1,
                "factor_type": "contradict",
            },
            {
                "key": "wrong_selection",
                "claim": "Despite deriving reactant C as cyclohexane-1,3-dione, the model selects option C.",
                "source": "Answer: **C**",
                "posterior": 0.74,
                "strength": 0.70,
                "factor_weight": 0.58,
            },
        ],
    },
    "gpqa_diamond-58": {
        "domain": "chemistry.thermal_decomposition",
        "decision_label": "C: 17 atoms",
        "confidence": 0.95,
        "status": "auto_executed",
        "posterior_delta": 0.37,
        "beliefs": [
            {
                "key": "water_amount",
                "claim": "The 3.60 g mass gain in the drying tube corresponds to 0.20 mol water.",
                "source": "increases by 3.60 g",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.74,
            },
            {
                "key": "oxygen_amount",
                "claim": "The copper tube mass gain corresponds to 0.025 mol oxygen gas.",
                "source": "n(\\text{O}_2) = \\frac{0.05 \\text{ mol}}{2} = 0.025",
                "posterior": 0.91,
                "strength": 0.87,
                "factor_weight": 0.76,
            },
            {
                "key": "gas_c_nitrogen",
                "claim": "The remaining 2.24 L gas has molar mass 28 g/mol and is nitrogen.",
                "source": "A common gas with $M=28$ is nitrogen",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.79,
            },
            {
                "key": "salt_pair",
                "claim": "The gas ratios and mass balance identify ammonium nitrite and ammonium nitrate.",
                "source": "NH}_4\\text{NO}_2$ and B is $\\text{NH}_4\\text{NO}_3",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.86,
            },
            {
                "key": "atom_count",
                "claim": "The total atom count is 8 + 9 = 17.",
                "source": "Total number of atoms = $8 + 9 = 17$",
                "posterior": 0.97,
                "strength": 0.95,
                "factor_weight": 0.92,
            },
        ],
    },
    "gpqa_diamond-116": {
        "domain": "chemistry.aromatic_synthesis",
        "decision_label": "B: selected, conflicts with answer key A",
        "confidence": 0.39,
        "status": "ask_human",
        "posterior_delta": -0.24,
        "beliefs": [
            {
                "key": "target_pattern",
                "claim": "The target has acetyl, bromo, and nitro groups in a 1,3,5 pattern.",
                "source": "positions 1, 3, and 5 respectively",
                "posterior": 0.84,
                "strength": 0.79,
                "factor_weight": 0.62,
            },
            {
                "key": "acylation_first",
                "claim": "The model treats Friedel-Crafts acylation first as a robust route to acetophenone.",
                "source": "Friedel-Crafts Acylation of benzene yields **acetophenone**",
                "posterior": 0.82,
                "strength": 0.77,
                "factor_weight": 0.60,
            },
            {
                "key": "meta_director",
                "claim": "The acetyl group is used as a meta director for bromination.",
                "source": "acetyl group is a **meta-director**",
                "posterior": 0.80,
                "strength": 0.75,
                "factor_weight": 0.57,
            },
            {
                "key": "option_b_strategy",
                "claim": "The model claims option B is the more chemically robust placeholder strategy.",
                "source": "Option B provides a chemically robust strategy",
                "posterior": 0.73,
                "strength": 0.68,
                "factor_weight": 0.57,
            },
            {
                "key": "option_a_counterevidence",
                "claim": "The answer key route A is partially supported by the model's own note about amine directing.",
                "source": "Option A places the substituents in a logical order using the amine group as a directing block",
                "posterior": 0.86,
                "strength": 0.82,
                "factor_weight": 0.92,
                "direction": -1,
                "factor_type": "contradict",
                "source_type": "answer_key",
            },
        ],
    },
    "gpqa_diamond-138": {
        "domain": "chemistry.spectroscopy",
        "decision_label": "C: 4-chlorobenzoic acid",
        "confidence": 0.95,
        "status": "auto_executed",
        "posterior_delta": 0.37,
        "beliefs": [
            {
                "key": "chlorine_isotope",
                "claim": "The M and M+2 isotope pattern supports one chlorine atom.",
                "source": "diagnostic isotope pattern for Chlorine",
                "posterior": 0.91,
                "strength": 0.87,
                "factor_weight": 0.76,
            },
            {
                "key": "carboxy_ir",
                "claim": "The broad 3500-2700 cm-1 band and 1720 cm-1 carbonyl indicate carboxylic acid.",
                "source": "carboxylic acid functional group",
                "posterior": 0.93,
                "strength": 0.90,
                "factor_weight": 0.83,
            },
            {
                "key": "acidic_proton",
                "claim": "The 11 ppm singlet is diagnostic of a carboxylic acid proton.",
                "source": "diagnostic for the acidic proton",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.73,
            },
            {
                "key": "para_pattern",
                "claim": "Two aromatic doublets integrating to two protons each imply para disubstitution.",
                "source": "para-disubstituted benzene ring",
                "posterior": 0.95,
                "strength": 0.92,
                "factor_weight": 0.88,
            },
            {
                "key": "final_structure",
                "claim": "The combined spectra identify 4-chlorobenzoic acid.",
                "source": "4-chlorobenzoic acid",
                "posterior": 0.96,
                "strength": 0.94,
                "factor_weight": 0.91,
            },
        ],
    },
    "gpqa_diamond-179": {
        "domain": "physics.quantum_spin",
        "decision_label": "A: 0.64, 0.36 and hbar/7",
        "confidence": 0.94,
        "status": "auto_executed",
        "posterior_delta": 0.35,
        "beliefs": [
            {
                "key": "normalization",
                "claim": "The unnormalized coefficient norm is 7.",
                "source": "Total sum = $2 + 5 = 7$",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.78,
            },
            {
                "key": "operator_sx",
                "claim": "The off-diagonal hbar/2 operator is Sx.",
                "source": "This matrix corresponds to the spin operator $S_x$",
                "posterior": 0.91,
                "strength": 0.86,
                "factor_weight": 0.75,
            },
            {
                "key": "eigenvalues",
                "claim": "The operator eigenvalues are plus and minus hbar/2.",
                "source": "eigenvalues $\\lambda_1 = +\\hbar/2$ and $\\lambda_2 = -\\hbar/2$",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.74,
            },
            {
                "key": "prob_plus",
                "claim": "Projection onto the plus eigenstate gives probability 9/14, about 0.64.",
                "source": "Probability $P_1 = \\left| \\frac{3}{\\sqrt{14}} \\right|^2 = \\frac{9}{14}",
                "posterior": 0.94,
                "strength": 0.91,
                "factor_weight": 0.85,
            },
            {
                "key": "expectation",
                "claim": "The expectation value is hbar/7.",
                "source": "Both methods yield $\\hbar/7$",
                "posterior": 0.95,
                "strength": 0.92,
                "factor_weight": 0.88,
            },
        ],
    },
    "gpqa_diamond-88": {
        "domain": "chemistry.diels_alder_stereochemistry",
        "decision_label": "B: selected, conflicts with answer key C",
        "confidence": 0.41,
        "status": "ask_human",
        "posterior_delta": -0.21,
        "beliefs": [
            {
                "key": "reaction_class",
                "claim": "The model identifies the transformation as a Diels-Alder [4+2] cycloaddition.",
                "source": "[4+2] cycloaddition",
                "posterior": 0.88,
                "strength": 0.84,
                "factor_weight": 0.70,
            },
            {
                "key": "dienophile",
                "claim": "Furan-2,5-dione is interpreted as maleic anhydride.",
                "source": "Maleic Anhydride",
                "posterior": 0.87,
                "strength": 0.82,
                "factor_weight": 0.66,
            },
            {
                "key": "epithio_skeleton",
                "claim": "The sulfur bridge supports an epithioisobenzofuran skeleton rather than an epoxy skeleton.",
                "source": "epithioisobenzofuran-1,3-dione",
                "posterior": 0.90,
                "strength": 0.86,
                "factor_weight": 0.76,
            },
            {
                "key": "exo_assignment",
                "claim": "The model assigns the Option B stereochemistry to the exo isomer.",
                "source": "Option B) corresponds to the **exo** isomer",
                "posterior": 0.73,
                "strength": 0.68,
                "factor_weight": 0.58,
            },
            {
                "key": "answer_key_conflict",
                "claim": "The answer key selects option C, conflicting with the model's stereochemical assignment.",
                "source": "Option C",
                "posterior": 0.90,
                "strength": 0.87,
                "factor_weight": 0.94,
                "direction": -1,
                "factor_type": "contradict",
                "source_type": "answer_key",
            },
        ],
    },
    "gpqa_diamond-6": {
        "domain": "astronomy.observability",
        "decision_label": "C: Star3 and Star5",
        "confidence": 0.90,
        "status": "auto_executed",
        "posterior_delta": 0.30,
        "beliefs": [
            {
                "key": "strict_limit",
                "claim": "Detection by both instruments requires satisfying the stricter HIRES limit.",
                "source": "$m_V \\le 16$",
                "posterior": 0.88,
                "strength": 0.83,
                "factor_weight": 0.71,
            },
            {
                "key": "declination_window",
                "claim": "The combined latitude constraints define a shared observable declination window.",
                "source": "Stars with $\\delta < -71^\\circ$ are not visible",
                "posterior": 0.87,
                "strength": 0.82,
                "factor_weight": 0.68,
            },
            {
                "key": "star3_valid",
                "claim": "Star 3 passes both observability and brightness criteria under the observed-magnitude interpretation.",
                "source": "**Star 3:**",
                "posterior": 0.84,
                "strength": 0.79,
                "factor_weight": 0.64,
            },
            {
                "key": "star5_valid",
                "claim": "Star 5 is visible from both observatories and has apparent magnitude around 15.",
                "source": "**Star 5:**",
                "posterior": 0.89,
                "strength": 0.85,
                "factor_weight": 0.74,
            },
            {
                "key": "final_pair",
                "claim": "The final valid pair is Star3 and Star5.",
                "source": "Star3 and Star5",
                "posterior": 0.92,
                "strength": 0.88,
                "factor_weight": 0.81,
            },
        ],
    },
}


def _manual_gpqa_belief_memory_for_sample(
    record: dict[str, Any],
    sample: dict[str, Any],
    sample_index: int,
    annotation: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(record.get("task_id") or record.get("problem_id") or "task")
    sample_id = str(sample.get("trajectory_id") or sample.get("sample_index") or sample_index)
    suffix = _stable_suffix("manual", task_id, sample_id)
    answer = str(sample.get("extracted_answer") or "?")
    observed_at = str(record.get("completed_iso") or _iso(record.get("completed_at")) or "")
    step_count = max(1, int(sample.get("num_steps") or 1))
    confidence = float(annotation.get("confidence") or (0.94 if sample.get("is_correct") else 0.41))
    status = str(annotation.get("status") or ("auto_executed" if sample.get("is_correct") else "ask_human"))
    domain = str(annotation.get("domain") or "gpqa")
    beliefs = [b for b in annotation.get("beliefs", []) if isinstance(b, dict)]

    decision_id = f"decision_{suffix}"
    output_bv_id = f"bv_answer_{suffix}"
    factor_ids: dict[str, str] = {}
    nodes: list[dict[str, Any]] = [
        {
            "id": output_bv_id,
            "type": "BeliefVariable",
            "label": annotation.get("decision_label") or f"answer {answer}",
            "belief_kind": "effective",
            "prior": 0.5,
            "posterior": confidence,
            "status": "active" if confidence >= 0.6 else "conflicted",
            "valid_from": observed_at,
            "valid_to": "",
            "x": 505,
            "y": 210,
        },
        {
            "id": decision_id,
            "type": "Decision",
            "label": annotation.get("decision_label") or f"answer {answer}",
            "action_name": f"select_option_{answer}",
            "justification": confidence,
            "posterior": confidence,
            "status": status,
            "x": 600,
            "y": 210,
        },
    ]
    edges: list[dict[str, Any]] = [
        {
            "source": output_bv_id,
            "target": decision_id,
            "type": "REQUIRED_BY",
            "direction": 1 if confidence >= 0.6 else -1,
            "weight": round(confidence, 2),
        }
    ]

    def factor_id_for(factor_type: str) -> str:
        if factor_type in factor_ids:
            return factor_ids[factor_type]
        fid = f"factor_{factor_type}_{suffix}"
        factor_ids[factor_type] = fid
        y = 165 if factor_type == "support" else 265
        nodes.append(
            {
                "id": fid,
                "type": "Factor",
                "label": f"{factor_type} aggregation",
                "factor_type": factor_type,
                "weight": 0.90 if factor_type == "support" else 0.95,
                "confidence": confidence if factor_type == "support" else 0.92,
                "domain": domain,
                "activation_condition_threshold": 0.60,
                "posterior": confidence if factor_type == "support" else 0.92,
                "status": "active",
                "x": 405,
                "y": y,
            }
        )
        edges.append(
            {
                "source": fid,
                "target": output_bv_id,
                "type": "OUTPUT_TO",
                "direction": 1 if factor_type != "contradict" else -1,
                "weight": 0.90 if factor_type != "contradict" else 0.95,
            }
        )
        return fid

    step_spans: list[dict[str, Any]] = []
    for offset, belief in enumerate(beliefs):
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(belief.get("key") or offset)).strip("_")
        direction = int(belief.get("direction", 1) or 1)
        factor_type = str(belief.get("factor_type") or ("support" if direction >= 0 else "contradict"))
        factor_id = factor_id_for(factor_type)
        node_suffix = f"{suffix}_{key}"
        evidence_id = f"evidence_{node_suffix}"
        claim_id = f"claim_{node_suffix}"
        belief_id = f"bv_{node_suffix}"
        posterior = float(belief.get("posterior", 0.75))
        strength = float(belief.get("strength", max(0.55, posterior - 0.04)))
        weight = float(belief.get("factor_weight", max(0.4, posterior - 0.10)))
        source = str(belief.get("source") or belief.get("claim") or key)
        claim = str(belief.get("claim") or source)
        y = 52 + offset * 62
        nodes.extend(
            [
                {
                    "id": evidence_id,
                    "type": "Evidence",
                    "label": source,
                    "source_type": belief.get("source_type") or "model_reasoning",
                    "source_reliability": float(belief.get("source_reliability", 0.86)),
                    "linguistic_certainty": float(belief.get("linguistic_certainty", 0.88)),
                    "extractor_confidence": float(belief.get("extractor_confidence", 0.90)),
                    "strength": strength,
                    "observed_at": observed_at,
                    "posterior": strength,
                    "status": "observed",
                    "x": 55,
                    "y": y,
                },
                {
                    "id": claim_id,
                    "type": "Claim",
                    "label": claim,
                    "natural_language_description": claim,
                    "valid_from": observed_at,
                    "valid_to": "",
                    "posterior": posterior,
                    "status": "active",
                    "x": 170,
                    "y": y,
                },
                {
                    "id": belief_id,
                    "type": "BeliefVariable",
                    "label": claim,
                    "belief_kind": belief.get("belief_kind") or "truth",
                    "prior": float(belief.get("prior", 0.50)),
                    "posterior": posterior,
                    "status": "active" if direction >= 0 else "conflict",
                    "valid_from": observed_at,
                    "valid_to": "",
                    "half_life": belief.get("half_life") or "P30D",
                    "x": 290,
                    "y": y,
                },
            ]
        )
        edges.extend(
            [
                {
                    "source": evidence_id,
                    "target": belief_id,
                    "type": "EVALUATED_BY",
                    "direction": direction,
                    "weight": round(strength, 2),
                },
                {
                    "source": claim_id,
                    "target": belief_id,
                    "type": "HAS_BELIEF",
                    "direction": 1,
                    "weight": 1.0,
                },
                {
                    "source": belief_id,
                    "target": claim_id,
                    "type": "OWNED_BY",
                    "direction": 1,
                    "weight": 1.0,
                },
                {
                    "source": belief_id,
                    "target": factor_id,
                    "type": "INPUT_TO",
                    "direction": direction,
                    "weight": round(weight, 2),
                },
            ]
        )
        step_spans.append(
            {
                "text": source,
                "kind": "support" if direction >= 0 else "contradict",
                "node_id": belief_id,
                "confidence": round(posterior, 2),
            }
        )

    steps = [
        {
            "step_index": step_index,
            "state": "propagated" if confidence >= 0.6 else "conflicted",
            "source_spans": step_spans,
            "posterior_delta": annotation.get("posterior_delta", round(confidence - 0.5, 3)),
        }
        for step_index in range(step_count)
    ]
    return _normalize_belief_memory_graph(
        {
        "schema_version": 1,
        "mode": "mock:manual-first10",
        "memgraph_lab_url": "http://172.25.10.2:12345/lab",
        "task_id": task_id,
        "sample_id": sample_id,
        "title": f"Belief Memory - {task_id}",
        "description": "Manual GPQA Diamond belief annotation shaped for Memgraph-backed graph memory.",
        "nodes": nodes,
        "edges": edges,
        "steps": steps,
        "summary": {
            "claims": sum(1 for n in nodes if n.get("type") == "Claim"),
            "belief_variables": sum(1 for n in nodes if n.get("type") == "BeliefVariable"),
            "evidence": sum(1 for n in nodes if n.get("type") == "Evidence"),
            "factors": sum(1 for n in nodes if n.get("type") == "Factor"),
            "decision": annotation.get("decision_label") or answer,
            "confidence": confidence,
        },
        }
    )


def _mock_belief_memory_for_sample(
    record: dict[str, Any],
    sample: dict[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    task_id = str(record.get("task_id") or record.get("problem_id") or "task")
    sample_id = str(sample.get("trajectory_id") or sample.get("sample_index") or sample_index)
    manual_annotation = _GPQA_DIAMOND_FIRST10_BELIEF_ANNOTATIONS.get(task_id)
    if manual_annotation:
        return _manual_gpqa_belief_memory_for_sample(record, sample, sample_index, manual_annotation)
    suffix = _stable_suffix(task_id, sample_id)
    question = str(record.get("question") or _sample_message(sample, "user").get("content") or "")
    answer = str(sample.get("extracted_answer") or "?")
    model_steps = sample.get("model_steps") if isinstance(sample.get("model_steps"), list) else []
    step_count = max(1, int(sample.get("num_steps") or len(model_steps) or 1))
    first_step_text = _assistant_text_for_step(sample, 0)
    phrases = _candidate_source_phrases(question, first_step_text)
    if not phrases:
        phrases = ["problem statement", "model reasoning", f"answer {answer}"]

    nodes: list[dict[str, Any]] = [
        {
            "id": f"decision_{suffix}",
            "type": "Decision",
            "label": f"answer {answer}",
            "posterior": 0.94 if sample.get("is_correct") else 0.41,
            "status": "auto_executed" if sample.get("is_correct") else "ask_human",
            "x": 545,
            "y": 205,
        },
        {
            "id": f"factor_{suffix}",
            "type": "Factor",
            "label": "support aggregation",
            "posterior": 0.86 if sample.get("is_correct") else 0.52,
            "status": "active",
            "x": 395,
            "y": 205,
        },
    ]
    edges: list[dict[str, Any]] = [
        {
            "source": f"factor_{suffix}",
            "target": f"decision_{suffix}",
            "type": "OUTPUT_TO",
            "direction": 1,
            "weight": 0.91,
        }
    ]
    step_payloads: list[dict[str, Any]] = []
    for step_index in range(step_count):
        assistant_text = _assistant_text_for_step(sample, step_index)
        step_spans: list[dict[str, Any]] = []
        for offset, phrase in enumerate(phrases[:5]):
            node_suffix = _stable_suffix(suffix, phrase)
            claim_id = f"claim_{node_suffix}"
            belief_id = f"bv_{node_suffix}"
            evidence_id = f"evidence_{node_suffix}"
            y = 70 + offset * 68
            posterior = max(0.55, min(0.98, 0.91 - offset * 0.07))
            if step_index == 0:
                nodes.extend(
                    [
                        {
                            "id": evidence_id,
                            "type": "Evidence",
                            "label": phrase,
                            "posterior": posterior,
                            "status": "observed",
                            "x": 65,
                            "y": y,
                        },
                        {
                            "id": claim_id,
                            "type": "Claim",
                            "label": phrase,
                            "posterior": posterior,
                            "status": "active",
                            "x": 185,
                            "y": y,
                        },
                        {
                            "id": belief_id,
                            "type": "BeliefVariable",
                            "label": phrase[:38],
                            "posterior": posterior,
                            "status": "active",
                            "x": 305,
                            "y": y,
                        },
                    ]
                )
                edges.extend(
                    [
                        {
                            "source": evidence_id,
                            "target": claim_id,
                            "type": "SUPPORTS",
                            "direction": 1,
                            "weight": round(0.78 - offset * 0.04, 2),
                        },
                        {
                            "source": claim_id,
                            "target": belief_id,
                            "type": "HAS_BELIEF",
                            "direction": 1,
                            "weight": 1.0,
                        },
                        {
                            "source": belief_id,
                            "target": f"factor_{suffix}",
                            "type": "INPUT_TO",
                            "direction": 1,
                            "weight": round(0.72 - offset * 0.05, 2),
                        },
                    ]
                )
            if _phrase_present(assistant_text, phrase) or _phrase_present(question, phrase):
                step_spans.append(
                    {
                        "text": phrase,
                        "kind": "support",
                        "node_id": belief_id,
                        "confidence": round(posterior, 2),
                    }
                )
        step_payloads.append(
            {
                "step_index": step_index,
                "state": "propagated",
                "source_spans": step_spans[:5],
                "posterior_delta": round(0.18 / (step_index + 1), 3),
            }
        )

    return _normalize_belief_memory_graph(
        {
        "schema_version": 1,
        "mode": "mock",
        "memgraph_lab_url": "http://172.25.10.2:12345/lab",
        "task_id": task_id,
        "sample_id": sample_id,
        "title": f"Belief Memory - {task_id}",
        "description": "Mock topology shaped after the Memgraph PRD; replace with live extraction when observe()/compile() is wired.",
        "nodes": nodes,
        "edges": edges,
        "steps": step_payloads,
        "summary": {
            "claims": sum(1 for n in nodes if n.get("type") == "Claim"),
            "belief_variables": sum(1 for n in nodes if n.get("type") == "BeliefVariable"),
            "evidence": sum(1 for n in nodes if n.get("type") == "Evidence"),
            "factors": 1,
            "decision": answer,
            "confidence": 0.94 if sample.get("is_correct") else 0.41,
        },
        }
    )


def _attach_mock_belief_memory(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    benchmark = str(
        (payload.get("summary") if isinstance(payload.get("summary"), dict) else {}).get("benchmark")
        or payload.get("benchmark")
        or ""
    ).lower()
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    attached = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if attached >= 10:
            break
        record_benchmark = str(record.get("data_source") or benchmark).lower()
        if benchmark and "gpqa" not in benchmark and "gpqa" not in record_benchmark:
            continue
        samples = record.get("samples") if isinstance(record.get("samples"), list) else []
        for idx, sample in enumerate(samples):
            if isinstance(sample, dict) and "belief_memory" not in sample:
                sample["belief_memory"] = _mock_belief_memory_for_sample(record, sample, idx)
                attached += 1
                break
    return payload


def _memgraph_uri() -> str:
    return os.environ.get("BELIEF_TRACER_MEMGRAPH_URI") or os.environ.get("MEMGRAPH_URI") or DEFAULT_MEMGRAPH_URI


def _memgraph_auth() -> Any:
    user = os.environ.get("BELIEF_TRACER_MEMGRAPH_USER") or os.environ.get("MEMGRAPH_USER")
    password = os.environ.get("BELIEF_TRACER_MEMGRAPH_PASSWORD") or os.environ.get("MEMGRAPH_PASSWORD")
    if not user:
        return None
    try:
        from neo4j import basic_auth
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError("neo4j Python driver is required for Memgraph auth") from exc
    return basic_auth(user, password or "")


def _memgraph_driver() -> Any:
    try:
        from neo4j import GraphDatabase
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError("neo4j Python driver is required to connect to Memgraph") from exc
    return GraphDatabase.driver(_memgraph_uri(), auth=_memgraph_auth())


def _memgraph_memory_key(memory_key: str | None, memory: dict[str, Any] | None = None) -> str:
    if memory_key:
        return str(memory_key)
    if memory:
        existing = memory.get("memory_key")
        if existing:
            return str(existing)
        task_id = memory.get("task_id") or "task"
        sample_id = memory.get("sample_id") or "sample"
        return f"{task_id}:{sample_id}"
    raise ValueError("missing memory_key")


def _memgraph_safe_label(value: Any) -> str:
    label = str(value or "")
    return label if label in MEMGRAPH_NODE_LABELS else "Claim"


def _memgraph_safe_edge_type(value: Any) -> str:
    edge_type = re.sub(r"[^A-Z0-9_]+", "_", str(value or "").upper()).strip("_")
    return edge_type if edge_type in MEMGRAPH_EDGE_TYPES else "SUPPORTS"


def _json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _memgraph_decode_payload(value: Any, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    return dict(fallback or {})


def _memgraph_upsert_belief_memory(memory_key: str, memory: dict[str, Any]) -> dict[str, Any]:
    memory = _normalize_belief_memory_graph(memory)
    nodes = [n for n in memory.get("nodes", []) if isinstance(n, dict) and n.get("id")]
    edges = [
        e
        for e in memory.get("edges", [])
        if isinstance(e, dict) and e.get("source") and e.get("target")
    ]
    if not nodes:
        raise ValueError("belief memory has no nodes to store")
    now = datetime.now().isoformat(timespec="seconds")
    graph_payload = dict(memory)
    graph_payload["memory_key"] = memory_key
    graph_payload["memgraph_synced_at"] = now
    graph_payload.pop("nodes", None)
    graph_payload.pop("edges", None)
    graph_props = {
        "memory_key": memory_key,
        "task_id": str(memory.get("task_id") or ""),
        "sample_id": str(memory.get("sample_id") or ""),
        "title": str(memory.get("title") or ""),
        "mode": str(memory.get("mode") or ""),
        "synced_at": now,
        "payload_json": _json_payload(graph_payload),
    }
    with _memgraph_driver() as driver:
        with driver.session() as session:
            session.run(
                "MATCH (n:BeliefMemoryNode {memory_key: $memory_key}) DETACH DELETE n",
                memory_key=memory_key,
            ).consume()
            session.run(
                "MATCH (g:MemoryGraph {memory_key: $memory_key}) DETACH DELETE g",
                memory_key=memory_key,
            ).consume()
            session.run("CREATE (g:MemoryGraph) SET g = $props", props=graph_props).consume()
            for idx, node in enumerate(nodes):
                label = _memgraph_safe_label(node.get("type"))
                payload = dict(node)
                props = {
                    "memory_key": memory_key,
                    "id": str(node.get("id")),
                    "node_id": str(node.get("id")),
                    "node_type": label,
                    "label": str(node.get("label") or node.get("id") or ""),
                    "posterior": float(node.get("posterior"))
                    if isinstance(node.get("posterior"), (int, float))
                    else None,
                    "status": str(node.get("status") or ""),
                    "x": float(node.get("x", 0) or 0),
                    "y": float(node.get("y", 0) or 0),
                    "order": idx,
                    "payload_json": _json_payload(payload),
                }
                props = {k: v for k, v in props.items() if v is not None}
                session.run(
                    f"CREATE (n:BeliefMemoryNode:{label}) SET n = $props",
                    props=props,
                ).consume()
            for idx, edge in enumerate(edges):
                edge_type = _memgraph_safe_edge_type(edge.get("type"))
                payload = dict(edge)
                props = {
                    "memory_key": memory_key,
                    "source": str(edge.get("source")),
                    "target": str(edge.get("target")),
                    "edge_type": edge_type,
                    "direction": int(edge.get("direction", 1) or 1),
                    "weight": float(edge.get("weight"))
                    if isinstance(edge.get("weight"), (int, float))
                    else None,
                    "order": idx,
                    "payload_json": _json_payload(payload),
                }
                props = {k: v for k, v in props.items() if v is not None}
                session.run(
                    f"""
                    MATCH (a:BeliefMemoryNode {{memory_key: $memory_key, id: $source}})
                    MATCH (b:BeliefMemoryNode {{memory_key: $memory_key, id: $target}})
                    CREATE (a)-[r:{edge_type}]->(b)
                    SET r = $props
                    """,
                    memory_key=memory_key,
                    source=str(edge.get("source")),
                    target=str(edge.get("target")),
                    props=props,
                ).consume()
    return _memgraph_fetch_belief_memory(memory_key)


def _memgraph_fetch_belief_memory(memory_key: str) -> dict[str, Any]:
    with _memgraph_driver() as driver:
        with driver.session() as session:
            graph_record = session.run(
                "MATCH (g:MemoryGraph {memory_key: $memory_key}) RETURN g.payload_json AS payload",
                memory_key=memory_key,
            ).single()
            if not graph_record:
                raise FileNotFoundError(f"no belief memory found in Memgraph for {memory_key}")
            memory = _memgraph_decode_payload(graph_record.get("payload"))
            node_records = session.run(
                """
                MATCH (n:BeliefMemoryNode {memory_key: $memory_key})
                RETURN n.payload_json AS payload, n.id AS id, n.node_type AS node_type,
                       n.label AS label, n.posterior AS posterior, n.status AS status,
                       n.x AS x, n.y AS y
                ORDER BY n.order
                """,
                memory_key=memory_key,
            )
            nodes: list[dict[str, Any]] = []
            for record in node_records:
                node = _memgraph_decode_payload(
                    record.get("payload"),
                    {
                        "id": record.get("id"),
                        "type": record.get("node_type"),
                        "label": record.get("label"),
                        "posterior": record.get("posterior"),
                        "status": record.get("status"),
                        "x": record.get("x"),
                        "y": record.get("y"),
                    },
                )
                nodes.append(node)
            edge_records = session.run(
                """
                MATCH (a:BeliefMemoryNode {memory_key: $memory_key})-[r]->(b:BeliefMemoryNode {memory_key: $memory_key})
                RETURN r.payload_json AS payload, a.id AS source, b.id AS target,
                       type(r) AS edge_type, r.direction AS direction, r.weight AS weight
                ORDER BY r.order
                """,
                memory_key=memory_key,
            )
            raw_edges: list[dict[str, Any]] = []
            for record in edge_records:
                edge = _memgraph_decode_payload(
                    record.get("payload"),
                    {
                        "source": record.get("source"),
                        "target": record.get("target"),
                        "type": record.get("edge_type"),
                        "direction": record.get("direction"),
                        "weight": record.get("weight"),
                    },
                )
                raw_edges.append(edge)
    memory["memory_key"] = memory_key
    memory["mode"] = "memgraph"
    memory["memgraph_uri"] = _memgraph_uri()
    memory["nodes"] = nodes
    memory["edges"] = raw_edges
    raw_edge_count = len(raw_edges)
    memory = _normalize_belief_memory_graph(memory)
    if len(memory.get("edges", [])) != raw_edge_count:
        memory["_deduped_edge_count"] = raw_edge_count - len(memory.get("edges", []))
    memory.setdefault("summary", {})
    if isinstance(memory["summary"], dict):
        memory["summary"].update(
            {
                "claims": sum(1 for n in nodes if n.get("type") == "Claim"),
                "belief_variables": sum(1 for n in nodes if n.get("type") == "BeliefVariable"),
                "evidence": sum(1 for n in nodes if n.get("type") == "Evidence"),
                "factors": sum(1 for n in nodes if n.get("type") == "Factor"),
            }
        )
    return memory


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <title>BeliefTracer</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657282;
      --line: #d7dde5;
      --accent: #176b87;
      --accent-2: #5f6f52;
      --bad: #a73932;
      --good: #247145;
      --warn: #9a6700;
      --code: #101820;
      --forest-green: #228b22;
      --brick-red: #b22222;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111418;
        --panel: #181d23;
        --ink: #e6edf3;
        --muted: #9aa7b5;
        --line: #2f3742;
        --code: #0d1117;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 14px;
      padding-right: 6px;
    }
    #clock {
      min-width: 150px;
      text-align: right;
    }
    button, select {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    main {
      display: grid;
      grid-template-columns: var(--sidebar-width, 360px) minmax(0, 1fr) var(--memory-width, 420px);
      min-height: calc(100vh - 56px);
    }
    main.sidebar-collapsed { grid-template-columns: 0 minmax(0, 1fr) var(--memory-width, 420px); }
    main.memory-collapsed { grid-template-columns: var(--sidebar-width, 360px) minmax(0, 1fr) 0; }
    main.sidebar-collapsed.memory-collapsed { grid-template-columns: 0 minmax(0, 1fr) 0; }
    main.memory-fullscreen { grid-template-columns: var(--sidebar-width, 360px) minmax(0, 1fr) 0; }
    aside {
      position: relative;
      border-right: 1px solid var(--line);
      background: var(--panel);
      overflow: visible;
      max-height: calc(100vh - 56px);
      min-width: 0;
    }
    main.sidebar-collapsed aside { border-right: 0; }
    .sidebar-inner {
      height: calc(100vh - 56px);
      overflow: auto;
      background: var(--panel);
    }
    main.sidebar-collapsed .sidebar-inner {
      width: 0;
      overflow: hidden;
      pointer-events: none;
    }
    .sidebar-edge-toggle {
      position: absolute;
      top: 50%;
      right: -8px;
      width: 16px;
      height: 34px;
      padding: 0;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
      transform: translateY(-50%);
      z-index: 20;
      cursor: ew-resize;
      display: flex;
      align-items: center;
      justify-content: center;
      touch-action: none;
    }
    .sidebar-edge-toggle::before {
      content: "";
      width: 6px;
      height: 6px;
      border-right: 1.5px solid currentColor;
      border-bottom: 1.5px solid currentColor;
      transform: rotate(135deg);
      transition: transform 120ms ease;
    }
    main.sidebar-collapsed .sidebar-edge-toggle::before { transform: rotate(-45deg); }
    body.sidebar-resizing {
      cursor: ew-resize;
      user-select: none;
    }
    section { min-width: 0; }
    .sidebar-block { padding: 14px; border-bottom: 1px solid var(--line); }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .panel-toggle {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .panel-toggle::after {
      content: "";
      width: 7px;
      height: 7px;
      border-right: 1.5px solid currentColor;
      border-bottom: 1.5px solid currentColor;
      transform: rotate(45deg);
      transition: transform 120ms ease;
      flex: 0 0 auto;
    }
    .sidebar-block.collapsed .panel-toggle::after { transform: rotate(-45deg); }
    .sidebar-block.collapsed .panel-body { display: none; }
    .run, .result {
      width: 100%;
      text-align: left;
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: transparent;
    }
    .run.active, .result.active { border-color: var(--accent); outline: 2px solid color-mix(in srgb, var(--accent) 20%, transparent); }
    .title { font-weight: 650; overflow-wrap: anywhere; }
    .muted { color: var(--muted); font-size: 13px; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid var(--line);
      margin-top: 6px;
    }
    .status.running, .status.starting, .status.live { color: var(--warn); }
    .status.complete, .status.completed { color: var(--good); }
    .status.failed, .status.unreadable { color: var(--bad); }
    .bar { height: 8px; border-radius: 999px; background: color-mix(in srgb, var(--line) 65%, transparent); overflow: hidden; margin-top: 8px; }
    .bar > div { height: 100%; background: var(--accent); width: 0%; }
    .content { padding: 18px; overflow: auto; max-height: calc(100vh - 56px); }
    .content,
    .sidebar-inner,
    .memory-inner {
      scroll-behavior: smooth;
      overscroll-behavior: contain;
    }
    .memory-sidebar {
      border-right: 0;
      border-left: 1px solid var(--line);
      min-width: 0;
      background: var(--panel);
    }
    main.memory-collapsed .memory-sidebar {
      border-left: 0;
      overflow: visible;
    }
    main.memory-fullscreen .memory-sidebar {
      position: fixed;
      top: 56px;
      right: 0;
      bottom: 0;
      width: min(1180px, 100vw);
      max-height: none;
      z-index: 800;
      box-shadow: -16px 0 36px rgba(0, 0, 0, 0.18);
    }
    .memory-inner {
      height: calc(100vh - 56px);
      overflow: auto;
      background: var(--panel);
    }
    main.memory-collapsed .memory-inner {
      width: 0;
      overflow: hidden;
      pointer-events: none;
    }
    main.memory-fullscreen .memory-inner { height: calc(100vh - 56px); }
    .memory-resize {
      position: absolute;
      left: -5px;
      top: 0;
      bottom: 0;
      width: 10px;
      cursor: ew-resize;
      z-index: 10;
      touch-action: none;
    }
    main.memory-collapsed .memory-resize,
    main.memory-fullscreen .memory-resize { display: none; }
    .memory-open-tab {
      position: fixed;
      top: 50%;
      right: 0;
      transform: translateY(-50%);
      z-index: 25;
      writing-mode: vertical-rl;
      border-radius: 6px 0 0 6px;
      padding: 10px 5px;
      display: none;
    }
    main.memory-collapsed ~ .memory-open-tab { display: block; }
    .memory-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .memory-title { font-weight: 700; overflow-wrap: anywhere; }
    .memory-actions { display: flex; gap: 6px; }
    .memory-actions button,
    .memory-open {
      padding: 5px 8px;
      font-size: 12px;
    }
    .memory-actions button:disabled {
      opacity: 0.6;
      cursor: progress;
    }
    .memory-body { padding: 12px; }
    .memory-toolbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }
    .memory-toolbar select { width: 100%; min-width: 0; }
    .memory-mode {
      color: var(--warn);
      border: 1px solid color-mix(in srgb, var(--warn) 45%, var(--line));
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      text-transform: uppercase;
    }
    .memory-sync-status {
      min-height: 16px;
      margin: -4px 0 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .memory-sync-status.error { color: var(--bad); }
    .memory-sync-status.ok { color: var(--good); }
    .memory-graph {
      position: relative;
      height: 360px;
      min-height: 280px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background:
        linear-gradient(color-mix(in srgb, var(--line) 34%, transparent) 1px, transparent 1px),
        linear-gradient(90deg, color-mix(in srgb, var(--line) 34%, transparent) 1px, transparent 1px),
        color-mix(in srgb, var(--panel) 94%, var(--bg));
      background-size: 28px 28px;
      overflow: hidden;
      transition:
        height 320ms ease,
        border-radius 320ms ease,
        box-shadow 320ms ease,
        inset 320ms ease;
    }
    main.memory-fullscreen .memory-graph { height: min(68vh, 720px); }
    .memory-graph.graph-expanded {
      position: fixed;
      inset: 72px 18px 18px 18px;
      height: auto;
      min-height: 0;
      border-radius: 8px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
      z-index: 1200;
    }
    .memory-graph-tools {
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 5;
      display: flex;
      gap: 6px;
      opacity: 0;
      pointer-events: none;
      transform: translateY(-4px) scale(0.96);
      transition: opacity 150ms ease, transform 150ms ease;
    }
    .memory-graph:hover .memory-graph-tools,
    .memory-graph:focus-within .memory-graph-tools {
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0) scale(1);
    }
    .memory-graph-tool {
      display: grid;
      place-items: center;
      width: 28px;
      height: 28px;
      padding: 0;
      border-radius: 6px;
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
      transition: opacity 150ms ease, transform 150ms ease, border-color 150ms ease;
    }
    .memory-graph-tool[aria-pressed="true"] {
      border-color: color-mix(in srgb, var(--forest-green) 62%, var(--line));
      background: color-mix(in srgb, var(--forest-green) 12%, var(--panel));
    }
    .memory-graph-expand-icon {
      width: 14px;
      height: 14px;
      display: block;
      background:
        linear-gradient(var(--ink), var(--ink)) left top / 6px 1.5px no-repeat,
        linear-gradient(var(--ink), var(--ink)) left top / 1.5px 6px no-repeat,
        linear-gradient(var(--ink), var(--ink)) right top / 6px 1.5px no-repeat,
        linear-gradient(var(--ink), var(--ink)) right top / 1.5px 6px no-repeat,
        linear-gradient(var(--ink), var(--ink)) left bottom / 6px 1.5px no-repeat,
        linear-gradient(var(--ink), var(--ink)) left bottom / 1.5px 6px no-repeat,
        linear-gradient(var(--ink), var(--ink)) right bottom / 6px 1.5px no-repeat,
        linear-gradient(var(--ink), var(--ink)) right bottom / 1.5px 6px no-repeat;
    }
    .memory-graph-pin-icon {
      width: 14px;
      height: 14px;
      display: block;
      transform: rotate(35deg);
      background:
        radial-gradient(circle at 50% 3px, var(--ink) 0 3px, transparent 3.5px),
        linear-gradient(var(--ink), var(--ink)) center 6px / 9px 2px no-repeat,
        linear-gradient(var(--ink), var(--ink)) center 7px / 2px 9px no-repeat;
    }
    .memory-graph-scatter-icon {
      width: 15px;
      height: 15px;
      display: block;
      background:
        radial-gradient(circle at 50% 50%, var(--ink) 0 2px, transparent 2.5px),
        radial-gradient(circle at 12% 18%, var(--ink) 0 1.7px, transparent 2.2px),
        radial-gradient(circle at 82% 22%, var(--ink) 0 1.7px, transparent 2.2px),
        radial-gradient(circle at 18% 82%, var(--ink) 0 1.7px, transparent 2.2px),
        radial-gradient(circle at 84% 78%, var(--ink) 0 1.7px, transparent 2.2px);
    }
    .belief-svg { width: 100%; height: 100%; display: block; user-select: none; touch-action: none; }
    .belief-node,
    .belief-edge,
    .belief-edge-label { will-change: transform; }
    .belief-edge { stroke: color-mix(in srgb, var(--muted) 58%, transparent); stroke-width: 1.35; vector-effect: non-scaling-stroke; }
    .belief-edge.support { stroke: var(--forest-green); }
    .belief-edge.contradict { stroke: var(--bad); }
    .belief-edge-label { pointer-events: none; }
    .belief-edge-label rect {
      fill: color-mix(in srgb, var(--panel) 94%, transparent);
      stroke: color-mix(in srgb, var(--line) 72%, transparent);
      stroke-width: 1;
      rx: 4;
    }
    .belief-edge-label text {
      dominant-baseline: middle;
      fill: var(--ink);
      font-size: 9.5px;
      font-weight: 750;
      letter-spacing: 0;
      text-anchor: middle;
    }
    .belief-edge-label.support text { fill: var(--forest-green); }
    .belief-node { cursor: grab; }
    .belief-node.dragging { cursor: grabbing; }
    .belief-node circle {
      fill: var(--panel);
      stroke: var(--accent);
      stroke-width: 2;
      filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.18));
    }
    .belief-node.Evidence circle { stroke: #8a6f1d; }
    .belief-node.Claim circle { stroke: #176b87; }
    .belief-node.BeliefVariable circle { stroke: var(--forest-green); }
    .belief-node.Factor circle { stroke: #7b4aa0; }
    .belief-node.Decision circle { stroke: #b45f06; }
    .belief-node text {
      fill: var(--ink);
      font-size: 11px;
      font-weight: 650;
      paint-order: stroke;
      stroke: var(--panel);
      stroke-width: 3px;
      stroke-linejoin: round;
      pointer-events: none;
    }
    .memory-inspector,
    .memory-source-list {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: color-mix(in srgb, var(--panel) 92%, var(--bg));
      font-size: 12px;
    }
    .memory-source-list button {
      display: block;
      position: relative;
      width: 100%;
      text-align: left;
      margin-top: 6px;
      padding: 6px 8px;
      border-radius: 5px;
      background: transparent;
      overflow: hidden;
    }
    .memory-source-fill {
      position: absolute;
      inset: 0 auto 0 0;
      width: 0;
      background: color-mix(in srgb, var(--forest-green) 18%, transparent);
    }
    .memory-source-text {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    mark.belief-highlight {
      background: color-mix(in srgb, var(--forest-green) 24%, transparent);
      color: inherit;
      border-bottom: 2px solid var(--forest-green);
      border-radius: 3px;
      padding: 0 2px;
    }
    mark.belief-highlight.navigate-glow {
      animation: belief-highlight-pulse 520ms ease-in-out 6;
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--forest-green) 0%, transparent);
    }
    @keyframes belief-highlight-pulse {
      0%, 100% {
        background: color-mix(in srgb, var(--forest-green) 24%, transparent);
        box-shadow: 0 0 0 0 color-mix(in srgb, var(--forest-green) 0%, transparent);
      }
      50% {
        background: color-mix(in srgb, var(--forest-green) 38%, transparent);
        box-shadow: 0 0 16px 4px color-mix(in srgb, var(--forest-green) 36%, transparent);
      }
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: var(--panel);
    }
    .metric strong { display: block; font-size: 20px; margin-top: 3px; }
    details {
      border: 1px solid var(--line);
      border-radius: 6px;
      margin: 10px 0;
      background: var(--panel);
      overflow: hidden;
    }
    details.record,
    details.sample,
    .step,
    .message,
    .turn-section {
      content-visibility: auto;
      contain-intrinsic-size: 1px 220px;
    }
    .record-body,
    .detail-body,
    .step-body {
      contain: layout paint;
    }
    summary {
      cursor: pointer;
      padding: 12px;
      list-style-position: inside;
    }
    .detail-body { padding: 0 12px 12px; border-top: 1px solid var(--line); }
    .question { white-space: pre-wrap; margin: 8px 0 12px; }
    .sample { margin: 10px 0; }
    .sample > summary {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 8px;
      list-style: none;
    }
    .sample > summary::-webkit-details-marker { display: none; }
    .sample > summary::before,
    .step > summary::before {
      content: "";
      width: 7px;
      height: 7px;
      border-right: 1.5px solid currentColor;
      border-bottom: 1.5px solid currentColor;
      transform: rotate(-45deg);
      transition: transform 120ms ease;
      flex: 0 0 auto;
      margin-right: 2px;
    }
    .sample[open] > summary::before,
    .step[open] > summary::before { transform: rotate(45deg); }
    .sample-mark {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      font-size: 15px;
      font-weight: 800;
      line-height: 1;
    }
    .sample-mark.correct { color: var(--forest-green); }
    .sample-mark.incorrect { color: var(--brick-red); }
    .message {
      border-left: 4px solid var(--line);
      padding: 8px 10px;
      margin: 8px 0;
      background: color-mix(in srgb, var(--panel) 88%, var(--bg));
    }
    .message.assistant { border-left-color: var(--accent); }
    .message.user { border-left-color: var(--accent-2); }
    .message.tool { border-left-color: var(--warn); }
    .role { font-weight: 650; font-size: 13px; margin-bottom: 5px; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 24px;
      background: var(--panel);
    }
    .steps { margin: 10px 0; }
    .step {
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 8px;
      background: color-mix(in srgb, var(--panel) 92%, var(--bg));
      overflow: hidden;
    }
    .step-head {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 8px;
      padding: 8px 10px;
      background: color-mix(in srgb, var(--panel) 80%, var(--bg));
      list-style: none;
      cursor: pointer;
    }
    .step[open] > .step-head { border-bottom: 1px solid var(--line); }
    .step-head::-webkit-details-marker { display: none; }
    .step-head .idx { font-weight: 650; }
    .step-head .finish { font-size: 12px; color: var(--muted); }
    .step-raw-toggle {
      margin-left: auto;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 11px;
      color: var(--muted);
      background: color-mix(in srgb, var(--panel) 92%, var(--bg));
    }
    .step-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 6px 14px;
      padding: 8px 10px;
      font-size: 12px;
    }
    .step-grid .kv { display: flex; flex-direction: column; }
    .step-grid .k { color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.03em; }
    .step-grid .v { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .step-preview {
      border-top: 1px dashed var(--line);
      padding: 8px 10px;
    }
    .step-preview .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 4px; }
    .step-tokens {
      border-top: 1px dashed var(--line);
      padding: 6px 10px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 4px 12px;
    }
    .step-tokens .flag.ok { color: var(--good); }
    .step-tokens .flag.no { color: var(--bad); }
    .prompt-block {
      border: 1px solid var(--line);
      border-radius: 6px;
      margin: 8px 0;
      background: color-mix(in srgb, var(--panel) 94%, var(--bg));
      overflow: hidden;
    }
    .prompt-block > summary {
      display: flex;
      align-items: baseline;
      gap: 8px;
      padding: 8px 10px;
      list-style: none;
      background: color-mix(in srgb, var(--panel) 86%, var(--bg));
    }
    .prompt-block > summary::-webkit-details-marker { display: none; }
    .prompt-block > summary::before {
      content: "";
      width: 7px;
      height: 7px;
      border-right: 1.5px solid currentColor;
      border-bottom: 1.5px solid currentColor;
      transform: rotate(-45deg);
      transition: transform 120ms ease;
      flex: 0 0 auto;
    }
    .prompt-block[open] > summary::before { transform: rotate(45deg); }
    .prompt-body,
    .step-body { padding: 0; }
    .turn-section {
      border-top: 1px dashed var(--line);
      padding: 8px 10px;
    }
    .turn-section:first-child { border-top: 0; }
    .turn-details > summary {
      cursor: pointer;
      list-style: none;
    }
    .turn-details > summary .turn-label { margin-bottom: 0; }
    .turn-details > summary::-webkit-details-marker { display: none; }
    .turn-details > summary::before {
      content: "";
      display: inline-block;
      width: 7px;
      height: 7px;
      border-right: 1.5px solid currentColor;
      border-bottom: 1.5px solid currentColor;
      transform: rotate(-45deg);
      transition: transform 120ms ease;
      margin-right: 7px;
      vertical-align: 1px;
    }
    .turn-details[open] > summary::before { transform: rotate(45deg); }
    .turn-details pre { margin-top: 4px; }
    .markdown-body {
      font-size: 13px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }
    .markdown-body p { margin: 0 0 8px; }
    .markdown-body p:last-child { margin-bottom: 0; }
    .markdown-body ul,
    .markdown-body ol {
      margin: 4px 0 8px 22px;
      padding: 0;
    }
    .markdown-body li { margin: 2px 0; }
    .markdown-body h1,
    .markdown-body h2,
    .markdown-body h3 {
      margin: 8px 0 6px;
      font-size: 14px;
      line-height: 1.35;
    }
    .markdown-body code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: color-mix(in srgb, var(--line) 38%, transparent);
      border-radius: 4px;
      padding: 1px 4px;
    }
    .markdown-body pre {
      background: var(--code);
      color: #e6edf3;
      border-radius: 6px;
      padding: 8px;
      overflow: auto;
      margin: 6px 0;
    }
    .step.raw-mode .markdown-body { display: none; }
    .step:not(.raw-mode) .raw-body { display: none; }
    .turn-label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin-bottom: 4px;
      font-weight: 650;
    }
    .tool-response {
      border-left: 3px solid var(--warn);
      background: color-mix(in srgb, var(--warn) 7%, transparent);
    }
    .sample-meta {
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 8px;
    }
    .sample-meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 6px 14px;
      margin-top: 6px;
      font-size: 12px;
    }
    .sample-meta-grid .kv { display: flex; flex-direction: column; }
    .sample-meta-grid .k {
      color: var(--muted);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.03em;
    }
    .sample-meta-grid .v {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }
    @media (max-width: 860px) {
      main, main.sidebar-collapsed { display: block; }
      aside { max-height: none; border-right: 0; }
      .sidebar-inner { height: auto; max-height: none; }
      main.sidebar-collapsed .sidebar-inner { width: auto; pointer-events: auto; }
      .sidebar-edge-toggle { display: none; }
      .memory-sidebar { border-left: 0; border-top: 1px solid var(--line); }
      .memory-inner { height: auto; max-height: none; }
      main.memory-collapsed .memory-inner { width: auto; pointer-events: auto; }
      .memory-resize, .memory-open-tab { display: none !important; }
      .content { max-height: none; }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
    .ctx-menu {
      position: fixed;
      z-index: 1000;
      min-width: 180px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
      padding: 4px;
      display: none;
    }
    .ctx-menu.open { display: block; }
    .ctx-menu button {
      width: 100%;
      text-align: left;
      border: 0;
      background: transparent;
      color: var(--ink);
      padding: 8px 10px;
      border-radius: 4px;
      font: inherit;
    }
    .ctx-menu button:hover { background: color-mix(in srgb, var(--accent) 12%, transparent); }
    .ctx-menu button.danger { color: var(--bad); }
    .ctx-menu .sep { height: 1px; background: var(--line); margin: 4px 2px; }
    .run, .result { position: relative; }
    .run .custom-tag, .result .custom-tag {
      display: inline-block;
      font-size: 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--accent);
      border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
      border-radius: 999px;
      padding: 1px 6px;
      margin-left: 6px;
      vertical-align: middle;
    }
    .tag-row {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
    }
    .tag-row .tag {
      font-size: 11px;
      padding: 1px 6px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: color-mix(in srgb, var(--accent-2) 14%, transparent);
      color: var(--ink);
    }
    .notes-line {
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.45);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 999;
    }
    .modal-backdrop.open { display: flex; }
    .modal {
      background: var(--panel);
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      width: min(480px, 92vw);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.3);
    }
    .modal h2 { margin: 0 0 10px; font-size: 16px; }
    .modal label { display: block; font-size: 12px; color: var(--muted); margin-top: 10px; }
    .modal input, .modal textarea {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: var(--bg);
      color: var(--ink);
      font: inherit;
      margin-top: 4px;
    }
    .modal textarea { min-height: 88px; resize: vertical; }
    .modal .btn-row {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 14px;
    }
    .modal.config-modal {
      width: min(1120px, 96vw);
      max-height: 92vh;
      overflow: auto;
    }
    .config-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .config-top h2 {
      margin: 0;
      color: var(--muted);
      font-size: 22px;
      font-weight: 450;
    }
    .config-help {
      margin: 0 0 8px;
      color: color-mix(in srgb, var(--muted) 92%, var(--ink));
      font-size: 15px;
    }
    .config-help em { font-style: italic; }
    .config-help .learn {
      color: #008a9a;
      font-weight: 700;
      font-style: italic;
      margin-left: 4px;
    }
    .config-search {
      position: relative;
      margin: 8px 0 14px;
    }
    .config-search input {
      width: 100%;
      height: 38px;
      padding: 7px 12px 7px 38px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      font-size: 15px;
    }
    .config-search-icon {
      position: absolute;
      left: 13px;
      top: 50%;
      width: 15px;
      height: 15px;
      border: 1.8px solid var(--muted);
      border-radius: 50%;
      transform: translateY(-58%);
      pointer-events: none;
    }
    .config-search-icon::after {
      content: "";
      position: absolute;
      width: 7px;
      height: 1.8px;
      right: -6px;
      bottom: -4px;
      background: var(--muted);
      transform: rotate(45deg);
      transform-origin: left center;
      border-radius: 99px;
    }
    .config-regex-error {
      display: none;
      color: var(--bad);
      font-size: 12px;
      margin-top: -8px;
      margin-bottom: 10px;
    }
    .config-regex-error.open { display: block; }
    .config-tree {
      border: 1px solid var(--line);
      border-radius: 5px;
      background: var(--panel);
      padding: 14px 16px;
      max-height: min(64vh, 680px);
      overflow: auto;
      font-size: 15px;
    }
    .config-tree details {
      border: 0;
      border-radius: 0;
      margin: 0;
      background: transparent;
      overflow: visible;
    }
    .config-tree summary {
      list-style: none;
      cursor: pointer;
      padding: 6px 0;
    }
    .config-tree summary::-webkit-details-marker { display: none; }
    .config-row {
      display: flex;
      align-items: baseline;
      min-height: 24px;
      gap: 8px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .config-caret {
      width: 0;
      height: 0;
      border-top: 6px solid transparent;
      border-bottom: 6px solid transparent;
      border-left: 8px solid color-mix(in srgb, var(--muted) 58%, transparent);
      flex: 0 0 auto;
      transform: rotate(0deg);
      transition: transform 120ms ease;
    }
    .config-node[open] > summary .config-caret { transform: rotate(90deg); }
    .config-key {
      color: color-mix(in srgb, var(--ink) 74%, var(--muted));
      font-weight: 700;
    }
    .config-type {
      color: color-mix(in srgb, var(--accent-2) 72%, var(--muted));
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-weight: 500;
    }
    .config-count { color: #bf6b00; }
    .config-children { margin-left: 28px; }
    .config-leaf {
      display: flex;
      align-items: baseline;
      gap: 8px;
      min-height: 30px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .config-leaf::before {
      content: "";
      width: 8px;
      flex: 0 0 auto;
    }
    .config-value {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #008a9a;
    }
    .config-value.string { color: var(--good); }
    .config-value.null { color: var(--muted); }
    .config-empty {
      color: var(--muted);
      padding: 10px 0;
    }
    .config-raw {
      max-height: min(70vh, 720px);
      overflow: auto;
      background: var(--bg);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .modal button.danger { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 40%, var(--line)); }
    .modal button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    .toast {
      position: fixed;
      bottom: 18px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--panel);
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 14px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
      z-index: 1100;
      font-size: 13px;
      max-width: 80vw;
      display: none;
    }
    .toast.open { display: block; }
    .toast.error { border-color: color-mix(in srgb, var(--bad) 50%, var(--line)); }
  </style>
</head>
<body>
  <header>
    <h1>BeliefTracer</h1>
    <div class="header-actions">
      <span id="clock" class="muted"></span>
      <button id="refresh">Refresh</button>
    </div>
  </header>
  <main id="appMain">
    <aside id="sidebar">
      <div class="sidebar-inner">
        <div class="sidebar-block" data-sidebar-panel="runs">
          <button type="button" class="label panel-toggle" data-panel-toggle="runs" aria-expanded="true">Runs</button>
          <div id="runs" class="panel-body"></div>
        </div>
      </div>
      <button
        type="button"
        id="sidebarEdgeToggle"
        class="sidebar-edge-toggle"
        aria-label="Collapse sidebar"
        aria-expanded="true"
        title="Drag to resize, click to collapse"
      ></button>
    </aside>
    <section class="content">
      <div id="summary"></div>
      <div id="records" class="empty">Select a live stream or result file to inspect trajectories.</div>
    </section>
    <aside id="memorySidebar" class="memory-sidebar" aria-label="Belief memory graph">
      <div id="memoryResize" class="memory-resize" title="Drag to resize graph memory"></div>
      <div class="memory-inner">
        <div class="memory-head">
          <div>
            <div id="memoryTitle" class="memory-title">Graph Memory</div>
            <div id="memorySubtitle" class="muted">Select a trajectory sample.</div>
          </div>
          <div class="memory-actions">
            <button type="button" id="memoryUpdate" title="Write current graph to Memgraph and fetch latest topology">Update</button>
            <button type="button" id="memoryFullscreen" title="Toggle fullscreen">Full</button>
            <button type="button" id="memoryCollapse" title="Collapse graph memory">Hide</button>
          </div>
        </div>
        <div class="memory-body">
          <div class="memory-toolbar">
            <select id="memoryStepSelect" aria-label="Belief memory step"></select>
            <span id="memoryMode" class="memory-mode">mock</span>
          </div>
          <div id="memorySyncStatus" class="memory-sync-status"></div>
          <div id="memoryGraph" class="memory-graph"></div>
          <div id="memoryInspector" class="memory-inspector">No graph memory selected.</div>
          <div id="memorySources" class="memory-source-list"></div>
        </div>
      </div>
    </aside>
  </main>
  <button type="button" id="memoryOpenTab" class="memory-open-tab">Memory</button>
  <div id="ctxmenu" class="ctx-menu" role="menu">
    <button type="button" data-action="view-config">Config</button>
    <button type="button" data-action="rename">Rename</button>
    <button type="button" data-action="edit-meta">Edit notes & tags</button>
    <button type="button" data-action="stop">Stop run</button>
    <div class="sep"></div>
    <button type="button" data-action="copy-path">Copy path</button>
    <button type="button" data-action="reset">Reset name</button>
    <div class="sep"></div>
    <button type="button" class="danger" data-action="delete">Delete</button>
  </div>
  <div id="metaModal" class="modal-backdrop" role="dialog" aria-modal="true">
    <div class="modal">
      <h2 id="metaTitle">Edit run</h2>
      <label for="metaName">Display name</label>
      <input id="metaName" type="text" maxlength="200" autocomplete="off">
      <label for="metaNotes">Notes</label>
      <textarea id="metaNotes" maxlength="4000"></textarea>
      <label for="metaTags">Tags (comma-separated)</label>
      <input id="metaTags" type="text" autocomplete="off">
      <div class="btn-row">
        <button type="button" id="metaCancel">Cancel</button>
        <button type="button" id="metaSave" class="primary">Save</button>
      </div>
    </div>
  </div>
  <div id="configModal" class="modal-backdrop" role="dialog" aria-modal="true">
    <div class="modal config-modal">
      <div class="config-top">
        <h2 id="configTitle">Config</h2>
        <button type="button" id="configRawToggle">View raw data</button>
      </div>
      <p class="config-help"><em>Config parameters are your model's inputs.</em><span class="learn">Learn more</span></p>
      <div class="config-search">
        <span class="config-search-icon" aria-hidden="true"></span>
        <input id="configSearch" type="text" placeholder="Search keys with regex" autocomplete="off">
      </div>
      <div id="configRegexError" class="config-regex-error"></div>
      <div id="configTree" class="config-tree"></div>
      <pre id="configRaw" class="config-raw" hidden></pre>
      <div class="btn-row">
        <button type="button" id="configClose" class="primary">Close</button>
      </div>
    </div>
  </div>
  <div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script>
    const maxTrajectories = __MAX_TRAJECTORIES__;
    let selectedPath = "";
    let selectedKind = "";
    let stateCache = null;
    let refreshInFlight = false;
    let ctxTarget = null;
    let metaSaveHandler = null;
    let configPayload = null;
    let configTreePayload = null;
    let configRawVisible = false;
    const sidebarPanelStorageKey = "belieftracer.sidebar.collapsed";
    const sidebarWidthStorageKey = "belieftracer.sidebar.width";
    const sidebarCollapsedStorageKey = "belieftracer.sidebar.mainCollapsed";
    const sidebarMinWidth = 240;
    const sidebarDefaultWidth = 360;
    const sidebarCollapseThreshold = 140;
    const memoryWidthStorageKey = "belieftracer.memory.width";
    const memoryCollapsedStorageKey = "belieftracer.memory.collapsed";
    const memoryFullscreenStorageKey = "belieftracer.memory.fullscreen";
    const memoryMinWidth = 320;
    const memoryDefaultWidth = 420;
    const memoryCollapseThreshold = 180;
    let memorySamples = new Map();
    let currentMemoryKey = "";
    let currentBeliefMemory = null;
    let selectedMemoryStep = "all";
    let memorySyncStatus = "";
    let memorySyncStatusKind = "";
    let graphDrag = null;
    let memoryGraphExpanded = false;
    let memoryGraphPinned = false;
    let memoryGraphTransformMode = "original";
    let graphTransformOrigin = null;
    let graphLayoutAnimation = null;
    let graphDomCache = null;
    let graphDomUpdateFrame = null;
    let graphDirtyNodeIds = null;
    let graphDragFrame = null;
    let currentRenderingMemoryKey = "";
    let currentRecordsPayload = null;
    let memoryLocationIndex = new Map();
    const payloadCache = new Map();
    const payloadInflight = new Map();
    const payloadCacheLimit = 8;

    function readSidebarPanelState() {
      try {
        return JSON.parse(localStorage.getItem(sidebarPanelStorageKey) || "{}") || {};
      } catch {
        return {};
      }
    }
    function writeSidebarPanelState(state) {
      try {
        localStorage.setItem(sidebarPanelStorageKey, JSON.stringify(state || {}));
      } catch {
        return;
      }
    }
    function applySidebarPanelState() {
      const state = readSidebarPanelState();
      for (const block of document.querySelectorAll("[data-sidebar-panel]")) {
        const key = block.dataset.sidebarPanel || "";
        const collapsed = Boolean(state[key]);
        block.classList.toggle("collapsed", collapsed);
        const btn = block.querySelector("[data-panel-toggle]");
        if (btn) btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      }
    }
    function initSidebarPanels() {
      for (const btn of document.querySelectorAll("[data-panel-toggle]")) {
        btn.addEventListener("click", () => {
          const key = btn.dataset.panelToggle || "";
          const state = readSidebarPanelState();
          state[key] = !state[key];
          writeSidebarPanelState(state);
          applySidebarPanelState();
        });
      }
      applySidebarPanelState();
    }
    function sidebarMaxWidth() {
      return Math.max(sidebarMinWidth, Math.min(760, window.innerWidth - 360));
    }
    function clampSidebarWidth(value) {
      const width = Number(value);
      if (!Number.isFinite(width)) return sidebarDefaultWidth;
      return Math.max(sidebarMinWidth, Math.min(sidebarMaxWidth(), width));
    }
    function readSidebarWidth() {
      try {
        return clampSidebarWidth(Number(localStorage.getItem(sidebarWidthStorageKey)));
      } catch {
        return sidebarDefaultWidth;
      }
    }
    function writeSidebarWidth(width) {
      try {
        localStorage.setItem(sidebarWidthStorageKey, String(Math.round(clampSidebarWidth(width))));
      } catch {
        return;
      }
    }
    function readSidebarCollapsed() {
      try {
        return localStorage.getItem(sidebarCollapsedStorageKey) === "1";
      } catch {
        return false;
      }
    }
    function writeSidebarCollapsed(collapsed) {
      try {
        localStorage.setItem(sidebarCollapsedStorageKey, collapsed ? "1" : "0");
      } catch {
        return;
      }
    }
    function applySidebarLayout() {
      const main = document.getElementById("appMain");
      const edge = document.getElementById("sidebarEdgeToggle");
      if (!main) return;
      const collapsed = readSidebarCollapsed();
      main.style.setProperty("--sidebar-width", `${readSidebarWidth()}px`);
      main.classList.toggle("sidebar-collapsed", collapsed);
      if (edge) {
        edge.setAttribute("aria-expanded", collapsed ? "false" : "true");
        edge.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
        edge.title = collapsed ? "Click to expand sidebar" : "Drag to resize, click to collapse";
      }
    }
    function setSidebarCollapsed(collapsed) {
      writeSidebarCollapsed(collapsed);
      applySidebarLayout();
    }
    function setSidebarWidth(width) {
      writeSidebarWidth(width);
      writeSidebarCollapsed(false);
      applySidebarLayout();
    }
    function initSidebarResize() {
      const edge = document.getElementById("sidebarEdgeToggle");
      const main = document.getElementById("appMain");
      if (!edge || !main) return;
      let dragging = false;
      let moved = false;
      let startX = 0;
      let startWidth = 0;

      edge.addEventListener("pointerdown", ev => {
        if (ev.button !== 0) return;
        dragging = true;
        moved = false;
        startX = ev.clientX;
        startWidth = readSidebarCollapsed() ? 0 : readSidebarWidth();
        edge.setPointerCapture(ev.pointerId);
        document.body.classList.add("sidebar-resizing");
        ev.preventDefault();
      });
      edge.addEventListener("pointermove", ev => {
        if (!dragging) return;
        const nextWidth = startWidth + ev.clientX - startX;
        if (Math.abs(ev.clientX - startX) > 3) moved = true;
        if (!moved) return;
        if (nextWidth < sidebarCollapseThreshold) {
          writeSidebarCollapsed(true);
          applySidebarLayout();
        } else {
          writeSidebarWidth(nextWidth);
          writeSidebarCollapsed(false);
          applySidebarLayout();
        }
      });
      edge.addEventListener("pointerup", ev => {
        if (!dragging) return;
        dragging = false;
        document.body.classList.remove("sidebar-resizing");
        try { edge.releasePointerCapture(ev.pointerId); } catch {}
        if (!moved) setSidebarCollapsed(!readSidebarCollapsed());
        setTimeout(() => { moved = false; }, 0);
      });
      edge.addEventListener("click", ev => {
        if (moved) {
          ev.preventDefault();
          ev.stopPropagation();
        }
      });
      window.addEventListener("resize", () => {
        writeSidebarWidth(readSidebarWidth());
        applySidebarLayout();
      });
      applySidebarLayout();
    }

    function memoryMaxWidth() {
      return Math.max(memoryMinWidth, Math.min(900, window.innerWidth - 420));
    }
    function clampMemoryWidth(value) {
      const width = Number(value);
      if (!Number.isFinite(width)) return memoryDefaultWidth;
      return Math.max(memoryMinWidth, Math.min(memoryMaxWidth(), width));
    }
    function readMemoryWidth() {
      try {
        return clampMemoryWidth(Number(localStorage.getItem(memoryWidthStorageKey)));
      } catch {
        return memoryDefaultWidth;
      }
    }
    function writeMemoryWidth(width) {
      try {
        localStorage.setItem(memoryWidthStorageKey, String(Math.round(clampMemoryWidth(width))));
      } catch {
        return;
      }
    }
    function readMemoryCollapsed() {
      try {
        return localStorage.getItem(memoryCollapsedStorageKey) === "1";
      } catch {
        return false;
      }
    }
    function writeMemoryCollapsed(collapsed) {
      try {
        localStorage.setItem(memoryCollapsedStorageKey, collapsed ? "1" : "0");
      } catch {
        return;
      }
    }
    function readMemoryFullscreen() {
      try {
        return localStorage.getItem(memoryFullscreenStorageKey) === "1";
      } catch {
        return false;
      }
    }
    function writeMemoryFullscreen(fullscreen) {
      try {
        localStorage.setItem(memoryFullscreenStorageKey, fullscreen ? "1" : "0");
      } catch {
        return;
      }
    }
    function applyMemoryLayout() {
      const main = document.getElementById("appMain");
      const fullBtn = document.getElementById("memoryFullscreen");
      const collapseBtn = document.getElementById("memoryCollapse");
      if (!main) return;
      const fullscreen = readMemoryFullscreen();
      const collapsed = readMemoryCollapsed() && !fullscreen;
      main.style.setProperty("--memory-width", `${readMemoryWidth()}px`);
      main.classList.toggle("memory-collapsed", collapsed);
      main.classList.toggle("memory-fullscreen", fullscreen);
      if (fullBtn) fullBtn.textContent = fullscreen ? "Dock" : "Full";
      if (collapseBtn) collapseBtn.textContent = collapsed ? "Show" : "Hide";
    }
    function setMemoryCollapsed(collapsed) {
      writeMemoryCollapsed(collapsed);
      if (collapsed) writeMemoryFullscreen(false);
      applyMemoryLayout();
    }
    function setMemoryFullscreen(fullscreen) {
      writeMemoryFullscreen(fullscreen);
      if (fullscreen) writeMemoryCollapsed(false);
      applyMemoryLayout();
      renderMemoryPanel();
    }
    function initMemoryResize() {
      const handle = document.getElementById("memoryResize");
      if (!handle) return;
      let dragging = false;
      let moved = false;
      let startX = 0;
      let startWidth = 0;
      handle.addEventListener("pointerdown", ev => {
        if (ev.button !== 0) return;
        dragging = true;
        moved = false;
        startX = ev.clientX;
        startWidth = readMemoryCollapsed() ? 0 : readMemoryWidth();
        handle.setPointerCapture(ev.pointerId);
        document.body.classList.add("sidebar-resizing");
        ev.preventDefault();
      });
      handle.addEventListener("pointermove", ev => {
        if (!dragging) return;
        const nextWidth = startWidth + startX - ev.clientX;
        if (Math.abs(ev.clientX - startX) > 3) moved = true;
        if (!moved) return;
        if (nextWidth < memoryCollapseThreshold) {
          writeMemoryCollapsed(true);
          writeMemoryFullscreen(false);
        } else {
          writeMemoryWidth(nextWidth);
          writeMemoryCollapsed(false);
        }
        applyMemoryLayout();
      });
      handle.addEventListener("pointerup", ev => {
        if (!dragging) return;
        dragging = false;
        document.body.classList.remove("sidebar-resizing");
        try { handle.releasePointerCapture(ev.pointerId); } catch {}
      });
      window.addEventListener("resize", () => {
        writeMemoryWidth(readMemoryWidth());
        applyMemoryLayout();
      });
      applyMemoryLayout();
    }

    function findEntry(path) {
      if (!stateCache || !path) return null;
      const all = [
        ...(stateCache.runs || []),
        ...(stateCache.streams || []),
        ...(stateCache.results || []),
      ];
      return all.find(e => e.path === path) || null;
    }
    function showToast(message, kind) {
      const el = document.getElementById("toast");
      if (!el) return;
      el.textContent = message;
      el.className = `toast open${kind === "error" ? " error" : ""}`;
      clearTimeout(showToast._t);
      showToast._t = setTimeout(() => { el.classList.remove("open"); }, 3500);
    }
    function closeCtxMenu() {
      const menu = document.getElementById("ctxmenu");
      if (menu) menu.classList.remove("open");
      ctxTarget = null;
    }
    function openCtxMenu(event, entry) {
      event.preventDefault();
      const menu = document.getElementById("ctxmenu");
      if (!menu) return;
      ctxTarget = {
        path: entry.dataset.path || "",
        defaultName: entry.dataset.defaultName || "",
        kind: entry.dataset.kind || "",
        pid: entry.dataset.pid || "",
        status: entry.dataset.status || "",
      };
      const titleEl = entry.querySelector(".title");
      const hasCustom = !!(titleEl && titleEl.querySelector(".custom-tag"));
      const isRun = ctxTarget.kind === "run";
      const isStoppable = isRun && ctxTarget.pid && ["running", "starting"].includes(ctxTarget.status);
      const visibility = {
        "view-config": isRun,
        rename: true,
        "edit-meta": true,
        stop: isStoppable,
        "copy-path": true,
        reset: hasCustom,
        delete: true,
      };
      for (const btn of menu.querySelectorAll("button[data-action]")) {
        btn.style.display = visibility[btn.dataset.action] ? "" : "none";
      }
      menu.classList.add("open");
      const rect = menu.getBoundingClientRect();
      const w = rect.width || 200;
      const h = rect.height || 120;
      const x = Math.min(event.clientX, window.innerWidth - w - 8);
      const y = Math.min(event.clientY, window.innerHeight - h - 8);
      menu.style.left = `${Math.max(4, x)}px`;
      menu.style.top = `${Math.max(4, y)}px`;
    }
    function attachContextMenu(container) {
      const entries = container.querySelectorAll("[data-path]");
      for (const el of entries) {
        el.addEventListener("contextmenu", ev => openCtxMenu(ev, el));
      }
    }
    async function postJson(url, body) {
      const res = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body || {}),
      });
      let parsed = null;
      try { parsed = await res.json(); } catch { parsed = null; }
      if (!res.ok) {
        const msg = (parsed && parsed.error) || `${res.status} ${res.statusText}`;
        throw new Error(msg);
      }
      return parsed || {};
    }
    function openMetaModal(target) {
      const entry = findEntry(target.path) || {};
      const modal = document.getElementById("metaModal");
      const titleEl = document.getElementById("metaTitle");
      const nameInput = document.getElementById("metaName");
      const notesInput = document.getElementById("metaNotes");
      const tagsInput = document.getElementById("metaTags");
      if (titleEl) titleEl.textContent = `Edit ${target.kind || "entry"}`;
      nameInput.value = entry.display_name || "";
      nameInput.placeholder = target.defaultName || "";
      notesInput.value = entry.notes || "";
      tagsInput.value = (entry.tags || []).join(", ");
      modal.classList.add("open");
      setTimeout(() => nameInput.focus(), 0);
      metaSaveHandler = async () => {
        const tags = tagsInput.value.split(",").map(s => s.trim()).filter(Boolean);
        try {
          await postJson("/api/meta", {
            path: target.path,
            display_name: nameInput.value.trim(),
            notes: notesInput.value,
            tags,
          });
          closeMetaModal();
          await refresh({reloadSelected: false});
        } catch (err) {
          showToast(`Save failed: ${err.message}`, "error");
        }
      };
    }
    function closeMetaModal() {
      const modal = document.getElementById("metaModal");
      if (modal) modal.classList.remove("open");
      metaSaveHandler = null;
    }
    function closeConfigModal() {
      const modal = document.getElementById("configModal");
      if (modal) modal.classList.remove("open");
    }
    function hasConfigValue(value) {
      if (value === null || value === undefined) return false;
      if (typeof value === "string") return value !== "";
      if (Array.isArray(value)) return value.length > 0;
      if (value && typeof value === "object") return Object.keys(value).length > 0;
      return true;
    }
    function compactConfigValue(value) {
      if (Array.isArray(value)) {
        return value
          .map(compactConfigValue)
          .filter(hasConfigValue);
      }
      if (value && typeof value === "object") {
        const out = {};
        for (const [key, item] of Object.entries(value)) {
          const compacted = compactConfigValue(item);
          if (hasConfigValue(compacted)) out[key] = compacted;
        }
        return out;
      }
      return value;
    }
    function fallbackEffectiveConfig(payload) {
      const cfg = (payload && payload.config) || {};
      const sampling = (payload && payload.sampling_params) || {};
      const engine = {
        backend: cfg.backend,
        max_model_len: cfg.vllm_max_model_len || ((cfg.max_prompt_length || 0) + (cfg.max_response_length || 0)) || null,
        max_prompt_length: cfg.max_prompt_length,
        max_response_length: cfg.max_response_length,
        tensor_parallel_size: cfg.tensor_parallel_size,
        gpu_memory_utilization: cfg.gpu_memory_utilization,
        data_parallel_size: cfg.data_parallel_size,
        data_parallel_devices: cfg.data_parallel_devices,
        trust_remote_code: cfg.vllm_trust_remote_code,
      };
      if (cfg.backend === "sglang" || cfg.backend === "sglang_dp") {
        engine.sglang_context_length = engine.max_model_len;
        engine.sglang_max_total_tokens = engine.max_model_len;
        engine.sglang_disable_cuda_graph = true;
        engine.sglang_disable_piecewise_cuda_graph = true;
      }
      return compactConfigValue({
        model: {
          path: cfg.model,
          enable_thinking: cfg.enable_thinking,
        },
        benchmarks: {
          tasks: cfg.tasks,
          max_problems: cfg.max_problems,
          shuffle: cfg.shuffle,
          shuffle_seed: cfg.shuffle_seed,
          num_samples: cfg.num_samples,
          passk: cfg.passk,
        },
        agent: {
          parser_name: cfg.parser_name,
          tools: cfg.tools,
          max_steps: cfg.max_steps,
          system_prompt_enabled: Boolean(cfg.system_prompt),
          system_prompt: cfg.system_prompt,
        },
        engine,
        sampling,
        retrieval: {
          server_url: cfg.retrieval_server_url,
          max_results: cfg.retrieval_max_results,
          timeout: cfg.retrieval_timeout,
        },
        scheduler: {
          n_parallel_tasks: cfg.n_parallel_tasks,
          retry_limit: cfg.retry_limit,
          mixed_rollouts: cfg.mixed_rollouts,
        },
        artifacts: {
          output_dir: cfg.output_dir,
          overwrite: cfg.overwrite,
          auto_ui: cfg.auto_ui,
        },
      });
    }
    function buildConfigTreePayload(payload) {
      payload = payload || {};
      const effective = payload.effective_config || fallbackEffectiveConfig(payload);
      return compactConfigValue({
        effective_config: effective,
        run: {
          run_id: payload.run_id,
          status: payload.status,
          phase: payload.phase,
          created_iso: payload.created_iso,
          updated_iso: payload.updated_iso,
          completed_iso: payload.completed_iso,
          elapsed_seconds: payload.elapsed_seconds,
          source: payload.source,
          config_path: payload._path,
          run_state_path: payload.run_state_path,
        },
        process: payload.process,
        machine: payload.machine,
        artifacts: payload.artifacts,
        summaries: payload.summaries,
        raw_cli_config: compactConfigValue(payload.config || {}),
        error: payload.error,
      });
    }
    function configObjectLabel(value) {
      if (Array.isArray(value)) return ["[]", `${value.length} items`];
      if (value && typeof value === "object") {
        const n = Object.keys(value).length;
        return ["{}", `${n} ${n === 1 ? "key" : "keys"}`];
      }
      return ["", ""];
    }
    function configPath(parent, key) {
      const part = String(key);
      return parent ? `${parent}.${part}` : part;
    }
    function renderConfigValue(value) {
      if (value === null) return `<span class="config-value null">null</span>`;
      if (typeof value === "string") return `<span class="config-value string">"${esc(value)}"</span>`;
      if (typeof value === "boolean") return `<span class="config-value">${value ? "true" : "false"}</span>`;
      if (typeof value === "number") return `<span class="config-value">${esc(value)}</span>`;
      return `<span class="config-value">${esc(String(value))}</span>`;
    }
    function renderConfigNode(key, value, path, depth, matcher) {
      const isRoot = depth === 0;
      const label = isRoot ? "Config parameters:" : `${key}:`;
      const searchable = [path, String(key || "")].filter(Boolean).join(" ");
      const selfMatch = matcher ? matcher(searchable) : false;
      const isObject = value && typeof value === "object";
      if (!isObject) {
        const visible = !matcher || selfMatch;
        return {
          html: visible
            ? `<div class="config-leaf"><span class="config-key">${esc(label)}</span>${renderConfigValue(value)}</div>`
            : "",
          matched: visible,
        };
      }

      const entries = Array.isArray(value)
        ? value.map((item, idx) => [idx, item])
        : Object.entries(value);
      const children = [];
      let childMatch = false;
      for (const [childKey, childValue] of entries) {
        const child = renderConfigNode(
          childKey,
          childValue,
          configPath(isRoot ? "" : path, childKey),
          depth + 1,
          matcher && !selfMatch ? matcher : null
        );
        if (child.matched) childMatch = true;
        if (child.html) children.push(child.html);
      }
      const visible = !matcher || selfMatch || childMatch;
      if (!visible) return {html: "", matched: false};
      const [typeLabel, countLabel] = configObjectLabel(value);
      const open = isRoot || Boolean(matcher);
      const body = children.length
        ? children.join("")
        : `<div class="config-empty">${matcher ? "No matching nested keys." : "No keys."}</div>`;
      return {
        html: `<details class="config-node" ${open ? "open" : ""}>
          <summary>
            <div class="config-row">
              <span class="config-caret"></span>
              <span class="config-key">${esc(label)}</span>
              <span class="config-type">${esc(typeLabel)}</span>
              <span class="config-count">${esc(countLabel)}</span>
            </div>
          </summary>
          <div class="config-children">${body}</div>
        </details>`,
        matched: true,
      };
    }
    function renderConfigPanel() {
      const tree = document.getElementById("configTree");
      const raw = document.getElementById("configRaw");
      const toggle = document.getElementById("configRawToggle");
      const search = document.getElementById("configSearch");
      const error = document.getElementById("configRegexError");
      if (!tree || !raw || !toggle || !search || !error) return;

      const payload = configPayload || {};
      const treePayload = configTreePayload || buildConfigTreePayload(payload);
      raw.textContent = JSON.stringify(payload, null, 2);
      raw.hidden = !configRawVisible;
      tree.hidden = configRawVisible;
      search.disabled = configRawVisible;
      toggle.textContent = configRawVisible ? "View config tree" : "View raw data";

      let matcher = null;
      const query = search.value.trim();
      error.classList.remove("open");
      error.textContent = "";
      if (query) {
        try {
          const regex = new RegExp(query, "i");
          matcher = text => regex.test(String(text || ""));
        } catch (err) {
          error.textContent = `Invalid regex: ${err.message}`;
          error.classList.add("open");
        }
      }
      const rendered = renderConfigNode(null, treePayload, "", 0, matcher);
      tree.innerHTML = rendered.html || `<div class="config-empty">No matching keys.</div>`;
    }
    async function openRunConfig(target) {
      const res = await fetch(`/api/run/config?path=${encodeURIComponent(target.path)}`);
      let payload = null;
      try { payload = await res.json(); } catch { payload = null; }
      if (!res.ok) {
        const msg = (payload && payload.error) || `${res.status} ${res.statusText}`;
        throw new Error(msg);
      }
      const title = document.getElementById("configTitle");
      const search = document.getElementById("configSearch");
      if (title) title.textContent = "Config";
      if (search) search.value = "";
      configPayload = payload || {};
      configTreePayload = buildConfigTreePayload(configPayload);
      configRawVisible = false;
      renderConfigPanel();
      document.getElementById("configModal").classList.add("open");
    }
    async function confirmStop(target) {
      const entry = findEntry(target.path) || {};
      const label = entry.display_name || target.defaultName || target.path;
      const ok = window.confirm(
        `Send SIGTERM to PID ${target.pid} for "${label}"?\n\n` +
        `If the process is inside the docker container the server will fall back to docker exec.`
      );
      if (!ok) return;
      try {
        const result = await postJson("/api/run/stop", {path: target.path});
        showToast(result.method ? `Stop signal sent via ${result.method}` : (result.error || "Stop failed"), result.method ? "" : "error");
        await refresh({reloadSelected: false});
      } catch (err) {
        showToast(`Stop failed: ${err.message}`, "error");
      }
    }
    async function confirmDelete(target) {
      const entry = findEntry(target.path) || {};
      const label = entry.display_name || target.defaultName || target.path;
      const isRun = target.kind === "run";
      const what = isRun ? "the entire run directory" : "this artifact file";
      const typed = window.prompt(
        `Type DELETE to remove ${what} for:\n${label}\n\nThis cannot be undone.`,
        ""
      );
      if (typed === null) return;
      if (typed.trim().toUpperCase() !== "DELETE") {
        showToast("Delete cancelled (confirmation text did not match)", "error");
        return;
      }
      try {
        const result = await postJson("/api/run/delete", {path: target.path, confirm: true});
        if (selectedPath === target.path) {
          selectedPath = "";
          selectedKind = "";
          document.getElementById("records").className = "empty";
          document.getElementById("records").textContent = "Select a live stream or result file to inspect trajectories.";
          document.getElementById("summary").innerHTML = "";
        }
        showToast(`Removed ${result.removed || target.path}`);
        await refresh({reloadSelected: false});
      } catch (err) {
        showToast(`Delete failed: ${err.message}`, "error");
      }
    }
    async function handleCtxAction(action) {
      if (!ctxTarget) return;
      const target = ctxTarget;
      closeCtxMenu();
      try {
        if (action === "rename") {
          const entry = findEntry(target.path) || {};
          const current = entry.display_name || target.defaultName || "";
          const next = window.prompt("New name (leave blank to reset):", current);
          if (next === null) return;
          await postJson("/api/rename", {path: target.path, name: next.trim()});
          await refresh({reloadSelected: false});
        } else if (action === "reset") {
          await postJson("/api/rename", {path: target.path, name: ""});
          await refresh({reloadSelected: false});
        } else if (action === "edit-meta") {
          openMetaModal(target);
        } else if (action === "view-config") {
          await openRunConfig(target);
        } else if (action === "copy-path") {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(target.path);
            showToast("Path copied");
          }
        } else if (action === "stop") {
          await confirmStop(target);
        } else if (action === "delete") {
          await confirmDelete(target);
        }
      } catch (err) {
        showToast(`${action} failed: ${err.message}`, "error");
      }
    }
    document.addEventListener("click", () => closeCtxMenu());
    document.addEventListener("keydown", ev => {
      if (ev.key === "Escape") {
        closeCtxMenu();
        closeMetaModal();
        closeConfigModal();
      }
    });
    window.addEventListener("blur", () => closeCtxMenu());
    {
      const menu = document.getElementById("ctxmenu");
      if (menu) {
        menu.addEventListener("click", ev => {
          const btn = ev.target.closest("button[data-action]");
          if (!btn) return;
          ev.stopPropagation();
          handleCtxAction(btn.dataset.action);
        });
      }
      const modal = document.getElementById("metaModal");
      if (modal) {
        modal.addEventListener("click", ev => {
          if (ev.target === modal) closeMetaModal();
        });
      }
      const configModal = document.getElementById("configModal");
      if (configModal) {
        configModal.addEventListener("click", ev => {
          if (ev.target === configModal) closeConfigModal();
        });
      }
      const configClose = document.getElementById("configClose");
      if (configClose) configClose.addEventListener("click", () => closeConfigModal());
      const configRawToggle = document.getElementById("configRawToggle");
      if (configRawToggle) configRawToggle.addEventListener("click", () => {
        configRawVisible = !configRawVisible;
        renderConfigPanel();
      });
      const configSearch = document.getElementById("configSearch");
      if (configSearch) configSearch.addEventListener("input", () => renderConfigPanel());
      initSidebarPanels();
      initSidebarResize();
      initMemoryResize();
      const memoryCollapse = document.getElementById("memoryCollapse");
      if (memoryCollapse) memoryCollapse.addEventListener("click", () => setMemoryCollapsed(!readMemoryCollapsed()));
      const memoryOpenTab = document.getElementById("memoryOpenTab");
      if (memoryOpenTab) memoryOpenTab.addEventListener("click", () => setMemoryCollapsed(false));
      const memoryFullscreen = document.getElementById("memoryFullscreen");
      if (memoryFullscreen) memoryFullscreen.addEventListener("click", () => setMemoryFullscreen(!readMemoryFullscreen()));
      const memoryUpdate = document.getElementById("memoryUpdate");
      if (memoryUpdate) memoryUpdate.addEventListener("click", () => updateMemoryFromMemgraph());
      const memoryStepSelect = document.getElementById("memoryStepSelect");
      if (memoryStepSelect) memoryStepSelect.addEventListener("change", () => {
        selectedMemoryStep = memoryStepSelect.value || "all";
        renderMemoryPanel();
      });
      const cancelBtn = document.getElementById("metaCancel");
      if (cancelBtn) cancelBtn.addEventListener("click", () => closeMetaModal());
      const saveBtn = document.getElementById("metaSave");
      if (saveBtn) saveBtn.addEventListener("click", () => {
        if (metaSaveHandler) metaSaveHandler();
      });
      const recordsEl = document.getElementById("records");
      if (recordsEl) recordsEl.addEventListener("click", ev => {
        const memoryBtn = ev.target.closest(".memory-open");
        if (memoryBtn) {
          ev.preventDefault();
          ev.stopPropagation();
          activateMemoryForKey(memoryBtn.dataset.memoryKey || "", memoryBtn.dataset.stepIndex || "all");
          return;
        }
        const btn = ev.target.closest(".step-raw-toggle");
        if (!btn) return;
        ev.preventDefault();
        ev.stopPropagation();
        const step = btn.closest(".step");
        if (!step) return;
        const rawMode = !step.classList.contains("raw-mode");
        step.classList.toggle("raw-mode", rawMode);
        btn.textContent = rawMode ? "markdown" : "raw";
        btn.title = rawMode ? "Show markdown rendering" : "Show raw content";
      });
      const memorySources = document.getElementById("memorySources");
      if (memorySources) memorySources.addEventListener("click", ev => {
        const source = ev.target.closest("[data-memory-key][data-text]");
        if (!source) return;
        ev.preventDefault();
        ev.stopPropagation();
        void navigateToSupportSpan(
          source.dataset.memoryKey || currentMemoryKey,
          source.dataset.stepIndex || selectedMemoryStep || "all",
          source.dataset.text || "",
          source.dataset.nodeId || "",
        );
      });
    }

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function pct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return `${(Number(value) * 100).toFixed(1)}%`;
    }
    function confidenceWidth(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
      return Math.max(0, Math.min(100, Number(value) * 100));
    }
    function count(value) {
      if (value === null || value === undefined || value === "") return "n/a";
      return String(value);
    }
    function statusPill(status) {
      return `<span class="status ${esc(status)}">${esc(status || "unknown")}</span>`;
    }
    function progressBar(progress) {
      const width = Math.max(0, Math.min(100, Number(progress || 0) * 100));
      return `<div class="bar"><div style="width:${width}%"></div></div>`;
    }
    function metric(label, value) {
      return `<div class="metric"><span class="label">${esc(label)}</span><strong>${esc(value)}</strong></div>`;
    }
    async function fetchJson(url) {
      const res = await fetch(url, {cache: "no-store"});
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return await res.json();
    }
    function payloadKey(kind, path) {
      return `${kind}:${path}`;
    }
    function rememberPayload(key, payload) {
      if (payloadCache.has(key)) payloadCache.delete(key);
      payloadCache.set(key, payload);
      while (payloadCache.size > payloadCacheLimit) {
        const oldest = payloadCache.keys().next().value;
        if (!oldest) break;
        payloadCache.delete(oldest);
      }
      return payload;
    }
    function payloadUrl(kind, path) {
      if (kind === "stream") {
        return `/api/stream?path=${encodeURIComponent(path)}&limit=${maxTrajectories}`;
      }
      return `/api/result?path=${encodeURIComponent(path)}`;
    }
    async function loadPayloadCached(kind, path) {
      const key = payloadKey(kind, path);
      if (payloadCache.has(key)) return payloadCache.get(key);
      if (payloadInflight.has(key)) return payloadInflight.get(key);
      const promise = fetchJson(payloadUrl(kind, path))
        .then(payload => rememberPayload(key, payload))
        .finally(() => payloadInflight.delete(key));
      payloadInflight.set(key, promise);
      return promise;
    }
    function prefetchPayload(kind, path) {
      if (!kind || !path || kind !== "result") return;
      const key = payloadKey(kind, path);
      if (payloadCache.has(key) || payloadInflight.has(key)) return;
      void loadPayloadCached(kind, path);
    }
    function scheduleIdlePrefetch(task) {
      if (typeof requestIdleCallback === "function") {
        requestIdleCallback(() => task(), {timeout: 1200});
      } else {
        setTimeout(task, 120);
      }
    }
    function warmSidebarPayloads(state) {
      const items = buildSidebarItems(state)
        .map(sidebarItemLoadTarget)
        .filter(load => load.kind === "result" && load.path)
        .slice(0, 4);
      scheduleIdlePrefetch(() => {
        for (const load of items) prefetchPayload(load.kind, load.path);
      });
    }
    function renderTagsAndNotes(entry) {
      const tags = (entry && entry.tags) || [];
      const notes = (entry && entry.notes) || "";
      const tagRow = tags.length
        ? `<div class="tag-row">${tags.map(t => `<span class="tag">${esc(t)}</span>`).join("")}</div>`
        : "";
      const notesLine = notes
        ? `<div class="notes-line">${esc(notes)}</div>`
        : "";
      return tagRow + notesLine;
    }
    function sidebarItemKey(model, benchmark, fallback) {
      const m = model || "";
      const b = benchmark || "";
      const f = fallback || "";
      return (m || b) ? `${m} / ${b} / ${f}` : f;
    }
    function buildSidebarItems(state) {
      state = state || {};
      const byKey = new Map();
      const byResultPath = new Map();
      const byStreamPath = new Map();
      const ensure = (key) => {
        if (!byKey.has(key)) byKey.set(key, {key});
        return byKey.get(key);
      };
      for (const result of state.results || []) {
        const key = sidebarItemKey(result.model, result.benchmark, result.path);
        const item = ensure(key);
        item.result = result;
        item.model = item.model || result.model;
        item.benchmark = item.benchmark || result.benchmark;
        byResultPath.set(result.path, item);
      }
      for (const stream of state.streams || []) {
        const key = sidebarItemKey(stream.model, stream.benchmark, stream.path);
        const item = ensure(key);
        item.stream = stream;
        item.model = item.model || stream.model;
        item.benchmark = item.benchmark || stream.benchmark;
        byStreamPath.set(stream.path, item);
      }
      for (const run of state.runs || []) {
        let matched = false;
        for (const path of run.live_paths || []) {
          const item = byStreamPath.get(path);
          if (item) {
            item.run = run;
            item.model = item.model || run.model;
            item.benchmark = item.benchmark || run.current_benchmark;
            matched = true;
          }
        }
        for (const path of run.result_paths || []) {
          const item = byResultPath.get(path);
          if (item) {
            item.run = run;
            item.model = item.model || run.model;
            item.benchmark = item.benchmark || run.current_benchmark;
            matched = true;
          }
        }
        if (!matched) {
          const key = sidebarItemKey(run.model, run.current_benchmark, run.path);
          const item = ensure(key);
          item.run = run;
          item.model = item.model || run.model;
          item.benchmark = item.benchmark || run.current_benchmark;
        }
      }
      const items = Array.from(byKey.values());
      const statusRank = item => {
        const status = sidebarItemStatus(item);
        if (["running", "starting"].includes(status)) return 0;
        if (status === "live") return 1;
        if (status === "stale") return 2;
        if (["failed", "unreadable"].includes(status)) return 3;
        return 4;
      };
      return items.sort((a, b) => {
        const rank = statusRank(a) - statusRank(b);
        if (rank) return rank;
        return sidebarItemTime(b) - sidebarItemTime(a);
      });
    }
    function sidebarItemStatus(item) {
      const runStatus = item.run && item.run.status;
      if (["starting", "running", "stale", "failed", "unreadable"].includes(runStatus)) return runStatus;
      if (item.result) return item.result.status || "completed";
      if (item.stream) return item.stream.status || "live";
      return runStatus || "unknown";
    }
    function sidebarItemTime(item) {
      return Number(
        (item.run && item.run.updated_at) ||
        (item.result && item.result.mtime) ||
        (item.stream && item.stream.mtime) ||
        0
      );
    }
    function sidebarItemPrimaryEntry(item) {
      return item.run || item.result || item.stream || {};
    }
    function sidebarItemLoadTarget(item) {
      const status = sidebarItemStatus(item);
      if (["starting", "running", "stale", "live"].includes(status) && item.stream) {
        return {kind: "stream", path: item.stream.path};
      }
      if (item.result) return {kind: "result", path: item.result.path};
      if (item.stream) return {kind: "stream", path: item.stream.path};
      return {kind: "", path: ""};
    }
    function renderSidebarRuns(state) {
      const target = document.getElementById("runs");
      const items = buildSidebarItems(state);
      if (!items.length) {
        target.innerHTML = `<div class="muted" style="margin-top:8px">No runs found.</div>`;
        return;
      }
      target.innerHTML = items.map(item => {
        const entry = sidebarItemPrimaryEntry(item);
        const load = sidebarItemLoadTarget(item);
        const status = sidebarItemStatus(item);
        const defaultTitle = item.benchmark ? `${item.model || ""} / ${item.benchmark}` : (item.model || entry.path || "");
        const title = entry.display_name || defaultTitle;
        const tag = entry.display_name ? `<span class="custom-tag">renamed</span>` : "";
        const ctx = item.run || item.result || item.stream || {};
        const ctxKind = item.run ? "run" : item.result ? "result" : "stream";
        const pidAttr = item.run && item.run.pid ? `data-pid="${esc(item.run.pid)}"` : "";
        const active = load.path === selectedPath && load.kind === selectedKind ? "active" : "";
        const sampleLine = item.run
          ? `${count(item.run.completed_samples)} / ${count(item.run.total_samples)} samples`
          : item.result
          ? `${count(item.result.num_correct)} / ${count(item.result.num_samples)} correct`
          : item.stream
          ? `${count(item.stream.num_samples)} completed`
          : "";
        const timeLine = (item.run && item.run.updated_iso)
          || (item.result && item.result.mtime_iso)
          || (item.stream && item.stream.mtime_iso)
          || "";
        const phase = item.run && item.run.phase ? `${item.run.phase}${item.run.current_benchmark ? " - " + item.run.current_benchmark : ""}` : "";
        return `
        <button class="result ${active}" data-path="${esc(ctx.path || "")}" data-load-path="${esc(load.path)}" data-load-kind="${esc(load.kind)}" data-default-name="${esc(defaultTitle)}" data-kind="${esc(ctxKind)}" data-status="${esc(status)}" ${pidAttr}>
          <div class="title">${esc(title)}${tag}</div>
          ${phase ? `<div class="muted">${esc(phase)}</div>` : ""}
          ${statusPill(status)}
          ${item.run ? progressBar(item.run.progress) : ""}
          ${sampleLine ? `<div class="muted">${esc(sampleLine)}</div>` : ""}
          ${timeLine ? `<div class="muted">updated ${esc(timeLine)}</div>` : ""}
          ${renderTagsAndNotes(entry)}
        </button>
      `;
      }).join("");
      for (const el of target.querySelectorAll("button.result")) {
        el.addEventListener("click", () => {
          if (el.dataset.loadKind === "result") loadResult(el.dataset.loadPath);
          else if (el.dataset.loadKind === "stream") loadStream(el.dataset.loadPath);
        });
        el.addEventListener("mouseenter", () => {
          prefetchPayload(el.dataset.loadKind || "", el.dataset.loadPath || "");
        }, {passive: true});
      }
      attachContextMenu(target);
      warmSidebarPayloads(state);
    }
    function renderSummary(payload) {
      const summary = payload.summary || {};
      const accuracy = summary.accuracy_mean ?? (payload.num_samples ? payload.num_correct / payload.num_samples : null);
      const sampleText = summary.live
        ? `${count(summary.num_completed_trajectories)} completed`
        : (summary.num_samples_per_task ? `${summary.num_samples_per_task} each` : payload.num_samples ?? 0);
      document.getElementById("summary").innerHTML = `
        <div class="metrics">
          ${metric("Benchmark", summary.benchmark || payload.benchmark || "")}
          ${metric("Tasks", summary.num_tasks ?? payload.records?.length ?? 0)}
          ${metric("Samples", sampleText)}
          ${metric("Accuracy", pct(accuracy))}
        </div>
      `;
    }
    function renderMessage(msg, idx) {
      const role = msg.role || "message";
      let content = msg.content;
      if (Array.isArray(content) || (content && typeof content === "object")) {
        content = JSON.stringify(content, null, 2);
      }
      const reasoning = msg.reasoning_content || msg.reasoning || "";
      const reasoningText = reasoning ? `REASONING:\n${reasoning}\n\n` : "";
      const messageText = msg.raw_content || (reasoningText + (content ?? ""));
      const toolCalls = msg.tool_calls ? `\n\nTOOL_CALLS:\n${JSON.stringify(msg.tool_calls, null, 2)}` : "";
      return `<div class="message ${esc(role)}">
        <div class="role">${idx + 1}. ${esc(role)}</div>
        <pre>${esc(messageText + toolCalls)}</pre>
      </div>`;
    }
    function formatMessageContent(content) {
      if (Array.isArray(content) || (content && typeof content === "object")) {
        return JSON.stringify(content, null, 2);
      }
      return content ?? "";
    }
    function splitLeadingThinkingBlock(text) {
      const source = String(text ?? "");
      const startToken = "<think>";
      const endToken = "</think>";
      const start = source.indexOf(startToken);
      if (start < 0 || source.slice(0, start).trim()) {
        return {reasoning: "", action: source};
      }
      const afterStart = start + startToken.length;
      const end = source.indexOf(endToken, afterStart);
      if (end >= 0) {
        return {
          reasoning: source.slice(afterStart, end).trim(),
          action: source.slice(end + endToken.length).trim(),
        };
      }
      const toolStart = source.indexOf("<tool_call>", afterStart);
      if (toolStart >= 0) {
        return {
          reasoning: source.slice(afterStart, toolStart).trim(),
          action: source.slice(toolStart).trim(),
        };
      }
      return {reasoning: source.slice(afterStart).trim(), action: ""};
    }
    function normalizedHighlightSpans(spans) {
      return (spans || [])
        .filter(s => s && s.text)
        .map(s => ({...s, text: String(s.text)}))
        .sort((a, b) => b.text.length - a.text.length)
        .slice(0, 8);
    }
    function renderTextWithHighlights(text, spans) {
      const source = String(text ?? "");
      const highlights = normalizedHighlightSpans(spans);
      if (!source || !highlights.length) return esc(source);
      const ranges = [];
      const lower = source.toLowerCase();
      for (const span of highlights) {
        const needle = span.text.toLowerCase();
        if (!needle) continue;
        let start = lower.indexOf(needle);
        while (start >= 0) {
          const end = start + needle.length;
          const overlaps = ranges.some(r => start < r.end && end > r.start);
          if (!overlaps) ranges.push({start, end, span});
          start = lower.indexOf(needle, end);
        }
      }
      if (!ranges.length) return esc(source);
      ranges.sort((a, b) => a.start - b.start);
      let out = "";
      let cursor = 0;
      for (const range of ranges) {
        out += esc(source.slice(cursor, range.start));
        const title = `${range.span.kind || "support"} ${range.span.confidence ? `p=${range.span.confidence}` : ""}`.trim();
        out += `<mark class="belief-highlight" data-node-id="${esc(range.span.node_id || "")}" title="${esc(title)}">${esc(source.slice(range.start, range.end))}</mark>`;
        cursor = range.end;
      }
      out += esc(source.slice(cursor));
      return out;
    }
    function renderMarkdownInline(text, spans = []) {
      return renderTextWithHighlights(text, spans)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>");
    }
    function renderMarkdownText(text, spans = []) {
      const source = String(text || "").replace(/\r\n/g, "\n");
      const chunks = source.split(/```/);
      return chunks.map((chunk, idx) => {
        if (idx % 2 === 1) return `<pre><code>${esc(chunk.trim())}</code></pre>`;
        const lines = chunk.split("\n");
        const out = [];
        let list = null;
        const closeList = () => {
          if (list) {
            out.push(`</${list}>`);
            list = null;
          }
        };
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) {
            closeList();
            continue;
          }
          const heading = /^(#{1,3})\s+(.+)$/.exec(trimmed);
          if (heading) {
            closeList();
            const level = heading[1].length;
            out.push(`<h${level}>${renderMarkdownInline(heading[2], spans)}</h${level}>`);
            continue;
          }
          const bullet = /^[-*]\s+(.+)$/.exec(trimmed);
          if (bullet) {
            if (list !== "ul") {
              closeList();
              out.push("<ul>");
              list = "ul";
            }
            out.push(`<li>${renderMarkdownInline(bullet[1], spans)}</li>`);
            continue;
          }
          const numbered = /^\d+[.)]\s+(.+)$/.exec(trimmed);
          if (numbered) {
            if (list !== "ol") {
              closeList();
              out.push("<ol>");
              list = "ol";
            }
            out.push(`<li>${renderMarkdownInline(numbered[1], spans)}</li>`);
            continue;
          }
          closeList();
          out.push(`<p>${renderMarkdownInline(trimmed, spans)}</p>`);
        }
        closeList();
        return out.join("");
      }).join("");
    }
    function renderSeconds(value) {
      if (value === undefined || value === null || value === "" || Number.isNaN(Number(value))) return "";
      return `${Number(value).toFixed(2)} s`;
    }
    function renderTurnSection(label, text, cls = "", collapsible = false, open = true, markdown = false, highlights = []) {
      if (text === undefined || text === null || text === "") return "";
      if (collapsible) {
        return `<details class="turn-section turn-details ${esc(cls)}" ${open ? "open" : ""}>
          <summary><span class="turn-label">${esc(label)}</span></summary>
          ${markdown ? `<div class="markdown-body">${renderMarkdownText(text, highlights)}</div>` : ""}
          <pre class="${markdown ? "raw-body" : ""}">${renderTextWithHighlights(text, highlights)}</pre>
        </details>`;
      }
      return `<div class="turn-section ${esc(cls)}">
        <div class="turn-label">${esc(label)}</div>
        <pre>${renderTextWithHighlights(text, highlights)}</pre>
      </div>`;
    }
    function renderPromptBlock(title, msg, open = true) {
      if (!msg) return "";
      const text = formatMessageContent(msg.content);
      return `<details class="prompt-block" ${open ? "open" : ""}>
        <summary><strong>${esc(title)}</strong><span class="muted">${esc(msg.role || "")}</span></summary>
        <div class="prompt-body">
          ${renderTurnSection("content", text)}
        </div>
      </details>`;
    }
    function renderAssistantSections(msg, highlights = []) {
      if (!msg) return renderTurnSection("assistant", "No assistant message saved.");
      const content = formatMessageContent(msg.content);
      const splitContent = splitLeadingThinkingBlock(content);
      const reasoning = msg.reasoning_content || msg.reasoning || splitContent.reasoning || "";
      const raw = msg.raw_content || "";
      const toolCalls = msg.tool_calls
        ? JSON.stringify(msg.tool_calls, null, 2)
        : "";
      const reasoningHasThinkStart = String(reasoning).includes("<think>");
      const reasoningHasThinkEnd = String(reasoning).includes("</think>");
      const rawHasThinkStart = String(raw).includes("<think>");
      const rawHasThinkEnd = String(raw).includes("</think>");
      const reasoningText = reasoning
        ? `${rawHasThinkStart && !reasoningHasThinkStart ? "<think>\n" : ""}${reasoning}${rawHasThinkEnd && !reasoningHasThinkEnd ? "\n</think>" : ""}`
        : "";
      const actionText = splitContent.action;
      const rawOnly = raw && raw !== content && !reasoning ? raw : "";
      return [
        renderTurnSection("reasoning", reasoningText, "", true, true, true, highlights),
        renderTurnSection("action", actionText, "", true, true, true, highlights),
        renderTurnSection("assistant tool calls", toolCalls),
        renderTurnSection("raw model output", rawOnly, "", false, true, false, highlights),
      ].join("") || renderTurnSection("assistant", "No assistant content saved.");
    }
    function renderToolResponse(msg, idx) {
      const name = msg.name ? ` - ${msg.name}` : "";
      const callId = msg.tool_call_id ? ` (${msg.tool_call_id})` : "";
      return renderTurnSection(
        `tool response ${idx + 1}${name}${callId}`,
        formatMessageContent(msg.content),
        "tool-response"
      );
    }
    function memorySteps(memory) {
      return (memory && Array.isArray(memory.steps)) ? memory.steps : [];
    }
    function memoryStepData(memory, stepIndex) {
      const steps = memorySteps(memory);
      if (stepIndex === undefined || stepIndex === null || stepIndex === "all") {
        return {source_spans: steps.flatMap(s => s.source_spans || [])};
      }
      const idx = Number(stepIndex);
      return steps.find(s => Number(s.step_index) === idx) || {source_spans: []};
    }
    function memorySourceEntries(memory, stepIndex) {
      const steps = memorySteps(memory);
      if (stepIndex === undefined || stepIndex === null || stepIndex === "all") {
        return steps.flatMap(step =>
          normalizedHighlightSpans(step.source_spans || []).map(span => ({
            ...span,
            step_index: Number(step.step_index),
          }))
        );
      }
      const idx = Number(stepIndex);
      const step = steps.find(s => Number(s.step_index) === idx);
      return normalizedHighlightSpans(step?.source_spans || []).map(span => ({
        ...span,
        step_index: idx,
      }));
    }
    function memoryStepHighlights(memory, stepIndex) {
      return normalizedHighlightSpans(memoryStepData(memory, stepIndex).source_spans || []);
    }
    function setMemorySyncStatus(message, kind = "") {
      memorySyncStatus = message || "";
      memorySyncStatusKind = kind || "";
      const el = document.getElementById("memorySyncStatus");
      if (el) {
        el.textContent = memorySyncStatus;
        el.className = `memory-sync-status ${memorySyncStatusKind}`;
      }
    }
    function activateMemoryForKey(key, stepIndex = "all") {
      if (!key || !memorySamples.has(key)) return;
      currentMemoryKey = key;
      currentBeliefMemory = memorySamples.get(key);
      if (currentBeliefMemory) currentBeliefMemory.memory_key = currentMemoryKey;
      selectedMemoryStep = stepIndex === undefined ? "all" : String(stepIndex);
      memoryGraphTransformMode = "original";
      graphTransformOrigin = null;
      setMemoryCollapsed(false);
      setMemorySyncStatus("");
      renderMemoryPanel();
    }
    async function updateMemoryFromMemgraph() {
      if (!currentBeliefMemory || !currentMemoryKey) {
        setMemorySyncStatus("Select a graph memory sample first.", "error");
        return;
      }
      const startedAt = performance.now();
      const btn = document.getElementById("memoryUpdate");
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Updating";
      }
      setMemorySyncStatus("Fetching latest topology from Memgraph; seeding current graph only if missing...");
      try {
        currentBeliefMemory.memory_key = currentMemoryKey;
        const payload = await postJson("/api/memory/update", {
          memory_key: currentMemoryKey,
          memory: currentBeliefMemory,
        });
        if (!payload || !payload.memory) throw new Error("Memgraph response did not include memory");
        currentBeliefMemory = payload.memory;
        currentBeliefMemory.memory_key = currentMemoryKey;
        memorySamples.set(currentMemoryKey, currentBeliefMemory);
        memoryGraphTransformMode = "original";
        graphTransformOrigin = null;
        selectedMemoryStep = document.getElementById("memoryStepSelect")?.value || selectedMemoryStep || "all";
        renderMemoryPanel();
        await new Promise(resolve => requestAnimationFrame(() => resolve()));
        const elapsedMs = Math.max(1, Math.round(performance.now() - startedAt));
        const elapsedLabel = elapsedMs >= 1000 ? `${(elapsedMs / 1000).toFixed(1)}s` : `${elapsedMs}ms`;
        setMemorySyncStatus(`Memgraph updated: ${(currentBeliefMemory.nodes || []).length} nodes, ${(currentBeliefMemory.edges || []).length} edges (${elapsedLabel}).`, "ok");
      } catch (err) {
        setMemorySyncStatus(`Memgraph update failed: ${err.message || err}`, "error");
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Update";
        }
      }
    }
    function graphPoint(svg, ev) {
      const pt = svg.createSVGPoint();
      pt.x = ev.clientX;
      pt.y = ev.clientY;
      return pt.matrixTransform(svg.getScreenCTM().inverse());
    }
    function edgeRelationLabel(edge) {
      return String(edge.type || edge.label || edge.relationship || "related_to")
        .trim()
        .replace(/\s+/g, "_")
        .toLowerCase();
    }
    function edgePriority(edge) {
      const priority = {
        evaluated_by: 100,
        has_belief: 90,
        input_to: 80,
        output_to: 70,
        required_by: 60,
        supports: 50,
        owned_by: 10,
      };
      return priority[edgeRelationLabel(edge)] || 0;
    }
    function uniqueDirectedEdges(edges) {
      const selected = new Map();
      (edges || []).forEach((edge, idx) => {
        if (!edge || !edge.source || !edge.target || edge.source === edge.target) return;
        const pairKey = [String(edge.source), String(edge.target)].sort().join("\u0000");
        const score = edgePriority(edge) * 1000 + Math.abs(Number(edge.weight || 0)) - idx / 100000;
        const current = selected.get(pairKey);
        if (!current || score > current.score) selected.set(pairKey, {edge, score, idx});
      });
      return Array.from(selected.values()).sort((a, b) => a.idx - b.idx).map(item => item.edge);
    }
    function edgeLabelPosition(source, targetNode) {
      const sx = Number(source.x || 0);
      const sy = Number(source.y || 0);
      const tx = Number(targetNode.x || 0);
      const ty = Number(targetNode.y || 0);
      const dx = tx - sx;
      const dy = ty - sy;
      const len = Math.hypot(dx, dy) || 1;
      return {
        x: (sx + tx) / 2 + (-dy / len) * 9,
        y: (sy + ty) / 2 + (dx / len) * 9,
      };
    }
    function edgeRenderPoints(source, targetNode) {
      const sx = Number(source.x || 0);
      const sy = Number(source.y || 0);
      const tx = Number(targetNode.x || 0);
      const ty = Number(targetNode.y || 0);
      const dx = tx - sx;
      const dy = ty - sy;
      const len = Math.hypot(dx, dy) || 1;
      const startOffset = 30;
      const endOffset = 31;
      if (len <= startOffset + endOffset + 4) {
        return {x1: sx, y1: sy, x2: tx, y2: ty};
      }
      return {
        x1: sx + (dx / len) * startOffset,
        y1: sy + (dy / len) * startOffset,
        x2: tx - (dx / len) * endOffset,
        y2: ty - (dy / len) * endOffset,
      };
    }
    function expandedGraphCanvas(memory) {
      const counts = new Map();
      (memory?.nodes || []).forEach(node => counts.set(node.type || "Other", (counts.get(node.type || "Other") || 0) + 1));
      const maxColumn = Math.max(1, ...counts.values());
      return {width: 820, height: Math.max(520, maxColumn * 76 + 120)};
    }
    function expandedGraphLayout(memory) {
      const canvas = expandedGraphCanvas(memory);
      const columns = {
        Evidence: 72,
        Claim: 235,
        BeliefVariable: 405,
        Factor: 585,
        Decision: 748,
        Other: 405,
      };
      const groups = new Map();
      (memory?.nodes || []).forEach(node => {
        const type = columns[node.type] !== undefined ? node.type : "Other";
        if (!groups.has(type)) groups.set(type, []);
        groups.get(type).push(node);
      });
      const layout = new Map();
      for (const [type, items] of groups.entries()) {
        const top = 68;
        const bottom = canvas.height - 68;
        const step = items.length > 1 ? (bottom - top) / (items.length - 1) : 0;
        items.forEach((node, idx) => {
          layout.set(node.id, {
            x: columns[type],
            y: items.length > 1 ? top + step * idx : canvas.height / 2,
          });
        });
      }
      return {canvas, layout};
    }
    function compactGraphLayout(memory) {
      const layout = new Map();
      (memory?.nodes || []).forEach(node => {
        layout.set(node.id, {
          x: Number(node._compact_x ?? node.x ?? 0),
          y: Number(node._compact_y ?? node.y ?? 0),
        });
      });
      return {canvas: {width: 640, height: 420}, layout};
    }
    function currentGraphCanvas() {
      const svg = document.querySelector("#memoryGraph .belief-svg");
      if (!svg) return memoryGraphExpanded ? expandedGraphCanvas(currentBeliefMemory) : {width: 640, height: 420};
      const viewBox = svg.viewBox.baseVal;
      return {width: Number(viewBox.width || 640), height: Number(viewBox.height || 420)};
    }
    function captureGraphTransformOrigin(memory) {
      graphTransformOrigin = {
        canvas: currentGraphCanvas(),
        layout: new Map((memory?.nodes || []).map(node => [
          node.id,
          {x: Number(node.x || 0), y: Number(node.y || 0)},
        ])),
      };
    }
    function starScatterGraphLayout(memory) {
      const nodes = memory?.nodes || [];
      const centerNode = nodes.find(node => node.type === "Decision")
        || nodes.find(node => String(node.id || "").startsWith("bv_answer_"))
        || nodes[0];
      const outer = nodes.filter(node => node.id !== centerNode?.id);
      const radius = Math.max(210, (outer.length * 68) / (2 * Math.PI));
      const canvas = {
        width: Math.max(820, Math.ceil(radius * 2 + 180)),
        height: Math.max(560, Math.ceil(radius * 2 + 180)),
      };
      const center = {x: canvas.width / 2, y: canvas.height / 2};
      const layout = new Map();
      if (centerNode) layout.set(centerNode.id, center);
      outer.forEach((node, idx) => {
        const angle = -Math.PI / 2 + (idx / Math.max(1, outer.length)) * Math.PI * 2;
        layout.set(node.id, {
          x: center.x + Math.cos(angle) * radius,
          y: center.y + Math.sin(angle) * radius,
        });
      });
      return {canvas, layout};
    }
    function treeGraphLayout(memory) {
      const nodes = memory?.nodes || [];
      const levelFor = node => {
        if (node.type === "Evidence") return 0;
        if (node.type === "Claim") return 1;
        if (node.type === "BeliefVariable" && !String(node.id || "").startsWith("bv_answer_")) return 2;
        if (node.type === "Factor") return 3;
        if (node.type === "BeliefVariable") return 4;
        if (node.type === "Decision") return 5;
        return 2;
      };
      const groups = new Map();
      nodes.forEach(node => {
        const level = levelFor(node);
        if (!groups.has(level)) groups.set(level, []);
        groups.get(level).push(node);
      });
      const maxGroup = Math.max(1, ...Array.from(groups.values()).map(group => group.length));
      const canvas = {width: 900, height: Math.max(560, maxGroup * 76 + 120)};
      const layout = new Map();
      for (const [level, group] of groups.entries()) {
        const x = 70 + level * ((canvas.width - 140) / 5);
        const top = 68;
        const bottom = canvas.height - 68;
        const step = group.length > 1 ? (bottom - top) / (group.length - 1) : 0;
        group.forEach((node, idx) => {
          layout.set(node.id, {
            x,
            y: group.length > 1 ? top + step * idx : canvas.height / 2,
          });
        });
      }
      return {canvas, layout};
    }
    function setGraphViewBox(canvas) {
      const svg = document.querySelector("#memoryGraph .belief-svg");
      if (svg) svg.setAttribute("viewBox", `0 0 ${canvas.width} ${canvas.height}`);
    }
    function syncGraphExpandButton() {
      const graph = document.getElementById("memoryGraph");
      const btn = document.getElementById("memoryGraphExpand");
      const pinBtn = document.getElementById("memoryGraphPin");
      const scatterBtn = document.getElementById("memoryGraphScatter");
      if (graph) graph.classList.toggle("graph-expanded", memoryGraphExpanded);
      if (btn) {
        btn.setAttribute("aria-pressed", memoryGraphExpanded ? "true" : "false");
        btn.title = memoryGraphExpanded ? "Dock graph" : "Expand graph";
      }
      if (pinBtn) {
        pinBtn.setAttribute("aria-pressed", memoryGraphPinned ? "true" : "false");
        pinBtn.title = memoryGraphPinned ? "Unpin graph drag" : "Pin graph drag";
      }
      if (scatterBtn) {
        scatterBtn.setAttribute("aria-pressed", memoryGraphTransformMode !== "original" ? "true" : "false");
        scatterBtn.title = memoryGraphTransformMode === "original"
          ? "Scatter to star topology"
          : (memoryGraphTransformMode === "star" ? "Transform to tree topology" : "Restore original topology");
      }
    }
    function animateGraphTo(layout, canvas) {
      if (!currentBeliefMemory) return;
      if (graphLayoutAnimation) cancelAnimationFrame(graphLayoutAnimation);
      const dirtyNodeIds = new Set(layout.keys());
      const starts = new Map((currentBeliefMemory.nodes || []).map(node => [
        node.id,
        {x: Number(node.x || 0), y: Number(node.y || 0)},
      ]));
      const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 360;
      const started = performance.now();
      setGraphViewBox(canvas);
      const ease = t => 1 - Math.pow(1 - t, 3);
      if (duration === 0) {
        for (const node of currentBeliefMemory.nodes || []) {
          const target = layout.get(node.id);
          if (!target) continue;
          node.x = target.x;
          node.y = target.y;
        }
        scheduleGraphDomUpdate(dirtyNodeIds);
        graphLayoutAnimation = null;
        return;
      }
      const frame = now => {
        const progress = Math.min(1, (now - started) / duration);
        const eased = ease(progress);
        (currentBeliefMemory.nodes || []).forEach(node => {
          const start = starts.get(node.id) || {x: Number(node.x || 0), y: Number(node.y || 0)};
          const target = layout.get(node.id);
          if (!target) return;
          node.x = start.x + (target.x - start.x) * eased;
          node.y = start.y + (target.y - start.y) * eased;
        });
        scheduleGraphDomUpdate(dirtyNodeIds);
        if (progress < 1) {
          graphLayoutAnimation = requestAnimationFrame(frame);
        } else {
          graphLayoutAnimation = null;
        }
      };
      graphLayoutAnimation = requestAnimationFrame(frame);
    }
    function setMemoryGraphExpanded(expanded) {
      if (!currentBeliefMemory) return;
      memoryGraphTransformMode = "original";
      graphTransformOrigin = null;
      memoryGraphExpanded = Boolean(expanded);
      syncGraphExpandButton();
      if (memoryGraphExpanded) {
        (currentBeliefMemory.nodes || []).forEach(node => {
          node._compact_x = Number(node.x || 0);
          node._compact_y = Number(node.y || 0);
        });
        const {canvas, layout} = expandedGraphLayout(currentBeliefMemory);
        animateGraphTo(layout, canvas);
      } else {
        const {canvas, layout} = compactGraphLayout(currentBeliefMemory);
        animateGraphTo(layout, canvas);
      }
    }
    function setMemoryGraphPinned(pinned) {
      memoryGraphPinned = Boolean(pinned);
      syncGraphExpandButton();
    }
    function cycleMemoryGraphTransform() {
      if (!currentBeliefMemory) return;
      if (memoryGraphTransformMode === "original") {
        captureGraphTransformOrigin(currentBeliefMemory);
        memoryGraphTransformMode = "star";
        syncGraphExpandButton();
        const {canvas, layout} = starScatterGraphLayout(currentBeliefMemory);
        animateGraphTo(layout, canvas);
        return;
      }
      if (memoryGraphTransformMode === "star") {
        memoryGraphTransformMode = "tree";
        syncGraphExpandButton();
        const {canvas, layout} = treeGraphLayout(currentBeliefMemory);
        animateGraphTo(layout, canvas);
        return;
      }
      memoryGraphTransformMode = "original";
      syncGraphExpandButton();
      const fallback = memoryGraphExpanded ? expandedGraphLayout(currentBeliefMemory) : compactGraphLayout(currentBeliefMemory);
      const canvas = graphTransformOrigin?.canvas || fallback.canvas;
      const layout = graphTransformOrigin?.layout || fallback.layout;
      graphTransformOrigin = null;
      animateGraphTo(layout, canvas);
    }
    function buildGraphDomCache() {
      const graph = document.getElementById("memoryGraph");
      if (!graph || !currentBeliefMemory) {
        graphDomCache = null;
        return;
      }
      const nodes = new Map((currentBeliefMemory.nodes || []).map(n => [n.id, n]));
      const nodeEls = new Map(Array.from(graph.querySelectorAll(".belief-node")).map(el => [el.dataset.nodeId, el]));
      const edgeEls = Array.from(graph.querySelectorAll(".belief-edge")).map(el => ({
        sourceId: el.dataset.source,
        targetId: el.dataset.target,
        el,
      }));
      const labelEls = Array.from(graph.querySelectorAll(".belief-edge-label")).map(el => ({
        sourceId: el.dataset.source,
        targetId: el.dataset.target,
        el,
      }));
      const incidentEdges = new Map(Array.from(nodes.keys()).map(id => [id, []]));
      const incidentLabels = new Map(Array.from(nodes.keys()).map(id => [id, []]));
      const adjacency = new Map(Array.from(nodes.keys()).map(id => [id, new Set()]));
      for (const record of edgeEls) {
        incidentEdges.get(record.sourceId)?.push(record);
        incidentEdges.get(record.targetId)?.push(record);
        adjacency.get(record.sourceId)?.add(record.targetId);
        adjacency.get(record.targetId)?.add(record.sourceId);
      }
      for (const record of labelEls) {
        incidentLabels.get(record.sourceId)?.push(record);
        incidentLabels.get(record.targetId)?.push(record);
      }
      graphDomCache = {
        nodes,
        nodeEls,
        edgeEls,
        labelEls,
        incidentEdges,
        incidentLabels,
        adjacency,
        influenceCache: new Map(),
      };
    }
    function updateGraphDom(dirtyNodeIds = null) {
      if (!currentBeliefMemory) return;
      if (!graphDomCache) buildGraphDomCache();
      const cache = graphDomCache;
      if (!cache) return;
      const dirty = dirtyNodeIds ? new Set(dirtyNodeIds) : null;
      const nodeEntries = dirty
        ? Array.from(dirty, id => [id, cache.nodeEls.get(id)]).filter(([, el]) => el)
        : Array.from(cache.nodeEls.entries());
      for (const [id, g] of nodeEntries) {
        const node = cache.nodes.get(id);
        if (node) g.setAttribute("transform", `translate(${Number(node.x || 0)}, ${Number(node.y || 0)})`);
      }
      const edgeRecords = dirty
        ? Array.from(new Set(Array.from(dirty).flatMap(id => cache.incidentEdges.get(id) || [])))
        : cache.edgeEls;
      for (const record of edgeRecords) {
        const source = cache.nodes.get(record.sourceId);
        const target = cache.nodes.get(record.targetId);
        if (!source || !target) continue;
        const points = edgeRenderPoints(source, target);
        record.el.setAttribute("x1", points.x1);
        record.el.setAttribute("y1", points.y1);
        record.el.setAttribute("x2", points.x2);
        record.el.setAttribute("y2", points.y2);
      }
      const labelRecords = dirty
        ? Array.from(new Set(Array.from(dirty).flatMap(id => cache.incidentLabels.get(id) || [])))
        : cache.labelEls;
      for (const record of labelRecords) {
        const source = cache.nodes.get(record.sourceId);
        const target = cache.nodes.get(record.targetId);
        if (!source || !target) continue;
        const pos = edgeLabelPosition(source, target);
        record.el.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
      }
    }
    function scheduleGraphDomUpdate(dirtyNodeIds = null) {
      if (dirtyNodeIds === null) {
        graphDirtyNodeIds = null;
      } else if (graphDirtyNodeIds !== null) {
        for (const id of dirtyNodeIds) graphDirtyNodeIds.add(id);
      } else {
        graphDirtyNodeIds = new Set(dirtyNodeIds);
      }
      if (graphDomUpdateFrame) return;
      graphDomUpdateFrame = requestAnimationFrame(() => {
        graphDomUpdateFrame = null;
        const dirty = graphDirtyNodeIds;
        graphDirtyNodeIds = null;
        updateGraphDom(dirty);
      });
    }
    function graphDragInfluence(startId) {
      if (!currentBeliefMemory || memoryGraphPinned) return new Map([[startId, 1]]);
      if (!graphDomCache) buildGraphDomCache();
      const cache = graphDomCache;
      if (!cache) return new Map([[startId, 1]]);
      const cached = cache.influenceCache.get(startId);
      if (cached) return cached;
      const influence = new Map([[startId, 1]]);
      const queue = [{id: startId, hop: 0}];
      const seen = new Set([startId]);
      while (queue.length) {
        const current = queue.shift();
        const nextHop = current.hop + 1;
        for (const next of cache.adjacency.get(current.id) || []) {
          if (seen.has(next)) continue;
          seen.add(next);
          const strength = Math.max(0.12, Math.pow(0.58, nextHop));
          influence.set(next, strength);
          queue.push({id: next, hop: nextHop});
        }
      }
      cache.influenceCache.set(startId, influence);
      return influence;
    }
    function clampGraphPoint(svg, x, y) {
      const viewBox = svg.viewBox.baseVal;
      return {
        x: Math.max(28, Math.min((viewBox.width || 640) - 28, x)),
        y: Math.max(28, Math.min((viewBox.height || 420) - 28, y)),
      };
    }
    function bindGraphDrag(svg) {
      const processGraphDragFrame = () => {
        graphDragFrame = null;
        if (!graphDrag || !graphDrag.pendingPoint) return;
        const p = graphDrag.pendingPoint;
        const primaryStart = graphDrag.starts.get(graphDrag.node.id) || {x: Number(graphDrag.node.x || 0), y: Number(graphDrag.node.y || 0)};
        const primaryTarget = clampGraphPoint(svg, p.x + graphDrag.dx, p.y + graphDrag.dy);
        const deltaX = primaryTarget.x - primaryStart.x;
        const deltaY = primaryTarget.y - primaryStart.y;
        for (const [id, strength] of graphDrag.influence.entries()) {
          const node = graphDrag.nodesById.get(id);
          const start = graphDrag.starts.get(id);
          if (!node || !start) continue;
          const target = clampGraphPoint(svg, start.x + deltaX * strength, start.y + deltaY * strength);
          node.x = target.x;
          node.y = target.y;
        }
        scheduleGraphDomUpdate(graphDrag.influence.keys());
      };
      svg.addEventListener("pointerdown", ev => {
        const nodeEl = ev.target.closest(".belief-node");
        if (!nodeEl || !currentBeliefMemory) return;
        if (!graphDomCache) buildGraphDomCache();
        const cache = graphDomCache;
        const node = cache?.nodes.get(nodeEl.dataset.nodeId);
        if (!node) return;
        const p = graphPoint(svg, ev);
        const influence = graphDragInfluence(node.id);
        const starts = new Map();
        for (const id of influence.keys()) {
          const influenced = cache?.nodes.get(id);
          if (influenced) starts.set(id, {x: Number(influenced.x || 0), y: Number(influenced.y || 0)});
        }
        graphDrag = {
          node,
          dx: Number(node.x || 0) - p.x,
          dy: Number(node.y || 0) - p.y,
          nodeEl,
          influence,
          starts,
          nodesById: cache?.nodes || new Map(),
          pendingPoint: p,
        };
        nodeEl.classList.add("dragging");
        svg.setPointerCapture(ev.pointerId);
        ev.preventDefault();
      });
      svg.addEventListener("pointermove", ev => {
        if (!graphDrag) return;
        graphDrag.pendingPoint = graphPoint(svg, ev);
        if (!graphDragFrame) graphDragFrame = requestAnimationFrame(processGraphDragFrame);
      });
      const endDrag = ev => {
        if (!graphDrag) return;
        graphDrag.nodeEl.classList.remove("dragging");
        try { svg.releasePointerCapture(ev.pointerId); } catch {}
        if (graphDragFrame) {
          cancelAnimationFrame(graphDragFrame);
          graphDragFrame = null;
        }
        graphDrag = null;
      };
      svg.addEventListener("pointerup", endDrag);
      svg.addEventListener("pointercancel", endDrag);
    }
    function renderBeliefGraph(memory) {
      const target = document.getElementById("memoryGraph");
      if (!target) return;
      graphDomCache = null;
      graphDirtyNodeIds = null;
      if (graphDomUpdateFrame) {
        cancelAnimationFrame(graphDomUpdateFrame);
        graphDomUpdateFrame = null;
      }
      if (!memory || !(memory.nodes || []).length) {
        memoryGraphExpanded = false;
        memoryGraphTransformMode = "original";
        graphTransformOrigin = null;
        target.classList.remove("graph-expanded");
        target.innerHTML = `<div class="empty" style="margin:12px">No graph memory for this sample.</div>`;
        return;
      }
      if (memoryGraphExpanded) {
        const expanded = expandedGraphLayout(memory);
        (memory.nodes || []).forEach(node => {
          if (node._compact_x === undefined) node._compact_x = Number(node.x || 0);
          if (node._compact_y === undefined) node._compact_y = Number(node.y || 0);
          const targetPos = expanded.layout.get(node.id);
          if (targetPos) {
            node.x = targetPos.x;
            node.y = targetPos.y;
          }
        });
      }
      target.classList.toggle("graph-expanded", memoryGraphExpanded);
      const nodes = new Map((memory.nodes || []).map(n => [n.id, n]));
      const edges = uniqueDirectedEdges(memory.edges || []).filter(e => nodes.has(e.source) && nodes.has(e.target));
      const edgeHtml = edges.map(e => {
        const source = nodes.get(e.source);
        const targetNode = nodes.get(e.target);
        const cls = Number(e.direction || 1) >= 0 ? "support" : "contradict";
        const relation = edgeRelationLabel(e);
        const label = relation.length > 18 ? `${relation.slice(0, 17)}...` : relation;
        const labelWidth = Math.max(48, Math.min(124, label.length * 6.2 + 14));
        const labelPos = edgeLabelPosition(source, targetNode);
        const points = edgeRenderPoints(source, targetNode);
        const marker = cls === "support" ? "arrow-support" : "arrow-contradict";
        const title = `${relation} w=${e.weight ?? ""}`;
        return `<g class="belief-edge-wrap">
          <line class="belief-edge ${cls}" data-source="${esc(e.source)}" data-target="${esc(e.target)}" x1="${esc(points.x1)}" y1="${esc(points.y1)}" x2="${esc(points.x2)}" y2="${esc(points.y2)}" marker-end="url(#${marker})"><title>${esc(title)}</title></line>
          <g class="belief-edge-label ${cls}" data-source="${esc(e.source)}" data-target="${esc(e.target)}" transform="translate(${esc(labelPos.x)}, ${esc(labelPos.y)})">
            <rect x="${esc(-labelWidth / 2)}" y="-8.5" width="${esc(labelWidth)}" height="17"></rect>
            <text y="0">${esc(label)}</text>
            <title>${esc(title)}</title>
          </g>
        </g>`;
      }).join("");
      const nodeHtml = (memory.nodes || []).map(n => {
        const label = String(n.label || n.id || "").length > 20 ? `${String(n.label || n.id).slice(0, 19)}...` : String(n.label || n.id || "");
        const posterior = n.posterior !== undefined ? `${Math.round(Number(n.posterior) * 100)}%` : "";
        return `<g class="belief-node ${esc(n.type || "")}" data-node-id="${esc(n.id)}" transform="translate(${esc(n.x)}, ${esc(n.y)})">
          <circle r="24"></circle>
          <text text-anchor="middle" y="-31">${esc(n.type || "")}</text>
          <text text-anchor="middle" y="4">${esc(label)}</text>
          <text text-anchor="middle" y="18">${esc(posterior)}</text>
          <title>${esc(n.type || "Node")}: ${esc(n.label || n.id)}\nposterior=${esc(n.posterior ?? "")}\nstatus=${esc(n.status || "")}</title>
        </g>`;
      }).join("");
      const canvas = memoryGraphExpanded ? expandedGraphCanvas(memory) : {width: 640, height: 420};
      target.innerHTML = `<div class="memory-graph-tools">
        <button type="button" id="memoryGraphPin" class="memory-graph-tool memory-graph-pin" aria-label="Pin graph drag" aria-pressed="${memoryGraphPinned ? "true" : "false"}" title="${memoryGraphPinned ? "Unpin graph drag" : "Pin graph drag"}">
          <span class="memory-graph-pin-icon" aria-hidden="true"></span>
        </button>
        <button type="button" id="memoryGraphScatter" class="memory-graph-tool memory-graph-scatter" aria-label="Scatter transform graph" aria-pressed="${memoryGraphTransformMode !== "original" ? "true" : "false"}" title="Scatter to star topology">
          <span class="memory-graph-scatter-icon" aria-hidden="true"></span>
        </button>
        <button type="button" id="memoryGraphExpand" class="memory-graph-tool memory-graph-expand" aria-label="Expand graph" aria-pressed="${memoryGraphExpanded ? "true" : "false"}" title="${memoryGraphExpanded ? "Dock graph" : "Expand graph"}">
          <span class="memory-graph-expand-icon" aria-hidden="true"></span>
        </button>
      </div>
      <svg class="belief-svg" viewBox="0 0 ${esc(canvas.width)} ${esc(canvas.height)}" role="img" aria-label="Belief memory topology">
        <defs>
          <marker id="arrow-support" markerWidth="5" markerHeight="5" refX="4.6" refY="2.5" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,5 L5,2.5 z" fill="var(--forest-green)"></path>
          </marker>
          <marker id="arrow-contradict" markerWidth="5" markerHeight="5" refX="4.6" refY="2.5" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,5 L5,2.5 z" fill="var(--bad)"></path>
          </marker>
        </defs>
        ${edgeHtml}
        ${nodeHtml}
      </svg>`;
      const svg = target.querySelector("svg");
      buildGraphDomCache();
      if (svg) bindGraphDrag(svg);
      const pinBtn = document.getElementById("memoryGraphPin");
      if (pinBtn) pinBtn.addEventListener("click", () => setMemoryGraphPinned(!memoryGraphPinned));
      const scatterBtn = document.getElementById("memoryGraphScatter");
      if (scatterBtn) scatterBtn.addEventListener("click", () => cycleMemoryGraphTransform());
      const expandBtn = document.getElementById("memoryGraphExpand");
      if (expandBtn) expandBtn.addEventListener("click", () => setMemoryGraphExpanded(!memoryGraphExpanded));
      syncGraphExpandButton();
    }
    function renderMemoryPanel() {
      const memory = currentBeliefMemory;
      const title = document.getElementById("memoryTitle");
      const subtitle = document.getElementById("memorySubtitle");
      const mode = document.getElementById("memoryMode");
      const select = document.getElementById("memoryStepSelect");
      const inspector = document.getElementById("memoryInspector");
      const sources = document.getElementById("memorySources");
      const syncStatus = document.getElementById("memorySyncStatus");
      if (!title || !subtitle || !mode || !select || !inspector || !sources) return;
      if (!memory) {
        title.textContent = "Graph Memory";
        subtitle.textContent = "Select a trajectory sample.";
        mode.textContent = "mock";
        select.innerHTML = `<option value="all">All steps</option>`;
        inspector.textContent = "No graph memory selected.";
        sources.innerHTML = "";
        if (syncStatus) {
          syncStatus.textContent = "";
          syncStatus.className = "memory-sync-status";
        }
        renderBeliefGraph(null);
        return;
      }
      if (syncStatus) {
        syncStatus.textContent = memorySyncStatus;
        syncStatus.className = `memory-sync-status ${memorySyncStatusKind}`;
      }
      title.textContent = memory.title || "Graph Memory";
      subtitle.textContent = memory.description || "";
      mode.textContent = memory.mode || "mock";
      const stepOptions = [`<option value="all">All steps</option>`].concat(
        memorySteps(memory).map(s => `<option value="${esc(s.step_index)}">Step ${esc(s.step_index)}</option>`)
      );
      select.innerHTML = stepOptions.join("");
      select.value = selectedMemoryStep;
      const summary = memory.summary || {};
      const selected = memoryStepData(memory, selectedMemoryStep);
      const sourceEntries = memorySourceEntries(memory, selectedMemoryStep);
      inspector.innerHTML = `
        <div><strong>${esc(summary.decision ? `decision: ${summary.decision}` : "belief state")}</strong></div>
        <div class="muted">claims=${count(summary.claims)} belief_variables=${count(summary.belief_variables)} evidence=${count(summary.evidence)} confidence=${pct(summary.confidence)}</div>
        <div class="muted">step=${esc(selectedMemoryStep)} posterior_delta=${esc(selected.posterior_delta ?? "n/a")} lab=${esc(memory.memgraph_lab_url || "")}</div>
      `;
      sources.innerHTML = sourceEntries.length
        ? `<div class="turn-label">support spans</div>${sourceEntries.map(s => {
            const width = confidenceWidth(s.confidence);
            return `<button
              type="button"
              title="${esc(s.node_id || "")}"
              data-memory-key="${esc(currentMemoryKey)}"
              data-step-index="${esc(s.step_index)}"
              data-node-id="${esc(s.node_id || "")}"
              data-text="${esc(s.text)}">
              <span class="memory-source-fill" style="width:${esc(width)}%"></span>
              <span class="memory-source-text"
                data-memory-key="${esc(currentMemoryKey)}"
                data-step-index="${esc(s.step_index)}"
                data-node-id="${esc(s.node_id || "")}"
                data-text="${esc(s.text)}">
                <span>${esc(s.text)}</span><span class="muted">${pct(s.confidence)}</span>
              </span>
            </button>`;
          }).join("")}`
        : `<div class="muted">No highlighted support spans for this step.</div>`;
      renderBeliefGraph(memory);
    }
    function buildTrajectoryBlocks(sample) {
      const messages = sample.trajectory || [];
      const modelSteps = sample.model_steps || [];
      const used = new Set();
      const systemIndex = messages.findIndex(m => (m.role || "") === "system");
      const userIndex = messages.findIndex(m => (m.role || "") === "user");
      const blocks = {
        system: systemIndex >= 0 ? messages[systemIndex] : null,
        user: userIndex >= 0 ? messages[userIndex] : null,
        steps: [],
        extras: [],
      };
      if (systemIndex >= 0) used.add(systemIndex);
      if (userIndex >= 0) used.add(userIndex);

      let current = null;
      let assistantStep = 0;
      for (let i = 0; i < messages.length; i += 1) {
        if (used.has(i)) continue;
        const msg = messages[i] || {};
        const role = msg.role || "message";
        if (role === "assistant") {
          current = {
            index: assistantStep,
            assistant: msg,
            tools: [],
            extras: [],
            meta: modelSteps[assistantStep] || {},
          };
          blocks.steps.push(current);
          assistantStep += 1;
        } else if (role === "tool" && current) {
          current.tools.push(msg);
        } else if (current) {
          current.extras.push(msg);
        } else {
          blocks.extras.push(msg);
        }
      }

      while (assistantStep < modelSteps.length) {
        blocks.steps.push({
          index: assistantStep,
          assistant: null,
          tools: [],
          extras: [],
          meta: modelSteps[assistantStep] || {},
        });
        assistantStep += 1;
      }
      return blocks;
    }
    function renderStepTokens(tokens) {
      if (!tokens || typeof tokens !== "object") return "";
      const flag = (label, value) => {
        if (value === undefined || value === null) return "";
        const cls = value ? "ok" : "no";
        return `<span class="flag ${cls}">${esc(label)}: ${value ? "yes" : "no"}</span>`;
      };
      const ids = [];
      if (tokens.think_start_token_id !== undefined && tokens.think_start_token_id !== null) {
        ids.push(`start_id=${esc(tokens.think_start_token_id)}`);
      }
      if (tokens.think_end_token_id !== undefined && tokens.think_end_token_id !== null) {
        ids.push(`end_id=${esc(tokens.think_end_token_id)}`);
      }
      const flags = [
        flag("prompt_start", tokens.prompt_has_think_start),
        flag("prompt_end", tokens.prompt_has_think_end),
        flag("completion_start", tokens.completion_has_think_start),
        flag("completion_has_think_end" in tokens ? "completion_end" : "", tokens.completion_has_think_end),
      ].filter(Boolean);
      const parts = ids.map(esc).concat(flags);
      if (!parts.length) return "";
      return `<div class="step-tokens"><span class="k">qwen tokens</span>${parts.map(p => `<span>${p}</span>`).join("")}</div>`;
    }
    function renderStepCard(block, idx, memory = null) {
      const step = block.meta || {};
      const stepIndex = step.step_index ?? block.index ?? idx;
      const highlights = memoryStepHighlights(memory, stepIndex);
      const fields = [
        ["prompt_tokens", step.prompt_tokens ?? step.prompt_length],
        ["completion_tokens", step.completion_tokens ?? step.completion_length],
        ["reasoning_tokens", step.reasoning_tokens ?? step.reasoning_chars],
        ["action_tokens", step.content_tokens ?? step.content_chars],
      ];
      const grid = fields
        .filter(([, v]) => v !== undefined && v !== null)
        .map(([k, v]) => `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`)
        .join("");
      const finish = step.finish_reason ? `<span class="finish">finish_reason: ${esc(step.finish_reason)}</span>` : "";
      const previewBlock = (label, text) => {
        if (!text) return "";
        return `<div class="step-preview"><div class="k">${esc(label)}</div><pre>${esc(text)}</pre></div>`;
      };
      const assistantAndTools = [
        renderAssistantSections(block.assistant, highlights),
        ...(block.tools || []).map(renderToolResponse),
        ...(block.extras || []).map(renderMessage),
      ].join("");
      return `<details class="step" data-step-index="${esc(stepIndex)}" ${idx === 0 ? "open" : ""}>
        <summary class="step-head">
          <span class="idx">Step ${count(stepIndex)}</span>
          ${finish}
          ${memory ? `<button type="button" class="memory-open" data-memory-key="${esc(currentRenderingMemoryKey || "")}" data-step-index="${esc(stepIndex)}" title="Open graph memory for this step">memory</button>` : ""}
          <button type="button" class="step-raw-toggle" title="Show raw content">raw</button>
        </summary>
        <div class="step-body">
          ${grid ? `<div class="step-grid">${grid}</div>` : ""}
          ${assistantAndTools}
          ${renderStepTokens(step.qwen_thinking_tokens)}
          ${previewBlock("reasoning preview", !block.assistant ? step.reasoning_preview : "")}
          ${previewBlock("text preview", !block.assistant ? step.text_preview : "")}
        </div>
      </details>`;
    }
    function renderTrajectoryBlocks(sample, sampleMetadataHtml = "", memory = null) {
      const blocks = buildTrajectoryBlocks(sample);
      const promptBlocks = [
        renderPromptBlock("System prompt", blocks.system, true),
        sampleMetadataHtml,
        renderPromptBlock("User prompt", blocks.user, true),
        ...blocks.extras.map((msg, idx) => renderPromptBlock(`Context ${idx + 1}`, msg, false)),
      ].join("");
      const steps = blocks.steps.length
        ? `<div class="steps">${blocks.steps.map((block, idx) => renderStepCard(block, idx, memory)).join("")}</div>`
        : "";
      return promptBlocks + steps;
    }
    function renderSampleMetadata(sample, idx) {
      const modelSteps = sample.model_steps || [];
      const sumStepMetric = (...keys) => modelSteps.reduce((total, step) => {
        for (const key of keys) {
          const value = step && step[key];
          if (value !== undefined && value !== null && value !== "") return total + Number(value || 0);
        }
        return total;
      }, 0);
      const promptTokens = sumStepMetric("prompt_tokens", "prompt_length");
      const generatedTokens = sumStepMetric("generated_tokens", "completion_tokens", "completion_length");
      const reasoningTokens = sumStepMetric("reasoning_tokens", "reasoning_length");
      const actionTokens = sumStepMetric("action_tokens", "content_tokens", "content_length");
      const elapsedSeconds = Number(sample.elapsed_seconds);
      const hasElapsed = Number.isFinite(elapsedSeconds);
      const elapsedTime = hasElapsed ? renderSeconds(elapsedSeconds) : "-";
      const completedTime = sample.completed_iso || "-";
      const fields = [
        ["prompt_tokens", promptTokens],
        ["generated_tokens", generatedTokens],
        ["reasoning_tokens", reasoningTokens],
        ["action_tokens", actionTokens],
        ["elapsed_time", elapsedTime],
        ["completed_time", completedTime],
        ["sample_index", sample.sample_index ?? idx],
        ["is_correct", Boolean(sample.is_correct)],
        ["num_steps", sample.num_steps],
        ["termination_reason", sample.termination_reason],
        ["exact_match", sample.exact_match],
        ["f1_score", sample.f1_score],
        ["extracted_answer", sample.extracted_answer],
        ["trajectory_id", sample.trajectory_id],
      ]
        .filter(([, value]) => value !== undefined && value !== null && value !== "")
        .map(([key, value]) => {
          const rendered = typeof value === "object" ? JSON.stringify(value) : String(value);
          return `<div class="kv"><span class="k">${esc(key)}</span><span class="v">${esc(rendered)}</span></div>`;
        })
        .join("");
      if (!fields) return "";
      return `<div class="sample-meta">
        <div class="turn-label">sample metadata</div>
        <div class="sample-meta-grid">${fields}</div>
      </div>`;
    }
    function renderSample(sample, idx, recordIndex = 0, record = {}) {
      const memory = sample.belief_memory || null;
      const memoryKey = memory ? `${record.problem_id || record.task_id || recordIndex}:${sample.trajectory_id || sample.sample_index || idx}` : "";
      if (memory && memoryKey) memory.memory_key = memoryKey;
      if (memoryKey) memorySamples.set(memoryKey, memory);
      currentRenderingMemoryKey = memoryKey;
      const metadata = renderSampleMetadata(sample, idx);
      const organized = renderTrajectoryBlocks(sample, metadata, memory);
      currentRenderingMemoryKey = "";
      const sampleCorrect = Boolean(sample.is_correct);
      const sampleMark = sampleCorrect
        ? `<span class="sample-mark correct" aria-label="correct sample" title="correct">✓</span>`
        : `<span class="sample-mark incorrect" aria-label="incorrect sample" title="incorrect">✗</span>`;
      return `<details class="sample" ${idx === 0 ? "open" : ""}>
        <summary>
          <span>Sample ${idx}</span>
          ${sampleMark}
          <span class="muted">idx=${count(sample.sample_index ?? idx)} steps=${count(sample.num_steps)} termination=${esc(sample.termination_reason || "")}</span>
          ${memory ? `<button type="button" class="memory-open" data-memory-key="${esc(memoryKey)}" data-step-index="all" title="Open graph memory">graph memory</button>` : ""}
        </summary>
        <div class="detail-body">
          ${organized || `<div class="empty">No trajectory messages saved.</div>`}
        </div>
      </details>`;
    }
    function registerSampleMemory(sample, idx, recordIndex = 0, record = {}) {
      const memory = sample.belief_memory || null;
      const memoryKey = memory ? `${record.problem_id || record.task_id || recordIndex}:${sample.trajectory_id || sample.sample_index || idx}` : "";
      if (memory && memoryKey) {
        memory.memory_key = memoryKey;
        memorySamples.set(memoryKey, memory);
        memoryLocationIndex.set(memoryKey, {recordIndex, sampleIndex: idx});
      }
      return memoryKey;
    }
    function renderSampleShell(sample, idx, recordIndex = 0, record = {}) {
      const memoryKey = registerSampleMemory(sample, idx, recordIndex, record);
      const sampleCorrect = Boolean(sample.is_correct);
      const sampleMark = sampleCorrect
        ? `<span class="sample-mark correct" aria-label="correct sample" title="correct">✓</span>`
        : `<span class="sample-mark incorrect" aria-label="incorrect sample" title="incorrect">✗</span>`;
      return `<details class="sample" data-record-index="${esc(recordIndex)}" data-sample-index="${esc(idx)}">
        <summary>
          <span>Sample ${idx}</span>
          ${sampleMark}
          <span class="muted">idx=${count(sample.sample_index ?? idx)} steps=${count(sample.num_steps)} termination=${esc(sample.termination_reason || "")}</span>
          ${memoryKey ? `<button type="button" class="memory-open" data-memory-key="${esc(memoryKey)}" data-step-index="all" title="Open graph memory">graph memory</button>` : ""}
        </summary>
        <div class="detail-body lazy-sample-body"><div class="muted">Open sample to render trajectory.</div></div>
      </details>`;
    }
    function hydrateRecordDetails(recordDetails) {
      if (!currentRecordsPayload || recordDetails.dataset.rendered === "1") return;
      const recordIndex = Number(recordDetails.dataset.recordIndex);
      const record = currentRecordsPayload.records?.[recordIndex];
      if (!record) return;
      recordDetails.dataset.rendered = "1";
      const samplesHtml = (record.samples || [])
        .map((sample, sampleIdx) => renderSampleShell(sample, sampleIdx, recordIndex, record))
        .join("");
      recordDetails.innerHTML = `
        <div class="question">${esc(record.question || "")}</div>
        <div class="muted">ground truth: ${esc(JSON.stringify(record.ground_truth ?? ""))}</div>
        ${samplesHtml}
      `;
    }
    function hydrateSampleDetails(sampleDetails) {
      if (!currentRecordsPayload || sampleDetails.dataset.rendered === "1") return;
      const sampleEl = sampleDetails.closest("details.sample");
      if (!sampleEl) return;
      const recordIndex = Number(sampleEl.dataset.recordIndex);
      const sampleIndex = Number(sampleEl.dataset.sampleIndex);
      const record = currentRecordsPayload.records?.[recordIndex];
      const sample = record?.samples?.[sampleIndex];
      if (!record || !sample) return;
      sampleDetails.dataset.rendered = "1";
      const memory = sample.belief_memory || null;
      const memoryKey = registerSampleMemory(sample, sampleIndex, recordIndex, record);
      if (memoryKey) currentRenderingMemoryKey = memoryKey;
      const metadata = renderSampleMetadata(sample, sampleIndex);
      sampleDetails.innerHTML = renderTrajectoryBlocks(sample, metadata, memory) || `<div class="empty">No trajectory messages saved.</div>`;
      currentRenderingMemoryKey = "";
    }
    function nextFrame() {
      return new Promise(resolve => requestAnimationFrame(() => resolve()));
    }
    async function ensureMemorySampleRendered(memoryKey) {
      const location = memoryLocationIndex.get(memoryKey);
      if (!location) return null;
      const record = document.querySelector(`details.record[data-record-index="${CSS.escape(String(location.recordIndex))}"]`);
      if (!record) return null;
      if (!record.open) {
        record.open = true;
        await nextFrame();
      }
      const recordBody = record.querySelector(".record-body");
      if (recordBody) hydrateRecordDetails(recordBody);
      await nextFrame();
      const sample = record.querySelector(`details.sample[data-record-index="${CSS.escape(String(location.recordIndex))}"][data-sample-index="${CSS.escape(String(location.sampleIndex))}"]`);
      if (!sample) return null;
      if (!sample.open) {
        sample.open = true;
        await nextFrame();
      }
      const sampleBody = sample.querySelector(".lazy-sample-body");
      if (sampleBody) hydrateSampleDetails(sampleBody);
      await nextFrame();
      return sample;
    }
    function triggerHighlightGlow(mark) {
      if (!mark) return;
      mark.classList.remove("navigate-glow");
      void mark.offsetWidth;
      mark.classList.add("navigate-glow");
      setTimeout(() => mark.classList.remove("navigate-glow"), 3000);
    }
    async function navigateToSupportSpan(memoryKey, stepIndex, text, nodeId = "") {
      const sample = await ensureMemorySampleRendered(memoryKey);
      if (!sample) return;
      const step = sample.querySelector(`details.step[data-step-index="${CSS.escape(String(stepIndex))}"]`);
      if (step && !step.open) {
        step.open = true;
        await nextFrame();
      }
      const scope = step || sample;
      let candidates = Array.from(scope.querySelectorAll("mark.belief-highlight"));
      if (nodeId) {
        const byNode = candidates.filter(mark => String(mark.dataset.nodeId || "") === String(nodeId));
        if (byNode.length) candidates = byNode;
      }
      const needle = String(text || "").trim().toLowerCase();
      const target = candidates.find(mark => mark.textContent.trim().toLowerCase() === needle)
        || candidates.find(mark => mark.textContent.toLowerCase().includes(needle))
        || candidates[0];
      if (!target) return;
      target.scrollIntoView({behavior: "smooth", block: "center", inline: "nearest"});
      triggerHighlightGlow(target);
    }
    function bindLazyRecordRendering(target) {
      if (target.dataset.lazyBound === "1") return;
      target.dataset.lazyBound = "1";
      target.addEventListener("toggle", ev => {
        const record = ev.target.closest?.("details.record");
        if (record && ev.target === record && record.open) {
          hydrateRecordDetails(record.querySelector(".record-body"));
          return;
        }
        const sample = ev.target.closest?.("details.sample");
        if (sample && sample.open) hydrateSampleDetails(sample.querySelector(".lazy-sample-body"));
      }, true);
    }
    function renderRecords(payload) {
      const records = payload.records || [];
      const target = document.getElementById("records");
      currentRecordsPayload = payload;
      memorySamples = new Map();
      memoryLocationIndex = new Map();
      if (!records.length) {
        target.className = "empty";
        target.innerHTML = "No records found in this result file.";
        currentBeliefMemory = null;
        currentMemoryKey = "";
        renderMemoryPanel();
        return;
      }
      target.className = "";
      for (const [idx, record] of records.entries()) {
        for (const [sampleIdx, sample] of (record.samples || []).entries()) {
          registerSampleMemory(sample, sampleIdx, idx, record);
        }
      }
      target.innerHTML = records.map((record, idx) => `
        <details class="record" data-record-index="${esc(idx)}">
          <summary>
            ${esc(record.task_id || record.problem_id || `Problem ${idx + 1}`)}
            <span class="muted">${count(record.num_correct)} / ${count(record.num_samples)} correct</span>
          </summary>
          <div class="detail-body record-body" data-record-index="${esc(idx)}"><div class="muted">Open problem to render samples.</div></div>
        </details>
      `).join("");
      bindLazyRecordRendering(target);
      if (!currentMemoryKey || !memorySamples.has(currentMemoryKey)) {
        const firstKey = memorySamples.keys().next().value || "";
        if (firstKey) {
          currentMemoryKey = firstKey;
          currentBeliefMemory = memorySamples.get(firstKey);
          selectedMemoryStep = "all";
        } else {
          currentBeliefMemory = null;
          currentMemoryKey = "";
        }
      }
      renderMemoryPanel();
    }
    async function loadResult(path) {
      selectedPath = path;
      selectedKind = "result";
      renderSidebarRuns(stateCache);
      const key = payloadKey("result", path);
      document.getElementById("records").className = "empty";
      document.getElementById("records").textContent = payloadCache.has(key) ? "Opening cached result..." : "Loading result file...";
      if (!payloadCache.has(key)) await new Promise(resolve => requestAnimationFrame(() => resolve()));
      const payload = await loadPayloadCached("result", path);
      renderSummary(payload);
      renderRecords(payload);
    }
    async function loadStream(path) {
      selectedPath = path;
      selectedKind = "stream";
      renderSidebarRuns(stateCache);
      const key = payloadKey("stream", path);
      document.getElementById("records").className = "empty";
      document.getElementById("records").textContent = payloadCache.has(key) ? "Opening cached trajectories..." : "Loading live trajectories...";
      if (!payloadCache.has(key)) await new Promise(resolve => requestAnimationFrame(() => resolve()));
      const payload = await loadPayloadCached("stream", path);
      renderSummary(payload);
      renderRecords(payload);
    }
    async function refresh(options = {}) {
      const reloadSelected = Boolean(options.reloadSelected);
      if (refreshInFlight) return;
      refreshInFlight = true;
      const refreshButton = document.getElementById("refresh");
      refreshButton.disabled = true;
      try {
        stateCache = await fetchJson("/api/state");
        document.getElementById("clock").textContent = stateCache.generated_iso || "";
        renderSidebarRuns(stateCache);
        if (!reloadSelected) return;
        if (selectedKind === "stream" && selectedPath) {
          await loadStream(selectedPath);
        } else if (selectedKind === "result" && selectedPath) {
          await loadResult(selectedPath);
        } else if (!selectedPath && stateCache.streams?.length) {
          await loadStream(stateCache.streams[0].path);
        } else if (!selectedPath && stateCache.results?.length) {
          await loadResult(stateCache.results[0].path);
        }
      } catch (err) {
        document.getElementById("records").className = "empty";
        document.getElementById("records").textContent = `UI error: ${err.message}`;
      } finally {
        refreshInFlight = false;
        refreshButton.disabled = false;
      }
    }
    document.getElementById("refresh").addEventListener("click", () => refresh({reloadSelected: true}));
    refresh();
  </script>
</body>
</html>
"""


class BeliefTracerUiHandler(BaseHTTPRequestHandler):
    config: UiConfig

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _send_error_json(self, exc: Exception, status: HTTPStatus) -> None:
        self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                body = HTML.replace(
                    "__MAX_TRAJECTORIES__", str(self.config.max_trajectories)
                )
                self._send(body.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path in {"/favicon.ico", "/favicon.svg"}:
                self._send(_favicon_asset_path().read_bytes(), "image/svg+xml")
                return
            if parsed.path == "/api/state":
                self._send_json(
                    _scan_artifacts(self.config.artifacts_dir, self.config.max_results)
                )
                return
            if parsed.path == "/api/run/config":
                query = parse_qs(parsed.query)
                state_path = _resolve_artifact_path(
                    self.config.artifacts_dir, query.get("path", [""])[0]
                )
                self._send_json(_read_run_config_for_state(state_path))
                return
            if parsed.path == "/api/result":
                query = parse_qs(parsed.query)
                result_path = _resolve_artifact_path(
                    self.config.artifacts_dir, query.get("path", [""])[0]
                )
                payload = _json_read(result_path)
                if isinstance(payload, dict):
                    payload["_path"] = str(result_path)
                    payload["_mtime_iso"] = _iso(result_path.stat().st_mtime)
                    _attach_mock_belief_memory(payload)
                self._send_json(payload)
                return
            if parsed.path == "/api/stream":
                query = parse_qs(parsed.query)
                stream_path = _resolve_artifact_path(
                    self.config.artifacts_dir, query.get("path", [""])[0]
                )
                limit_raw = query.get("limit", [str(self.config.max_trajectories)])[0]
                try:
                    limit = max(1, min(2000, int(limit_raw)))
                except ValueError:
                    limit = self.config.max_trajectories
                self._send_json(_attach_mock_belief_memory(_read_stream_payload(stream_path, limit)))
                return
            content_type = mimetypes.guess_type(parsed.path)[0] or "text/plain"
            self._send(b"not found", content_type, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._send_error_json(exc, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_error_json(exc, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_error_json(exc, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/memory/update":
                self._handle_memory_update(payload)
                return
            if parsed.path == "/api/rename":
                self._handle_rename(payload)
                return
            if parsed.path == "/api/meta":
                self._handle_meta(payload)
                return
            if parsed.path == "/api/run/stop":
                self._handle_stop(payload)
                return
            if parsed.path == "/api/run/delete":
                self._handle_delete(payload)
                return
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._send_error_json(exc, HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._send_error_json(exc, HTTPStatus.FORBIDDEN)
        except ValueError as exc:
            self._send_error_json(exc, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_error_json(exc, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        return data

    def _resolve_target(self, payload: dict[str, Any]) -> Path:
        raw_path = str(payload.get("path") or "")
        if not raw_path:
            raise ValueError("missing path")
        return _resolve_artifact_path(self.config.artifacts_dir, raw_path)

    def _handle_memory_update(self, payload: dict[str, Any]) -> None:
        memory = payload.get("memory")
        if memory is not None and not isinstance(memory, dict):
            raise ValueError("memory must be an object")
        memory_key = _memgraph_memory_key(
            str(payload.get("memory_key") or "") or None,
            memory if isinstance(memory, dict) else None,
        )
        write_current = bool(payload.get("write_current"))
        if write_current and isinstance(memory, dict) and memory.get("nodes"):
            memory["memory_key"] = memory_key
            updated = _memgraph_upsert_belief_memory(memory_key, memory)
        else:
            try:
                updated = _memgraph_fetch_belief_memory(memory_key)
                if int(updated.pop("_deduped_edge_count", 0) or 0) > 0:
                    updated = _memgraph_upsert_belief_memory(memory_key, updated)
            except FileNotFoundError:
                if not isinstance(memory, dict) or not memory.get("nodes"):
                    raise
                memory["memory_key"] = memory_key
                updated = _memgraph_upsert_belief_memory(memory_key, memory)
        self._send_json(
            {
                "ok": True,
                "memory_key": memory_key,
                "memgraph_uri": _memgraph_uri(),
                "memory": updated,
            }
        )

    def _handle_rename(self, payload: dict[str, Any]) -> None:
        target = self._resolve_target(payload)
        root = self.config.artifacts_dir.resolve()
        meta = _load_meta(root)
        key = str(target)
        entry = dict(meta.get(key) or _empty_meta_entry())
        new_name = payload.get("name")
        if new_name is None or str(new_name).strip() == "":
            entry["display_name"] = ""
        else:
            entry["display_name"] = str(new_name).strip()[:200]
        if _meta_is_empty(entry):
            meta.pop(key, None)
        else:
            meta[key] = entry
        _save_meta(root, meta)
        self._send_json({"path": key, **entry})

    def _handle_meta(self, payload: dict[str, Any]) -> None:
        target = self._resolve_target(payload)
        root = self.config.artifacts_dir.resolve()
        meta = _load_meta(root)
        key = str(target)
        entry = dict(meta.get(key) or _empty_meta_entry())
        if "display_name" in payload:
            name = payload.get("display_name")
            entry["display_name"] = str(name).strip()[:200] if name else ""
        if "notes" in payload:
            notes = payload.get("notes")
            entry["notes"] = str(notes)[:4000] if notes else ""
        if "tags" in payload:
            tags = payload.get("tags")
            if isinstance(tags, list):
                entry["tags"] = [str(t).strip()[:60] for t in tags if str(t).strip()][:32]
            elif isinstance(tags, str):
                entry["tags"] = [t.strip()[:60] for t in tags.split(",") if t.strip()][:32]
            elif tags is None:
                entry["tags"] = []
        if _meta_is_empty(entry):
            meta.pop(key, None)
        else:
            meta[key] = entry
        _save_meta(root, meta)
        self._send_json({"path": key, **entry})

    def _handle_stop(self, payload: dict[str, Any]) -> None:
        target = self._resolve_target(payload)
        if target.name != "run_state.json":
            raise ValueError("stop is only valid for run_state.json entries")
        try:
            state = _json_read(target)
        except Exception as exc:
            raise ValueError(f"could not read run state: {exc}") from exc
        pid_raw = state.get("pid") if isinstance(state, dict) else None
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("run_state.json has no valid pid") from exc
        if pid <= 0:
            raise ValueError("run_state.json has no valid pid")
        sig_name = str(payload.get("signal") or "TERM").upper()
        try:
            sig = getattr(signal, f"SIG{sig_name}")
        except AttributeError as exc:
            raise ValueError(f"unsupported signal: {sig_name}") from exc
        method, error = self._send_signal(pid, sig)
        result = {"path": str(target), "pid": pid, "method": method, "signal": sig_name}
        if error:
            result["error"] = error
        if method:
            self._mark_run_stopping(target, state)
        self._send_json(result)

    def _send_signal(self, pid: int, sig: int) -> tuple[str | None, str | None]:
        try:
            os.kill(pid, sig)
            return "host", None
        except ProcessLookupError:
            host_err: str | None = "no such process on host"
        except PermissionError as exc:
            host_err = f"host kill denied: {exc}"
        except OSError as exc:
            host_err = f"host kill failed: {exc}"
        container = (self.config.container_name or "").strip()
        if not container:
            return None, host_err
        try:
            proc = subprocess.run(
                ["docker", "exec", container, "kill", f"-{sig}", str(pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            return None, f"{host_err}; docker CLI not installed"
        except subprocess.TimeoutExpired:
            return None, f"{host_err}; docker exec timed out"
        if proc.returncode == 0:
            return f"docker:{container}", None
        stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
        return None, f"{host_err}; docker exec rc={proc.returncode}: {stderr}"

    def _mark_run_stopping(self, target: Path, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            state = dict(state)
            state["status"] = "stopping"
            state["updated_at"] = time.time()
            tmp = target.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            tmp.replace(target)
        except Exception:
            return

    def _handle_delete(self, payload: dict[str, Any]) -> None:
        target = self._resolve_target(payload)
        if not bool(payload.get("confirm")):
            raise ValueError("delete requires confirm: true")
        root = self.config.artifacts_dir.resolve()
        if target.name == "run_state.json":
            run_dir = target.parent.resolve()
            if run_dir == root or root not in run_dir.parents:
                raise ValueError("refusing to delete artifacts root")
            shutil.rmtree(run_dir)
            removed = str(run_dir)
            kind = "run"
        else:
            target.unlink()
            removed = str(target)
            kind = "file"
        meta = _load_meta(root)
        removed_path = Path(removed)
        cleaned = False
        for key in list(meta.keys()):
            try:
                kp = Path(key).resolve()
            except Exception:
                continue
            if kp == removed_path or removed_path in kp.parents:
                meta.pop(key, None)
                cleaned = True
        if cleaned:
            _save_meta(root, meta)
        self._send_json({"removed": removed, "kind": kind})


class BeliefTracerUiServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _fetch_ui_state(host: str, port: int, timeout: float = 0.75) -> dict[str, Any] | None:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", "/api/state")
        resp = conn.getresponse()
        if resp.status != HTTPStatus.OK:
            return None
        body = resp.read()
        data = json.loads(body.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except OSError:
        return None
    except Exception:
        return None
    finally:
        conn.close()


def _ui_is_running(host: str, port: int) -> bool:
    return _fetch_ui_state(host, port) is not None


def ensure_ui_running(
    *,
    artifacts_dir: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    wait_seconds: float = 15.0,
) -> bool:
    """Start the monitor UI in the background if no healthy UI is listening."""
    if os.environ.get("BELIEF_TRACER_AUTO_UI", "").lower() in {"0", "false", "no"}:
        return False

    root = artifacts_dir.resolve()
    state = _fetch_ui_state(host, port)
    if state is not None:
        running_root = state.get("root")
        if running_root and Path(str(running_root)).resolve() != root:
            print(
                "BeliefTracer UI already running at "
                f"http://{host}:{port}, but it monitors {running_root} "
                f"instead of {root}.",
                file=sys.stderr,
            )
        else:
            print(f"BeliefTracer UI already running: http://{host}:{port}")
        return True

    root.mkdir(parents=True, exist_ok=True)
    AUTO_UI_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = AUTO_UI_LOG.open("wb", buffering=0)
    cmd = [
        sys.executable,
        "-m",
        "bcg.agent.cli",
        "ui",
        "--host",
        host,
        "--port",
        str(port),
        "--artifacts-dir",
        str(root),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(Path.cwd()),
        )
    except Exception as exc:
        log.close()
        print(f"Could not auto-start BeliefTracer UI: {exc}", file=sys.stderr)
        return False
    finally:
        try:
            log.close()
        except Exception:
            pass

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            print(
                "BeliefTracer UI exited during startup; see "
                f"{AUTO_UI_LOG} for details.",
                file=sys.stderr,
            )
            return False
        if _ui_is_running(host, port):
            print(f"BeliefTracer UI started: http://{host}:{port}")
            print(f"BeliefTracer UI log: {AUTO_UI_LOG}")
            return True
        time.sleep(0.2)

    print(f"BeliefTracer UI starting in background: http://{host}:{port}")
    print(f"BeliefTracer UI log: {AUTO_UI_LOG}")
    return True


def serve(config: UiConfig) -> None:
    root = config.artifacts_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    class _ConfiguredHandler(BeliefTracerUiHandler):
        pass

    _ConfiguredHandler.config = UiConfig(
        host=config.host,
        port=config.port,
        artifacts_dir=root,
        poll_seconds=config.poll_seconds,
        max_results=config.max_results,
        max_trajectories=config.max_trajectories,
        container_name=config.container_name,
    )
    httpd = BeliefTracerUiServer((config.host, config.port), _ConfiguredHandler)
    print(f"BeliefTracer UI: http://{config.host}:{config.port}")
    print(f"Artifacts root: {root}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BeliefTracer UI")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = RichArgumentParser(
        prog=prog or "bcg agent ui",
        description="Serve the BeliefTracer trajectory monitor web UI.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="HTTP bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP bind port")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/belief_tracer",
        help="BeliefTracer artifact root to monitor",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help=(
            "Compatibility option retained for older launch commands; "
            "the web UI refreshes trajectory panes only on button click."
        ),
    )
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=200,
        help="Maximum live JSONL trajectory rows to render per stream.",
    )
    parser.add_argument(
        "--container-name",
        default=DEFAULT_CONTAINER,
        help=(
            "Docker container that hosts belief_tracer processes. Used as a "
            "fallback when stopping a run whose PID is not visible to the host."
        ),
    )
    args = parser.parse_args(argv)
    serve(
        UiConfig(
            host=args.host,
            port=args.port,
            artifacts_dir=Path(args.artifacts_dir),
            poll_seconds=args.poll_seconds,
            max_results=args.max_results,
            max_trajectories=args.max_trajectories,
            container_name=args.container_name,
        )
    )


if __name__ == "__main__":
    main()
