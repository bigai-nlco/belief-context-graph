"""Deterministic run-level annotation statistics. No model or trajectory access.

python -m tools.case_study_studio.annotation_stats --run-id 7 --output /tmp/run-7-stats
python -m tools.case_study_studio.annotation_stats --input /tmp/run-7-stats/input_snapshot.json --output /tmp/replay
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from collections import Counter
from contextlib import closing
from pathlib import Path

VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = {
    "A1": "长程信息复用", "A2": "减少无效重复搜索", "A3": "定位关键证据缺口",
    "A4": "合理分配验证精力", "A5": "发现冲突并修正判断", "A6": "结论与证据可追溯",
    "A7": "降低整体上下文开销", "A8": "改善多 Agent 协作",
}
OPTIONS = {
    "opportunity": ["有", "无", "无法判断"],
    "utilization": ["充分利用", "部分利用", "未利用", "误用", "无法判断", "不适用"],
    "effect": ["正向", "无明显变化", "负向", "待验证", "不适用"],
    "evidence": ["明确可见", "有较强迹象", "证据不足"],
    "fault": ["无明显失效", "信息获取", "建图", "Graph 选择", "注入表达", "Agent 使用", "团队协作", "系统执行", "无法判断"],
    "result": ["正确 / 达标", "部分正确 / 部分达标", "错误 / 未达标", "未完成", "尚未评判"],
    "kind": ["benefit", "missed_opportunity", "misuse"],
    "search": ["合理核验", "新增信息搜索", "无效重复", "无效工具重试", "证据不足"],
}
MISSING = "未填写"
UNKNOWN = "未知值"
FOCUS = [v for v in OPTIONS["fault"] if v not in {"无明显失效", "无法判断"}]


class StatsError(ValueError):
    """Invalid or unavailable saved annotation input."""


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list:
    return value if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    return sorted({_text(v).strip() for v in _list(value) if _text(v).strip()})


def _child(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise StatsError("统计来源路径超出允许目录")
    return path


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StatsError("无法读取统计所需的 JSON 记录") from exc
    if not isinstance(value, dict):
        raise StatsError("统计所需的 JSON 记录必须为 object")
    return value


def project_annotation(value: object) -> dict:
    """Keep labels/links only, never large evidence, raw report or credentials."""
    if not isinstance(value, dict) or not isinstance(value.get("dimensions"), dict) or not isinstance(value.get("events"), list):
        raise StatsError("标注缺少 dimensions 或 events")
    events = []
    seen = set()
    for raw in value["events"]:
        item = _object(raw)
        event_id = _text(item.get("id"))
        if not event_id or event_id in seen:
            raise StatsError("关键过程 ID 缺失或重复")
        seen.add(event_id)
        events.append({"id": event_id, "label": _text(item.get("label")),
            "kind": _text(item.get("ai_kind")), "strength": _text(item.get("strength")),
            "fault": _text(item.get("fault")), "dimensions": _strings(item.get("dimensions")),
            "search": _text(_object(item.get("searchAssessment")).get("verdict"))})
    custom = []
    seen_custom = set(DIMENSIONS)
    for raw in _list(value.get("customDimensions")):
        item = _object(raw)
        local_id, title = _text(item.get("id")), _text(item.get("title")).strip()
        if not local_id or local_id in seen_custom or not title:
            raise StatsError("自定义维度名称或 ID 缺失 / 重复")
        seen_custom.add(local_id)
        custom.append({"id": local_id, "title": title, "question": _text(item.get("question")).strip()})
    dimensions = {}
    for key, raw in value["dimensions"].items():
        item = _object(raw)
        dimensions[key] = {k: _text(item.get(k)) for k in ("opportunity", "utilization", "effect")}
        dimensions[key]["eventIds"] = _strings(item.get("eventIds"))
    conclusion = _object(value.get("conclusion"))
    return {"status": _text(value.get("status")), "result": _text(value.get("result")),
        "noApplicableDimensions": value.get("noApplicableDimensions"),
        "dimensions": dimensions, "customDimensions": sorted(custom, key=lambda d: d["id"]),
        "events": sorted(events, key=lambda e: e["id"]),
        "conclusion": {"focus": _strings(conclusion.get("focus")), "evidence": _text(conclusion.get("evidence"))}}


def collect_snapshot(root: Path, state: Path, run_id: int, *, annotation_sink: dict | None = None) -> dict:
    """Read a DB snapshot and small result/annotation files; never writes to source."""
    if run_id < 1:
        raise StatsError("运行编号必须为正整数")
    database = _child(state, "studio.sqlite3")
    if not database.is_file():
        raise StatsError("找不到标注数据库")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        run = db.execute("SELECT id,source,title,config FROM runs WHERE id=?", (run_id,)).fetchone()
        if run is None:
            raise StatsError("找不到这个运行编号")
        humans = [dict(row) for row in db.execute(
            "SELECT r.benchmark,r.task_id,r.user_id,r.revision,r.annotation,r.evidence_sha,u.name "
            "FROM reviews r JOIN users u ON u.id=r.user_id WHERE r.run_id=? ORDER BY r.user_id,r.benchmark,r.task_id", (run_id,))]
        jobs = [dict(row) for row in db.execute(
            "SELECT id,benchmark,task_id,status,evidence_sha FROM ai_reviews WHERE run_id=? ORDER BY id", (run_id,))]
    folder = _child(root / "results", run["source"])
    config = json.loads(run["config"])
    meta_path = _child(folder, "drb2", "run.json")
    meta = _read(meta_path) if meta_path.is_file() else {}
    ids = meta.get("task_ids", _object(config.get("tasks")).get("drb2", []))
    if not isinstance(ids, list) or any(not isinstance(t, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", t) for t in ids):
        raise StatsError("Run 的任务清单无效")
    if len(ids) != len(set(ids)):
        raise StatsError("Run 的任务清单包含重复 ID")
    tasks = []
    for task_id in sorted(ids, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else t)):
        path = _child(folder, "drb2", "trials", task_id, "result.json")
        result = _read(path) if path.is_file() else {}
        tasks.append({"benchmark": "drb2", "task_id": task_id, "status": _text(result.get("status")) or "not_completed"})
    records, warnings = [], []
    for row in humans:
        if row["benchmark"] != "drb2":
            continue
        record = {"benchmark": row["benchmark"], "task_id": row["task_id"], "source": "human",
            "reviewer_id": row["user_id"], "reviewer": row["name"], "revision": row["revision"],
            "evidence_sha": row["evidence_sha"], "state": "invalid", "annotation": None}
        try:
            original = json.loads(row["annotation"])
            record["annotation"] = project_annotation(original)
            record["state"] = "completed" if record["annotation"]["status"] == "reviewed" else "draft"
            if annotation_sink is not None:
                annotation_sink[("human", row["user_id"], row["benchmark"], row["task_id"])] = original
        except (ValueError, TypeError):
            warnings.append(f"人工标注 {row['name']} · {row['task_id']} 格式无效，未计入")
        records.append(record)
    for job in jobs:
        if job["benchmark"] != "drb2":
            continue
        record = {"benchmark": job["benchmark"], "task_id": job["task_id"], "source": "ai",
            "reviewer_id": "ai", "reviewer": "AI", "model": "", "prompt_version": "", "reasoning_effort": "",
            "evidence_sha": job["evidence_sha"], "state": job["status"], "annotation": None}
        if job["status"] == "completed":
            try:
                saved = _read(_child(state, "ai-reviews", str(job["id"]), "review.json"))
                if str(saved.get("task_id")) != job["task_id"] or saved.get("benchmark") not in {"drb2", "deepresearch_bench_ii"}:
                    raise StatsError("AI 标注对应的任务不一致")
                # DB digest hashes the gzip bundle; review digest hashes the
                # evidence identity inside it. They intentionally differ.
                record["review_evidence_sha"] = _text(saved.get("evidence_sha256"))
                model = saved.get("model")
                record.update(model=_text(model) or _text(_object(model).get("model")),
                    reasoning_effort=_text(_object(model).get("reasoning_effort")),
                    prompt_version=_text(saved.get("prompt_version")),
                    annotation=project_annotation(saved.get("annotation")))
                record["reviewer"] = record["model"] or "未记录模型"
                if annotation_sink is not None:
                    annotation_sink[("ai", "ai", job["benchmark"], job["task_id"])] = saved["annotation"]
            except StatsError:
                record["state"] = "invalid"
                warnings.append(f"AI 标注 · {job['task_id']} 缺失、格式无效或与任务不匹配，未计入")
        records.append(record)
    return {"format": "bcg-run-annotations", "version": VERSION,
        "run": {"id": run_id, "number": f"RUN-{run_id:04d}", "title": run["title"]},
        "tasks": tasks, "records": records, "warnings": sorted(warnings)}


def _label(value: object, options: list[str]) -> str:
    if not value or value == MISSING:
        return MISSING
    return value if isinstance(value, str) and value in options else UNKNOWN


def _distribution(values: list, options: list[str]) -> dict[str, int]:
    counts = Counter(_label(value, options) for value in values)
    return {key: counts[key] for key in [*options, MISSING, UNKNOWN]}


def _cohort(record: dict) -> dict:
    source = record["source"]
    identity = {"source": source, "reviewer": record["reviewer"]}
    if source == "ai":
        identity.update({key: record.get(key, "") for key in ("model", "prompt_version", "reasoning_effort")})
        label = "AI · " + identity["reviewer"]
        label += f" · {identity['prompt_version'] or '未记录模板版本'} · {identity['reasoning_effort'] or '未记录思考强度'}"
    else:
        identity["reviewer_id"] = record["reviewer_id"]
        label = "人工 · " + identity["reviewer"]
    return {"id": digest(identity)[:16], "label": label, **identity}


def summarize(snapshot: dict) -> dict:
    """Count exact categorical labels. No text interpretation, voting or weighting."""
    if snapshot.get("format") != "bcg-run-annotations" or snapshot.get("version") != VERSION:
        raise StatsError("输入不是受支持的标注统计快照")
    tasks = {(t["benchmark"], t["task_id"]): t for t in snapshot["tasks"]}
    if len(tasks) != len(snapshot["tasks"]):
        raise StatsError("任务清单重复")
    eligible = {key for key, task in tasks.items() if task["status"] == "completed"}
    groups, rows, events, exclusions = {}, [], [], []
    warnings = list(snapshot.get("warnings", []))
    covered = {"human": set(), "ai": set()}
    seen = set()
    for record in sorted(snapshot["records"], key=lambda r: (r["source"], r["reviewer_id"], r["benchmark"], r["task_id"])):
        key = (record["benchmark"], record["task_id"])
        unique = (*key, record["source"], record["reviewer_id"])
        if unique in seen:
            raise StatsError("同一题同一标注来源有多份记录，不能重复统计")
        seen.add(unique)
        if record["source"] not in covered:
            raise StatsError("未知标注来源")
        reason = ("任务不在本 Run" if key not in tasks else "任务未正常完成" if key not in eligible
                  else record["state"] if record["state"] != "completed" else "")
        if reason:
            exclusions.append({"task_id": key[1], "source": record["source"], "reviewer": record["reviewer"], "reason": reason})
            continue
        a = record["annotation"]
        if not isinstance(a, dict):
            raise StatsError("已完成记录缺少标注")
        cohort = _cohort(record)
        group = groups.setdefault(cohort["id"], {**cohort, "records": []})
        group["records"].append(record)
        covered[record["source"]].add(key)
        event_map = {e["id"]: e for e in a["events"]}
        if len(event_map) != len(a["events"]):
            raise StatsError("同一标注中存在重复关键过程 ID")
        for event in a["events"]:
            events.append({"cohort_id": cohort["id"], "task_id": key[1], "benchmark": key[0], **event})
        custom = {d["id"]: d for d in a["customDimensions"]}
        catalog = {k: {"title": title, "question": ""} for k, title in DIMENSIONS.items()} | custom
        selected = False
        for local_id, dimension in catalog.items():
            # Local C1 can describe different concepts in different tasks.
            dim_id = local_id if local_id in DIMENSIONS else "custom-" + digest([dimension["title"], dimension["question"]])[:16]
            rating = a["dimensions"].get(local_id, {})
            opportunity = _label(rating.get("opportunity"), OPTIONS["opportunity"])
            selected |= opportunity in {"有", "无法判断"}
            linked = sorted(set(rating.get("eventIds", [])))
            missing_refs = [event_id for event_id in linked if event_id not in event_map]
            if opportunity in {"有", "无法判断"} and (not linked or missing_refs):
                warnings.append(f"{cohort['label']} · {key[1]} · {local_id} 缺少关键过程关联或关联 ID 无效")
            for field in ("opportunity", "utilization", "effect"):
                if rating.get(field) and rating[field] not in OPTIONS[field]:
                    warnings.append(f"{cohort['label']} · {key[1]} · {local_id} 的 {field} 含未知值")
            rows.append({"cohort_id": cohort["id"], "benchmark": key[0], "task_id": key[1],
                "dimension_id": dim_id, "local_dimension_id": local_id, "title": dimension["title"],
                "description": dimension["question"], "opportunity": opportunity,
                # Deselected controls retain stale draft ratings in the UI; do not count them.
                "utilization": _label(rating.get("utilization"), OPTIONS["utilization"]) if opportunity == "有" else "",
                "effect": _label(rating.get("effect"), OPTIONS["effect"]) if opportunity == "有" else "",
                "event_ids": [i for i in linked if i in event_map], "missing_event_ids": missing_refs})
        unknown_dims = sorted(set(a["dimensions"]) - set(catalog))
        if unknown_dims:
            warnings.append(f"{cohort['label']} · {key[1]} 有未定义维度：{', '.join(unknown_dims)}，未计入")
        if selected and a["noApplicableDimensions"] is True:
            warnings.append(f"{cohort['label']} · {key[1]} 同时选择维度和“无相关维度”，以实际选择为准")
    cohorts = []
    for group in sorted(groups.values(), key=lambda g: (g["source"], g["label"], g["id"])):
        records = group.pop("records")
        group_rows = [row for row in rows if row["cohort_id"] == group["id"]]
        group_events = [row for row in events if row["cohort_id"] == group["id"]]
        dimensions = []
        for dim_id in dict.fromkeys(row["dimension_id"] for row in group_rows):
            ratings = [row for row in group_rows if row["dimension_id"] == dim_id]
            applicable = [row for row in ratings if row["opportunity"] == "有"]
            dimensions.append({"id": dim_id, "title": ratings[0]["title"], "description": ratings[0]["description"],
                "available_tasks": len(ratings), "applicable_tasks": len(applicable),
                "opportunity": _distribution([r["opportunity"] for r in ratings], OPTIONS["opportunity"]),
                "utilization": _distribution([r["utilization"] for r in applicable], OPTIONS["utilization"]),
                "effect": _distribution([r["effect"] for r in applicable], OPTIONS["effect"]),
                "task_ids": [r["task_id"] for r in applicable]})
        annotations = [r["annotation"] for r in records]
        focus = {label: sum(label in a["conclusion"]["focus"] for a in annotations) for label in FOCUS}
        no_dimensions = sum(not any(d.get("opportunity") in {"有", "无法判断"} for d in a["dimensions"].values())
            and (a["noApplicableDimensions"] is True or (a["noApplicableDimensions"] is None
                 and all(a["dimensions"].get(d, {}).get("opportunity") == "无" for d in DIMENSIONS)))
            for a in annotations)
        cohorts.append({**group, "tasks": len(records), "task_ids": [r["task_id"] for r in records],
            "no_applicable_dimensions": no_dimensions, "dimensions": dimensions,
            "results": _distribution([a["result"] for a in annotations], OPTIONS["result"]),
            "focus": focus, "focus_unfilled_tasks": sum(not a["conclusion"]["focus"] for a in annotations),
            "evidence": _distribution([a["conclusion"]["evidence"] for a in annotations], OPTIONS["evidence"]),
            "events": len(group_events),
            "event_kind": _distribution([e["kind"] for e in group_events], OPTIONS["kind"]),
            "event_strength": _distribution([e["strength"] for e in group_events], OPTIONS["evidence"]),
            "search": _distribution([e["search"] for e in group_events], OPTIONS["search"])})
    return {"format": "bcg-annotation-statistics", "version": VERSION, "input_sha256": digest(snapshot),
        "run": snapshot["run"], "coverage": {"total_tasks": len(tasks), "eligible_tasks": len(eligible),
            "ineligible_tasks": len(tasks) - len(eligible), "reviewed_tasks": len(covered["human"] | covered["ai"]),
            "human_only": len(covered["human"] - covered["ai"]), "ai_only": len(covered["ai"] - covered["human"]),
            "both": len(covered["human"] & covered["ai"]), "unreviewed": len(eligible - (covered["human"] | covered["ai"]))},
        "cohorts": cohorts, "dimension_rows": rows, "event_rows": events, "excluded_records": exclusions,
        "warnings": sorted(set(warnings))}


def _csv(rows: list[dict], fields: list[str]) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            # Quotes alone do not prevent spreadsheet formula injection.
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            values.append(value)
        writer.writerow(values)
    return b"\xef\xbb\xbf" + out.getvalue().encode("utf-8")


def report_files(snapshot: dict) -> dict[str, bytes]:
    from tools.case_study_studio.annotation_charts import render_report
    stats = summarize(snapshot)
    html, charts = render_report(stats)
    dimensions = []
    for cohort in stats["cohorts"]:
        for dimension in cohort["dimensions"]:
            for metric in ("opportunity", "utilization", "effect"):
                denominator = dimension["available_tasks"] if metric == "opportunity" else dimension["applicable_tasks"]
                for label, count in dimension[metric].items():
                    dimensions.append({"cohort": cohort["label"], "dimension": dimension["title"], "metric": metric,
                        "label": label, "tasks": count, "denominator": denominator,
                        "percent": round(count / denominator * 100, 4) if denominator else ""})
    return {"report.html": html.encode(), "input_snapshot.json": canonical(snapshot), "stats.json": canonical(stats),
        "dimensions.csv": _csv(dimensions, ["cohort", "dimension", "metric", "label", "tasks", "denominator", "percent"]),
        "task_dimensions.csv": _csv(stats["dimension_rows"], ["cohort_id", "benchmark", "task_id", "dimension_id",
            "local_dimension_id", "title", "description", "opportunity", "utilization", "effect", "event_ids", "missing_event_ids"]),
        "events.csv": _csv(stats["event_rows"], ["cohort_id", "benchmark", "task_id", "id", "label", "kind", "strength", "fault", "search", "dimensions"]),
        "excluded.csv": _csv(stats["excluded_records"], ["source", "reviewer", "task_id", "reason"]),
        **{f"charts/{name}.svg": svg.encode() for name, svg in charts.items()}}


def archive(files: dict[str, bytes]) -> bytes:
    """ZIP headers have a fixed timestamp/order too, for reproducible exports."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as pack:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            pack.writestr(info, data)
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", type=int)
    source.add_argument("--input", type=Path, help="Replay an exported input_snapshot.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args()
    try:
        snapshot = _read(args.input) if args.input else collect_snapshot(
            args.root, args.state_dir or args.root / "runtime/case-study-studio", args.run_id)
        files = report_files(snapshot)
        if args.output.exists() and any(args.output.iterdir()):
            raise StatsError("输出目录非空，请使用新目录，避免覆盖之前的统计")
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (args.output / "annotation-stats.zip").write_bytes(archive(files))
    except (StatsError, OSError, sqlite3.Error, ValueError) as exc:
        parser.exit(2, f"annotation-stats: {exc}\n")
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
