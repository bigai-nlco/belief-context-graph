"""``bcg benchmark judge``: official LLM-as-judge scoring of saved reports.

Judging is decoupled from running: it reads ``<run>/<benchmark>/<mode>/``
task results and reports, scores them with the official judges in
:mod:`bcg.apps.benchmark.judges`, and writes::

    <run>/<benchmark>/<mode>/judgements/<task>.<judge-id>.json   per task
    <run>/judge_summary.json                                     aggregates
    <run>/export/deepresearch_bench_ii/result.<judge-id>.jsonl   official aggregate_scores.py input
    <run>/export/drbench/scores.<judge-id>.jsonl                 official score_report dicts

``judge-id`` combines the judge model with a hash of every setting that can
change a score (request parameters, metrics, prompt hashes), so several judges
can score the same reports side by side. Existing judgements are reused unless
the report changed, ``--overwrite`` is given, or ``--retry-failed`` selects
judgements with failures. Raw judge replies are cached under
``<run>/.judge-cache/<judge-id>/`` so an interrupted run resumes cheaply.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bcg.apps.benchmark import drb2 as drb2_data
from bcg.apps.benchmark import drbench as drbench_data
from bcg.apps.benchmark.judges import drb2 as drb2_judge
from bcg.apps.benchmark.judges import drbench as drbench_judge
from bcg.apps.benchmark.judges.client import JudgeClient, JudgeEndpoint
from bcg.apps.benchmark.judges.official_sources import load_sources
from bcg.apps.benchmark.models import BenchmarkTask

JUDGE_SCHEMA = "bcg.judgement.v1"
JUDGE_VERSION = 1
JUDGED_BENCHMARKS = (drb2_data.NAME, drbench_data.NAME)
DEFAULT_MODELS = {
    drb2_data.NAME: drb2_judge.DEFAULT_MODEL,
    drbench_data.NAME: drbench_judge.DEFAULT_MODEL,
}

# --judge-param keys per benchmark: key -> (section, type). Sections are the
# request-parameter groups (DRBench) or settings (DRB2).
PARAM_KEYS: dict[str, dict[str, tuple[str, type]]] = {
    drb2_data.NAME: {
        "reasoning_effort": ("request", str),
        "max_completion_tokens": ("request", int),
        "temperature": ("request", float),
        "chunk_size": ("settings", int),
        "max_retries": ("settings", int),
        "max_paper_chars": ("settings", int),
    },
    drbench_data.NAME: {
        f"{section}.{name}": (section, kind)
        for section in ("extraction", "scoring", "quality")
        for name, kind in (
            ("temperature", float),
            ("max_tokens", int),
            ("max_completion_tokens", int),
            ("reasoning_effort", str),
        )
    },
}


class JudgeConfigError(ValueError):
    """Invalid judge CLI configuration."""


@dataclass
class JudgeSpec:
    """Everything that determines one benchmark's judge."""

    benchmark: str
    model: str
    overrides: dict[str, Any] = field(default_factory=dict)
    metrics: tuple[str, ...] = drbench_judge.DEFAULT_METRICS

    def request_params(self) -> Any:
        if self.benchmark == drb2_data.NAME:
            params = dict(drb2_judge.OFFICIAL_PARAMS)
            params.update(
                {
                    k: v
                    for k, v in self.overrides.items()
                    if PARAM_KEYS[self.benchmark][k][0] == "request"
                }
            )
            return params
        params = {
            section: dict(values)
            for section, values in drbench_judge.OFFICIAL_PARAMS.items()
        }
        for key, value in self.overrides.items():
            section, name = key.split(".", 1)
            params[section][name] = value
        return params

    def settings(self) -> dict[str, Any]:
        if self.benchmark != drb2_data.NAME:
            return {}
        settings = dict(drb2_judge.OFFICIAL_SETTINGS)
        settings.update(
            {
                k: v
                for k, v in self.overrides.items()
                if PARAM_KEYS[self.benchmark][k][0] == "settings"
            }
        )
        return settings

    @property
    def deviates_from_official(self) -> bool:
        return bool(self.overrides)

    def judge_id(self) -> str:
        prompts = load_sources()["prompts"]
        material = json.dumps(
            {
                "version": JUDGE_VERSION,
                "benchmark": self.benchmark,
                "model": self.model,
                "params": self.request_params(),
                "settings": self.settings(),
                "metrics": list(self.metrics)
                if self.benchmark == drbench_data.NAME
                else None,
                "prompts": {name: meta["sha256"] for name, meta in prompts.items()},
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:8]
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", self.model).strip("-").lower() or "judge"
        return f"{slug}-{digest}"

    def describe(self) -> dict[str, Any]:
        sources = load_sources()
        return {
            "judge_id": self.judge_id(),
            "judge": drb2_judge.NAME
            if self.benchmark == drb2_data.NAME
            else drbench_judge.NAME,
            "model": self.model,
            "request_params": self.request_params(),
            "settings": self.settings(),
            "metrics": list(self.metrics)
            if self.benchmark == drbench_data.NAME
            else None,
            "overrides": self.overrides,
            "deviates_from_official": self.deviates_from_official,
            "official_source": sources["repositories"][
                "DeepResearch-Bench-II"
                if self.benchmark == drb2_data.NAME
                else "drbench"
            ],
            "prompt_sha256": {
                name: meta["sha256"]
                for name, meta in sources["prompts"].items()
                if name.startswith(
                    "drb2" if self.benchmark == drb2_data.NAME else "drbench"
                )
            },
        }


def parse_judge_models(
    values: list[str] | None, *, canonicalize: Callable[[str], str]
) -> tuple[str | None, dict[str, str]]:
    """``MODEL`` applies to every benchmark; ``BENCH=MODEL`` to one."""

    unscoped: str | None = None
    scoped: dict[str, str] = {}
    for raw in values or []:
        head, sep, tail = raw.partition("=")
        if sep:
            if not head.strip() or not tail.strip():
                raise JudgeConfigError(
                    f"Invalid --judge-model {raw!r}; expected MODEL or BENCH=MODEL."
                )
            scoped[canonicalize(head.strip())] = tail.strip()
        elif raw.strip():
            unscoped = raw.strip()
    return unscoped, scoped


def parse_judge_params(
    values: list[str] | None, *, canonicalize: Callable[[str], str]
) -> dict[str, dict[str, Any]]:
    """``BENCH.KEY=VALUE``, e.g. ``drb2.reasoning_effort=high``, ``drbench.scoring.temperature=0``."""

    parsed: dict[str, dict[str, Any]] = {}
    for raw in values or []:
        target, sep, value = raw.partition("=")
        bench, dot, key = target.partition(".")
        if not sep or not dot:
            raise JudgeConfigError(
                f"Invalid --judge-param {raw!r}; expected BENCH.KEY=VALUE."
            )
        name = canonicalize(bench.strip())
        allowed = PARAM_KEYS.get(name)
        if allowed is None or key.strip() not in allowed:
            raise JudgeConfigError(
                f"Unknown --judge-param key {key!r} for {name}; supported: "
                + ", ".join(sorted(allowed or {}))
            )
        kind = allowed[key.strip()][1]
        try:
            parsed.setdefault(name, {})[key.strip()] = kind(value.strip())
        except ValueError as exc:
            raise JudgeConfigError(f"Invalid value in --judge-param {raw!r}.") from exc
    return parsed


def build_specs(
    benchmarks: list[str],
    *,
    model_values: list[str] | None,
    param_values: list[str] | None,
    metrics: list[str] | None,
    canonicalize: Callable[[str], str],
) -> dict[str, JudgeSpec]:
    unscoped, scoped = parse_judge_models(model_values, canonicalize=canonicalize)
    params = parse_judge_params(param_values, canonicalize=canonicalize)
    selected_metrics = tuple(metrics or drbench_judge.DEFAULT_METRICS)
    unknown = set(selected_metrics) - set(drbench_judge.METRICS)
    if unknown:
        raise JudgeConfigError(
            f"Unknown --metrics {sorted(unknown)}; supported: {', '.join(drbench_judge.METRICS)}."
        )
    return {
        name: JudgeSpec(
            benchmark=name,
            model=scoped.get(name) or unscoped or DEFAULT_MODELS[name],
            overrides=params.get(name, {}),
            metrics=selected_metrics,
        )
        for name in benchmarks
        if name in JUDGED_BENCHMARKS
    }


# --- discovery -------------------------------------------------------------------


@dataclass
class JudgeItem:
    benchmark: str
    mode: str
    task_id: str
    result_path: Path
    report_path: Path | None
    status: str


def discover(
    run_dir: Path, benchmarks: list[str], modes: list[str] | None
) -> list[JudgeItem]:
    items: list[JudgeItem] = []
    for benchmark in benchmarks:
        for result_path in sorted((run_dir / benchmark).glob("*/tasks/*.json")):
            mode = result_path.parents[1].name
            if modes and mode not in modes:
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            report = result.get("report")
            report_path = Path(report) if report else None
            if report_path and not report_path.is_file():
                local = result_path.parents[1] / "reports" / report_path.name
                report_path = local if local.is_file() else None
            items.append(
                JudgeItem(
                    benchmark,
                    mode,
                    str(result.get("task_id")),
                    result_path,
                    report_path,
                    str(result.get("status")),
                )
            )
    return items


def load_reference_tasks(
    benchmark: str, data_root: Path, data_file: Path | None = None
) -> dict[str, BenchmarkTask]:
    if benchmark == drb2_data.NAME:
        return {
            task.task_id: task
            for task in drb2_data.load_tasks(data_root, data_file=data_file)
        }
    tasks: dict[str, BenchmarkTask] = {}
    for subset in ("val", "sanity"):
        try:
            for task in drbench_data.load_tasks(
                data_root, data_file=data_file, subset=subset
            ):
                tasks[task.task_id] = task
        except drbench_data.DRBenchDataError:
            continue
    return tasks


# --- execution -------------------------------------------------------------------


def judgement_path(item: JudgeItem, judge_id: str) -> Path:
    return (
        item.result_path.parents[1]
        / "judgements"
        / f"{item.result_path.stem}.{judge_id}.json"
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def needs_judging(
    path: Path, report_sha: str, *, overwrite: bool, retry_failed: bool
) -> bool:
    if overwrite or not path.is_file():
        return True
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if existing.get("report_sha256") != report_sha:
        return True
    return bool(
        retry_failed
        and (existing.get("status") != "ok" or existing.get("judge_failures"))
    )


def judge_one(
    item: JudgeItem,
    task: BenchmarkTask,
    spec: JudgeSpec,
    client: JudgeClient,
) -> dict[str, Any]:
    assert item.report_path is not None
    report = item.report_path.read_text(encoding="utf-8")
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    if spec.benchmark == drb2_data.NAME:
        result = drb2_judge.judge_report(
            task, report, client, params=spec.request_params(), settings=spec.settings()
        )
    else:
        result = drbench_judge.judge_report(
            task, report, client, metrics=spec.metrics, params=spec.request_params()
        )
    judgement = {
        "schema": JUDGE_SCHEMA,
        "benchmark": item.benchmark,
        "mode": item.mode,
        "task_id": item.task_id,
        "judge": {
            **spec.describe(),
            "base_url": client.endpoint.base_url,
            "param_adaptations": dict(client.adaptations),
        },
        "report": str(item.report_path),
        "report_sha256": _sha256_file(item.report_path),
        "report_chars": len(report),
        "agent_status": item.status,
        "started_at": started_at,
        "seconds": round(time.monotonic() - started, 3),
        "judge_failures": result.get("judge_failures", 0),
        **result,
    }
    return judgement


@dataclass
class JudgeRunResult:
    judged: int = 0
    reused: int = 0
    no_report: int = 0
    missing_reference: int = 0
    errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def plan(
    run_dir: Path,
    specs: dict[str, JudgeSpec],
    *,
    data_root: Path,
    modes: list[str] | None = None,
    task_filter: Callable[[str, list[str]], list[str]] | None = None,
    overwrite: bool = False,
    retry_failed: bool = False,
    data_files: dict[str, Path] | None = None,
) -> tuple[
    list[tuple[JudgeItem, BenchmarkTask]],
    JudgeRunResult,
    dict[str, dict[str, BenchmarkTask]],
]:
    """Select the (item, task) pairs that need a judge call."""

    run_dir = Path(run_dir).expanduser().resolve()
    outcome = JudgeRunResult()
    todo: list[tuple[JudgeItem, BenchmarkTask]] = []
    references: dict[str, dict[str, BenchmarkTask]] = {}
    items = discover(run_dir, list(specs), modes)
    for benchmark in specs:
        references[benchmark] = load_reference_tasks(
            benchmark, data_root, (data_files or {}).get(benchmark)
        )
    if task_filter is not None:
        keep: set[tuple[str, str]] = set()
        for benchmark in specs:
            ids = sorted(
                {item.task_id for item in items if item.benchmark == benchmark}
            )
            keep |= {(benchmark, value) for value in task_filter(benchmark, ids)}
        items = [item for item in items if (item.benchmark, item.task_id) in keep]
    for item in items:
        if item.report_path is None:
            outcome.no_report += 1
            continue
        task = references[item.benchmark].get(item.task_id)
        if task is None:
            outcome.missing_reference += 1
            continue
        spec = specs[item.benchmark]
        path = judgement_path(item, spec.judge_id())
        if needs_judging(
            path,
            _sha256_file(item.report_path),
            overwrite=overwrite,
            retry_failed=retry_failed,
        ):
            todo.append((item, task))
        else:
            outcome.reused += 1
    return todo, outcome, references


def run_judges(
    run_dir: Path,
    specs: dict[str, JudgeSpec],
    endpoint_for: Callable[[str], JudgeEndpoint],
    *,
    data_root: Path,
    modes: list[str] | None = None,
    task_filter: Callable[[str, list[str]], list[str]] | None = None,
    workers: int = 4,
    overwrite: bool = False,
    retry_failed: bool = False,
    data_files: dict[str, Path] | None = None,
    progress: Callable[[str], None] = print,
    transport: Any = None,
) -> JudgeRunResult:
    run_dir = Path(run_dir).expanduser().resolve()
    todo, outcome, _references = plan(
        run_dir,
        specs,
        data_root=data_root,
        modes=modes,
        task_filter=task_filter,
        overwrite=overwrite,
        retry_failed=retry_failed,
        data_files=data_files,
    )
    clients = {
        benchmark: JudgeClient(
            endpoint_for(spec.model),
            cache_dir=run_dir / ".judge-cache" / spec.judge_id(),
            transport=transport,
        )
        for benchmark, spec in specs.items()
    }
    total = len(todo)

    def work(pair: tuple[JudgeItem, BenchmarkTask]) -> tuple[JudgeItem, dict[str, Any]]:
        item, task = pair
        return item, judge_one(
            item, task, specs[item.benchmark], clients[item.benchmark]
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, pair) for pair in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                item, judgement = future.result()
            except Exception as exc:  # A bug in one task must not lose the others.
                outcome.errors.append(str(exc))
                progress(f"[judge {done}/{total}] error: {exc}")
                continue
            spec = specs[item.benchmark]
            path = judgement_path(item, spec.judge_id())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(judgement, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _link_into_result(item, spec.judge_id(), judgement, path)
            outcome.judged += 1
            flag = (
                "✓"
                if judgement["status"] == "ok" and not judgement.get("judge_failures")
                else "!"
            )
            brief = ", ".join(
                f"{k}={v:.3f}"
                for k, v in judgement.get("scores", {}).items()
                if isinstance(v, (int, float))
            )
            error = judgement.get("error")
            failures = judgement.get("judge_failures")
            progress(
                f"[judge {done}/{total}] {flag} {item.benchmark}/{item.mode}/{item.task_id}: "
                f"{judgement['status']}{f' ({brief})' if brief else ''}"
                f"{f' · {failures} failed judge call(s)' if failures else ''}"
                f"{f' · {error}' if error else ''}"
            )
    outcome.summary = summarize(run_dir, specs)
    write_exports(run_dir, specs)
    return outcome


def _link_into_result(
    item: JudgeItem, judge_id: str, judgement: dict[str, Any], path: Path
) -> None:
    try:
        result = json.loads(item.result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    result.setdefault("judgements", {})[judge_id] = {
        "status": judgement["status"],
        "scores": judgement.get("scores", {}),
        "judge_failures": judgement.get("judge_failures", 0),
        "path": str(path),
    }
    item.result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# --- summary and exports -------------------------------------------------------------


def _judgements(run_dir: Path, benchmark: str, judge_id: str) -> list[dict[str, Any]]:
    values = []
    for path in sorted((run_dir / benchmark).glob(f"*/judgements/*.{judge_id}.json")):
        try:
            values.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return values


def _mean(values: list[float]) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(numbers) / len(numbers) if numbers else None


def summarize(run_dir: Path, specs: dict[str, JudgeSpec]) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    path = run_dir / "judge_summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    summary.setdefault("benchmarks", {})
    for benchmark, spec in specs.items():
        judge_id = spec.judge_id()
        keys = (
            drb2_judge.METRIC_KEYS
            if benchmark == drb2_data.NAME
            else drbench_judge.METRIC_KEYS
        )
        group_key = "language" if benchmark == drb2_data.NAME else "difficulty"
        by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for judgement in _judgements(run_dir, benchmark, judge_id):
            by_mode[judgement["mode"]].append(judgement)
        tasks_by_mode: dict[str, int] = defaultdict(int)
        for result_path in (run_dir / benchmark).glob("*/tasks/*.json"):
            tasks_by_mode[result_path.parents[1].name] += 1
        entry = summary["benchmarks"].setdefault(benchmark, {})
        for mode, judgements in sorted(by_mode.items()):
            ok = [j for j in judgements if j.get("status") == "ok"]
            usage: dict[str, int] = defaultdict(int)
            for judgement in judgements:
                for key, value in (judgement.get("usage") or {}).items():
                    usage[key] += int(value)
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for judgement in ok:
                groups[
                    str(_task_meta(run_dir, judgement).get(group_key) or "unknown")
                ].append(judgement)
            entry.setdefault(mode, {})[judge_id] = {
                "judge": spec.describe(),
                "tasks": tasks_by_mode.get(mode, 0),
                "judged": len(judgements),
                "ok": len(ok),
                "errors": len(judgements) - len(ok),
                "judge_failures": sum(
                    int(j.get("judge_failures", 0)) for j in judgements
                ),
                "coverage": len(ok) / tasks_by_mode[mode]
                if tasks_by_mode.get(mode)
                else None,
                # Official aggregation averages successfully judged tasks only.
                "scores": {
                    key: _mean([j["scores"].get(key) for j in ok]) for key in keys
                },
                f"scores_by_{group_key}": {
                    name: {
                        "tasks": len(items),
                        **{
                            key: _mean([j["scores"].get(key) for j in items])
                            for key in keys
                        },
                    }
                    for name, items in sorted(groups.items())
                },
                "usage": dict(usage),
            }
    summary["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _task_meta(run_dir: Path, judgement: dict[str, Any]) -> dict[str, Any]:
    result_path = (
        run_dir
        / judgement["benchmark"]
        / judgement["mode"]
        / "tasks"
        / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(judgement['task_id'])).strip('_') or 'item'}.json"
    )
    try:
        return json.loads(result_path.read_text(encoding="utf-8")).get("metadata") or {}
    except (OSError, ValueError):
        return {}


def write_exports(run_dir: Path, specs: dict[str, JudgeSpec]) -> None:
    run_dir = Path(run_dir).expanduser().resolve()
    for benchmark, spec in specs.items():
        judge_id = spec.judge_id()
        judgements = _judgements(run_dir, benchmark, judge_id)
        if not judgements:
            continue
        directory = run_dir / "export" / benchmark
        directory.mkdir(parents=True, exist_ok=True)
        if benchmark == drb2_data.NAME:
            lines = [
                {
                    "model": j["mode"],
                    "idx": int(j["task_id"]),
                    "result": j["official_result"],
                }
                for j in judgements
            ]
            target = directory / f"result.{judge_id}.jsonl"
        else:
            lines = [
                {
                    "mode": j["mode"],
                    "task_id": j["task_id"],
                    **j.get("official_result", {}),
                }
                for j in judgements
            ]
            target = directory / f"scores.{judge_id}.jsonl"
        target.write_text(
            "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
            encoding="utf-8",
        )


# --- dry-run estimate -------------------------------------------------------------------


def _tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3000-\u9fff\uff00-\uffef]", text))
    return int(cjk + (len(text) - cjk) / 4)


def estimate(
    todo: list[tuple[JudgeItem, BenchmarkTask]], specs: dict[str, JudgeSpec]
) -> dict[str, dict[str, int]]:
    """Rough call and input-token counts for the pending judgements."""

    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tasks": 0, "calls": 0, "input_tokens": 0}
    )
    for item, task in todo:
        report = (
            item.report_path.read_text(encoding="utf-8") if item.report_path else ""
        )
        bucket = totals[item.benchmark]
        bucket["tasks"] += 1
        if item.benchmark == drb2_data.NAME:
            settings = specs[item.benchmark].settings()
            text = report[: settings["max_paper_chars"]]
            rubric = (task.reference.get("official_content") or {}).get("rubric") or {}
            items = [
                i for dim in drb2_judge.DIMENSIONS for i in (rubric.get(dim) or [])
            ]
            batches = (
                max(1, math.ceil(len(items) / settings["chunk_size"]))
                if settings["chunk_size"] > 0
                else 1
            )
            bucket["calls"] += batches
            bucket["input_tokens"] += batches * (_tokens(text) + 700) + _tokens(
                "".join(items)
            )
        else:
            text = report[: drbench_judge.MAX_REPORT_LENGTH]
            qa = task.reference.get("qa") or []
            metrics = specs[item.benchmark].metrics
            scoring = sum(
                1
                for q in qa
                if ("insights_recall" in metrics and q.get("qa_type") == "insight")
                or ("distractor_recall" in metrics and q.get("qa_type") == "distractor")
            )
            quality = 1 if "report_quality" in metrics else 0
            claims = int(
                _tokens(text) * 0.4
            )  # extracted claims are a fraction of the report
            bucket["calls"] += 1 + scoring + quality
            bucket["input_tokens"] += (
                (_tokens(text) + 900)
                + scoring * (claims + 450)
                + quality * (_tokens(text) + 1200)
            )
    return dict(totals)


__all__ = [
    "DEFAULT_MODELS",
    "JUDGED_BENCHMARKS",
    "JudgeConfigError",
    "JudgeItem",
    "JudgeRunResult",
    "JudgeSpec",
    "build_specs",
    "discover",
    "estimate",
    "parse_judge_models",
    "parse_judge_params",
    "plan",
    "run_judges",
    "summarize",
    "write_exports",
]
