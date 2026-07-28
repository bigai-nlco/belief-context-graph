"""Loaders for agentic benchmarks under experiments/artifacts/benchmarks/.

Each benchmark directory has its own layout (data.json, sub-folders with
parquet, csv, or heterogeneous JSON). This module normalizes all of them into
a single :class:`AgenticTask` shape compatible with
:func:`rllm.rewards.reward_fn.search_reward_fn` and
:class:`rllm.environments.tools.tool_env.ToolEnvironment`.

The public entry point is :func:`load_benchmark`. Unknown benchmarks raise
``KeyError`` so missing coverage is loud rather than silent.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_BENCHMARKS_DIR = PROJECT_ROOT / "datasets"
_RLLM_ARTIFACTS_BENCHMARKS_DIR = Path(
    "/share/nlp/liuyang/workspace/gem/rllm/experiments/artifacts/benchmarks"
)


def _default_benchmarks_dir() -> Path:
    env_dir = (
        os.environ.get("BELIEF_TRACER_BENCHMARKS_DIR")
        or os.environ.get("BELIEFTRACER_BENCHMARKS_DIR")
        or os.environ.get("AGENTIC_ARTIFACTS_DIR")
    )
    if env_dir:
        return Path(env_dir)
    if LOCAL_BENCHMARKS_DIR.is_dir():
        return LOCAL_BENCHMARKS_DIR
    return _RLLM_ARTIFACTS_BENCHMARKS_DIR


ARTIFACTS_BENCHMARKS_DIR = _default_benchmarks_dir()


_QA_HINTS = (
    "Please answer the above question",
    "Answer the above question",
    "When ready, output the final answer",
    "When ready, please output",
    "Output the final answer enclosed in",
    "Put your final answer",
)


def _strip_qa_suffix(q: str) -> str:
    """Remove common instruction suffixes appended to benchmark questions."""
    if not isinstance(q, str):
        return str(q)
    out = q
    for hint in _QA_HINTS:
        idx = out.find(hint)
        if idx > 0:
            out = out[:idx]
    return out.strip()


def _as_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v) != ""]
    if isinstance(value, (tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


@dataclass
class AgenticTask:
    """Normalized representation of one benchmark example."""

    task_id: str
    question: str
    ground_truth: list[str]
    data_source: str
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_env_task(self) -> dict[str, Any]:
        """Build the task dict consumed by ToolEnvironment + search_reward_fn.

        rllm expects ``ground_truth`` (str or list) and ``data_source``.
        """
        gt: Any = self.ground_truth
        if len(gt) == 1:
            gt = gt[0]
        return {
            "question": self.question,
            "ground_truth": gt,
            "data_source": self.data_source,
            "extra_info": self.extra_info,
        }


# -------- Per-benchmark loaders -----------------------------------------------


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_parquet(path: Path) -> list[dict]:
    import pandas as pd  # local to avoid hard dep when unused

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return _load_json(path)
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".csv":
        return _load_csv(path)
    if path.suffix == ".parquet":
        return _load_parquet(path)
    raise ValueError(f"Unsupported benchmark file type: {path}")


def _extract_boxed_answer(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    marker = "\\boxed{"
    start = s.find(marker)
    if start < 0:
        return s
    i = start + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(s) and depth > 0:
        ch = s[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() or s


LOCAL_BENCHMARK_FILES: dict[str, tuple[str, ...]] = {
    "aime24": ("AIME24/aime24.parquet",),
    "aime25": ("AIME25/aime25.json",),
    "amc23": ("AMC23/amc23.json",),
    "bbeh": ("BBEH/bbeh_processed.json", "BBEH/bbeh.json"),
    "gpqa": ("GPQA_Diamond/gpqa_diamond_processed.json",),
    "gpqa_diamond": ("GPQA_Diamond/gpqa_diamond_processed.json",),
    "hmmt": ("HMMT_Feb_2025/hmmt_feb_2025.json",),
    "hmmt_feb_2025": ("HMMT_Feb_2025/hmmt_feb_2025.json",),
    "math500": ("MATH500/math500.json",),
    "minerva": ("Minerva/minerva.json",),
    "mmlu": ("MMLU-Pro/mmlu_pro_test_processed.json",),
    "mmlu_pro": ("MMLU-Pro/mmlu_pro_test_processed.json",),
    "olympiad": ("Olympiad/olympiad.json",),
    "zebra_logic": ("ZebraLogic/zebra_logic.json",),
}


def _resolve_local_benchmark_file(name: str, root: Path) -> Path | None:
    for rel in LOCAL_BENCHMARK_FILES.get(name, ()):
        p = root / rel
        if p.is_file():
            return p
    return None


def _row_question(row: dict[str, Any]) -> str:
    q = (
        row.get("question")
        or row.get("input")
        or row.get("problem")
        or row.get("prompt")
        or row.get("query")
    )
    if q is None and row.get("puzzle"):
        q = row.get("puzzle")
    parts = [str(q or "").strip()]
    if row.get("context"):
        parts.insert(0, str(row["context"]).strip())
    if row.get("puzzle") and row.get("question") and row.get("puzzle") != row.get("question"):
        parts = [str(row["puzzle"]).strip(), str(row["question"]).strip()]
    choices = row.get("choices")
    if choices:
        if isinstance(choices, dict):
            choices_text = "\n".join(f"{k}. {v}" for k, v in choices.items())
        elif isinstance(choices, list):
            choices_text = "\n".join(str(v) for v in choices)
        else:
            choices_text = str(choices)
        parts.append(choices_text)
    return "\n\n".join(p for p in parts if p)


def _row_answer(row: dict[str, Any]) -> list[str]:
    ans = None
    for key in (
        "answer",
        "ground_truth_answer",
        "gt_answer",
        "target",
        "final_answer",
    ):
        if row.get(key) is not None:
            ans = row[key]
            break
    if ans is None and row.get("solution") is not None:
        ans = _extract_boxed_answer(row["solution"])
    return _as_answer_list(ans)


def _load_local_benchmark_file(name: str, path: Path) -> list[AgenticTask]:
    rows = _load_rows(path)
    tasks: list[AgenticTask] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question = _row_question(row)
        if not question:
            continue
        task_id = str(
            row.get("task_id")
            or row.get("id")
            or row.get("problem_idx")
            or row.get("unique_id")
            or f"{name}-{i}"
        )
        tasks.append(
            AgenticTask(
                task_id=task_id,
                question=question,
                ground_truth=_row_answer(row),
                data_source=name,
                extra_info={
                    k: v
                    for k, v in row.items()
                    if k
                    not in (
                        "question",
                        "input",
                        "problem",
                        "prompt",
                        "query",
                        "answer",
                        "ground_truth_answer",
                        "gt_answer",
                        "target",
                        "final_answer",
                        "solution",
                    )
                },
            )
        )
    return tasks


def _load_data_json(
    bench_dir: Path,
    data_source: str,
    question_keys: tuple[str, ...] = ("question", "query", "input", "problem"),
    answer_keys: tuple[str, ...] = (
        "answer",
        "gt_answer",
        "ground_truth_answer",
        "target",
    ),
    strip_suffix: bool = True,
) -> list[AgenticTask]:
    for rel in ("data/data.json", "data.json"):
        p = bench_dir / rel
        if p.is_file():
            samples = _load_json(p)
            break
    else:
        raise FileNotFoundError(f"No data.json under {bench_dir}")

    tasks: list[AgenticTask] = []
    for i, s in enumerate(samples):
        q = None
        for k in question_keys:
            if s.get(k):
                q = s[k]
                break
        if q is None:
            # Look into extra_info for nested question
            ei = s.get("extra_info") or {}
            for k in question_keys:
                if isinstance(ei, dict) and ei.get(k):
                    q = ei[k]
                    break
        if q is None:
            continue

        ans = None
        for k in answer_keys:
            if s.get(k) is not None:
                ans = s[k]
                break
        if ans is None:
            ei = s.get("extra_info") or {}
            for k in answer_keys:
                if isinstance(ei, dict) and ei.get(k) is not None:
                    ans = ei[k]
                    break

        task_id = str(s.get("task_id") or s.get("id") or f"{data_source}-{i}")
        tasks.append(
            AgenticTask(
                task_id=task_id,
                question=_strip_qa_suffix(q) if strip_suffix else str(q),
                ground_truth=_as_answer_list(ans),
                data_source=data_source,
                extra_info={
                    k: v
                    for k, v in s.items()
                    if k not in question_keys and k not in answer_keys
                },
            )
        )
    return tasks


def load_bamboogle(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(bench_dir, data_source="bamboogle")


def load_hotpotqa(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(bench_dir, data_source="hotpotqa")


def load_2wiki(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir, data_source="2wiki", question_keys=("query", "question", "input")
    )


def load_musique(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir, data_source="musique", question_keys=("query", "question", "input")
    )


def _clean_gaia_scalar(value: Any) -> str:
    """Normalize JSON/parquet scalar values, including pandas NaN."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _load_gaia_metadata(bench_dir: Path, split: str) -> list[dict[str, Any]] | None:
    for base in (bench_dir / "2023" / split, bench_dir / "raw" / "2023" / split):
        for name in ("metadata.jsonl", "metadata.parquet"):
            path = base / name
            if path.is_file():
                return _load_rows(path)
    return None


def load_gaia(bench_dir: Path) -> list[AgenticTask]:
    """Load GAIA 2023 validation/test metadata and preserve attachment paths.

    ``GAIA_SPLIT`` selects ``validation`` (default) or ``test``. ``GAIA_LEVEL``
    may be ``all`` (default), ``1``, ``2``, or ``3``. The validation split has
    public ground truth; the test split intentionally uses ``?`` answers.
    """
    split = os.environ.get("GAIA_SPLIT", "validation").strip().lower()
    if split not in {"validation", "test"}:
        raise ValueError("GAIA_SPLIT must be 'validation' or 'test'")
    level_filter = os.environ.get("GAIA_LEVEL", "all").strip().lower()
    if level_filter not in {"all", "1", "2", "3", "level1", "level2", "level3"}:
        raise ValueError("GAIA_LEVEL must be all, 1, 2, or 3")
    if level_filter.startswith("level"):
        level_filter = level_filter[-1]

    rows = _load_gaia_metadata(bench_dir, split)
    if rows is None:
        # Backward compatibility for the old rllm artifact layout.
        return _load_data_json(bench_dir, data_source="gaia")

    tasks: list[AgenticTask] = []
    metadata_dir = bench_dir / "2023" / split
    if not metadata_dir.is_dir():
        metadata_dir = bench_dir / "raw" / "2023" / split
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question = _clean_gaia_scalar(
            row.get("Question") or row.get("question") or row.get("input")
        )
        if not question:
            continue
        level = _clean_gaia_scalar(row.get("Level") or row.get("level"))
        if level_filter != "all" and level != level_filter:
            continue

        file_name = _clean_gaia_scalar(row.get("file_name") or row.get("file"))
        file_path = metadata_dir / file_name if file_name else None
        try:
            relative_attachment = file_path.relative_to(bench_dir) if file_path else None
        except ValueError:
            relative_attachment = None
        file_url = (
            f"file://gaia/{relative_attachment.as_posix()}"
            if relative_attachment is not None
            else ""
        )
        if file_name:
            question = (
                f"{question}\n\n"
                "This GAIA task includes an attached local file.\n"
                f"Attachment: {file_url}\n"
                "Use the read_file tool when the attachment is text-compatible. "
                "The local file must also be available to any richer document or "
                "multimodal inspection tool used by the run."
            )

        answer = _clean_gaia_scalar(
            row.get("Final answer")
            or row.get("final_answer")
            or row.get("answer")
        )
        ground_truth = [] if answer in {"", "?"} else [answer]
        task_id = _clean_gaia_scalar(row.get("task_id") or row.get("id"))
        tasks.append(
            AgenticTask(
                task_id=task_id or f"gaia-{split}-{index}",
                question=question,
                ground_truth=ground_truth,
                data_source="gaia",
                extra_info={
                    "level": int(level) if level.isdigit() else level,
                    "split": split,
                    "file_name": file_name,
                    "file_path": str(file_path.resolve()) if file_path else "",
                    "file_url": file_url,
                    "source": "GAIA 2023",
                },
            )
        )
    return tasks


def load_medqa(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(bench_dir, data_source="medqa")


def load_browse_comp(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir,
        data_source="browsecomp",
        question_keys=("input", "question", "query"),
        answer_keys=("ground_truth_answer", "answer", "gt_answer"),
        strip_suffix=False,
    )


def load_browsecomp_plus(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir,
        data_source="browsecomp_plus",
        question_keys=("input", "question", "query"),
        answer_keys=("ground_truth_answer", "answer", "gt_answer"),
        strip_suffix=False,
    )


def load_simpleqa_verified(bench_dir: Path) -> list[AgenticTask]:
    # CSV in the folder root
    for fname in ("simpleqa_verified.csv", "data.csv"):
        p = bench_dir / fname
        if p.is_file():
            rows = _load_csv(p)
            break
    else:
        # fall back to data.json if present
        return _load_data_json(
            bench_dir, data_source="simpleqa_verified", strip_suffix=False
        )

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or r.get("input")
        ans = r.get("answer") or r.get("ground_truth_answer")
        if not q:
            continue
        tasks.append(
            AgenticTask(
                task_id=str(r.get("task_id") or f"simpleqa_verified-{i}"),
                question=_strip_qa_suffix(q),
                ground_truth=_as_answer_list(ans),
                data_source="simpleqa_verified",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k not in ("problem", "question", "input", "answer")
                },
            )
        )
    return tasks


def _xbench_xor_decrypt(b64_text: str, key: str) -> str:
    """Decode xbench-style (base64 + per-row XOR) ciphertext to UTF-8.

    The xbench ScienceQA shard (``ScienceQA.csv``) stores ``prompt`` and
    ``answer`` as base64(XOR(plaintext, canary)) to keep benchmark data out
    of scraped web indices. Without decryption the model is given pure
    ciphertext, which crashes pass@1 to ~0. Mirrors
    ``experiments/artifacts/benchmarks/scienceqa/xbench_evals.py``.
    """
    if not b64_text or not key:
        return b64_text
    raw = base64.b64decode(b64_text)
    kb = key.encode("utf-8")
    kl = len(kb) or 1
    return bytes(raw[i] ^ kb[i % kl] for i in range(len(raw))).decode("utf-8")


def load_scienceqa(bench_dir: Path) -> list[AgenticTask]:
    for fname in ("ScienceQA.csv", "scienceqa.csv", "data.csv"):
        p = bench_dir / fname
        if p.is_file():
            rows = _load_csv(p)
            break
    else:
        return _load_data_json(bench_dir, data_source="scienceqa")

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        canary = r.get("canary") or ""
        q_raw = (
            r.get("prompt") or r.get("problem") or r.get("question") or r.get("input")
        )
        ans_raw = r.get("answer") or r.get("ground_truth_answer")
        if not q_raw:
            continue
        # xbench ships ScienceQA with base64+XOR ciphertext keyed on
        # ``canary``. Decode both sides; plain-text rows (no canary or
        # decode failure) fall through verbatim.
        q = q_raw
        ans = ans_raw
        if canary:
            try:
                q = _xbench_xor_decrypt(q_raw, canary)
            except Exception:
                q = q_raw
            if ans_raw:
                try:
                    ans = _xbench_xor_decrypt(ans_raw, canary)
                except Exception:
                    ans = ans_raw
        tasks.append(
            AgenticTask(
                task_id=str(r.get("id") or r.get("task_id") or f"scienceqa-{i}"),
                question=_strip_qa_suffix(q),
                ground_truth=_as_answer_list(ans),
                data_source="scienceqa",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k
                    not in (
                        "problem",
                        "question",
                        "prompt",
                        "input",
                        "answer",
                        "canary",
                    )
                },
            )
        )
    return tasks


def load_hle(bench_dir: Path) -> list[AgenticTask]:
    # Real content lives in the parquet shard, not data.json.
    for rel in (
        "data/test-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
        "data.parquet",
    ):
        p = bench_dir / rel
        if p.is_file():
            rows = _load_parquet(p)
            break
    else:
        return _load_data_json(bench_dir, data_source="hle", strip_suffix=False)

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        q = r.get("question") or r.get("problem") or r.get("input")
        ans = r.get("answer") or r.get("ground_truth_answer")
        if not q:
            continue
        tasks.append(
            AgenticTask(
                task_id=str(r.get("id") or r.get("task_id") or f"hle-{i}"),
                question=str(q),
                ground_truth=_as_answer_list(ans),
                data_source="hle",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k in ("category", "answer_type", "subject")
                },
            )
        )
    return tasks


def load_deepsearchqa(bench_dir: Path) -> list[AgenticTask]:
    for fname in ("DSQA-full.csv", "data.csv", "deepsearchqa.csv"):
        p = bench_dir / fname
        if p.is_file():
            rows = _load_csv(p)
            break
    else:
        return _load_data_json(bench_dir, data_source="deepsearchqa")

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or r.get("query") or r.get("input")
        ans = r.get("answer") or r.get("ground_truth_answer")
        if not q:
            continue
        tasks.append(
            AgenticTask(
                task_id=str(r.get("task_id") or f"deepsearchqa-{i}"),
                question=_strip_qa_suffix(q),
                ground_truth=_as_answer_list(ans),
                data_source="deepsearchqa",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k not in ("problem", "question", "query", "input", "answer")
                },
            )
        )
    return tasks


AVERITEC_LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)


def _resolve_averitec_dir(bench_dir: Path) -> Path:
    env_dir = os.environ.get("AVERITEC_DATA_DIR") or os.environ.get("BELIEF_TRACER_AVERITEC_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            bench_dir,
            bench_dir.parent / "AVeriTeC",
            bench_dir.parent / "averitec",
            bench_dir.parent / "sub_AVeriTeC",
            Path("/data/user/baijun/datasets/AVeriTeC"),
        ]
    )
    for candidate in candidates:
        if (candidate / "data" / "dev.json").is_file():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "AVeriTeC dev.json not found. Set AVERITEC_DATA_DIR to the dataset root. "
        f"Searched: {searched}"
    )


def _format_optional_field(label: str, value: Any) -> str:
    if value in (None, "", [], {}):
        return f"{label}: Unknown"
    return f"{label}: {value}"


def _format_averitec_question(sample: dict[str, Any], claim_id: str) -> str:
    lines = [
        "Verify the following real-world claim using the AVeriTeC label set.",
        "",
        f"Claim ID: {claim_id}",
        f"Claim: {sample.get('claim', '')}",
        _format_optional_field("Claim date", sample.get("claim_date")),
        _format_optional_field("Speaker", sample.get("speaker")),
        _format_optional_field("Reporting source", sample.get("reporting_source")),
        _format_optional_field("Location ISO code", sample.get("location_ISO_code")),
        "",
        "Choose exactly one label:",
    ]
    lines.extend(f"- {label}" for label in AVERITEC_LABELS)
    lines.extend(
        [
            "",
            "Use the search tool to gather evidence, then select the label that best fits the evidence.",
        ]
    )
    return "\n".join(lines).strip()


def load_averitec(bench_dir: Path) -> list[AgenticTask]:
    dataset_dir = _resolve_averitec_dir(bench_dir)
    data_file = os.environ.get("AVERITEC_DATA_FILE", "dev.json")
    rows = _load_json(dataset_dir / "data" / data_file)
    tasks: list[AgenticTask] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("claim"):
            continue
        claim_id = str(row.get("claim_id") or row.get("id") or i)
        label = str(row.get("label") or "").strip()
        tasks.append(
            AgenticTask(
                task_id=claim_id,
                question=_format_averitec_question(row, claim_id),
                ground_truth=[label],
                data_source="averitec",
                extra_info={
                    "claim_id": claim_id,
                    "claim": row.get("claim", ""),
                    "label": label,
                    "justification": row.get("justification", ""),
                    "claim_date": row.get("claim_date"),
                    "speaker": row.get("speaker"),
                    "original_claim_url": row.get("original_claim_url"),
                    "fact_checking_article": row.get("fact_checking_article"),
                    "reporting_source": row.get("reporting_source"),
                    "location_ISO_code": row.get("location_ISO_code"),
                    "claim_types": row.get("claim_types") or [],
                    "fact_checking_strategies": row.get("fact_checking_strategies") or [],
                    "questions": row.get("questions") or [],
                    "dataset_dir": str(dataset_dir),
                },
            )
        )
    return tasks


def load_gpqa_diamond(bench_dir: Path) -> list[AgenticTask]:
    # data.json under experiments/artifacts/benchmarks/gpqa_diamond/ exposes
    # `input` (multiple-choice prompt) and `ground_truth_answer` / `target`
    # (one of A/B/C/D). Strip QA suffixes is unnecessary here; the prompt is
    # already a clean MCQ block.
    return _load_data_json(
        bench_dir,
        data_source="gpqa_diamond",
        question_keys=("input", "question", "problem"),
        answer_keys=("ground_truth_answer", "target", "gt_answer", "answer"),
        strip_suffix=False,
    )


BENCHMARK_REGISTRY: dict[str, Callable[[Path], list[AgenticTask]]] = {
    "bamboogle": load_bamboogle,
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "musique": load_musique,
    "gaia": load_gaia,
    "medqa": load_medqa,
    "browse_comp": load_browse_comp,
    "browsecomp": load_browse_comp,
    "browsecomp_plus": load_browsecomp_plus,
    "simpleqa_verified": load_simpleqa_verified,
    "scienceqa": load_scienceqa,
    "hle": load_hle,
    "deepsearchqa": load_deepsearchqa,
    "gpqa_diamond": load_gpqa_diamond,
    "averitec": load_averitec,
}

AVAILABLE_BENCHMARKS = tuple(
    sorted(set(BENCHMARK_REGISTRY) | set(LOCAL_BENCHMARK_FILES))
)


def load_benchmark(
    name: str,
    artifacts_dir: Path | str | None = None,
    max_problems: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 0,
) -> list[AgenticTask]:
    """Load a benchmark by name and return a list of normalized tasks.

    When ``shuffle`` is true (the default), tasks are deterministically
    permuted with ``random.Random(shuffle_seed)`` *before* ``max_problems``
    truncation, so capped runs sample across the whole benchmark rather
    than taking a head slice. Pass ``shuffle=False`` to preserve original
    file order.

    Raises :class:`KeyError` if ``name`` is not registered.
    """
    root = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_BENCHMARKS_DIR
    name = name.lower()
    if name not in BENCHMARK_REGISTRY:
        local_file = _resolve_local_benchmark_file(name, root)
        if local_file is None and not artifacts_dir:
            local_file = _resolve_local_benchmark_file(name, LOCAL_BENCHMARKS_DIR)
        if local_file is None:
            raise KeyError(
                f"Unknown agentic benchmark '{name}'. Registered: {list(AVAILABLE_BENCHMARKS)}"
            )
        tasks = _load_local_benchmark_file(name, local_file)
        if shuffle:
            random.Random(shuffle_seed).shuffle(tasks)
        if max_problems is not None:
            tasks = tasks[:max_problems]
        return tasks
    directory_name = {"browsecomp": "browse_comp"}.get(name, name)
    bench_dir = root / directory_name
    if name == "averitec" and not bench_dir.is_dir():
        tasks = BENCHMARK_REGISTRY[name](bench_dir)
        if shuffle:
            random.Random(shuffle_seed).shuffle(tasks)
        if max_problems is not None:
            tasks = tasks[:max_problems]
        return tasks
    if not bench_dir.is_dir():
        local_file = _resolve_local_benchmark_file(name, root)
        if local_file is None and not artifacts_dir:
            local_file = _resolve_local_benchmark_file(name, LOCAL_BENCHMARKS_DIR)
        if local_file is None:
            raise FileNotFoundError(f"Benchmark directory not found: {bench_dir}")
        tasks = _load_local_benchmark_file(name, local_file)
    else:
        tasks = BENCHMARK_REGISTRY[name](bench_dir)
    if shuffle:
        random.Random(shuffle_seed).shuffle(tasks)
    if max_problems is not None:
        tasks = tasks[:max_problems]
    return tasks
