"""Dataset adapters for the benchmarks supported by ``bcg benchmark``."""

from __future__ import annotations

import csv
import json
import math
import string
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from bcg.apps.benchmark.models import BenchmarkTask

BENCHMARKS = (
    "browsecomp",
    "gaia",
    "hotpotqa",
    "mmlu_pro",
)

_DIRECTORY_CANDIDATES = {
    "browsecomp": ("browsecomp", "browse_comp", "BrowseComp"),
    "gaia": ("gaia", "GAIA"),
    "hotpotqa": ("hotpotqa", "hotpot_qa", "HotpotQA"),
    "mmlu_pro": ("mmlu_pro", "MMLU-Pro", "MMLU_Pro"),
}

_GAIA_MULTIMODAL_TOOL_MARKERS = (
    "image",
    "video",
    "audio",
    "ocr",
    "visual",
    "gif",
    "photo",
    "picture",
    "color recognition",
    "colour recognition",
    "computer vision",
)


class BenchmarkDataError(ValueError):
    """Raised when benchmark files are absent or have an unsupported schema."""


def load_benchmark(
    name: str,
    data_root: Path,
    *,
    data_file: Path | None = None,
    split: str | None = None,
    gaia_level: int | None = None,
    gaia_text_only: bool = False,
) -> list[BenchmarkTask]:
    """Load one benchmark and normalize it to :class:`BenchmarkTask` objects."""

    canonical = canonical_name(name)
    root = Path(data_root).expanduser().resolve()
    if canonical == "gaia":
        return _load_gaia(
            root,
            data_file=data_file,
            split=split or "validation",
            level=gaia_level,
            text_only=gaia_text_only,
        )

    source = (
        Path(data_file).expanduser().resolve()
        if data_file is not None
        else _find_data_file(canonical, root, split)
    )
    rows = _read_rows(source, split=split)
    loader = {
        "browsecomp": _load_browsecomp_rows,
        "hotpotqa": _load_hotpotqa_rows,
        "mmlu_pro": _load_mmlu_pro_rows,
    }[canonical]
    tasks = loader(rows)
    if not tasks:
        raise BenchmarkDataError(f"No valid {canonical} tasks found in {source}.")
    return tasks


def canonical_name(name: str) -> str:
    canonical = name.strip().lower().replace("-", "_")
    if canonical not in BENCHMARKS:
        supported = ", ".join(BENCHMARKS)
        raise BenchmarkDataError(
            f"Unsupported benchmark {name!r}. Supported benchmarks: {supported}."
        )
    return canonical


def _benchmark_dir(name: str, data_root: Path) -> Path:
    for directory in _DIRECTORY_CANDIDATES[name]:
        candidate = data_root / directory
        if candidate.is_dir():
            return candidate
    searched = ", ".join(
        str(data_root / value) for value in _DIRECTORY_CANDIDATES[name]
    )
    raise BenchmarkDataError(
        f"Dataset directory for {name} was not found. Searched: {searched}."
    )


def _find_data_file(name: str, data_root: Path, split: str | None) -> Path:
    directory = _benchmark_dir(name, data_root)
    stems = [split] if split else []
    stems.extend(("data", "test", "validation", "dev"))
    candidates: list[Path] = []
    for stem in stems:
        if not stem:
            continue
        for suffix in (".json", ".jsonl", ".csv", ".parquet"):
            candidates.extend(
                (directory / f"{stem}{suffix}", directory / "data" / f"{stem}{suffix}")
            )

    if name == "mmlu_pro":
        candidates.insert(0, directory / "mmlu_pro_test_processed.json")
    elif name == "hotpotqa":
        candidates[:0] = [
            directory / "hotpot_dev_fullwiki_v1.json",
            directory / "hotpot_dev_distractor_v1.json",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(path) for path in candidates[:8])
    raise BenchmarkDataError(
        f"No data file found for {name} under {directory}. "
        f"Use --data-file {name}=PATH. Expected one of: {expected}."
    )


def _read_rows(path: Path, *, split: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BenchmarkDataError(f"Benchmark data file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            selected = value.get(split or "") if split else None
            selected = selected or value.get("data") or value.get("examples")
            if not isinstance(selected, list):
                raise BenchmarkDataError(
                    f"Expected a JSON list (or data/examples list) in {path}."
                )
            rows = selected
        else:
            raise BenchmarkDataError(f"Expected a JSON list or object in {path}.")
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise BenchmarkDataError(
                "Reading Parquet requires the benchmark extras: "
                "`uv sync --extra benchmarks`."
            ) from exc
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise BenchmarkDataError(f"Unsupported benchmark file type: {path}")

    return [row for row in rows if isinstance(row, dict)]


def _first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None or (isinstance(value, str) and not value):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return None


def _answers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def _task_id(row: dict[str, Any], prefix: str, index: int) -> str:
    value = _first(
        row,
        (
            "task_id",
            "id",
            "_id",
            "query_id",
            "question_id",
            "problem_id",
            "unique_id",
        ),
    )
    return str(value) if value is not None else f"{prefix}-{index:04d}"


def _load_browsecomp_rows(rows: list[dict[str, Any]]) -> list[BenchmarkTask]:
    tasks = []
    for index, row in enumerate(rows):
        question = _first(row, ("input", "question", "query", "problem"))
        answers = _answers(
            _first(row, ("ground_truth_answer", "answer", "target", "reference_answer"))
        )
        if question is None or not answers:
            continue
        tasks.append(
            BenchmarkTask(
                benchmark="browsecomp",
                task_id=_task_id(row, "browsecomp", index),
                question=str(question).strip(),
                answers=answers,
                metadata=_without(
                    row,
                    {
                        "input",
                        "question",
                        "query",
                        "problem",
                        "ground_truth_answer",
                        "answer",
                        "target",
                        "reference_answer",
                    },
                ),
            )
        )
    return tasks


def _load_hotpotqa_rows(rows: list[dict[str, Any]]) -> list[BenchmarkTask]:
    tasks = []
    for index, row in enumerate(rows):
        question = _first(row, ("question", "input", "query"))
        answers = _answers(_first(row, ("answer", "ground_truth_answer", "target")))
        if question is None or not answers:
            continue
        tasks.append(
            BenchmarkTask(
                benchmark="hotpotqa",
                task_id=_task_id(row, "hotpotqa", index),
                question=str(question).strip(),
                answers=answers,
                metadata=_without(
                    row,
                    {
                        "question",
                        "input",
                        "query",
                        "answer",
                        "ground_truth_answer",
                        "target",
                    },
                ),
            )
        )
    return tasks


def _choice_lines(choices: Any) -> tuple[list[str], dict[str, str]]:
    labels = list(string.ascii_uppercase)
    mapping: dict[str, str] = {}
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    if isinstance(choices, dict):
        for key, value in choices.items():
            mapping[str(key).strip().upper()] = str(value)
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        mapping = {labels[index]: str(value) for index, value in enumerate(choices)}
    lines = [f"{label}. {value}" for label, value in mapping.items()]
    return lines, mapping


def _mmlu_answer(row: dict[str, Any], mapping: dict[str, str]) -> tuple[str, ...]:
    answer = _first(row, ("answer", "answer_index", "label", "target"))
    if isinstance(answer, int) or (
        isinstance(answer, str) and answer.strip().isdigit()
    ):
        index = int(answer)
        if 0 <= index < len(mapping):
            return (list(mapping)[index],)
    text = str(answer or "").strip().upper()
    if text in mapping:
        return (text,)
    for label, value in mapping.items():
        if text == value.strip().upper():
            return (label,)
    return _answers(answer)


def _load_mmlu_pro_rows(rows: list[dict[str, Any]]) -> list[BenchmarkTask]:
    tasks = []
    for index, row in enumerate(rows):
        question = _first(row, ("question", "input", "problem"))
        lines, mapping = _choice_lines(_first(row, ("options", "choices")))
        answers = _mmlu_answer(row, mapping)
        if question is None or not lines or not answers:
            continue
        prompt = f"{str(question).strip()}\n\n" + "\n".join(lines)
        tasks.append(
            BenchmarkTask(
                benchmark="mmlu_pro",
                task_id=_task_id(row, "mmlu_pro", index),
                question=prompt,
                answers=answers,
                metadata={
                    **_without(
                        row,
                        {
                            "question",
                            "input",
                            "problem",
                            "options",
                            "choices",
                            "answer",
                            "answer_index",
                            "label",
                            "target",
                        },
                    ),
                    "choice_labels": list(mapping),
                },
            )
        )
    return tasks


def _load_gaia(
    data_root: Path,
    *,
    data_file: Path | None,
    split: str,
    level: int | None,
    text_only: bool,
) -> list[BenchmarkTask]:
    if split not in {"validation", "test"}:
        raise BenchmarkDataError("GAIA split must be 'validation' or 'test'.")
    if data_file is not None:
        source = Path(data_file).expanduser().resolve()
        split_dir = source.parent
    else:
        directory = _benchmark_dir("gaia", data_root)
        split_candidates = (
            directory / "2023" / split,
            directory / split,
        )
        split_dir = next((path for path in split_candidates if path.is_dir()), None)
        if split_dir is None:
            raise BenchmarkDataError(
                f"GAIA {split} directory was not found under {directory}."
            )
        source = next(
            (
                split_dir / filename
                for filename in ("metadata.jsonl", "metadata.json", "metadata.parquet")
                if (split_dir / filename).is_file()
            ),
            None,
        )
        if source is None:
            raise BenchmarkDataError(f"GAIA metadata was not found in {split_dir}.")

    tasks = []
    for index, row in enumerate(_read_rows(source, split=split)):
        row_level = _first(row, ("Level", "level"))
        try:
            normalized_level = int(row_level) if row_level is not None else None
        except (TypeError, ValueError):
            normalized_level = None
        if level is not None and normalized_level != level:
            continue

        filename = str(_first(row, ("file_name", "filename", "file")) or "").strip()
        requires_multimodal = _gaia_requires_multimodal_tools(row)
        if text_only and (filename or requires_multimodal):
            continue
        attachment = split_dir / filename if filename else None
        if attachment is not None and not attachment.is_file():
            raise BenchmarkDataError(
                f"GAIA attachment for task {_task_id(row, 'gaia', index)} "
                f"does not exist: {attachment}"
            )

        question = _first(row, ("Question", "question", "input"))
        answer = _first(row, ("Final answer", "final_answer", "answer"))
        answers = _answers(answer)
        if question is None:
            continue
        if split == "validation" and not answers:
            continue
        tasks.append(
            BenchmarkTask(
                benchmark="gaia",
                task_id=_task_id(row, "gaia", index),
                question=str(question).strip(),
                answers=answers,
                attachment=attachment,
                metadata={
                    **_without(
                        row,
                        {
                            "Question",
                            "question",
                            "input",
                            "Final answer",
                            "final_answer",
                            "answer",
                            "file_name",
                            "filename",
                            "file",
                        },
                    ),
                    "level": normalized_level,
                    "split": split,
                    "modality": (
                        "multimodal" if filename or requires_multimodal else "text"
                    ),
                },
            )
        )
    if not tasks:
        filters = " after applying the requested filters" if level or text_only else ""
        raise BenchmarkDataError(f"No valid GAIA tasks found in {source}{filters}.")
    return tasks


def _gaia_requires_multimodal_tools(row: dict[str, Any]) -> bool:
    """Use GAIA's own annotator tool declaration to reject hidden media tasks."""

    annotation = _first(row, ("Annotator Metadata", "annotator_metadata"))
    if isinstance(annotation, dict):
        tools = _first(annotation, ("Tools", "tools"))
    else:
        tools = None
    if tools is None:
        tools = _first(row, ("Tools", "tools"))
    normalized = str(tools or "").casefold()
    return any(marker in normalized for marker in _GAIA_MULTIMODAL_TOOL_MARKERS)


def _without(row: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in excluded}


__all__ = [
    "BENCHMARKS",
    "BenchmarkDataError",
    "load_benchmark",
    "canonical_name",
]
