#!/usr/bin/env python3
"""Download and normalize BrowseComp and GAIA for BeliefTracer.

BrowseComp is downloaded from OpenAI's official simple-evals release. Its
question and answer fields are base64-encoded XOR ciphertext; conversion here
uses the exact decryption algorithm from ``openai/simple-evals``.

The canonical GAIA repository is gated. By default this script downloads the
public ``smolagents/GAIA-annotated`` mirror maintained by Hugging Face, which
contains the GAIA 2023 validation/test metadata and attachments. Set
``--gaia-official`` after accepting the dataset terms and exporting ``HF_TOKEN``
to download ``gaia-benchmark/GAIA`` instead.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bcg.env import load_project_env

load_project_env()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets"
BROWSECOMP_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)
GAIA_PUBLIC_REPO = "smolagents/GAIA-annotated"
GAIA_OFFICIAL_REPO = "gaia-benchmark/GAIA"


def derive_key(password: str, length: int) -> bytes:
    """Derive the repeated SHA-256 key used by the BrowseComp release."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt_browsecomp(ciphertext_b64: str, password: str) -> str:
    """Decrypt one official BrowseComp field."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key, strict=True)).decode("utf-8")


def convert_browsecomp_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert official encrypted CSV rows to BeliefTracer's ``data.json`` shape."""
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        canary = str(row.get("canary") or "")
        encrypted_problem = str(row.get("problem") or "")
        encrypted_answer = str(row.get("answer") or "")
        if not canary or not encrypted_problem or not encrypted_answer:
            raise ValueError(
                f"BrowseComp row {index} is missing canary/problem/answer fields"
            )
        try:
            question = decrypt_browsecomp(encrypted_problem, canary)
            answer = decrypt_browsecomp(encrypted_answer, canary)
        except Exception as exc:
            raise ValueError(
                f"Could not decrypt BrowseComp row {index}: {exc}"
            ) from exc
        converted.append(
            {
                "task_id": f"browsecomp-{index:04d}",
                "input": question,
                "ground_truth_answer": answer,
                "extra_info": {
                    "source": "openai/simple-evals BrowseComp",
                    "source_row": index,
                },
            }
        )
    return converted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _download(url: str, destination: Path, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"[reuse] {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BeliefTracer benchmark downloader/1.0"},
    )
    print(f"[download] {url}")
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    print(f"[saved] {destination} ({destination.stat().st_size:,} bytes)")
    return destination


def prepare_browsecomp(
    data_root: Path,
    *,
    url: str = BROWSECOMP_URL,
    force: bool = False,
) -> dict[str, Any]:
    target = data_root / "browse_comp"
    raw_path = _download(
        url,
        target / "raw" / "browse_comp_test_set.csv",
        force=force,
    )
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        tasks = convert_browsecomp_rows(csv.DictReader(handle))
    if not tasks:
        raise RuntimeError("The downloaded BrowseComp CSV contained no examples")

    output_path = target / "data.json"
    _write_json(output_path, tasks)
    manifest = {
        "benchmark": "BrowseComp",
        "source_url": url,
        "source_sha256": _sha256(raw_path),
        "num_tasks": len(tasks),
        "prepared_at": datetime.now(UTC).isoformat(),
        "data_file": str(output_path),
    }
    _write_json(target / "manifest.json", manifest)
    print(f"[ready] BrowseComp: {len(tasks)} tasks -> {output_path}")
    return manifest


def _gaia_metadata_path(target: Path, split: str) -> Path | None:
    base = target / "2023" / split
    for name in ("metadata.jsonl", "metadata.parquet"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _gaia_metadata_stats(path: Path) -> tuple[int, int]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - runtime dependency guidance
            raise RuntimeError(
                "pandas and pyarrow are required for GAIA parquet"
            ) from exc
        rows = pd.read_parquet(path).to_dict(orient="records")
    attachments = sum(bool(str(row.get("file_name") or "").strip()) for row in rows)
    return len(rows), attachments


def prepare_gaia(
    data_root: Path,
    *,
    repo_id: str = GAIA_PUBLIC_REPO,
    force: bool = False,
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - runtime dependency guidance
        raise RuntimeError(
            "huggingface_hub is required; install the project's runtime dependencies"
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if repo_id == GAIA_OFFICIAL_REPO and not token:
        raise RuntimeError(
            "The official GAIA repository is gated. Accept its terms at "
            "https://huggingface.co/datasets/gaia-benchmark/GAIA and export HF_TOKEN."
        )

    target = data_root / "gaia"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] GAIA repository {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["2023/**", "GAIA.py", "README.md"],
        token=token,
        force_download=force,
    )

    split_stats: dict[str, Any] = {}
    for split in ("validation", "test"):
        metadata_path = _gaia_metadata_path(target, split)
        if metadata_path is None:
            raise RuntimeError(
                f"GAIA {split} metadata was not downloaded under {target}"
            )
        count, attachments = _gaia_metadata_stats(metadata_path)
        split_stats[split] = {
            "metadata_file": str(metadata_path),
            "num_tasks": count,
            "tasks_with_attachments": attachments,
        }

    revision = ""
    with suppress(Exception):
        revision = HfApi().dataset_info(repo_id=repo_id, token=token).sha or ""
    manifest = {
        "benchmark": "GAIA 2023",
        "repo_id": repo_id,
        "revision": revision,
        "prepared_at": datetime.now(UTC).isoformat(),
        "default_split": "validation",
        "splits": split_stats,
    }
    _write_json(target / "manifest.json", manifest)
    print(
        "[ready] GAIA: "
        + ", ".join(
            f"{name}={stats['num_tasks']}" for name, stats in split_stats.items()
        )
    )
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "browsecomp", "gaia"),
        default="all",
        help="Download both benchmarks or just one (default: all).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Benchmark root (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument("--browsecomp-url", default=BROWSECOMP_URL)
    parser.add_argument(
        "--gaia-repo",
        default=GAIA_PUBLIC_REPO,
        help=f"GAIA dataset repository (default: {GAIA_PUBLIC_REPO}).",
    )
    parser.add_argument(
        "--gaia-official",
        action="store_true",
        help=f"Use gated {GAIA_OFFICIAL_REPO}; requires accepted access and HF_TOKEN.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload source files even when local copies exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.only in {"all", "browsecomp"}:
            prepare_browsecomp(
                data_root,
                url=args.browsecomp_url,
                force=args.force,
            )
        if args.only in {"all", "gaia"}:
            prepare_gaia(
                data_root,
                repo_id=GAIA_OFFICIAL_REPO if args.gaia_official else args.gaia_repo,
                force=args.force,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
