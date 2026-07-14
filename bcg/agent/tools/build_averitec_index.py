#!/usr/bin/env python3
"""Pre-build AVeriTeC claim indices for rollout-time retrieval.

The cache files are pickle objects whose classes live in
``bcg.agent.tools.averitec_index_types``. Keep build and runtime imports on
that stable module path so caches can be loaded by ``AVeriTeCSearchTool`` during
``bcg agent run``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# resolve paths/imports
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _TOOLS_DIR / "averitec_search_config.json"
_PROJECT_SRC = _TOOLS_DIR.parents[1]
_PROJECT_ROOT = _PROJECT_SRC.parent
_DEFAULT_DATA_DIR = Path("/data/user/baijun/datasets/AVeriTeC")

for candidate in (_PROJECT_SRC, _PROJECT_ROOT / "rllm"):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from bcg.agent.tools.averitec_index_types import _Chunk, _ClaimIndex  # noqa: E402

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency.
    tqdm = None


def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.is_file():
        print(f"[build_index] Config not found: {_CONFIG_PATH}")
        sys.exit(1)
    with _CONFIG_PATH.open() as f:
        cfg = json.load(f)
    return cfg if isinstance(cfg, dict) else {}


# ---------------------------------------------------------------------------
# index building (mirrors AVeriTeCSearchTool internals)
# ---------------------------------------------------------------------------
def _tokenize(text: str, token_re: re.Pattern[str]) -> list[str]:
    return [m.group(0).lower() for m in token_re.finditer(str(text or ""))]


def _build_claim_index(
    member: str,
    knowledge_zip: Path,
    claim_id: str,
    max_chunk_chars: int,
    token_pattern: str,
) -> _ClaimIndex:
    chunks: list[_Chunk] = []
    token_re = re.compile(token_pattern)

    with zipfile.ZipFile(knowledge_zip) as zf:
        with zf.open(member) as f:
            for line_no, raw_line in enumerate(f):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                paragraphs = record.get("url2text") or []
                if isinstance(paragraphs, str):
                    paragraphs = [paragraphs]
                if not isinstance(paragraphs, list):
                    continue
                source_type = str(record.get("type") or "unknown")
                source_query = str(record.get("query") or "")
                url = str(record.get("url") or "")
                buffer: list[str] = []
                buffer_len = 0
                chunk_idx = 0

                def flush() -> None:
                    nonlocal chunk_idx, buffer, buffer_len
                    text = "\n".join(p for p in buffer if p).strip()
                    buffer = []
                    buffer_len = 0
                    if not text:
                        return
                    searchable_text = f"{source_query}\n{url}\n{text}"
                    tokens = tuple(_tokenize(searchable_text, token_re))
                    if not tokens:
                        return
                    chunks.append(
                        _Chunk(
                            chunk_id=f"{line_no}:{chunk_idx}",
                            source_type=source_type,
                            source_query=source_query,
                            url=url,
                            text=text,
                            tokens=tokens,
                            term_counts=Counter(tokens),
                        )
                    )
                    chunk_idx += 1

                for para in paragraphs:
                    para_text = re.sub(r"\s+", " ", str(para or "")).strip()
                    if not para_text:
                        continue
                    while len(para_text) > max_chunk_chars:
                        piece = para_text[:max_chunk_chars]
                        split_at = max(
                            piece.rfind(". "),
                            piece.rfind("; "),
                            piece.rfind(", "),
                        )
                        if split_at < max_chunk_chars // 2:
                            split_at = max_chunk_chars
                        buffer.append(para_text[:split_at].strip())
                        flush()
                        para_text = para_text[split_at:].strip()
                    if buffer_len + len(para_text) > max_chunk_chars and buffer:
                        flush()
                    buffer.append(para_text)
                    buffer_len += len(para_text)
                flush()

    doc_freq: Counter[str] = Counter()
    total_len = 0
    for chunk in chunks:
        doc_freq.update(set(chunk.tokens))
        total_len += len(chunk.tokens)
    num_docs = max(len(chunks), 1)
    idf = {
        term: math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }
    avg_doc_len = total_len / num_docs if chunks else 0.0
    return _ClaimIndex(
        claim_id=claim_id,
        chunks=tuple(chunks),
        idf=idf,
        avg_doc_len=avg_doc_len,
    )


def _cache_dir_for_config(dataset_dir: Path, max_chunk_chars: int) -> Path:
    """Return the cache directory used by AVeriTeCSearchTool._disk_cache_path."""
    return dataset_dir / "data_store" / "index_cache" / f"chunk{max_chunk_chars}"


def _valid_cache(cache_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    try:
        with cache_path.open("rb") as f:
            index = pickle.load(f)
    except Exception:
        return False
    return isinstance(index, _ClaimIndex)


def _build_one(args: tuple[str, Path, str, int, str, Path, bool]) -> tuple[str, str, float, int]:
    member, knowledge_zip, claim_id, max_chunk_chars, token_pattern, cache_dir, force = args
    cache_path = cache_dir / f"{claim_id}.pkl"
    if not force and _valid_cache(cache_path):
        return (claim_id, "cached", 0.0, 0)

    t0 = time.perf_counter()
    index = _build_claim_index(
        member,
        knowledge_zip,
        claim_id,
        max_chunk_chars,
        token_pattern,
    )
    elapsed = time.perf_counter() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    return (claim_id, "built", elapsed, len(index.chunks))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=None, help="AVeriTeC dataset root")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4))
    parser.add_argument("--force", action="store_true", help="Rebuild even if a valid cache exists")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--log-every", type=int, default=0, help="Print a per-claim line every N built indices; 0 disables")
    parser.add_argument(
        "--claim-id",
        action="append",
        dest="claim_ids",
        help="Only build one claim id. May be passed multiple times.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    cfg = _load_config()
    dataset_dir = Path(
        args.dataset_dir
        or os.environ.get("AVERITEC_DATA_DIR")
        or os.environ.get("BELIEF_TRACER_AVERITEC_DIR")
        or cfg.get("dataset_dir")
        or _DEFAULT_DATA_DIR
    )
    knowledge_zip = (
        dataset_dir / "data_store" / "knowledge_store" / "dev_knowledge_store.zip"
    )
    if not knowledge_zip.is_file():
        print(f"[build_index] Knowledge store not found: {knowledge_zip}")
        sys.exit(1)

    max_chunk_chars = int(os.environ.get("AVERITEC_MAX_CHUNK_CHARS", cfg.get("max_chunk_chars", 800)))
    token_pattern = str(cfg.get("token_pattern", r"[A-Za-z0-9]+"))
    cache_dir = _cache_dir_for_config(dataset_dir, max_chunk_chars)
    print(f"[build_index] Dataset:   {dataset_dir}")
    print(f"[build_index] Chunk size: {max_chunk_chars} chars")
    print(f"[build_index] Token regex: {token_pattern}")
    print(f"[build_index] Cache dir:  {cache_dir}")

    requested_claims = {str(cid) for cid in (args.claim_ids or [])}
    with zipfile.ZipFile(knowledge_zip) as zf:
        members = sorted(
            m for m in zf.namelist()
            if m.startswith("output_dev/") and m.endswith(".json") and m != "output_dev/.json"
        )
    if not members:
        print("[build_index] No claim files found in knowledge store.")
        sys.exit(1)

    claims: list[tuple[str, str]] = []
    for member in members:
        claim_id = member.split("/")[-1].replace(".json", "")
        if requested_claims and claim_id not in requested_claims:
            continue
        claims.append((member, claim_id))

    if not claims:
        print(f"[build_index] No matching claims for: {sorted(requested_claims)}")
        sys.exit(1)

    print(f"[build_index] Claims selected: {len(claims)}")

    if args.force:
        valid_existing = 0
        existing_files = sum(1 for _, cid in claims if (cache_dir / f"{cid}.pkl").exists())
        if existing_files:
            print(
                f"[build_index] Existing cache: {existing_files} files; "
                "--force skips expensive validation and rebuilds them."
            )
        to_build = len(claims)
    else:
        valid_existing = 0
        invalid_existing = 0
        for _, cid in claims:
            path = cache_dir / f"{cid}.pkl"
            if _valid_cache(path):
                valid_existing += 1
            elif path.exists():
                invalid_existing += 1
        if valid_existing or invalid_existing:
            print(
                f"[build_index] Existing cache: {valid_existing} valid, "
                f"{invalid_existing} incompatible/stale"
            )
        to_build = len(claims) - valid_existing
    if to_build == 0:
        print("[build_index] All selected indices already cached — nothing to do.")
        return

    workers = max(1, min(int(args.workers), len(claims)))
    print(f"[build_index] Building {to_build} indices with {workers} workers...")

    args_list = [
        (member, knowledge_zip, claim_id, max_chunk_chars, token_pattern, cache_dir, args.force)
        for member, claim_id in claims
    ]
    t_start = time.perf_counter()
    built = 0
    skipped = 0

    progress = None
    use_progress = bool(tqdm is not None and not args.no_progress)
    if use_progress:
        progress = tqdm(
            total=len(args_list),
            desc="[build_index] claims",
            unit="claim",
            dynamic_ncols=True,
        )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, a): a[2] for a in args_list}
        for future in as_completed(futures):
            cid, status, elapsed, chunks = future.result()
            if status == "cached":
                skipped += 1
            else:
                built += 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    built=built,
                    cached=skipped,
                    last=cid,
                    chunks=chunks,
                    refresh=False,
                )
            elif status == "built" or (args.log_every and skipped % args.log_every == 0):
                print(
                    f"  [{built + skipped}/{len(args_list)}] claim {cid}: "
                    f"{status:6s} {chunks:6d} chunks  {elapsed:6.1f}s"
                )
            if args.log_every and status == "built" and built % args.log_every == 0:
                line = (
                    f"[build_index] progress built={built} cached={skipped} "
                    f"last_claim={cid} chunks={chunks} elapsed={elapsed:.1f}s"
                )
                if progress is not None:
                    progress.write(line)
                else:
                    print(line)

    if progress is not None:
        progress.close()

    total_elapsed = time.perf_counter() - t_start
    print(
        f"[build_index] Done. {built} built, {skipped} cached, "
        f"{total_elapsed:.0f}s total."
    )


if __name__ == "__main__":
    main()
