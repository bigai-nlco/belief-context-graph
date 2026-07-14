"""Build the local artifact consumed by bcg.agent.tools.bcp_search.

Output layout:
  manifest.json
  embeddings.npy
  docids.json
  chunk_ids.json
  corpus.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import requests
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from bcg.env import load_project_env


load_project_env()

DEFAULT_MODEL = "jina-embeddings-v5-text-small"
DEFAULT_URL = "http://10.2.152.9:8016/v1/embeddings"
DEFAULT_TOKENIZER = "jinaai/jina-embeddings-v5-text-small"


class ChunkRecord(NamedTuple):
    row_index: int
    doc_index: int
    docid: str
    chunk_id: str
    chunk_index: int
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BrowseComp-Plus dense index for bcp_search.")
    parser.add_argument("--dataset-name", default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        default="../BrowseComp-Plus/indexes/hero-jina-embeddings-v5-text-small",
    )
    parser.add_argument(
        "--embedding-url",
        default=os.environ.get("HERO_EMBEDDING_URL", DEFAULT_URL),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("HERO_EMBEDDING_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--api-key", default=os.environ.get("HERO_EMBEDDING_API_KEY", "EMPTY"))
    parser.add_argument(
        "--api-key-header",
        default=os.environ.get("HERO_EMBEDDING_API_KEY_HEADER", "Authorization"),
    )
    parser.add_argument(
        "--api-key-prefix",
        default=os.environ.get("HERO_EMBEDDING_API_KEY_PREFIX", "Bearer "),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--chunk-mode",
        choices=("token", "char"),
        default="token",
        help="Use tokenizer-aware chunks by default; char mode is only a fallback.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=os.environ.get("HERO_TOKENIZER_MODEL", DEFAULT_TOKENIZER),
        help="Tokenizer used when --chunk-mode token.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per embedded corpus chunk when --chunk-mode token.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=512,
        help="Token overlap between adjacent chunks when --chunk-mode token.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=8000,
        help="Maximum characters per embedded corpus chunk when --chunk-mode char.",
    )
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=400,
        help="Character overlap between adjacent chunks.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--extra-json",
        default=None,
        help="Optional JSON object merged into every embedding request payload.",
    )
    parser.add_argument("--id-field", default="docid")
    parser.add_argument("--text-field", default="text")
    return parser.parse_args()


def headers(args: argparse.Namespace) -> dict[str, str]:
    out = {"Content-Type": "application/json"}
    if args.api_key:
        out[args.api_key_header] = f"{args.api_key_prefix}{args.api_key}"
    return out


def extract_embeddings(payload: Any, expected: int) -> list[list[float]]:
    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            return payload
        if expected == 1 and payload and isinstance(payload[0], (int, float)):
            return [payload]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported embedding response type: {type(payload).__name__}")
    if "data" in payload:
        rows = sorted(payload["data"], key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in rows]
    for key in ("embeddings", "vectors"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    value = payload.get("embedding")
    if isinstance(value, list):
        return [value]
    raise ValueError(f"Cannot find embeddings in response keys: {sorted(payload.keys())}")


def chunk_spans(text: str, chunk_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    if chunk_chars <= 0:
        raise ValueError("--chunk-chars must be positive")
    if overlap_chars < 0:
        raise ValueError("--chunk-overlap-chars must be non-negative")
    if overlap_chars >= chunk_chars:
        raise ValueError("--chunk-overlap-chars must be smaller than --chunk-chars")

    if not text:
        return [(0, 0)]
    n = len(text)
    if n <= chunk_chars:
        return [(0, n)]

    spans: list[tuple[int, int]] = []
    start = 0
    min_tail = max(256, chunk_chars // 8)
    while start < n:
        hard_end = min(start + chunk_chars, n)
        end = hard_end
        if hard_end < n:
            split_at = text.rfind(" ", start + min_tail, hard_end)
            if split_at > start:
                end = split_at
        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end >= n:
            break
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = start + max(1, chunk_chars - overlap_chars)
        start = next_start
    return spans


def token_spans(token_count: int, chunk_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    if chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("--chunk-overlap-tokens must be non-negative")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("--chunk-overlap-tokens must be smaller than --chunk-tokens")
    if token_count <= 0:
        return [(0, 0)]
    if token_count <= chunk_tokens:
        return [(0, token_count)]

    spans: list[tuple[int, int]] = []
    start = 0
    step = chunk_tokens - overlap_tokens
    while start < token_count:
        end = min(start + chunk_tokens, token_count)
        spans.append((start, end))
        if end >= token_count:
            break
        start += step
    return spans


def plan_chunks(
    ds: Any,
    id_field: str,
    text_field: str,
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for doc_index in tqdm(range(len(ds)), desc="Planning BCP chunks"):
        item = ds[int(doc_index)]
        docid = str(item[id_field])
        text = str(item.get(text_field) or "")
        if args.chunk_mode == "token":
            if tokenizer is None:
                raise ValueError("tokenizer is required for --chunk-mode token")
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            spans = token_spans(
                len(token_ids),
                args.chunk_tokens,
                args.chunk_overlap_tokens,
            )
        else:
            spans = chunk_spans(text, args.chunk_chars, args.chunk_overlap_chars)
        for chunk_index, (start, end) in enumerate(spans):
            chunk_id = f"{docid}#chunk-{chunk_index:04d}"
            records.append(
                ChunkRecord(
                    row_index=len(records),
                    doc_index=int(doc_index),
                    docid=docid,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    start=start,
                    end=end,
                )
            )
    return records


def record_text(
    ds: Any,
    text_field: str,
    record: ChunkRecord,
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> str:
    text = str(ds[int(record.doc_index)].get(text_field) or "")
    if args.chunk_mode == "token":
        if tokenizer is None:
            raise ValueError("tokenizer is required for --chunk-mode token")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunk = tokenizer.decode(
            token_ids[record.start : record.end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    else:
        chunk = text[record.start : record.end]
    return chunk if chunk else "empty"


def embed_batch(
    session: requests.Session,
    texts: list[str],
    args: argparse.Namespace,
    request_headers: dict[str, str],
    extra: dict[str, Any],
) -> np.ndarray:
    payload = {"model": args.model_name, "input": texts}
    payload.update(extra)
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            resp = session.post(
                args.embedding_url,
                headers=request_headers,
                json=payload,
                timeout=args.timeout,
            )
            if not resp.ok:
                raise RuntimeError(
                    f"{resp.status_code} {resp.reason}: {resp.text[:1000]}"
                )
            vectors = extract_embeddings(resp.json(), len(texts))
            if len(vectors) != len(texts):
                raise ValueError(f"Expected {len(texts)} vectors, got {len(vectors)}")
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D embedding array, got shape {arr.shape}")
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Embedding request failed after {args.retries} attempts: {last_error}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra = json.loads(args.extra_json) if args.extra_json else {}
    if not isinstance(extra, dict):
        raise ValueError("--extra-json must decode to a JSON object")

    ds = load_dataset(args.dataset_name, split=args.split)
    documents_total = len(ds)
    if documents_total <= 0:
        raise ValueError("Corpus dataset is empty")

    tokenizer = None
    if args.chunk_mode == "token":
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=True,
        )

    records = plan_chunks(
        ds,
        args.id_field,
        args.text_field,
        args,
        tokenizer,
    )
    total = len(records)
    if total <= 0:
        raise ValueError("Corpus dataset produced no chunks")

    db_path = output_dir / "corpus.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        "CREATE TABLE chunks ("
        "chunk_id TEXT PRIMARY KEY, "
        "docid TEXT NOT NULL, "
        "chunk_index INTEGER NOT NULL, "
        "text TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX idx_chunks_docid ON chunks(docid)")

    session = requests.Session()
    request_headers = headers(args)

    first_records = records[0 : min(args.batch_size, total)]
    first_texts = [record_text(ds, args.text_field, rec, args, tokenizer) for rec in first_records]
    first_vecs = embed_batch(session, first_texts, args, request_headers, extra)
    dim = int(first_vecs.shape[1])

    vectors_path = output_dir / "embeddings.npy"
    vectors = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, dim),
    )
    docids: list[str] = [""] * total
    chunk_ids: list[str] = [""] * total

    def write_rows(start: int, batch_records: list[ChunkRecord], texts: list[str], arr: np.ndarray) -> None:
        end = start + len(batch_records)
        vectors[start:end] = arr
        docids[start:end] = [rec.docid for rec in batch_records]
        chunk_ids[start:end] = [rec.chunk_id for rec in batch_records]
        conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, docid, chunk_index, text) VALUES (?, ?, ?, ?)",
            (
                (rec.chunk_id, rec.docid, rec.chunk_index, text)
                for rec, text in zip(batch_records, texts)
            ),
        )
        conn.commit()

    write_rows(0, first_records, first_texts, first_vecs)

    for start in tqdm(range(len(first_records), total, args.batch_size), desc="Encoding BCP chunks"):
        batch_records = records[start : min(start + args.batch_size, total)]
        texts = [record_text(ds, args.text_field, rec, args, tokenizer) for rec in batch_records]
        arr = embed_batch(session, texts, args, request_headers, extra)
        if arr.shape[1] != dim:
            raise ValueError(f"Embedding dim changed from {dim} to {arr.shape[1]}")
        write_rows(start, batch_records, texts, arr)

    vectors.flush()
    conn.close()

    (output_dir / "docids.json").write_text(
        json.dumps(docids, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "chunk_ids.json").write_text(
        json.dumps(chunk_ids, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model": args.model_name,
        "embedding_url": args.embedding_url,
        "normalized": True,
        "unit": "chunk",
        "chunk_mode": args.chunk_mode,
        "documents": documents_total,
        "chunks": total,
        "tokenizer": args.tokenizer_name if args.chunk_mode == "token" else None,
        "chunk_tokens": args.chunk_tokens if args.chunk_mode == "token" else None,
        "chunk_overlap_tokens": args.chunk_overlap_tokens if args.chunk_mode == "token" else None,
        "chunk_chars": args.chunk_chars,
        "chunk_overlap_chars": args.chunk_overlap_chars,
        "dimension": dim,
        "vectors_path": "embeddings.npy",
        "docids_path": "docids.json",
        "chunk_ids_path": "chunk_ids.json",
        "docstore_path": "corpus.sqlite",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote BrowseComp-Plus bcp_search index to {output_dir}")


if __name__ == "__main__":
    main()
