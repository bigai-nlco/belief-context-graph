"""BrowseComp-Plus local corpus retrieval tool."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from rllm.tools.tool_base import Tool, ToolOutput


_DEFAULT_ROOT = (
    Path(__file__).resolve().parents[3]
    / ".."
    / "BrowseComp-Plus"
    / "indexes"
    / "hero-jina-embeddings-v5-text-small"
).resolve()
_DEFAULT_EMBEDDING_MODEL = "jina-embeddings-v5-text-small"
_DEFAULT_EMBEDDING_URL = "http://10.2.152.9:8016/v1/embeddings"
_TASK_DESC = "Given a web search query, retrieve relevant passages that answer the query"


class BCPDenseIndex:
    """Lazy, process-local BrowseComp-Plus dense index."""

    _instances: dict[Path, "BCPDenseIndex"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def for_dir(cls, index_dir: str | Path) -> "BCPDenseIndex":
        path = Path(index_dir).expanduser().resolve()
        with cls._instances_lock:
            inst = cls._instances.get(path)
            if inst is None:
                inst = cls(path)
                cls._instances[path] = inst
            return inst

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        manifest_path = index_dir / "manifest.json"
        self.manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        vectors_path = Path(self.manifest.get("vectors_path", "embeddings.npy"))
        docids_path = Path(self.manifest.get("docids_path", "docids.json"))
        chunk_ids_path = Path(self.manifest.get("chunk_ids_path", "chunk_ids.json"))
        docstore_path = Path(self.manifest.get("docstore_path", "corpus.sqlite"))
        if not vectors_path.is_absolute():
            vectors_path = index_dir / vectors_path
        if not docids_path.is_absolute():
            docids_path = index_dir / docids_path
        if not chunk_ids_path.is_absolute():
            chunk_ids_path = index_dir / chunk_ids_path
        if not docstore_path.is_absolute():
            docstore_path = index_dir / docstore_path

        if not vectors_path.is_file():
            raise FileNotFoundError(f"BrowseComp-Plus vectors not found: {vectors_path}")
        if not docids_path.is_file():
            raise FileNotFoundError(f"BrowseComp-Plus docids not found: {docids_path}")
        if not docstore_path.is_file():
            raise FileNotFoundError(f"BrowseComp-Plus docstore not found: {docstore_path}")

        self.vectors = np.load(vectors_path, mmap_mode="r")
        with docids_path.open("r", encoding="utf-8") as f:
            self.docids = [str(v) for v in json.load(f)]
        if chunk_ids_path.is_file():
            with chunk_ids_path.open("r", encoding="utf-8") as f:
                self.chunk_ids = [str(v) for v in json.load(f)]
        else:
            self.chunk_ids = list(self.docids)
        if self.vectors.shape[0] != len(self.docids):
            raise ValueError(
                f"Vector/docid length mismatch: {self.vectors.shape[0]} vs {len(self.docids)}"
            )
        if self.vectors.shape[0] != len(self.chunk_ids):
            raise ValueError(
                f"Vector/chunk_id length mismatch: {self.vectors.shape[0]} vs {len(self.chunk_ids)}"
            )
        self.conn = sqlite3.connect(docstore_path, check_same_thread=False)
        self._db_lock = threading.Lock()
        self._has_chunks = self._table_exists("chunks")

    def _table_exists(self, table_name: str) -> bool:
        with self._db_lock:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
        return row is not None

    def get_chunk(self, chunk_id: str, docid: str) -> dict[str, Any] | None:
        with self._db_lock:
            if self._has_chunks:
                row = self.conn.execute(
                    "SELECT docid, chunk_index, text FROM chunks WHERE chunk_id = ?",
                    (str(chunk_id),),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT docid, 0, text FROM documents WHERE docid = ?",
                    (str(docid),),
                ).fetchone()
        if row is None:
            return None
        return {
            "docid": str(row[0]),
            "chunk_index": int(row[1]),
            "text": str(row[2]),
        }

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, str, float]]:
        vec = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        scores = self.vectors @ vec
        k = max(1, min(int(top_k), len(scores)))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            (self.docids[int(i)], self.chunk_ids[int(i)], float(scores[int(i)]))
            for i in idx
        ]


class BCPSearchTool(Tool):
    """Search the fixed BrowseComp-Plus local corpus."""

    NAME = "bcp_search"
    FEEDS_MEMORY = True
    DESCRIPTION = (
        "Search the local BrowseComp-Plus corpus for documents relevant to the "
        "current research question."
    )

    def __init__(
        self,
        name: str = NAME,
        description: str | None = None,
        index_dir: str | Path | None = None,
        embedding_model: str | None = None,
        embedding_url: str | None = None,
        api_key: str | None = None,
        api_key_header: str | None = None,
        api_key_prefix: str | None = None,
        max_results: int = 5,
        max_output_chars: int = 6000,
        batch_timeout: float = 120.0,
    ) -> None:
        self.index_dir = Path(
            index_dir
            or os.environ.get("BCP_INDEX_DIR")
            or _DEFAULT_ROOT
        )
        self.embedding_model = (
            embedding_model
            or os.environ.get("HERO_EMBEDDING_MODEL")
            or _DEFAULT_EMBEDDING_MODEL
        )
        self.embedding_url = (
            embedding_url
            or os.environ.get("HERO_EMBEDDING_URL")
            or _DEFAULT_EMBEDDING_URL
        )
        self.api_key = api_key if api_key is not None else os.environ.get("HERO_EMBEDDING_API_KEY", "EMPTY")
        self.api_key_header = api_key_header or os.environ.get("HERO_EMBEDDING_API_KEY_HEADER", "Authorization")
        self.api_key_prefix = api_key_prefix if api_key_prefix is not None else os.environ.get("HERO_EMBEDDING_API_KEY_PREFIX", "Bearer ")
        self.max_results = int(max_results)
        self.max_output_chars = int(max_output_chars)
        self.batch_timeout = float(batch_timeout)
        self._session = requests.Session()
        super().__init__(name=name, description=description or self.DESCRIPTION)

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query for BrowseComp-Plus.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of documents to return (default: {self.max_results}).",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def usage_prompt(self, hyde: bool = False, detail: bool = True) -> str:
        mr = self.max_results
        if not detail:
            return (
                "- bcp_search: search the fixed BrowseComp-Plus corpus. Call as:\n"
                "  <tool_call>\n"
                '  {"name": "bcp_search", "arguments": {"query": "..."}}\n'
                "  </tool_call>\n"
                f"  Optional top_k (integer, default {mr})."
            )
        return (
            f"- bcp_search(query: string, top_k?: integer): search the fixed "
            "BrowseComp-Plus corpus for relevant passages. Use complete natural "
            "language queries. Reformulate when the returned documents do not "
            "answer the question.\n"
            f"  top_k is optional; default {mr}.\n"
            "  Example:\n"
            "  <tool_call>\n"
            '  {"name": "bcp_search", "arguments": {"query": "Who founded the '
            'organization mentioned in the question?"}}\n'
            "  </tool_call>"
        )

    def forward(self, query: str, top_k: int | None = None) -> ToolOutput:
        if not query or not str(query).strip():
            return ToolOutput(name=self.name, error="query must be a non-empty string")
        try:
            index = BCPDenseIndex.for_dir(self.index_dir)
            query_vec = self._embed_query(str(query))
            hits = index.search(query_vec, top_k or self.max_results)
            evidences = []
            for rank, (docid, chunk_id, score) in enumerate(hits, 1):
                chunk = index.get_chunk(chunk_id, docid) or {}
                text = str(chunk.get("text") or "")
                evidences.append(
                    {
                        "rank": rank,
                        "docid": str(chunk.get("docid") or docid),
                        "chunk_id": chunk_id,
                        "chunk_index": int(chunk.get("chunk_index") or 0),
                        "score": score,
                        "text": text,
                        "url": "",
                    }
                )
            return ToolOutput(
                name=self.name,
                output=self._format_results(evidences),
                metadata={
                    "query": query,
                    "num_results": len(evidences),
                    "retriever_type": "bcp_dense_hero",
                    "index_dir": str(self.index_dir),
                    "retrieved_docids": list(dict.fromkeys(e["docid"] for e in evidences)),
                    "evidences": evidences,
                },
            )
        except Exception as exc:
            return ToolOutput(name=self.name, error=f"bcp_search failed: {exc}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = f"{self.api_key_prefix}{self.api_key}"
        return headers

    def _embed_query(self, query: str) -> np.ndarray:
        text = f"Instruct: {_TASK_DESC}\nQuery: {query}"
        payload = {"model": self.embedding_model, "input": [text]}
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = self._session.post(
                    self.embedding_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.batch_timeout,
                )
                resp.raise_for_status()
                vectors = _extract_embeddings(resp.json(), 1)
                arr = np.asarray(vectors[0], dtype=np.float32)
                if arr.ndim != 1:
                    raise ValueError(f"Expected 1D query embedding, got shape {arr.shape}")
                return arr
            except Exception as exc:  # noqa: BLE001 - retry and surface final error
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"embedding request failed after retries: {last_error}")

    def _format_results(self, evidences: list[dict[str, Any]]) -> str:
        if not evidences:
            return "No relevant BrowseComp-Plus documents found."
        parts: list[str] = []
        remaining = self.max_output_chars
        for item in evidences:
            text = str(item.get("text") or "")
            header = (
                f"[Evidence {item['rank']}]\n"
                f"docid: {item['docid']}\n"
                f"chunk_id: {item.get('chunk_id', item['docid'])}\n"
                f"score: {float(item['score']):.4f}\n"
                "text:\n"
            )
            budget = max(300, remaining - len(header))
            if len(text) > budget:
                text = text[:budget].rsplit(" ", 1)[0].strip() + "..."
            block = header + text
            parts.append(block)
            remaining -= len(block)
            if remaining <= 400:
                break
        return "\n\n".join(parts)


def _extract_embeddings(payload: Any, expected: int) -> list[list[float]]:
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


__all__ = ["BCPSearchTool", "BCPDenseIndex"]
