"""OpenAI-compatible retrieval embeddings without local model weights."""

from __future__ import annotations

import importlib
import threading
from collections import OrderedDict
from typing import Any

import httpx

from .client import GraphError
from .vector_store import TongGraphVectorStore


def _chunks(text: str, byte_limit: int) -> list[str]:
    """Lossless UTF-8 chunks, preferring whitespace near the end of each chunk.

    The default 240-byte cap conservatively fits MiniLM's 256-token WordPiece
    window (including special tokens), without downloading its tokenizer.
    This is NOT a token counter for arbitrary models: when changing models,
    configure this cap to fit the deployed tokenizer and context window.
    """
    if byte_limit < 4:
        raise GraphError("pcst_embedding_chunk_limit_invalid")
    raw = text.encode("utf-8")
    chunks: list[str] = []
    while raw:
        part = raw[:byte_limit].decode("utf-8", errors="ignore")
        if len(raw) > byte_limit:
            boundary = max((i + 1 for i, c in enumerate(part) if c.isspace()), default=0)
            if boundary and boundary >= len(part) // 2:
                part = part[:boundary]
        chunks.append(part)
        raw = raw[len(part.encode("utf-8")):]
    return chunks or [""]


class RemoteRelevance:
    """Batch, validate and cache remote vectors; normalize before cosine scoring."""

    def __init__(
        self, url: str, model_name: str, api_key: str, timeout: float, chunk_bytes: int,
        *, store: TongGraphVectorStore | None = None,
    ) -> None:
        self.url, self.model_name = url, model_name
        self.api_key, self.timeout, self.chunk_bytes = api_key, timeout, chunk_bytes
        self.lock = threading.Lock()
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.dimensions: int | None = None
        self.store = store
        self.persisted: set[str] = set()

    def _prepare(self, all_texts: list[str], texts: list[str], np: Any) -> dict[str, Any]:
        """Called under the encoder lock; persist only node text, never queries."""
        audit: dict[str, Any] = {"backend": "memory"}
        if self.store is not None:
            audit = {
                **self.store.audit(), "loaded_texts": 0, "loaded_chunks": 0,
                "written_texts": 0, "written_chunks": 0, "errors": [],
            }
            missing_nodes = list(dict.fromkeys(t for t in texts if t not in self.cache))
            try:
                for text, rows in self.store.load(missing_nodes).items():
                    vectors = np.asarray(rows, dtype=np.float64)
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    if (
                        vectors.ndim != 2 or len(vectors) != len(_chunks(text, self.chunk_bytes))
                        or not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any()
                        or (self.dimensions is not None and vectors.shape[1] != self.dimensions)
                    ):
                        continue
                    self.dimensions = vectors.shape[1]
                    self.cache[text] = vectors / norms
                    self.persisted.add(text)
                    audit["loaded_texts"] += 1
                    audit["loaded_chunks"] += len(vectors)
            except GraphError as exc:
                audit["errors"].append(str(exc))
        missing = list(dict.fromkeys(t for t in all_texts if t not in self.cache))
        chunks: list[str] = []
        spans: dict[str, tuple[int, int]] = {}
        for text in missing:
            start = len(chunks)
            chunks.extend(_chunks(text, self.chunk_bytes))
            spans[text] = (start, len(chunks))
        if chunks:
            vectors = self._encode(chunks, np)
            for text, (start, end) in spans.items():
                self.cache[text] = vectors[start:end]
        if self.store is not None and not audit["errors"]:
            pending = {t: self.cache[t].tolist() for t in dict.fromkeys(texts) if t not in self.persisted}
            try:
                self.store.save(pending)
                self.persisted.update(pending)
                audit["written_texts"] = len(pending)
                audit["written_chunks"] = sum(len(rows) for rows in pending.values())
            except GraphError as exc:
                audit["errors"].append(str(exc))
        return {
            "new_chunks_encoded": len(chunks), "persistent_cache": audit,
            "new_node_chunks_encoded": sum(spans[t][1] - spans[t][0] for t in set(texts) if t in spans),
        }

    def _trim(self, texts: list[str]) -> None:
        for text in dict.fromkeys(texts):
            self.cache.move_to_end(text)
        while len(self.cache) > 4096:
            text, _ = self.cache.popitem(last=False)
            self.persisted.discard(text)

    def persist_nodes(self, texts: list[str]) -> dict[str, Any]:
        """Warm and persist the final graph without encoding any query or running PCST."""
        try:
            np = importlib.import_module("numpy")
            with self.lock:
                audit = self._prepare(texts, texts, np)
                self._trim(texts)
                return {"model": self.model_name, "dimensions": self.dimensions, **audit}
        except GraphError:
            raise
        except Exception as exc:
            raise GraphError("pcst_embedding_failed") from exc

    def _encode(self, chunks: list[str], np: Any) -> Any:
        vectors = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # This internal endpoint must not inherit an unrelated model proxy/key.
        with httpx.Client(timeout=self.timeout, headers=headers, trust_env=False) as client:
            for offset in range(0, len(chunks), 32):
                batch = chunks[offset:offset + 32]
                response = client.post(self.url, json={
                    "model": self.model_name, "input": batch, "encoding_format": "float",
                })
                if response.is_error:
                    raise GraphError(f"pcst_embedding_http_{response.status_code}")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise GraphError("pcst_embedding_response_invalid")
                if payload.get("model", self.model_name) != self.model_name:
                    raise GraphError("pcst_embedding_model_mismatch")
                data = payload.get("data")
                if not isinstance(data, list) or len(data) != len(batch):
                    raise GraphError("pcst_embedding_response_invalid")
                ordered: dict[int, list[float]] = {}
                for item in data:
                    if not isinstance(item, dict):
                        raise GraphError("pcst_embedding_response_invalid")
                    index, vector = item.get("index"), item.get("embedding")
                    if (
                        type(index) is not int or not 0 <= index < len(batch) or index in ordered
                        or not isinstance(vector, list) or not vector
                        or any(type(v) not in {int, float} for v in vector)
                    ):
                        raise GraphError("pcst_embedding_response_invalid")
                    ordered[index] = vector
                vectors.extend(ordered[i] for i in range(len(batch)))
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise GraphError("pcst_embedding_vectors_invalid")
        if self.dimensions is not None and matrix.shape[1] != self.dimensions:
            raise GraphError("pcst_embedding_dimensions_changed")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise GraphError("pcst_embedding_vectors_invalid")
        self.dimensions = matrix.shape[1]
        return matrix / norms

    def score(
        self, question: str, focus: str, texts: list[str],
    ) -> tuple[list[float], dict[str, Any]]:
        try:
            np = importlib.import_module("numpy")
            with self.lock:
                all_texts = [question, focus or question, *texts]
                prepared = self._prepare(all_texts, texts, np)
                task_vectors, focus_vectors = (self.cache[t] for t in all_texts[:2])
                scores = []
                for text in texts:
                    vectors = self.cache[text]
                    task_score = max(0.0, float(np.max(vectors @ task_vectors.T)))
                    focus_score = max(0.0, float(np.max(vectors @ focus_vectors.T)))
                    scores.append(min(1.0, 0.5 * task_score + 0.5 * focus_score))
                audit = {
                    "provider": "openai", "url": self.url, "model": self.model_name,
                    "dimensions": self.dimensions, "chunking": "utf8_bytes",
                    "chunk_bytes": self.chunk_bytes,
                    "question_chunks": len(task_vectors), "focus_chunks": len(focus_vectors),
                    "node_chunks": sum(len(self.cache[t]) for t in texts),
                    **prepared,
                }
                self._trim(all_texts)
                return scores, audit
        except GraphError:
            raise
        except httpx.TimeoutException as exc:
            raise GraphError("pcst_embedding_timeout") from exc
        except httpx.RequestError as exc:
            raise GraphError("pcst_embedding_unavailable") from exc
        except Exception as exc:
            # Never persist HTTP response bodies or authentication in audit logs.
            raise GraphError("pcst_embedding_failed") from exc
