"""Persistent, content-addressed node-text embeddings in TongGraph.

Each text has a manifest node and one vector-indexed node per text chunk. The
manifest is published last, so interrupted writes cannot expose a partial text.
This namespace is an embedding cache; BCG snapshots remain authoritative for
belief identity, confidence and edges. No existing graph records are deleted.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote

import httpx

from .client import GraphError


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TongGraphVectorStore:
    def __init__(
        self, url: str, token: str, graph: str, timeout: float, descriptor: dict[str, Any],
    ) -> None:
        self.url, self.token, self.graph, self.timeout = url, token, graph, timeout
        self.descriptor = descriptor
        self.fingerprint = text_key(json.dumps(descriptor, sort_keys=True))
        self.logical_graph_id = "bcg_pcst_embeddings_" + self.fingerprint[:20]
        self.index = "bcg_pcst_" + self.fingerprint[:20]
        self.path = "/graphs/" + quote(graph, safe="")

    def audit(self) -> dict[str, Any]:
        return {
            "backend": "tonggraph", "url": self.url, "graph": self.graph,
            "logical_graph_id": self.logical_graph_id, "index": self.index,
            "encoder_fingerprint": self.fingerprint,
        }

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.url, headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout, trust_env=False,
        )

    def _request(
        self, client: httpx.Client, method: str, path: str, *,
        payload: dict[str, Any] | None = None, missing_ok: bool = False,
        conflict_ok: bool = False, scoped: bool = True,
    ) -> dict[str, Any]:
        params = None
        if scoped:
            if method == "GET":
                params = {"logical_graph_id": self.logical_graph_id}
            else:
                payload = {**(payload or {}), "logical_graph_id": self.logical_graph_id}
        try:
            response = client.request(method, self.path + path, params=params, json=payload)
            if missing_ok and response.status_code == 404:
                return {}
            if conflict_ok and response.status_code == 409:
                return {"conflict": True}
            if not response.is_success:
                raise GraphError(f"tonggraph_http_{response.status_code}")
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("expected object")
            return value
        except GraphError:
            raise
        except httpx.HTTPError as exc:
            raise GraphError("tonggraph_unavailable") from exc
        except ValueError as exc:
            raise GraphError("tonggraph_response_invalid") from exc

    def _node_id(self, client: httpx.Client, external_id: str) -> int | None:
        value = self._request(
            client, "GET", "/nodes/by-external-id/" + quote(external_id, safe=""), missing_ok=True,
        ).get("id")
        if value is not None and type(value) is not int:
            raise GraphError("tonggraph_node_id_invalid")
        return value

    def _ensure_nodes(self, client: httpx.Client, records: list[dict[str, Any]]) -> list[int]:
        # Stable external IDs also recover nodes created by an interrupted write.
        for _ in range(3):
            ids = [self._node_id(client, r["external_id"]) for r in records]
            missing = [i for i, node_id in enumerate(ids) if node_id is None]
            if not missing:
                return [int(i) for i in ids if i is not None]
            result = self._request(
                client, "POST", "/nodes/batch", conflict_ok=True,
                payload={"records": [records[i] for i in missing]},
            )
            if result.get("conflict"):
                continue  # Another writer may have created these immutable keys.
            created = result.get("ids")
            if not isinstance(created, list) or len(created) != len(missing) or any(type(i) is not int for i in created):
                raise GraphError("tonggraph_node_ids_invalid")
            for position, node_id in zip(missing, created, strict=True):
                ids[position] = node_id
            return [int(i) for i in ids if i is not None]
        raise GraphError("tonggraph_concurrent_write_conflict")

    def _external_id(self, text: str) -> str:
        return f"bcg:embedding:{self.fingerprint}:{text_key(text)}"

    def load(self, texts: list[str]) -> dict[str, list[list[float]]]:
        if not texts:
            return {}
        with self._client() as client:
            self._request(client, "POST", "/logical-graphs", conflict_ok=True)

            def read(text: str) -> tuple[str, list[list[float]] | None]:
                node_id = self._node_id(client, self._external_id(text))
                if node_id is None:
                    return text, None
                node = self._request(client, "GET", f"/nodes/{node_id}", missing_ok=True).get("node")
                if not isinstance(node, dict):
                    return text, None
                props = node.get("properties", {})
                if (
                    not isinstance(props, dict) or props.get("encoder_fingerprint") != self.fingerprint
                    or props.get("text_sha256") != text_key(text) or props.get("text") != text
                ):
                    return text, None
                try:
                    ids = json.loads(props.get("chunk_ids", "[]"))
                    dimensions = props.get("dimensions")
                    if (
                        not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                        or any(type(i) is not int for i in ids)
                        or type(dimensions) is not int or dimensions <= 0
                    ):
                        return text, None
                    vectors = []
                    for chunk_id in ids:
                        vector = self._request(
                            client, "GET", f"/vector/{self.index}/{chunk_id}", missing_ok=True,
                        ).get("vector")
                        if (
                            not isinstance(vector, list) or len(vector) != dimensions
                            or any(type(v) not in {int, float} or not math.isfinite(v) for v in vector)
                            or not any(v != 0 for v in vector)
                        ):
                            return text, None
                        vectors.append(vector)
                    return text, vectors
                except (TypeError, ValueError):
                    return text, None

            pool = ThreadPoolExecutor(max_workers=8)
            try:
                return {text: vectors for text, vectors in pool.map(read, texts) if vectors is not None}
            finally:
                # A failed server must not make every queued text wait for its timeout.
                pool.shutdown(wait=True, cancel_futures=True)

    def _ensure_index(self, client: httpx.Client, dimensions: int) -> None:
        def matches() -> bool:
            indexes = self._request(client, "GET", "/vector/indexes", scoped=False).get("indexes", [])
            for index in indexes:
                if index.get("name") == self.index:
                    if any(index.get(k) != v for k, v in {
                        "dimensions": dimensions, "target": "node", "metric": "cosine",
                        "model": self.descriptor["model"], "model_version": self.fingerprint,
                    }.items()):
                        raise GraphError("tonggraph_index_schema_mismatch")
                    return True
            return False
        if not matches():
            self._request(client, "POST", "/vector/indexes", scoped=False, conflict_ok=True, payload={
                "name": self.index, "dimensions": dimensions, "target": "node", "metric": "cosine",
                "model": self.descriptor["model"], "model_version": self.fingerprint,
            })
            if not matches():
                raise GraphError("tonggraph_index_missing")

    def save(self, texts: dict[str, list[list[float]]]) -> None:
        if not texts:
            return
        dimensions = len(next(iter(texts.values()))[0])
        with self._client() as client:
            self._request(client, "POST", "/logical-graphs", conflict_ok=True)
            self._ensure_index(client, dimensions)

            def write(item: tuple[str, list[list[float]]]) -> None:
                text, vectors = item
                external = self._external_id(text)
                props = {
                    "text": text, "text_sha256": text_key(text),
                    "encoder_fingerprint": self.fingerprint,
                    "encoder_config": json.dumps(self.descriptor, sort_keys=True),
                }
                parent = self._ensure_nodes(client, [{
                    "external_id": external, "labels": ["BCGRetrievalText"], "properties": props,
                }])[0]
                ids = self._ensure_nodes(client, [{
                    "external_id": f"{external}:chunk:{i}", "labels": ["BCGRetrievalChunk"],
                    "properties": {"text_sha256": text_key(text), "text_node_id": parent, "chunk_index": i},
                } for i in range(len(vectors))])
                for offset in range(0, len(ids), 32):
                    self._request(client, "PUT", f"/vector/{self.index}/batch", payload={
                        "vectors": dict(zip(ids[offset:offset + 32], vectors[offset:offset + 32], strict=True)),
                    })
                # Publish the complete ordered list only after every vector write succeeds.
                self._request(client, "PATCH", f"/nodes/{parent}", payload={"set_properties": {
                    **props, "dimensions": dimensions, "chunk_count": len(ids), "chunk_ids": json.dumps(ids),
                }})

            pool = ThreadPoolExecutor(max_workers=8)
            try:
                list(pool.map(write, texts.items()))
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
