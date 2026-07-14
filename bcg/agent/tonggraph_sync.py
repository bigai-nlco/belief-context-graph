"""Sync BeliefTracer graph snapshots into TongGraph Server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bcg.cli_help import RichArgumentParser

from bcg.env import load_project_env

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://10.2.152.51:8719"
DEFAULT_GRAPH = "agent_workspace"
DEFAULT_TEXT_INDEX = "agent_text"
DEFAULT_VECTOR_INDEX = "agent_embedding"
SERVER_LOGICAL_GRAPH_PROPERTY = "_tg_logical_graph_id"

class TongGraphSyncError(RuntimeError):
    """Raised when TongGraph Server rejects a sync request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id


@dataclass(frozen=True)
class TongGraphSyncResult:
    graph: str
    logical_graph_id: str
    nodes_seen: int
    nodes_created: int
    nodes_reused: int
    edges_seen: int
    edges_created: int
    edges_reused: int
    nodes_deleted: int
    edges_deleted: int
    text_index: str | None
    embedding_index: str | None = None
    embeddings_upserted: int = 0
    embedding_dimensions: int | None = None
    readback_verified: bool = False
    readback_mismatches: list[str] = field(default_factory=list)


class TongGraphHTTPClient:
    """Small stdlib HTTP client for the TongGraph endpoints used here."""

    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            encoded = urllib.parse.urlencode(
                {key: value for key, value in query.items() if value is not None}
            )
            if encoded:
                url = f"{url}?{encoded}"

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.token}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            raise _server_error(body, status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise TongGraphSyncError(f"TongGraph request failed: {exc}") from exc

        if not body:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TongGraphSyncError("TongGraph returned non-JSON response") from exc

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def create_logical_graph(self, graph: str, logical_graph_id: str) -> None:
        try:
            self.request(
                "POST",
                f"/graphs/{urllib.parse.quote(graph)}/logical-graphs",
                payload={"logical_graph_id": logical_graph_id},
            )
        except TongGraphSyncError as exc:
            if exc.code != "conflict":
                raise

    def get_node_id(self, graph: str, external_id: str, logical_graph_id: str) -> int | None:
        quoted_graph = urllib.parse.quote(graph)
        quoted_external_id = urllib.parse.quote(external_id, safe="")
        result = self.request(
            "GET",
            f"/graphs/{quoted_graph}/nodes/by-external-id/{quoted_external_id}",
            query={"logical_graph_id": logical_graph_id},
        )
        value = result.get("id")
        return int(value) if value is not None else None

    def get_node(self, graph: str, node_id: int, logical_graph_id: str) -> dict[str, Any]:
        result = self.request(
            "GET",
            f"/graphs/{urllib.parse.quote(graph)}/nodes/{node_id}",
            query={"logical_graph_id": logical_graph_id},
        )
        node = result.get("node") or {}
        return dict(node) if isinstance(node, dict) else {}

    def node_ids(self, graph: str, logical_graph_id: str) -> list[int]:
        result = self.request(
            "GET",
            f"/graphs/{urllib.parse.quote(graph)}/nodes",
            query={"logical_graph_id": logical_graph_id},
        )
        return [int(node_id) for node_id in result.get("ids", [])]

    def update_node(
        self,
        graph: str,
        node_id: int,
        logical_graph_id: str,
        *,
        add_labels: list[str] | None = None,
        remove_labels: list[str] | None = None,
        set_properties: dict[str, Any] | None = None,
        remove_properties: list[str] | None = None,
    ) -> dict[str, Any]:
        result = self.request(
            "PATCH",
            f"/graphs/{urllib.parse.quote(graph)}/nodes/{node_id}",
            payload={
                "add_labels": add_labels,
                "remove_labels": remove_labels,
                "set_properties": set_properties,
                "remove_properties": remove_properties,
                "logical_graph_id": logical_graph_id,
            },
        )
        node = result.get("node") or {}
        return dict(node) if isinstance(node, dict) else {}

    def delete_node(self, graph: str, node_id: int, logical_graph_id: str) -> bool:
        result = self.request(
            "DELETE",
            f"/graphs/{urllib.parse.quote(graph)}/nodes/{node_id}",
            query={"detach": "false", "logical_graph_id": logical_graph_id},
        )
        return bool(result.get("deleted"))

    def add_nodes(
        self,
        graph: str,
        logical_graph_id: str,
        records: list[dict[str, Any]],
    ) -> list[int]:
        if not records:
            return []
        result = self.request(
            "POST",
            f"/graphs/{urllib.parse.quote(graph)}/nodes/batch",
            payload={"logical_graph_id": logical_graph_id, "records": records},
        )
        return [int(value) for value in result.get("ids", [])]

    def edge_exists(
        self,
        graph: str,
        logical_graph_id: str,
        *,
        edge_type: str,
        agent_edge_id: str,
    ) -> bool:
        spec = {
            "match": [
                {"node": "source"},
                {
                    "edge": "edge",
                    "type": edge_type,
                    "direction": "out",
                    "properties": {"agent_edge_id": agent_edge_id},
                },
                {"node": "target"},
            ],
            "return": ["edge"],
            "limit": 1,
        }
        result = self.request(
            "POST",
            f"/graphs/{urllib.parse.quote(graph)}/query",
            payload={"logical_graph_id": logical_graph_id, "spec": spec},
        ).get("result", [])
        if isinstance(result, dict):
            rows = result.get("rows", [])
        else:
            rows = result
        return bool(rows)

    def edge_ids_with_property(
        self,
        graph: str,
        logical_graph_id: str,
        *,
        key: str,
        value: Any,
    ) -> list[int]:
        result = self.request(
            "GET",
            f"/graphs/{urllib.parse.quote(graph)}/edges/by-property",
            query={"key": key, "value": value, "logical_graph_id": logical_graph_id},
        )
        return [int(edge_id) for edge_id in result.get("ids", [])]

    def add_edges(
        self,
        graph: str,
        logical_graph_id: str,
        records: list[dict[str, Any]],
    ) -> list[int]:
        if not records:
            return []
        result = self.request(
            "POST",
            f"/graphs/{urllib.parse.quote(graph)}/edges/batch",
            payload={"logical_graph_id": logical_graph_id, "records": records},
        )
        return [int(value) for value in result.get("ids", [])]

    def get_edge(self, graph: str, edge_id: int, logical_graph_id: str) -> dict[str, Any]:
        result = self.request(
            "GET",
            f"/graphs/{urllib.parse.quote(graph)}/edges/{edge_id}",
            query={"logical_graph_id": logical_graph_id},
        )
        edge = result.get("edge") or {}
        return dict(edge) if isinstance(edge, dict) else {}

    def edge_ids(self, graph: str, logical_graph_id: str) -> list[int]:
        result = self.request(
            "GET",
            f"/graphs/{urllib.parse.quote(graph)}/edges",
            query={"logical_graph_id": logical_graph_id},
        )
        return [int(edge_id) for edge_id in result.get("ids", [])]

    def update_edge(
        self,
        graph: str,
        edge_id: int,
        logical_graph_id: str,
        *,
        set_properties: dict[str, Any] | None = None,
        remove_properties: list[str] | None = None,
    ) -> dict[str, Any]:
        result = self.request(
            "PATCH",
            f"/graphs/{urllib.parse.quote(graph)}/edges/{edge_id}",
            payload={
                "set_properties": set_properties,
                "remove_properties": remove_properties,
                "logical_graph_id": logical_graph_id,
            },
        )
        edge = result.get("edge") or {}
        return dict(edge) if isinstance(edge, dict) else {}

    def delete_edge(self, graph: str, edge_id: int, logical_graph_id: str) -> bool:
        result = self.request(
            "DELETE",
            f"/graphs/{urllib.parse.quote(graph)}/edges/{edge_id}",
            query={"logical_graph_id": logical_graph_id},
        )
        return bool(result.get("deleted"))

    def create_fulltext_index(
        self,
        graph: str,
        name: str,
        properties: list[str],
        *,
        target: str = "node",
    ) -> None:
        try:
            self.request(
                "POST",
                f"/graphs/{urllib.parse.quote(graph)}/fulltext/indexes",
                payload={"name": name, "properties": properties, "target": target},
            )
        except TongGraphSyncError as exc:
            if exc.code not in {"conflict", "invalid_request"}:
                raise
            logger.debug("Ignoring existing fulltext index %s: %s", name, exc)

    def vector_indexes(self, graph: str) -> list[dict[str, Any]]:
        result = self.request("GET", f"/graphs/{urllib.parse.quote(graph)}/vector/indexes")
        indexes = result.get("indexes") or []
        return [dict(index) for index in indexes if isinstance(index, dict)]

    def ensure_vector_index(
        self,
        graph: str,
        name: str,
        *,
        dimensions: int,
        target: str = "node",
        metric: str = "cosine",
        model: str | None = None,
    ) -> None:
        for index in self.vector_indexes(graph):
            if index.get("name") != name:
                continue
            existing_dimensions = index.get("dimensions")
            existing_target = index.get("target")
            if existing_dimensions not in {None, dimensions} or existing_target not in {None, target}:
                raise TongGraphSyncError(
                    f"vector index {name!r} exists with incompatible schema: {index}"
                )
            return
        self.request(
            "POST",
            f"/graphs/{urllib.parse.quote(graph)}/vector/indexes",
            payload={
                "name": name,
                "dimensions": dimensions,
                "target": target,
                "metric": metric,
                "model": model,
            },
        )

    def upsert_vectors(
        self,
        graph: str,
        index: str,
        logical_graph_id: str,
        vectors: dict[int, list[float]],
    ) -> int:
        if not vectors:
            return 0
        self.request(
            "PUT",
            f"/graphs/{urllib.parse.quote(graph)}/vector/{urllib.parse.quote(index)}/batch",
            payload={"logical_graph_id": logical_graph_id, "vectors": vectors},
        )
        return len(vectors)

    def search_vector(
        self,
        graph: str,
        index: str,
        logical_graph_id: str,
        query_vector: list[float],
        *,
        labels: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        result = self.request(
            "POST",
            f"/graphs/{urllib.parse.quote(graph)}/vector/{urllib.parse.quote(index)}/search",
            payload={
                "logical_graph_id": logical_graph_id,
                "query_vector": query_vector,
                "labels": labels,
                "limit": limit,
            },
        )
        rows = result.get("results") or []
        return [dict(row) for row in rows if isinstance(row, dict)]


class EmbeddingHTTPClient:
    """OpenAI-compatible embedding client used before writing vectors."""

    def __init__(self, url: str, model: str, *, batch_size: int = 16, timeout: float = 120.0) -> None:
        self.url = url
        self.model = model
        self.batch_size = max(1, int(batch_size or 16))
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        cleaned = [str(text or "")[:8000].strip() or "empty" for text in texts]
        vectors: list[list[float] | None] = [None] * len(cleaned)
        for start in range(0, len(cleaned), self.batch_size):
            chunk = cleaned[start:start + self.batch_size]
            for offset, vector in enumerate(self._embed_chunk(chunk)):
                vectors[start + offset] = vector
        missing = [index for index, vector in enumerate(vectors) if vector is None]
        if missing:
            raise TongGraphSyncError(f"embedding API did not return vectors for positions {missing}")
        return [vector for vector in vectors if vector is not None]

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise TongGraphSyncError(f"embedding API HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise TongGraphSyncError(f"embedding API request failed: {exc}") from exc

        payload = json.loads(body.decode("utf-8"))
        if "data" in payload:
            data_rows = sorted(payload["data"], key=lambda row: row.get("index", 0))
            return [[float(value) for value in row["embedding"]] for row in data_rows]
        if "embeddings" in payload and isinstance(payload["embeddings"], dict):
            return [[float(value) for value in vector] for vector in payload["embeddings"]["float"]]
        raise TongGraphSyncError("embedding API returned an unsupported response shape")


def _server_error(body: bytes, *, status_code: int) -> TongGraphSyncError:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return TongGraphSyncError(
            f"TongGraph HTTP {status_code}: {body.decode('utf-8', errors='replace')}",
            status_code=status_code,
        )
    error = payload.get("error") or {}
    message = error.get("message") or f"TongGraph HTTP {status_code}"
    return TongGraphSyncError(
        message,
        status_code=status_code,
        code=error.get("code"),
        request_id=error.get("request_id"),
    )


def load_graph_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict):
                return dict(item)
        raise ValueError(f"{path} contains a list but no graph object")
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a graph object or graph snapshot list")
    return payload


def infer_logical_graph_id(path: Path | None, payload: dict[str, Any], explicit: str = "") -> str:
    if explicit:
        return _safe_id(explicit)
    env_value = os.environ.get("TONGGRAPH_LOGICAL_GRAPH_ID", "")
    if env_value:
        return _safe_id(env_value)
    item_id = payload.get("item_id")
    if item_id:
        return _safe_id(str(item_id))
    if path is not None:
        if path.name == "final_graph.json" and path.parent.name:
            return _safe_id(path.parent.name)
        return _safe_id(path.stem)
    return "trajectory"


def sync_graph_file(
    path: Path,
    *,
    base_url: str,
    token: str,
    graph: str = DEFAULT_GRAPH,
    logical_graph_id: str = "",
    text_index: str | None = DEFAULT_TEXT_INDEX,
    embedding_url: str = "",
    embedding_model: str = "",
    embedding_index: str | None = None,
    embedding_batch_size: int = 16,
    timeout: float = 30.0,
    skip_existing_edges: bool = True,
    verify_readback: bool = True,
) -> TongGraphSyncResult:
    payload = load_graph_payload(path)
    resolved_logical = infer_logical_graph_id(path, payload, logical_graph_id)
    return sync_graph_payload(
        payload,
        base_url=base_url,
        token=token,
        graph=graph,
        logical_graph_id=resolved_logical,
        text_index=text_index,
        embedding_url=embedding_url,
        embedding_model=embedding_model,
        embedding_index=embedding_index,
        embedding_batch_size=embedding_batch_size,
        timeout=timeout,
        skip_existing_edges=skip_existing_edges,
        verify_readback=verify_readback,
    )


def sync_graph_payload(
    payload: dict[str, Any],
    *,
    base_url: str,
    token: str,
    graph: str = DEFAULT_GRAPH,
    logical_graph_id: str,
    text_index: str | None = DEFAULT_TEXT_INDEX,
    embedding_url: str = "",
    embedding_model: str = "",
    embedding_index: str | None = None,
    embedding_batch_size: int = 16,
    timeout: float = 30.0,
    skip_existing_edges: bool = True,
    verify_readback: bool = True,
) -> TongGraphSyncResult:
    client = TongGraphHTTPClient(base_url, token=token, timeout=timeout)
    client.create_logical_graph(graph, logical_graph_id)

    item_id = str(payload.get("item_id") or logical_graph_id)
    generated_at = payload.get("generated_at")
    nodes = list(iter_agent_nodes(payload))
    records = [
        build_node_record(node, item_id=item_id, generated_at=generated_at, logical_graph_id=logical_graph_id)
        for node in nodes
    ]
    logger.info(
        "[TongGraph] Latest-state sync started graph=%s logical_graph_id=%s "
        "snapshot_nodes=%d snapshot_relations=%d",
        graph,
        logical_graph_id,
        len(records),
        len(iter_relations(payload)),
    )

    id_map: dict[str, int] = {}
    missing_records: list[dict[str, Any]] = []
    missing_keys: list[str] = []
    reused_records: list[tuple[int, dict[str, Any]]] = []
    for node, record in zip(nodes, records, strict=True):
        key = _agent_id_key(node["id"])
        existing_id = client.get_node_id(graph, record["external_id"], logical_graph_id)
        if existing_id is None:
            missing_records.append(record)
            missing_keys.append(key)
        else:
            id_map[key] = existing_id
            reused_records.append((existing_id, record))

    created_ids = client.add_nodes(graph, logical_graph_id, missing_records)
    for key, internal_id in zip(missing_keys, created_ids, strict=True):
        id_map[key] = internal_id

    for internal_id, record in reused_records:
        _update_existing_node(client, graph, logical_graph_id, internal_id, record)

    edge_records = build_edge_records(
        payload,
        id_map=id_map,
        item_id=item_id,
        logical_graph_id=logical_graph_id,
    )
    new_edge_records: list[dict[str, Any]] = []
    new_edge_keys: list[str] = []
    edge_id_map: dict[str, int] = {}
    reused_edges = 0
    for record in edge_records:
        agent_edge_id = str((record.get("properties") or {}).get("agent_edge_id") or "")
        if skip_existing_edges and agent_edge_id:
            try:
                existing_edge_ids = client.edge_ids_with_property(
                    graph, logical_graph_id, key="agent_edge_id", value=agent_edge_id
                )
                if existing_edge_ids:
                    edge_id = existing_edge_ids[0]
                    _update_existing_edge(client, graph, logical_graph_id, edge_id, record)
                    edge_id_map[agent_edge_id] = edge_id
                    reused_edges += 1
                    continue
            except TongGraphSyncError as exc:
                logger.warning("Edge existence check failed; creating edge anyway: %s", exc)
        new_edge_records.append(record)
        new_edge_keys.append(agent_edge_id)

    created_edge_ids = client.add_edges(graph, logical_graph_id, new_edge_records)
    for key, edge_id in zip(new_edge_keys, created_edge_ids, strict=True):
        if key:
            edge_id_map[key] = edge_id

    edges_deleted, nodes_deleted = _delete_stale_graph_records(
        client,
        graph=graph,
        logical_graph_id=logical_graph_id,
        desired_node_ids=set(id_map.values()),
        desired_edge_ids=set(edge_id_map.values()),
    )
    logger.info(
        "[TongGraph] Latest-state reconciliation graph=%s logical_graph_id=%s "
        "desired_nodes=%d deleted_nodes=%d desired_edges=%d deleted_edges=%d",
        graph,
        logical_graph_id,
        len(id_map),
        nodes_deleted,
        len(edge_id_map),
        edges_deleted,
    )
    if text_index:
        client.create_fulltext_index(
            graph,
            text_index,
            ["text", "belief", "decision"],
            target="node",
        )

    embeddings_upserted = 0
    embedding_dimensions: int | None = None
    if embedding_url and embedding_model and embedding_index:
        texts_by_key = {
            _agent_id_key(node["id"]): str(
                properties_for(
                    node,
                    item_id=item_id,
                    generated_at=generated_at,
                    logical_graph_id=logical_graph_id,
                ).get("text", "")
            )
            for node in nodes
        }
        vector_entity_ids = [id_map[key] for key in texts_by_key if key in id_map]
        vector_texts = [texts_by_key[key] for key in texts_by_key if key in id_map]
        if vector_texts:
            embedder = EmbeddingHTTPClient(
                embedding_url,
                embedding_model,
                batch_size=embedding_batch_size,
                timeout=max(timeout, 120.0),
            )
            vectors = embedder.embed(vector_texts)
            embedding_dimensions = len(vectors[0]) if vectors else None
            if embedding_dimensions:
                client.ensure_vector_index(
                    graph,
                    embedding_index,
                    dimensions=embedding_dimensions,
                    target="node",
                    metric="cosine",
                    model=embedding_model,
                )
                embeddings_upserted = client.upsert_vectors(
                    graph,
                    embedding_index,
                    logical_graph_id,
                    {entity_id: vector for entity_id, vector in zip(vector_entity_ids, vectors, strict=True)},
                )

    readback_mismatches: list[str] = []
    if verify_readback:
        readback_mismatches = verify_tonggraph_readback(
            client,
            graph=graph,
            logical_graph_id=logical_graph_id,
            node_records=records,
            edge_records=edge_records,
            id_map={
                record["external_id"]: id_map[_agent_id_key(node["id"])]
                for node, record in zip(nodes, records, strict=True)
                if _agent_id_key(node["id"]) in id_map
            },
            edge_id_map=edge_id_map,
        )
        if readback_mismatches:
            preview = "; ".join(readback_mismatches[:5])
            suffix = "" if len(readback_mismatches) <= 5 else f"; +{len(readback_mismatches) - 5} more"
            logger.error(
                "[TongGraph] Full-graph readback failed graph=%s logical_graph_id=%s "
                "mismatches=%d: %s%s",
                graph,
                logical_graph_id,
                len(readback_mismatches),
                preview,
                suffix,
            )
            raise TongGraphSyncError(f"TongGraph full-graph readback verification failed: {preview}{suffix}")
        logger.info(
            "[TongGraph] Full-graph readback verified graph=%s logical_graph_id=%s "
            "nodes=%d edges=%d",
            graph,
            logical_graph_id,
            len(records),
            len(edge_records),
        )

    return TongGraphSyncResult(
        graph=graph,
        logical_graph_id=logical_graph_id,
        nodes_seen=len(records),
        nodes_created=len(created_ids),
        nodes_reused=len(records) - len(created_ids),
        edges_seen=len(edge_records),
        edges_created=len(created_edge_ids),
        edges_reused=reused_edges,
        nodes_deleted=nodes_deleted,
        edges_deleted=edges_deleted,
        text_index=text_index,
        embedding_index=embedding_index if embeddings_upserted else None,
        embeddings_upserted=embeddings_upserted,
        embedding_dimensions=embedding_dimensions,
        readback_verified=verify_readback,
        readback_mismatches=readback_mismatches,
    )


def iter_agent_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = payload.get("nodes")
    if isinstance(nodes, list) and nodes:
        return [_normalize_node(dict(node), index=index) for index, node in enumerate(nodes)]

    normalized: list[dict[str, Any]] = []
    for index, belief in enumerate(payload.get("beliefs") or []):
        if isinstance(belief, dict):
            node = dict(belief)
        else:
            node = {"belief": str(belief)}
        node.setdefault("node_type", "belief")
        node.setdefault("id", index)
        normalized.append(_normalize_node(node, index=index))

    offset = len(normalized)
    for index, decision in enumerate(payload.get("decisions") or []):
        if isinstance(decision, dict):
            node = dict(decision)
        else:
            node = {"decision": str(decision)}
        node.setdefault("node_type", "decision")
        node.setdefault("id", offset + index)
        normalized.append(_normalize_node(node, index=offset + index))
    return normalized


def build_node_record(
    node: dict[str, Any],
    *,
    item_id: str,
    generated_at: Any,
    logical_graph_id: str,
) -> dict[str, Any]:
    agent_id = node["id"]
    return {
        "external_id": f"agent:{logical_graph_id}:node:{agent_id}",
        "labels": labels_for(node),
        "properties": properties_for(
            node,
            item_id=item_id,
            generated_at=generated_at,
            logical_graph_id=logical_graph_id,
        ),
    }


def build_edge_records(
    payload: dict[str, Any],
    *,
    id_map: dict[str, int],
    item_id: str,
    logical_graph_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, relation in enumerate(iter_relations(payload)):
        from_key = _agent_id_key(relation.get("from_id"))
        to_key = _agent_id_key(relation.get("to_id"))
        source = id_map.get(from_key)
        target = id_map.get(to_key)
        if source is None or target is None:
            logger.warning("Skipping relation with missing endpoint: %s", relation)
            continue
        raw_type = str(relation.get("type") or relation.get("edge_type") or "related_to")
        normalized_type = edge_type(raw_type)
        agent_edge_id = f"agent:{logical_graph_id}:edge:{from_key}:{normalized_type}:{to_key}:{index}"
        records.append(
            {
                "source": source,
                "target": target,
                "edge_type": normalized_type,
                "properties": edge_properties_for(
                    relation,
                    item_id=item_id,
                    logical_graph_id=logical_graph_id,
                    agent_edge_id=agent_edge_id,
                ),
            }
        )
    return records


def iter_relations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    relations = payload.get("relations") or payload.get("forward_relations") or []
    return [dict(relation) for relation in relations if isinstance(relation, dict)]


def labels_for(node: dict[str, Any]) -> list[str]:
    labels = ["AgentNode"]
    kind = str(node.get("node_type") or "").strip().lower()
    if kind == "belief" or "belief" in node:
        labels.append("Belief")
    elif kind == "decision" or "decision" in node:
        labels.append("Decision")
    else:
        labels.append("AgentRecord")
    return labels


def properties_for(
    node: dict[str, Any],
    *,
    item_id: str,
    generated_at: Any,
    logical_graph_id: str,
) -> dict[str, Any]:
    text = node.get("belief") or node.get("decision") or node.get("text") or ""
    return _clean_properties({
        "agent_id": node.get("id"),
        "item_id": item_id,
        "logical_graph_id": logical_graph_id,
        "generated_at": generated_at,
        "node_type": node.get("node_type"),
        "belief": node.get("belief"),
        "decision": node.get("decision"),
        "text": text,
        "stance": node.get("stance"),
        "entities": node.get("entities") or [],
        "event_time": node.get("event_time"),
        "time_text": node.get("time_text"),
        "source": node.get("source") or {},
        "evidence": node.get("evidence") or [],
        "supporting_excerpts": node.get("supporting_excerpts") or [],
        "confidence": node.get("confidence"),
        "initial_confidence": node.get("initial_confidence"),
        "confidence_history": node.get("confidence_history") or [],
        "payload_json": node,
    })


def edge_properties_for(
    relation: dict[str, Any],
    *,
    item_id: str,
    logical_graph_id: str,
    agent_edge_id: str,
) -> dict[str, Any]:
    return _clean_properties({
        "agent_edge_id": agent_edge_id,
        "type": relation.get("type") or relation.get("edge_type"),
        "note": relation.get("note"),
        "item_id": item_id,
        "logical_graph_id": logical_graph_id,
        "from_agent_id": relation.get("from_id"),
        "to_agent_id": relation.get("to_id"),
        "payload_json": relation,
    })


def verify_tonggraph_readback(
    client: TongGraphHTTPClient,
    *,
    graph: str,
    logical_graph_id: str,
    node_records: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
    id_map: dict[str, int],
    edge_id_map: dict[str, int],
) -> list[str]:
    """Fetch the complete logical graph and compare it with the latest snapshot."""
    mismatches: list[str] = []
    actual_node_ids = set(client.node_ids(graph, logical_graph_id))
    expected_node_ids = set(id_map.values())
    missing_node_ids = sorted(expected_node_ids - actual_node_ids)
    extra_node_ids = sorted(actual_node_ids - expected_node_ids)
    if missing_node_ids:
        mismatches.append(f"logical graph: missing node ids {missing_node_ids}")
    if extra_node_ids:
        mismatches.append(f"logical graph: unexpected node ids {extra_node_ids}")

    actual_edge_ids = set(client.edge_ids(graph, logical_graph_id))
    expected_edge_ids = set(edge_id_map.values())
    missing_edge_ids = sorted(expected_edge_ids - actual_edge_ids)
    extra_edge_ids = sorted(actual_edge_ids - expected_edge_ids)
    if missing_edge_ids:
        mismatches.append(f"logical graph: missing edge ids {missing_edge_ids}")
    if extra_edge_ids:
        mismatches.append(f"logical graph: unexpected edge ids {extra_edge_ids}")

    for record in node_records:
        external_id = str(record.get("external_id") or "")
        node_id = id_map.get(external_id)
        if node_id is None:
            mismatches.append(f"node {external_id}: no internal id after sync")
            continue
        if node_id not in actual_node_ids:
            continue
        actual = client.get_node(graph, node_id, logical_graph_id)
        mismatches.extend(_compare_node_record(record, actual))

    for record in edge_records:
        agent_edge_id = str((record.get("properties") or {}).get("agent_edge_id") or "")
        edge_id = edge_id_map.get(agent_edge_id)
        if edge_id is None:
            mismatches.append(f"edge {agent_edge_id}: no internal id after sync")
            continue
        if edge_id not in actual_edge_ids:
            continue
        actual = client.get_edge(graph, edge_id, logical_graph_id)
        mismatches.extend(_compare_edge_record(record, actual))
    return mismatches


def _delete_stale_graph_records(
    client: TongGraphHTTPClient,
    *,
    graph: str,
    logical_graph_id: str,
    desired_node_ids: set[int],
    desired_edge_ids: set[int],
) -> tuple[int, int]:
    """Delete records outside the latest snapshot, with edges removed first."""
    existing_edge_ids = set(client.edge_ids(graph, logical_graph_id))
    stale_edge_ids = sorted(existing_edge_ids - desired_edge_ids)
    existing_node_ids = set(client.node_ids(graph, logical_graph_id))
    stale_node_ids = sorted(existing_node_ids - desired_node_ids)

    if stale_edge_ids or stale_node_ids:
        logger.info(
            "[TongGraph] Removing stale records graph=%s logical_graph_id=%s "
            "edge_ids=%s node_ids=%s",
            graph,
            logical_graph_id,
            stale_edge_ids,
            stale_node_ids,
        )

    edges_deleted = 0
    for edge_id in stale_edge_ids:
        if client.delete_edge(graph, edge_id, logical_graph_id):
            edges_deleted += 1

    nodes_deleted = 0
    for node_id in stale_node_ids:
        if client.delete_node(graph, node_id, logical_graph_id):
            nodes_deleted += 1

    return edges_deleted, nodes_deleted


def _update_existing_node(
    client: TongGraphHTTPClient,
    graph: str,
    logical_graph_id: str,
    node_id: int,
    record: dict[str, Any],
) -> None:
    existing = client.get_node(graph, node_id, logical_graph_id)
    existing_labels = set(existing.get("labels") or [])
    desired_labels = set(record.get("labels") or [])
    existing_props = _strip_server_properties(existing.get("properties") or {})
    desired_props = dict(record.get("properties") or {})
    remove_properties = sorted(set(existing_props) - set(desired_props))
    client.update_node(
        graph,
        node_id,
        logical_graph_id,
        add_labels=sorted(desired_labels - existing_labels) or None,
        remove_labels=sorted(existing_labels - desired_labels) or None,
        set_properties=desired_props,
        remove_properties=remove_properties or None,
    )


def _update_existing_edge(
    client: TongGraphHTTPClient,
    graph: str,
    logical_graph_id: str,
    edge_id: int,
    record: dict[str, Any],
) -> None:
    existing = client.get_edge(graph, edge_id, logical_graph_id)
    existing_props = _strip_server_properties(existing.get("properties") or {})
    desired_props = dict(record.get("properties") or {})
    remove_properties = sorted(set(existing_props) - set(desired_props))
    client.update_edge(
        graph,
        edge_id,
        logical_graph_id,
        set_properties=desired_props,
        remove_properties=remove_properties or None,
    )


def _compare_node_record(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    label = f"node {expected.get('external_id')}"
    if actual.get("external_id") != expected.get("external_id"):
        mismatches.append(
            f"{label}: external_id {actual.get('external_id')!r} != {expected.get('external_id')!r}"
        )
    actual_labels = set(actual.get("labels") or [])
    expected_labels = set(expected.get("labels") or [])
    if actual_labels != expected_labels:
        mismatches.append(
            f"{label}: labels {sorted(actual_labels)!r} != {sorted(expected_labels)!r}"
        )
    mismatches.extend(
        _compare_properties(
            label,
            expected.get("properties") or {},
            actual.get("properties") or {},
        )
    )
    return mismatches


def _compare_edge_record(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    props = expected.get("properties") or {}
    label = f"edge {props.get('agent_edge_id')}"
    mismatches: list[str] = []
    for key in ("source", "target", "edge_type"):
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{label}: {key} {actual.get(key)!r} != {expected.get(key)!r}")
    mismatches.extend(
        _compare_properties(
            label,
            props,
            actual.get("properties") or {},
        )
    )
    return mismatches


def _compare_properties(
    label: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    cleaned_actual = _strip_server_properties(actual)
    if cleaned_actual == expected:
        return []
    missing = sorted(set(expected) - set(cleaned_actual))
    extra = sorted(set(cleaned_actual) - set(expected))
    changed = sorted(
        key
        for key in set(expected) & set(cleaned_actual)
        if cleaned_actual[key] != expected[key]
    )
    return [
        f"{label}: properties differ missing={missing} extra={extra} changed={changed}"
    ]


def _strip_server_properties(properties: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(properties or {})
    cleaned.pop(SERVER_LOGICAL_GRAPH_PROPERTY, None)
    return cleaned


def edge_type(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).upper()
    return normalized or "RELATED_TO"


def _normalize_node(node: dict[str, Any], *, index: int) -> dict[str, Any]:
    node.setdefault("id", index)
    return node


def _agent_id_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _clean_properties(properties: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in properties.items():
        scalar = _property_value(value)
        if scalar is not None:
            cleaned[key] = scalar
    return cleaned


def _property_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return safe.strip("_") or "trajectory"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_project_env()
    default_embedding_url = os.environ.get("TONGGRAPH_EMBEDDING_URL") or os.environ.get("HERO_EMBEDDING_URL", "")
    default_embedding_model = os.environ.get("TONGGRAPH_EMBEDDING_MODEL") or os.environ.get("HERO_EMBEDDING_MODEL", "")
    parser = RichArgumentParser(
        prog="bcg agent tonggraph-sync",
        description="Ingest a BeliefTracer final graph or graph snapshot list into TongGraph Server.",
    )
    parser.add_argument("input", type=Path, help="Path to final_graph.json or a saved graph snapshot list")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TONGGRAPH_BASE_URL", DEFAULT_BASE_URL),
        help=f"TongGraph Server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TONGGRAPH_TOKEN") or os.environ.get("TONGGRAPH_AGENT_WRITER_TOKEN", ""),
        help="Bearer token; defaults to $TONGGRAPH_TOKEN or $TONGGRAPH_AGENT_WRITER_TOKEN",
    )
    parser.add_argument(
        "--graph",
        default=os.environ.get("TONGGRAPH_GRAPH", DEFAULT_GRAPH),
        help=f"Physical graph name (default: {DEFAULT_GRAPH})",
    )
    parser.add_argument(
        "--logical-graph-id",
        default=os.environ.get("TONGGRAPH_LOGICAL_GRAPH_ID", ""),
        help="Logical graph namespace; defaults to payload item_id or input path",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--index-name", default=DEFAULT_TEXT_INDEX)
    parser.add_argument("--no-index", action="store_true", help="Do not create the agent_text fulltext index")
    parser.add_argument(
        "--embedding-url",
        default=default_embedding_url,
        help="OpenAI-compatible embedding API URL; defaults to $TONGGRAPH_EMBEDDING_URL or $HERO_EMBEDDING_URL",
    )
    parser.add_argument(
        "--embedding-model",
        default=default_embedding_model,
        help="Embedding model name; defaults to $TONGGRAPH_EMBEDDING_MODEL or $HERO_EMBEDDING_MODEL",
    )
    parser.add_argument(
        "--embedding-index",
        default=os.environ.get("TONGGRAPH_EMBEDDING_INDEX", DEFAULT_VECTOR_INDEX),
        help=f"TongGraph vector index name (default: {DEFAULT_VECTOR_INDEX}); empty disables vector sync",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding generation and vector upsert",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.environ.get("TONGGRAPH_EMBEDDING_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--allow-duplicate-edges",
        action="store_true",
        help="Skip edge existence checks and always append edges",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.token:
        raise SystemExit("missing TongGraph token; set TONGGRAPH_TOKEN or pass --token")
    result = sync_graph_file(
        args.input,
        base_url=args.base_url,
        token=args.token,
        graph=args.graph,
        logical_graph_id=args.logical_graph_id,
        text_index=None if args.no_index else args.index_name,
        embedding_url="" if args.no_embedding else args.embedding_url,
        embedding_model="" if args.no_embedding else args.embedding_model,
        embedding_index=None if args.no_embedding or not args.embedding_index else args.embedding_index,
        embedding_batch_size=args.embedding_batch_size,
        timeout=args.timeout,
        skip_existing_edges=not args.allow_duplicate_edges,
    )
    print(
        "synced "
        f"graph={result.graph} logical_graph_id={result.logical_graph_id} "
        f"nodes={result.nodes_created} created/{result.nodes_reused} reused/{result.nodes_deleted} deleted "
        f"edges={result.edges_created} created/{result.edges_reused} reused/{result.edges_deleted} deleted "
        f"text_index={result.text_index or 'disabled'} "
        f"embedding_index={result.embedding_index or 'disabled'} "
        f"embeddings={result.embeddings_upserted}"
        f" readback={'verified' if result.readback_verified else 'skipped'}"
        + (f" dims={result.embedding_dimensions}" if result.embedding_dimensions else "")
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except TongGraphSyncError as exc:
        detail = f" ({exc.code}, request_id={exc.request_id})" if exc.code else ""
        print(f"TongGraph sync failed: {exc}{detail}", file=sys.stderr)
        raise SystemExit(1)
