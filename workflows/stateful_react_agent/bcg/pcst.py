"""Local, budgeted PCST retrieval over an acknowledged BCG snapshot.

This is a BCG adaptation, not a reproduction of G-Retriever: node prizes only,
original relation edges, no learned graph encoder or edge-text prizes. PCST
uses undirected connectivity; rendered edges retain their recorded direction.
The hard character constraint is enforced by a cost sweep and leaf pruning,
not by claiming that unconstrained PCST solves a budgeted optimum.
"""

from __future__ import annotations

import importlib
import math
import re
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from frontier_agent.core.messages import Message

from .client import GraphError
from .embeddings import RemoteRelevance
from .messages import assistant_text, tool_calls
from .rendering import GRAPH_OVERHEAD, _nodes, node_line, relation_line, render_graph
from .vector_store import TongGraphVectorStore

if TYPE_CHECKING:
    from .runtime import BCGSettings


def retrieval_focus(messages: list[Message]) -> str:
    """Latest action queries and readable assistant intent, excluding tool bodies.

    The question is encoded separately. A newer human input resets the focus;
    internal team reports are not human steering. Long reasoning is explicitly
    bounded here and then token-chunked, never silently cut by the encoder.
    """
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("bcg_origin") != "agent_report":
            break
        if message.get("role") != "assistant":
            continue
        queries: list[str] = []
        for call in tool_calls(message).values():
            for key in ("query", "q", "queries", "search_query"):
                value = call["arguments"].get(key)
                values = value if isinstance(value, list) else [value]
                queries.extend(v[:400] for v in values if isinstance(v, str) and v.strip())
        intent = re.sub(r"</?(?:think|thinking)>", "", assistant_text(message)).strip()
        parts = [*list(dict.fromkeys(queries))[:4], intent[-1200:]]
        focus = "\n".join(p for p in parts if p)
        if focus:
            return focus
    return ""


def _confidence(node: dict[str, Any]) -> float:
    value = node.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        return 0.5
    return max(0.0, min(1.0, float(value)))


class LocalRelevance:
    """Shared CPU/GPU encoder with token-sized chunks and bounded text caching."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name, self.device = model_name, device
        self.lock = threading.Lock()
        self.model: Any = None
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def score(
        self, question: str, focus: str, texts: list[str],
    ) -> tuple[list[float], dict[str, Any]]:
        try:
            np = importlib.import_module("numpy")
            with self.lock:
                if self.model is None:
                    module = importlib.import_module("sentence_transformers")
                    self.model = module.SentenceTransformer(self.model_name, device=self.device)
                model = self.model
                tokenizer = model.tokenizer
                max_tokens = int(model.max_seq_length)
                width = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
                if width < 8:
                    raise GraphError("pcst_embedding_token_limit_invalid")
                all_texts = [question, focus or question, *texts]
                missing = list(dict.fromkeys(t for t in all_texts if t not in self.cache))
                chunks: list[str] = []
                spans: dict[str, tuple[int, int]] = {}
                for text in missing:
                    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
                    start = len(chunks)
                    position = 0
                    while position < len(ids):
                        end = min(len(ids), position + width)
                        part = tokenizer.decode(ids[position:end], skip_special_tokens=True)
                        # Decoding a WordPiece boundary can expand when retokenized.
                        while len(tokenizer.encode(part, truncation=False, verbose=False)) > max_tokens:
                            end -= 1
                            if end <= position:
                                raise GraphError("pcst_embedding_chunk_failed")
                            part = tokenizer.decode(ids[position:end], skip_special_tokens=True)
                        chunks.append(part)
                        position = end
                    if start == len(chunks):
                        chunks.append("")
                    spans[text] = (start, len(chunks))
                if chunks:
                    vectors = model.encode(
                        chunks, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=32,
                    )
                    for text, (start, end) in spans.items():
                        self.cache[text] = vectors[start:end]
                task_vectors, focus_vectors = (self.cache[t] for t in all_texts[:2])
                scores = []
                for text in texts:
                    vectors = self.cache[text]
                    task_score = max(0.0, float(np.max(vectors @ task_vectors.T)))
                    focus_score = max(0.0, float(np.max(vectors @ focus_vectors.T)))
                    scores.append(min(1.0, 0.5 * task_score + 0.5 * focus_score))
                audit = {
                    "provider": "local",
                    "model": self.model_name, "device": self.device,
                    "max_seq_length": max_tokens,
                    "question_chunks": len(task_vectors), "focus_chunks": len(focus_vectors),
                    "node_chunks": sum(len(self.cache[t]) for t in texts),
                    "new_chunks_encoded": len(chunks),
                }
                for text in dict.fromkeys(all_texts):
                    self.cache.move_to_end(text)
                while len(self.cache) > 4096:
                    self.cache.popitem(last=False)
                return scores, audit
        except ImportError as exc:
            raise GraphError("pcst_embedding_dependency_missing: install frontier-agent[bcg-embeddings]") from exc
        except GraphError:
            raise
        except Exception as exc:
            raise GraphError("pcst_embedding_failed") from exc


@lru_cache(maxsize=2)
def _local_encoder(model_name: str, device: str) -> LocalRelevance:
    return LocalRelevance(model_name, device)


@lru_cache(maxsize=2)
def _remote_encoder(
    url: str, model_name: str, api_key: str, timeout: float, chunk_bytes: int,
    revision: str = "1", storage: tuple[str, str, str, float] | None = None,
) -> RemoteRelevance:
    store = TongGraphVectorStore(*storage, descriptor={
        "schema": 1, "provider": "openai", "url": url, "model": model_name,
        "revision": revision, "chunking": "utf8_bytes", "chunk_bytes": chunk_bytes, "normalization": "l2",
    }) if storage else None
    return RemoteRelevance(url, model_name, api_key, timeout, chunk_bytes, store=store)


def _encoder(settings: BCGSettings) -> LocalRelevance | RemoteRelevance:
    if settings.embedding_provider == "openai":
        return _remote_encoder(
            settings.embedding_url, settings.embedding_model, settings.embedding_api_key,
            settings.embedding_timeout, settings.embedding_chunk_bytes,
            settings.embedding_revision,
            (settings.tonggraph_url, settings.tonggraph_token, settings.tonggraph_graph, settings.tonggraph_timeout)
            if settings.embedding_store == "tonggraph" else None,
        )
    if settings.embedding_provider == "local":
        return _local_encoder(settings.embedding_model, settings.embedding_device)
    raise GraphError("invalid_pcst_embedding_provider")


def select_prized_tree(
    snapshot: dict[str, Any], prizes: dict[int, float], budget: int, *, edge_cost: float = 0.2,
) -> dict[str, Any]:
    """Return one connected tree within the *rendered* budget, or no selection.

    All feasible nodes remain available, including zero-prize bridges. For each
    size-penalty multiplier, solve ordinary PCST with pcst_fast strong pruning.
    Trim only leaves if necessary, then choose the feasible candidate with the
    largest original sum(prizes) - edge_cost * number_of_edges. Singletons are
    candidates too. This is a bounded heuristic for the budgeted problem.
    """
    if not math.isfinite(edge_cost) or edge_cost <= 0:
        raise GraphError("invalid_pcst_edge_cost")
    nodes = _nodes(snapshot)
    node_chars = {i: len(node_line(n)) + 1 for i, n in nodes.items()}
    # A whole node larger than the budget cannot participate in any feasible
    # output. This is a size feasibility check, never a confidence threshold.
    ids = sorted(i for i in nodes if GRAPH_OVERHEAD + node_chars[i] <= budget)
    rewards = {i: float(prizes.get(i, 0.0)) for i in ids}
    if any(not math.isfinite(p) or p < 0 for p in rewards.values()):
        raise GraphError("invalid_pcst_prize")
    result: dict[str, Any] = {
        "node_ids": [], "relation_ids": [], "rendered_chars": 0,
        "candidate_nodes": len(nodes), "feasible_nodes": len(ids),
        "oversized_nodes": len(nodes) - len(ids),
        "solver": "pcst_fast", "budget_method": "cost_sweep_and_leaf_pruning",
    }
    if not ids or max(rewards.values()) <= 0:
        return result
    try:
        np = importlib.import_module("numpy")
        solve = importlib.import_module("pcst_fast").pcst_fast
    except ImportError as exc:
        raise GraphError("pcst_dependency_missing: install frontier-agent[bcg-pcst]") from exc
    if int(np.__version__.split(".")[0]) >= 2:
        raise GraphError("pcst_requires_numpy_1: install frontier-agent[bcg-pcst]")

    # Parallel/antiparallel relations have the same zero prize. Use the one
    # cheapest to render; retain its original id, type, note and direction.
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    relations = snapshot.get("relations", [])
    if not isinstance(relations, list):
        raise GraphError("invalid_graph_relations")
    for edge in relations:
        if not isinstance(edge, dict) or type(edge.get("id")) is not int:
            continue
        a, b = edge.get("from_id"), edge.get("to_id")
        if type(a) is not int or type(b) is not int or a not in rewards or b not in rewards or a == b:
            continue
        if edge.get("type") not in {"depends_on", "supplements", "contradicts"}:
            continue
        pair = min(a, b), max(a, b)
        previous = by_pair.get(pair)
        if previous is None or (len(relation_line(edge)), edge["id"]) < (len(relation_line(previous)), previous["id"]):
            by_pair[pair] = edge
    pairs = sorted(by_pair)
    edges = [by_pair[pair] for pair in pairs]
    edge_chars = [len(relation_line(e)) + 1 for e in edges]
    index = {i: j for j, i in enumerate(ids)}
    best_id = max(ids, key=lambda i: (rewards[i], -node_chars[i], -i))
    best_nodes, best_edges = {best_id}, set()
    best_key = (rewards[best_id], -(GRAPH_OVERHEAD + node_chars[best_id]))
    best_penalty, best_pruned = 0.0, 0
    solver_calls = 0
    if edges:
        endpoints = np.asarray([(index[a], index[b]) for a, b in pairs], dtype=np.int64)
        prize_array = np.asarray([rewards[i] for i in ids], dtype=np.float64)
        # This size proxy guides the solver; exact additive rendering costs
        # below enforce the hard cap. Internal node text may be counted twice.
        size_proxy = np.asarray([
            edge_chars[k] + (node_chars[a] + node_chars[b]) / 2
            for k, (a, b) in enumerate(pairs)
        ], dtype=np.float64)
        scale = sum(rewards.values()) / max(1, budget)
        for multiplier in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            penalty = multiplier * scale
            try:
                vertices, chosen = solve(
                    endpoints, prize_array, edge_cost + penalty * size_proxy,
                    -1, 1, "strong", 0,
                )
            except Exception as exc:
                raise GraphError("pcst_solver_failed") from exc
            solver_calls += 1
            kept = {ids[int(i)] for i in vertices}
            kept_edges = {int(i) for i in chosen}
            if not kept:
                continue
            adjacency: dict[int, dict[int, int]] = {i: {} for i in kept}
            for k in kept_edges:
                a, b = pairs[k]
                if a not in kept or b not in kept:
                    raise GraphError("invalid_pcst_tree")
                adjacency[a][b] = k
                adjacency[b][a] = k
            # Enforce one component even for disconnected input graphs.
            visited, pending = set(), [min(kept)]
            while pending:
                i = pending.pop()
                if i not in visited:
                    visited.add(i)
                    pending.extend(adjacency[i])
            if visited != kept or len(kept_edges) != len(kept) - 1:
                raise GraphError("invalid_pcst_tree")
            size = GRAPH_OVERHEAD + sum(node_chars[i] for i in kept) + sum(edge_chars[k] for k in kept_edges)
            pruned = 0
            while size > budget and len(kept) > 1:
                leaves = [i for i in kept if len(adjacency[i]) == 1]

                leaf = min(
                    ((rewards[i] - edge_cost) / (
                        node_chars[i] + edge_chars[next(iter(adjacency[i].values()))]
                    ), i)
                    for i in leaves
                )[1]
                neighbor, k = next(iter(adjacency[leaf].items()))
                size -= node_chars[leaf] + edge_chars[k]
                del adjacency[neighbor][leaf]
                del adjacency[leaf]
                kept.remove(leaf)
                kept_edges.remove(k)
                pruned += 1
            if size > budget:
                continue
            objective = sum(rewards[i] for i in kept) - edge_cost * len(kept_edges)
            key = (objective, -size)
            if key > best_key:
                best_key = key
                best_nodes, best_edges = kept, kept_edges
                best_penalty, best_pruned = penalty, pruned
    result.update(
        node_ids=sorted(best_nodes), relation_ids=[edges[k]["id"] for k in sorted(best_edges)],
        objective=best_key[0], size_penalty=best_penalty, leaves_pruned=best_pruned,
        solver_calls=solver_calls,
        selected_prizes={str(i): rewards[i] for i in sorted(best_nodes)},
    )
    rendered, _, _ = render_graph(snapshot, result, budget, strict=True)
    result["rendered_chars"] = len(rendered)
    return result


class PCSTSelector:
    def __init__(self, settings: BCGSettings) -> None:
        self.settings = settings

    def persist_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        encoder = _encoder(self.settings)
        if not isinstance(encoder, RemoteRelevance) or encoder.store is None:
            return {"persistent_cache": {"backend": "memory"}}
        return encoder.persist_nodes([
            str(n.get("belief") or n.get("decision") or "") for n in _nodes(snapshot).values()
        ])

    def select(
        self, snapshot: dict[str, Any], question: str, focus: str, budget: int,
    ) -> dict[str, Any]:
        nodes = _nodes(snapshot)
        settings = self.settings
        embedding_audit: dict[str, Any] = {}
        relevance: dict[int, float] = {}
        if settings.retrieval == "query_pcst":
            if not question.strip():
                raise GraphError("pcst_missing_question")
            scores, embedding_audit = _encoder(settings).score(
                question, focus, [str(n.get("belief") or n.get("decision") or "") for n in nodes.values()],
            )
            relevance = dict(zip(nodes, scores, strict=True))
            floor = settings.pcst_confidence_floor
            prizes = {i: relevance[i] * (floor + (1 - floor) * _confidence(n)) for i, n in nodes.items()}
        elif settings.retrieval == "confidence_pcst":
            prizes = {i: _confidence(n) for i, n in nodes.items()}
        else:
            raise GraphError("invalid_pcst_strategy")
        # Pinned initial-user beliefs can bridge, but receive no reward for
        # duplicating a task that is already present verbatim in every request.
        for i, node in nodes.items():
            source = node.get("source")
            if isinstance(source, dict) and source.get("turn_id") in {0, 1}:
                prizes[i] = 0.0
        selection = select_prized_tree(snapshot, prizes, budget, edge_cost=settings.pcst_edge_cost)
        selection.update(
            strategy=settings.retrieval, problem_id=snapshot.get("problem_id"),
            retrieval="embedding" if settings.retrieval == "query_pcst" else "confidence",
            query={"question": question, "focus": focus}, embedding=embedding_audit,
            parameters={"edge_cost": settings.pcst_edge_cost, "confidence_floor": settings.pcst_confidence_floor},
            graph_chars_budget=budget,
            # Keep the scores from this invocation, including rejected candidates.
            # A viewer must not re-embed historical text or infer missing scores.
            node_scores={str(i): {
                "confidence": _confidence(node),
                "relevance": relevance.get(i),
                "confidence_factor": (
                    settings.pcst_confidence_floor + (1 - settings.pcst_confidence_floor) * _confidence(node)
                    if settings.retrieval == "query_pcst" else None
                ),
                "prize": prizes[i],
                "pinned_input": isinstance(node.get("source"), dict) and node["source"].get("turn_id") in {0, 1},
                "rendered_chars": GRAPH_OVERHEAD + len(node_line(node)) + 1,
                "feasible": GRAPH_OVERHEAD + len(node_line(node)) + 1 <= budget,
            } for i, node in nodes.items()},
        )
        return selection
