"""HerO two-stage retrieval tool: BM25 + Embedding Reranking.

This module implements HerO's retrieval method while maintaining the same
tool interface as AVeriTeCSearchTool, so agents can use it transparently.

Embedding backend is chosen by ``embedding_url``:
  - URL ending in ``/v2/embed``  → remote Jina-style API (``texts`` +
    ``input_type``; asymmetric query/document encoding handled server-side,
    no manual instruct prefix needed)
  - Other non-empty URL          → remote OpenAI-compatible ``/v1/embeddings``
    API (``input``; query side gets a manual instruct prefix, see
    ``_embedding_rerank``)
  - Empty string                 → local SentenceTransformer (singleton,
    loaded once; same instruct-prefix convention as the OpenAI-style API)
"""

from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from bcg.agent.tools.averitec_search import AVeriTeCSearchTool

logger = logging.getLogger(__name__)


class HerOSearchTool(AVeriTeCSearchTool):
    """
    HerO two-stage retrieval: BM25 + Embedding Reranking.

    Inherits AVeriTeCSearchTool and only overrides the _search method to add
    embedding-based reranking on top of BM25 results.

    Args:
        bm25_top_k: Number of candidates to retrieve with BM25 (default: 10)
        embedding_model: Model name/path — remote model id or local HF path
        embedding_url: Remote embedding API endpoint. Empty = use local model.
        batch_size: Batch size for embedding computation (default: 16)
        **kwargs: Arguments passed to AVeriTeCSearchTool
    """

    NAME = "averitec_search"  # Same name as base tool
    QUERY_SEPARATOR = " ||| "
    _shared_local_model = None
    _shared_local_model_name: str = ""
    _run_log_dir: "Path | None" = None
    _claim_round_counter: dict = {}
    # Guards _claim_round_counter's read-increment-write. Only used as a
    # fallback (see _next_round) when no (turn, call_index) label was set —
    # e.g. a standalone script calling .forward() directly outside
    # BeliefTracerEnvironment. Under that fallback, concurrent calls on the
    # same claim would still race on the +=1; the label path (the normal one,
    # set by BeliefTracerEnvironment._run_tool_labeled) avoids this entirely
    # because each call's position is fixed before dispatch, not negotiated
    # via shared state.
    _round_counter_lock = threading.Lock()
    # Thread-local (turn, call_index) label, set by
    # BeliefTracerEnvironment._run_tool_labeled just before invoking this
    # tool from a worker thread. Thread-local because concurrent tool calls
    # in the same turn run in different threads — each thread's label is
    # private, so there's nothing to lock.
    _call_label = threading.local()

    def __init__(
        self,
        name: str = NAME,
        bm25_top_k: int = 10,
        embedding_model: str = "SFR-Embedding-2_R",
        embedding_device: str = "cuda",
        batch_size: int = 16,
        embedding_url: str = "",
        four_stage: bool = False,
        stage1_bm25_k: int = 1000,
        stage2_embed_k: int = 64,
        stage3_rerank_k: int = 10,
        rerank_url: str = "http://10.2.152.9:8010/v1/rerank",
        rerank_model: str = "Qwen3-Reranker-0.6B",
        enable_judge: bool = True,
        judge_model: str = "",
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_max_workers: int = 10,
        judge_max_items: int = 10,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.bm25_top_k = bm25_top_k
        self.batch_size = batch_size
        self.embedding_device = embedding_device
        self._embedding_model_name = embedding_model
        self.embedding_url = embedding_url
        self._use_remote = bool(embedding_url)
        self._use_jina_embed = embedding_url.rstrip("/").endswith("/v2/embed")

        # Four-stage pipeline config
        self.four_stage = four_stage
        self.stage1_bm25_k = stage1_bm25_k
        self.stage2_embed_k = stage2_embed_k
        self.stage3_rerank_k = stage3_rerank_k
        self.rerank_url = rerank_url
        self.rerank_model = rerank_model
        self.enable_judge = enable_judge
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.judge_api_key = judge_api_key
        self.judge_max_workers = judge_max_workers
        self.judge_max_items = judge_max_items

        if self._use_remote:
            logger.info("[HerO] Embedding backend: remote API (%s)", embedding_url)
        else:
            logger.info("[HerO] Embedding backend: local SentenceTransformer (%s)", embedding_model)
        if self.four_stage:
            logger.info(
                "[HerO] 4-stage retrieval: BM25 min(%d,N) -> embed %d -> rerank %d -> judge=%s",
                self.stage1_bm25_k, self.stage2_embed_k, self.stage3_rerank_k, self.enable_judge,
            )

    # ------------------------------------------------------------------
    # Local SentenceTransformer (class-level singleton)
    # ------------------------------------------------------------------

    @property
    def _local_model(self):
        cls = HerOSearchTool
        if cls._shared_local_model is None or cls._shared_local_model_name != self._embedding_model_name:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.embedding_device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("[HerO] CUDA not available, falling back to CPU")
                device = "cpu"
            logger.info("[HerO] Loading local embedding model: %s (device=%s)", self._embedding_model_name, device)
            cls._shared_local_model = SentenceTransformer(
                self._embedding_model_name, device=device, trust_remote_code=True,
            )
            cls._shared_local_model_name = self._embedding_model_name
            logger.info("[HerO] Local embedding model loaded")
        return cls._shared_local_model

    def _get_embeddings_local(self, texts: list[str]) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self._local_model.encode(
                texts, batch_size=self.batch_size,
                show_progress_bar=False, convert_to_numpy=True,
            )

    # ------------------------------------------------------------------
    # Remote Jina-style /v2/embed API
    # ------------------------------------------------------------------

    def _get_embeddings_jina(
        self, texts: list[str], input_type: str, max_chars: int = 8000, max_workers: int = 32,
    ) -> np.ndarray:
        """Fetch embeddings from a remote Jina-style ``/v2/embed`` API.

        Request: ``{"model": ..., "texts": [...], "input_type": "query"|"document"}``.
        Response: ``{"embeddings": {"float": [[...], ...]}}`` — vectors come
        back in input order (no per-item index to sort by), unlike the OpenAI
        ``/v1/embeddings`` shape.
        """
        import requests

        cleaned = [str(t or "")[:max_chars].strip() or "empty" for t in texts]
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        }
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        embeddings: list[Any] = [None] * len(cleaned)

        batch = max(1, int(self.batch_size or 16))
        batches = [(start, cleaned[start:start + batch]) for start in range(0, len(cleaned), batch)]

        def _fetch(start: int, chunk_texts: list[str]):
            payload = {
                "model": self._embedding_model_name,
                "texts": chunk_texts,
                "input_type": input_type,
            }
            resp = session.post(self.embedding_url, json=payload, headers=headers, timeout=120)
            if not resp.ok:
                logger.warning("[HerO] Jina embedding API error (batch @%d, %d items): %d %s",
                               start, len(chunk_texts), resp.status_code, resp.text[:500])
                resp.raise_for_status()
            vecs = resp.json()["embeddings"]["float"]
            return start, vecs

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch, start, chunk) for start, chunk in batches]
            for future in as_completed(futures):
                start, embs = future.result()
                for offset, emb in enumerate(embs):
                    embeddings[start + offset] = emb

        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Remote OpenAI-compatible /v1/embeddings API
    # ------------------------------------------------------------------

    def _get_embeddings_remote(self, texts: list[str], max_chars: int = 8000, max_workers: int = 32) -> np.ndarray:
        """Fetch embeddings from a remote /v1/embeddings API.

        Sends ``input`` as a batched list (``batch_size`` items per request) and
        dispatches the batches concurrently. The response ``data`` entries carry
        an ``index`` field; we order by it so results map back to inputs even if
        the server reorders them.
        """
        import requests

        cleaned = [str(t or "")[:max_chars].strip() or "empty" for t in texts]
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        }
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        embeddings: list[Any] = [None] * len(cleaned)

        batch = max(1, int(self.batch_size or 16))
        batches = [(start, cleaned[start:start + batch]) for start in range(0, len(cleaned), batch)]

        def _fetch(start: int, chunk_texts: list[str]):
            payload = {"model": self._embedding_model_name, "input": chunk_texts}
            resp = session.post(self.embedding_url, json=payload, headers=headers, timeout=120)
            if not resp.ok:
                logger.warning("[HerO] Embedding API error (batch @%d, %d items): %d %s",
                               start, len(chunk_texts), resp.status_code, resp.text[:500])
                resp.raise_for_status()
            data = resp.json()["data"]
            # Sort by returned index so local position is correct.
            data = sorted(data, key=lambda d: d.get("index", 0))
            return start, [d["embedding"] for d in data]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch, start, chunk) for start, chunk in batches]
            for future in as_completed(futures):
                start, embs = future.result()
                for offset, emb in enumerate(embs):
                    embeddings[start + offset] = emb

        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Unified dispatch
    # ------------------------------------------------------------------

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> np.ndarray:
        """``input_type`` ("query" or "document") only matters for the Jina
        backend, which encodes the two sides asymmetrically server-side.
        The OpenAI-style and local backends ignore it — for those the caller
        (``_embedding_rerank``) still applies the manual instruct prefix.
        """
        if self._use_jina_embed:
            return self._get_embeddings_jina(texts, input_type=input_type)
        if self._use_remote:
            return self._get_embeddings_remote(texts)
        return self._get_embeddings_local(texts)

    def _search(
        self,
        index: Any,  # _ClaimIndex
        query: str,
        top_k: int,
    ) -> tuple[list[tuple[float, Any]], dict[str, str]]:
        """
        Two-stage retrieval aligned with HerO original:
        1. Split query by ||| into [claim, hypo_doc1, hypo_doc2, ...]
        2. BM25 with full concatenated query (claim + " " + " ".join(hypo_docs))
        3. Embedding reranking: average vector of instruct-wrapped query parts

        Returns ``(results, evidence_summaries)`` — see base class docstring.
        """
        # Split multi-part query: "claim ||| hypo1 ||| hypo2 ..."
        parts = [p.strip() for p in query.split(self.QUERY_SEPARATOR) if p.strip()]
        if not parts:
            parts = [query]
        claim = parts[0]
        hypo_docs = parts[1:] if len(parts) > 1 else []

        logger.info("[HerO] Query (%d parts): %s...", len(parts), claim[:80])
        for i, doc in enumerate(hypo_docs):
            logger.debug("[HerO]   HyDE[%d]: %s...", i + 1, doc[:120])

        if self.four_stage:
            return self._search_four_stage(index, claim, hypo_docs, top_k)

        # Allocate this call's round number once (see _next_round docstring —
        # concurrent tool calls on the same claim must not share one number
        # read fresh at each logging call).
        round_num = self._next_round()

        # BM25 query construction
        bm25_query = claim + " " + " ".join(hypo_docs) if hypo_docs else claim
        bm25_results = self._bm25_search(index, bm25_query, self.bm25_top_k)

        logger.info("[HerO] BM25 retrieved: %d candidates (requested top-%d)", len(bm25_results), self.bm25_top_k)
        if bm25_results:
            logger.info("[HerO] BM25 score range: [%.4f, %.4f]", bm25_results[-1][0], bm25_results[0][0])

        if bm25_results:
            self._log_bm25_results(bm25_query, bm25_results, round_num=round_num)

        if not bm25_results:
            return [], {}

        # Embedding reranking with all query parts
        reranked = self._embedding_rerank(claim, hypo_docs, bm25_results, top_k)

        logger.info("[HerO] After reranking: %d results (requested top-%d)", len(reranked), top_k)
        if reranked:
            logger.info("[HerO] Embedding score range: [%.4f, %.4f]", reranked[-1][0], reranked[0][0])
            for i, (score, chunk) in enumerate(reranked[:3]):
                bm25_score = next((s for s, c in bm25_results if c == chunk), None)
                bm25_str = f"{bm25_score:.4f}" if bm25_score is not None else "N/A"
                logger.info("[HerO]   [%d] Embedding: %.4f, BM25: %s", i + 1, score, bm25_str)

        if reranked:
            self._log_embedding_results(bm25_query, reranked, bm25_results, round_num=round_num)

        return reranked, {}

    # ------------------------------------------------------------------
    # Four-stage pipeline: BM25 -> embedding -> reranker -> LLM judge
    # ------------------------------------------------------------------

    def _search_four_stage(
        self,
        index: Any,
        claim: str,
        hypo_docs: list[str],
        top_k: int,
    ) -> tuple[list[tuple[float, Any]], dict[str, str]]:
        """BM25(min(stage1,N)) -> embedding(stage2) -> reranker(stage3) -> judge(soft).

        Returns ``(results, evidence_summaries)`` — see base class docstring.
        Summaries are returned rather than stashed on ``self`` so concurrent
        calls on the same tool instance can't cross-contaminate each other.
        """
        import time
        final_k = min(top_k or self.stage3_rerank_k, self.stage3_rerank_k)
        timings: dict[str, float] = {}

        # Allocate this call's round number ONCE, up front, and thread it
        # through every _log_* call below. Concurrent tool calls (parallel
        # tool execution) interleave their stages, so each call's stage logs
        # must share one round number — reading the shared counter fresh at
        # each stage would let a concurrent call's increment bleed in and
        # cause different stages of the SAME call to log under different (or
        # colliding) round numbers, silently overwriting each other's files.
        round_num = self._next_round()

        # Stage 1: BM25, pool size = min(stage1_bm25_k, N) (handled by _bm25_search cap).
        t0 = time.time()
        bm25_query = claim + " " + " ".join(hypo_docs) if hypo_docs else claim
        bm25_results = self._bm25_search(index, bm25_query, self.stage1_bm25_k)
        timings["bm25"] = time.time() - t0
        logger.info("[HerO4] Stage1 BM25: %d candidates (cap %d) [%.3fs]", len(bm25_results), self.stage1_bm25_k, timings["bm25"])
        if not bm25_results:
            return [], {}
        if bm25_results:
            self._log_bm25_results(bm25_query, bm25_results, round_num=round_num)

        # Stage 2: embedding rerank down to stage2_embed_k.
        t0 = time.time()
        embed_results = self._embedding_rerank(claim, hypo_docs, bm25_results, self.stage2_embed_k)
        timings["embedding"] = time.time() - t0
        logger.info("[HerO4] Stage2 embedding: %d survivors (target %d) [%.3fs]", len(embed_results), self.stage2_embed_k, timings["embedding"])
        if not embed_results:
            return [], {}
        self._log_embedding_results(bm25_query, embed_results, bm25_results, round_num=round_num)

        # Stage 3: reranker service down to stage3_rerank_k.
        t0 = time.time()
        rerank_results = self._rerank_service(claim, embed_results, self.stage3_rerank_k)
        timings["rerank"] = time.time() - t0
        logger.info("[HerO4] Stage3 rerank: %d survivors (target %d) [%.3fs]", len(rerank_results), self.stage3_rerank_k, timings["rerank"])
        self._log_stage("rerank", claim, rerank_results, round_num=round_num)
        if not rerank_results:
            # Degrade: fall back to embedding order truncated to final_k.
            return embed_results[:final_k], {}

        # Stage 4: LLM relevance judge (hard filter) down to final_k.
        if self.enable_judge:
            t0 = time.time()
            judged, verdicts, summaries = self._judge_relevance(claim, rerank_results, final_k)
            timings["judge"] = time.time() - t0
            logger.info("[HerO4] Stage4 judge: %d kept (final_k %d) [%.3fs]", len(judged), final_k, timings["judge"])
            self._log_stage("judge", claim, judged, round_num=round_num,
                            extra={"verdicts": verdicts, "summaries": summaries, "timings": timings})
            # Evidence summaries keyed by chunk text, returned to forward() to
            # attach a per-evidence summary to its metadata (used for archiving).
            judged_items = rerank_results[: self.judge_max_items]
            evidence_summaries = {
                chunk.text: summaries[i]
                for i, (_, chunk) in enumerate(judged_items)
                if i < len(summaries) and summaries[i]
            }
            return judged, evidence_summaries
        return rerank_results[:final_k], {}

    def _log_stage(
        self,
        stage: str,
        query: str,
        results: list[tuple[float, Any]],
        round_num: str,
        extra: dict | None = None,
    ) -> None:
        """Persist one stage's surviving results to a per-claim JSON log file.

        ``round_num`` must be the value allocated once by the caller (see
        ``_search_four_stage``), not re-derived here — otherwise a concurrent
        tool call's position could bleed into this call's file name.
        """
        import json
        from datetime import datetime

        try:
            claim_dir = self._get_claim_log_dir()
            claim_id = self.claim_id or "unknown"
            entry = {
                "timestamp": datetime.now().isoformat(),
                "claim_id": claim_id,
                "round": round_num,
                "stage": stage,
                "query": query,
                "num_results": len(results),
                "results": [
                    {
                        "rank": i + 1,
                        "score": float(score),
                        "text": chunk.text[:500],
                        "url": getattr(chunk, "url", ""),
                        "chunk_id": getattr(chunk, "chunk_id", ""),
                    }
                    for i, (score, chunk) in enumerate(results)
                ],
            }
            if extra:
                entry.update(extra)
            with open(claim_dir / f"{stage}_round_{round_num}.json", "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            logger.info("[HerO4] %s results saved to: %s/%s_round_%s.json", stage, claim_dir, stage, round_num)
        except Exception as exc:
            logger.warning("[HerO4] Failed to log %s stage: %s", stage, exc)

    # Official Qwen3-Reranker chat template pieces.
    _RERANK_PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _RERANK_INSTRUCT = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    # Above this many candidates in one rerank call, jina-reranker-v3's vLLM
    # serving path (JinaForRanking) has a confirmed bug: non-deterministic,
    # tail-biased ranking plus an O(n^2) prompt_tokens accounting artifact
    # (per-result duplication of the full n-doc prompt's token_ids during
    # response construction, not real context usage). Verified stable at 32,
    # broken at 64 — see jina-embedding-reranker-migration memory. Only
    # matters for _rerank_via_rerank_api (the batched native /rerank path);
    # _rerank_via_logprobs already scores one document per HTTP call, so it
    # has no equivalent batch-size exposure.
    _RERANK_SAFE_BATCH = 32

    def _rerank_service(
        self,
        query: str,
        candidates: list[tuple[float, Any]],
        top_n: int,
    ) -> list[tuple[float, Any]]:
        """Rerank candidates with Qwen3-Reranker.

        The sglang ``/v1/rerank`` endpoint on this deployment returns a frozen
        score for every input, so by default we score each (query, doc) pair via
        the model's official chat template on ``/v1/completions`` and read
        P("yes") from the next-token logprobs. Set ``rerank_url`` to a real
        ``/v1/rerank`` endpoint to use the native path instead.

        Degrades to the input (embedding) order on any error.

        When more than ``_RERANK_SAFE_BATCH`` candidates are passed to the
        native ``/rerank`` path, splits into safe-sized batches, reranks each
        independently, then reranks the union of each batch's survivors in a
        second (also safe-sized) pass — a plain per-batch top-n concat would
        be wrong here, since a listwise reranker's scores are relative to
        "what else is in this batch," not comparable across batches.
        """
        if not candidates:
            return []
        chunks = [chunk for _, chunk in candidates]

        if self.rerank_url.endswith("/rerank"):
            if len(chunks) > self._RERANK_SAFE_BATCH:
                return self._rerank_via_rerank_api_batched(query, chunks, top_n)
            return self._rerank_via_rerank_api(query, chunks, top_n)
        return self._rerank_via_logprobs(query, chunks, top_n)

    def _rerank_via_rerank_api_batched(
        self, query: str, chunks: list[Any], top_n: int
    ) -> list[tuple[float, Any]]:
        """Split > _RERANK_SAFE_BATCH chunks into safe-sized batches, rerank
        each independently, then rerank the merged survivors in one final
        safe-sized pass to get scores that are actually comparable.

        Each batch keeps up to ``top_n`` survivors (not just enough to fill
        the final top_n from one batch) so a batch that happens to hold most
        of the truly relevant candidates isn't starved by an even split.

        Known limitation (inherent to any split-batch rerank, not specific to
        this implementation): a candidate that ranks below top_n WITHIN its
        own batch never reaches the final pass, even if it would have beaten
        some other batch's survivors head-to-head. Verified in practice: with
        64 real candidates split into two batches of 32, one relevant
        candidate ranked 4th in its batch (so it did survive to the final
        pass) but was then edged out of the final top-5 by the final rerank's
        re-scoring — that part is normal (a listwise reranker's scores shift
        based on what else is being compared), not a batching bug. The
        strictly-batch-local elimination is the actual recall cost of
        batching versus a hypothetical correctly-working single n=64 call.
        """
        batch_size = self._RERANK_SAFE_BATCH
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(
            "[HerO4] Rerank batch too large (%d > %d): splitting into %d batches of <=%d",
            len(chunks), batch_size, len(batches), batch_size,
        )
        survivors: list[Any] = []
        for batch in batches:
            ranked = self._rerank_via_rerank_api(query, batch, top_n)
            survivors.extend(c for _, c in ranked)

        # The merged survivor set can itself exceed the safe batch size (e.g.
        # 3 batches x top_n=10 survivors = 30, still fine; but with a larger
        # top_n or more input batches it could recurse) — _rerank_service's
        # size check on the recursive call handles that without duplicating
        # the split-vs-single decision here.
        return self._rerank_service(query, [(0.0, c) for c in survivors], top_n)

    def _rerank_via_logprobs(
        self, query: str, chunks: list[Any], top_n: int
    ) -> list[tuple[float, Any]]:
        """Score each (query, doc) with the official Qwen3-Reranker template."""
        import math
        import requests

        # Derive the /v1/completions URL from the configured base.
        base = self.rerank_url
        for suffix in ("/v1/rerank", "/rerank", "/v1/completions", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        comp_url = base.rstrip("/") + "/v1/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}

        def _prompt(doc: str) -> str:
            return (
                f"{self._RERANK_PREFIX}<Instruct>: {self._RERANK_INSTRUCT}\n"
                f"<Query>: {query}\n<Document>: {doc}{self._RERANK_SUFFIX}"
            )

        def _score(i: int, doc: str) -> tuple[int, float]:
            try:
                resp = requests.post(comp_url, headers=headers, json={
                    "model": self.rerank_model,
                    "prompt": _prompt(doc[:4000]),
                    "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
                }, timeout=120)
                resp.raise_for_status()
                lp = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]
            except Exception as exc:
                logger.warning("[HerO4] Rerank logprob call failed (%s)", exc)
                return i, -1.0
            yes = math.exp(lp["yes"]) if "yes" in lp else 0.0
            no = math.exp(lp["no"]) if "no" in lp else (math.exp(lp["No"]) if "No" in lp else 0.0)
            return i, yes / (yes + no + 1e-30)

        scores: list[float] = [0.0] * len(chunks)
        any_ok = False
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.judge_max_workers)) as pool:
                futures = [pool.submit(_score, i, c.text) for i, c in enumerate(chunks)]
                for fut in as_completed(futures):
                    i, s = fut.result()
                    if s >= 0:
                        scores[i] = s
                        any_ok = True
        except Exception as exc:
            logger.warning("[HerO4] Rerank stage failed (%s); degrading to embedding order", exc)
            return [(0.0, c) for c in chunks][:top_n]
        if not any_ok:
            return [(0.0, c) for c in chunks][:top_n]

        ranked = sorted(zip(scores, chunks), key=lambda it: it[0], reverse=True)
        return ranked[:top_n]

    def _rerank_via_rerank_api(
        self, query: str, chunks: list[Any], top_n: int
    ) -> list[tuple[float, Any]]:
        """Native /v1/rerank path (kept for deployments where it works)."""
        import requests

        documents = [chunk.text for chunk in chunks]
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
        try:
            resp = requests.post(self.rerank_url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("[HerO4] Rerank service failed (%s); degrading to embedding order", exc)
            return [(0.0, c) for c in chunks][:top_n]

        items = data.get("results", data) if isinstance(data, dict) else data
        ranked: list[tuple[float, Any]] = []
        for item in items:
            idx = item.get("index")
            score = item.get("score", item.get("relevance_score", 0.0))
            if idx is None or not (0 <= idx < len(chunks)):
                continue
            ranked.append((float(score), chunks[idx]))
        if not ranked:
            logger.warning("[HerO4] Rerank returned no usable items; degrading")
            return [(0.0, c) for c in chunks][:top_n]
        ranked.sort(key=lambda it: it[0], reverse=True)
        return ranked[:top_n]

    def _bm25_search(
        self,
        index: Any,
        query: str,
        top_k: int
    ) -> list[tuple[float, Any]]:
        """
        BM25 retrieval without the 20-result limit.

        Override parent class to support large candidate pools for HerO.
        """
        from collections import Counter
        import math

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        query_counts = Counter(query_terms)
        top_k = max(1, int(top_k))  # Remove the min(top_k, 20) limit

        scored: list[tuple[float, Any]] = []
        for chunk in index.chunks:
            doc_len = len(chunk.tokens) or 1
            score = 0.0
            for term, qf in query_counts.items():
                tf = chunk.term_counts.get(term, 0)
                if tf <= 0:
                    continue
                idf = index.idf.get(term, 0.0)
                denom = tf + self.bm25_k1 * (
                    1.0 - self.bm25_b + self.bm25_b * doc_len / (index.avg_doc_len or 1.0)
                )
                score += idf * (tf * (self.bm25_k1 + 1.0) / denom) * (1.0 + math.log(qf))

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def _embedding_rerank(
        self,
        claim: str,
        hypo_docs: list[str],
        candidates: list[tuple[float, Any]],
        top_k: int
    ) -> list[tuple[float, Any]]:
        """
        Rerank BM25 candidates using average embedding (aligned with HerO reranking.py).

        Query side: [instruct(claim)] + [instruct(doc) for doc in hypo_docs] -> average vector
        Document side: plain text, no instruct prefix

        The Jina backend handles the query/document asymmetry itself via
        ``input_type``, so the manual "Instruct: ...\\nQuery: ..." wrapping
        below is skipped for it (double-wrapping would just add noise to the
        text Jina already encodes asymmetrically).
        """
        if not candidates:
            return []

        chunks = [chunk for _, chunk in candidates]
        sentences = [chunk.text for chunk in chunks]

        if self._use_jina_embed:
            queries = [claim] + [doc for doc in hypo_docs if doc.strip()]
        else:
            # Build instruct-wrapped query list
            task_desc = 'Given a web search query, retrieve relevant passages that answer the query'
            queries = [f'Instruct: {task_desc}\nQuery: {claim}']
            queries += [f'Instruct: {task_desc}\nQuery: {doc}' for doc in hypo_docs if doc.strip()]

        logger.debug("[HerO] Query parts: %d", len(queries))

        try:
            query_embs = self._get_embeddings(queries, input_type="query")
            avg_query_emb = np.mean(query_embs, axis=0, keepdims=True)

            sentence_embs = self._get_embeddings(sentences, input_type="document")

            avg_norm = avg_query_emb / (np.linalg.norm(avg_query_emb, axis=1, keepdims=True) + 1e-8)
            sent_norm = sentence_embs / (np.linalg.norm(sentence_embs, axis=1, keepdims=True) + 1e-8)
            scores = np.dot(avg_norm, sent_norm.T)[0]

        except Exception as e:
            logger.warning("[HerO] Embedding reranking failed: %s", e)
            return candidates[:top_k]

        ranked_indices = np.argsort(scores)[::-1]
        ranked_results = [(float(scores[i]), chunks[i]) for i in ranked_indices]

        deduplicated = self._deduplicate_results(claim, ranked_results, top_k)
        return deduplicated

    def _deduplicate_results(
        self,
        query: str,
        ranked_results: list[tuple[float, Any]],
        top_k: int
    ) -> list[tuple[float, Any]]:
        """
        Remove duplicate and query-similar results.

        Follows HerO's select_top_k logic:
        - Skip exact duplicate texts
        - Skip texts too similar to the query (likely the claim itself)
        - Skip texts where query is a large substring

        Args:
            query: Search query
            ranked_results: Embedding-ranked results
            top_k: Target number of unique results

        Returns:
            Deduplicated results
        """
        selected = []
        seen_texts = set()
        query_processed = self._preprocess_text(query)

        for score, chunk in ranked_results:
            if len(selected) >= top_k:
                break

            text = chunk.text
            text_processed = self._preprocess_text(text)

            # Skip exact duplicates
            if text_processed in seen_texts:
                continue

            # Skip texts too similar to query (likely claim itself)
            sim = self._text_similarity(query_processed, text_processed)
            if sim > 0.97:
                seen_texts.add(text_processed)
                continue

            # Skip if query is a large substring of text
            if query_processed in text_processed:
                if len(query_processed) / max(len(text_processed), 1) > 0.92:
                    seen_texts.add(text_processed)
                    continue

            selected.append((score, chunk))
            seen_texts.add(text_processed)

        return selected

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Preprocess text for similarity comparison."""
        return re.sub(r'[^\w\s]+', '', text).lower().strip()

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        Compute TF-IDF cosine similarity between two texts.

        Used for deduplication, not for ranking.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            if not text1 or not text2:
                return 0.0

            vectorizer = TfidfVectorizer().fit_transform([text1, text2])
            vectors = vectorizer.toarray()
            return float(cosine_similarity(vectors)[0][1])
        except Exception:
            return 0.0

    def _judge_relevance(
        self,
        query: str,
        candidates: list[tuple[float, Any]],
        final_k: int,
    ) -> tuple[list[tuple[float, Any]], list[bool | None], list[str]]:
        """LLM relevance filter (soft): keep RELEVANT items, preserving rerank order.

        Each candidate is judged with its own concurrent request (ThreadPoolExecutor,
        mirroring the embedding remote path). Soft semantics: if fewer than final_k
        survive, backfill from the rerank-ordered remainder. Degrades to the top
        final_k rerank results on judge failure.

        Returns ``(kept_results, verdicts, summaries)`` where ``verdicts[i]`` is the
        judge's boolean for ``items[i]`` (None if unjudged) and ``summaries[i]`` is
        the model's concise summary for relevant items ("" otherwise)."""
        if not candidates:
            return [], [], []
        items = candidates[: self.judge_max_items]

        model = self.judge_model or os.environ.get("MODEL", "")
        base_url = self.judge_base_url or os.environ.get("OPENAI_BASE_URL", "")
        api_key = self.judge_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        if not model or not base_url:
            logger.warning("[HerO4] Judge not configured (model/base_url); skipping judge")
            return candidates[:final_k], [], []

        verdicts: list[bool | None] = [None] * len(items)
        summaries: list[str] = [""] * len(items)

        def _judge_one(i: int, chunk: Any) -> tuple[int, bool, str]:
            relevant, summary = self._judge_call(
                model, base_url, api_key, query, chunk.text,
                claim=getattr(self, "claim_text", ""),
            )
            return i, relevant, summary

        try:
            with ThreadPoolExecutor(max_workers=max(1, self.judge_max_workers)) as pool:
                futures = [pool.submit(_judge_one, i, c) for i, (_, c) in enumerate(items)]
                for fut in as_completed(futures):
                    i, relevant, summary = fut.result()
                    verdicts[i] = relevant
                    summaries[i] = summary
        except Exception as exc:
            logger.warning("[HerO4] Judge stage failed (%s); degrading to rerank top-k", exc)
            return candidates[:final_k], verdicts, summaries

        kept = [items[i] for i, v in enumerate(verdicts) if v]
        n_relevant = len(kept)
        if not kept:
            kept = [items[0]]
            summaries[0] = "（目前没有搜到很匹配的证据，以下为相关性最高的结果仅供参考）"
            logger.info("[HerO4] Judge: 0/%d relevant, returning top-1 reranker fallback", len(items))
        else:
            logger.info("[HerO4] Judge: %d/%d relevant, returning %d (hard filter)",
                        n_relevant, len(items), len(kept))
        return kept[:final_k], verdicts, summaries

    def _judge_call(
        self,
        model: str,
        base_url: str,
        api_key: str,
        query: str,
        evidence: str,
        max_chars: int = 4000,
        claim: str = "",
    ) -> tuple[bool, str]:
        """Single relevance verdict via an OpenAI-compatible chat endpoint.

        The model is asked for a strict JSON object:
          - relevant:     {"relevant": true, "summary": "<concise summary>"}
          - not relevant: {"relevant": false}

        Returns ``(is_relevant, summary)``. ``summary`` is empty when the
        evidence is judged not relevant. When ``claim`` is provided the judge
        anchors on the original claim (broader than the model's search query),
        which avoids over-literal rejections when the query uses keywords that
        don't appear verbatim in otherwise on-topic evidence.
        """
        import json as _json
        import re as _re
        import requests

        url = base_url.rstrip("/") + "/chat/completions"
        # Anchor relevance on the original claim; the search query is secondary.
        target = (
            f"Claim being fact-checked:\n{claim}\n\nCurrent search query:\n{query}"
            if claim else f"Query:\n{query}"
        )
        prompt = (
            "You judge whether a piece of evidence is relevant to verifying a "
            "fact-checking claim. Evidence counts as relevant if it provides "
            "context, support, or contradiction for the claim, even if it does "
            "not mention every keyword.\n"
            "Respond with ONLY a JSON object, nothing else:\n"
            '- If relevant: {"relevant": true, "summary": "<concise summary of the '
            'evidence as it bears on the claim>"}\n'
            '- If not relevant: {"relevant": false}\n'
            "Do not output any text outside the JSON object.\n\n"
            f"{target}\n\nEvidence:\n{evidence[:max_chars]}"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            # Generous budget: reasoning models (e.g. DeepSeek V4) spend tokens on
            # hidden reasoning before emitting the JSON.
            "max_tokens": 2048,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            # Some reasoning models put the answer in content, others leave content
            # empty and expose reasoning_content; check both.
            content = (msg.get("content") or "") or (msg.get("reasoning_content") or "")
        except Exception as exc:
            logger.warning("[HerO4] Judge call failed (%s); treating as relevant", exc)
            return True, ""  # fail-open so we never drop evidence on transient errors

        if not content.strip():
            return True, ""  # empty -> keep (fail-open)

        # Extract the JSON object (handle ```json fences / surrounding prose).
        match = _re.search(r"\{.*\}", content, _re.DOTALL)
        if match:
            try:
                obj = _json.loads(match.group(0))
                relevant = bool(obj.get("relevant"))
                summary = str(obj.get("summary") or "").strip() if relevant else ""
                return relevant, summary
            except (ValueError, TypeError):
                pass

        # Fallback: couldn't parse JSON -> infer from text, keep on ambiguity.
        text = content.upper()
        if '"RELEVANT": FALSE' in text or "NOT_RELEVANT" in text or "NOT RELEVANT" in text:
            return False, ""
        return True, ""

    def _get_claim_log_dir(self) -> Path:
        """Get or create the log directory for the current claim within the run folder."""
        import json
        from datetime import datetime
        from pathlib import Path

        if HerOSearchTool._run_log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            HerOSearchTool._run_log_dir = Path("output/hero_logs") / f"run_{timestamp}"

        claim_id = self.claim_id or "unknown"
        claim_dir = HerOSearchTool._run_log_dir / f"claim_{claim_id}"
        claim_dir.mkdir(parents=True, exist_ok=True)
        return claim_dir

    def set_call_label(self, turn: int, call_index: int) -> None:
        """Called by BeliefTracerEnvironment._run_tool_labeled, in the worker
        thread that is about to invoke this call, just before ``forward()``.

        Stamps this thread's (turn, call_index) so ``_next_round`` can label
        hero_logs files by "which turn, which call within that turn" instead
        of racing other threads for a shared counter's next value.
        """
        HerOSearchTool._call_label.value = (int(turn), int(call_index))

    def _next_round(self) -> str:
        """Return this call's round label for hero_logs file names.

        Call this ONCE per search call (at the top of ``_search_four_stage`` /
        ``_search``) and thread the returned value through every ``_log_*``
        call for that same search — do not call it again, or have ``_log_*``
        re-derive it, partway through one call's stages. Two concurrent tool
        calls on the same claim (parallel tool execution) interleave their
        stages; a stage that re-derives the label fresh instead of using its
        call's allocated one can pick up a DIFFERENT call's position, causing
        both calls' stage logs to collide on one file name.

        Normal path: BeliefTracerEnvironment fixes (turn, call_index) for
        every pending tool call BEFORE dispatching to the thread pool (see
        ``_run_tool_labeled``), so each call's label is known upfront — no
        negotiation over shared state needed. Returns e.g. "1_2" (turn 1,
        2nd call in that turn).

        Fallback: a call made without going through the environment (e.g. a
        standalone script invoking ``.forward()`` directly, as
        ``scripts/test_hero_retrieval.py`` does) has no label set. Falls back
        to the pre-existing shared-counter behavior, which self-numbers but
        can race under concurrent calls on the same claim outside the
        environment's control.
        """
        label = getattr(HerOSearchTool._call_label, "value", None)
        if label is not None:
            turn, call_index = label
            return f"{turn}_{call_index}"

        claim_id = self.claim_id or "unknown"
        with HerOSearchTool._round_counter_lock:
            HerOSearchTool._claim_round_counter[claim_id] = (
                HerOSearchTool._claim_round_counter.get(claim_id, 0) + 1
            )
            return str(HerOSearchTool._claim_round_counter[claim_id])

    def _log_bm25_results(
        self, query: str, bm25_results: list[tuple[float, Any]], round_num: str,
    ) -> None:
        """Log BM25 results to file for analysis."""
        import json
        from datetime import datetime
        from pathlib import Path

        claim_dir = self._get_claim_log_dir()
        log_file = claim_dir / f"bm25_round_{round_num}.json"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "claim_id": self.claim_id,
            "round": round_num,
            "query": query,
            "num_results": len(bm25_results),
            "results": []
        }

        for i, (score, chunk) in enumerate(bm25_results[:100]):
            log_entry["results"].append({
                "rank": i + 1,
                "bm25_score": float(score),
                "text": chunk.text[:500],
                "full_text_length": len(chunk.text),
                "url": chunk.url if hasattr(chunk, 'url') else "",
                "chunk_id": chunk.chunk_id if hasattr(chunk, 'chunk_id') else ""
            })

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        logger.info("[HerO] BM25 results saved to: %s", log_file)

    def _log_embedding_results(
        self,
        query: str,
        embedding_results: list[tuple[float, Any]],
        bm25_results: list[tuple[float, Any]],
        round_num: str,
    ) -> None:
        """Log embedding reranking results to file for analysis."""
        import json
        from datetime import datetime
        from pathlib import Path

        claim_dir = self._get_claim_log_dir()
        log_file = claim_dir / f"embedding_round_{round_num}.json"

        bm25_scores = {id(chunk): score for score, chunk in bm25_results}

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "claim_id": self.claim_id,
            "round": round_num,
            "query": query,
            "num_results": len(embedding_results),
            "results": []
        }

        for i, (emb_score, chunk) in enumerate(embedding_results):
            bm25_score = bm25_scores.get(id(chunk), None)
            bm25_rank = None

            for j, (_, c) in enumerate(bm25_results):
                if id(c) == id(chunk):
                    bm25_rank = j + 1
                    break

            log_entry["results"].append({
                "embedding_rank": i + 1,
                "embedding_score": float(emb_score),
                "bm25_rank": bm25_rank,
                "bm25_score": float(bm25_score) if bm25_score is not None else None,
                "rank_change": (bm25_rank - (i + 1)) if bm25_rank else None,
                "text": chunk.text[:500],
                "full_text_length": len(chunk.text),
                "url": chunk.url if hasattr(chunk, 'url') else "",
                "chunk_id": chunk.chunk_id if hasattr(chunk, 'chunk_id') else ""
            })

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        logger.info("[HerO] Embedding results saved to: %s", log_file)


__all__ = ["HerOSearchTool"]
