"""Shared LLM infrastructure for both construct backends.

Extracted verbatim from the two backends' former ``llm.py`` copies: token
usage tracking, prompt/embedding audit logging, the OpenAI-compatible
``EmbeddingClient`` with its in-memory cache, and pairwise cosine similarity.
Modules here must not import from ``bcg.construct.api_based`` or
``bcg.construct.light``; backend-specific config loading and model calls stay
in each backend's ``llm.py``.
"""

from __future__ import annotations

import contextvars
import json
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


def _estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) when the API returns no usage block."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _coerce_usage(usage: Any) -> dict[str, int | None]:
    """Pull prompt/completion/reasoning/total counts from an SDK usage object."""
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    def _get(name: str) -> int | None:
        v = getattr(usage, name, None)
        if v is None and isinstance(usage, dict):
            v = usage.get(name)
        return v

    pt = _get("prompt_tokens")
    ct = _get("completion_tokens")
    tt = _get("total_tokens")
    reasoning: int | None = None

    details = getattr(usage, "completion_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", None)
        if reasoning is None and isinstance(details, dict):
            reasoning = details.get("reasoning_tokens")

    # Responses-style providers use output_tokens_details. Supporting it here
    # keeps the graph accounting correct for compatible gateways that expose
    # Responses usage through an otherwise Chat-Completions-shaped client.
    if reasoning is None:
        details = getattr(usage, "output_tokens_details", None)
        if details is None and isinstance(usage, dict):
            details = usage.get("output_tokens_details")
        if details is not None:
            reasoning = getattr(details, "reasoning_tokens", None)
            if reasoning is None and isinstance(details, dict):
                reasoning = details.get("reasoning_tokens")

    if pt is None and ct is None and tt is None:
        dump: dict[str, Any] | None = None
        if hasattr(usage, "model_dump"):
            try:
                dump = usage.model_dump()
            except Exception:
                dump = None
        elif isinstance(usage, dict):
            dump = usage
        if isinstance(dump, dict):
            pt = dump.get("prompt_tokens")
            ct = dump.get("completion_tokens")
            tt = dump.get("total_tokens")
            completion_details = dump.get("completion_tokens_details") or {}
            output_details = dump.get("output_tokens_details") or {}
            if reasoning is None and isinstance(completion_details, dict):
                reasoning = completion_details.get("reasoning_tokens")
            if reasoning is None and isinstance(output_details, dict):
                reasoning = output_details.get("reasoning_tokens")

    if tt is None and pt is not None and ct is not None:
        tt = pt + ct
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": reasoning,
        "total_tokens": tt,
    }


class TokenUsageTracker:
    """Accumulates per-call token usage so the cost of one input can be estimated.

    Thread-safe: ``record()`` / ``reset()`` are protected by an internal lock
    so concurrent callers (parallel LLM calls issued by merge.py's incremental
    merge-verify step, in particular) never race on ``records`` or produce
    duplicate ``index`` values.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._label: str = "unlabeled"
        self._lock = threading.Lock()

    # -- labelling -------------------------------------------------------
    def set_label(self, label: str) -> None:
        self._label = label or "unlabeled"

    @contextmanager
    def label(self, label: str):
        prev = self._label
        self.set_label(label)
        try:
            yield
        finally:
            self._label = prev

    # -- recording -------------------------------------------------------
    def record(
        self,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        reasoning_tokens: int | None = None,
        label: str | None = None,
        estimated: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            rec = {
                "index": len(self.records),
                "label": label if label is not None else self._label,
                "model": model,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "estimated": estimated,
            }
            self.records.append(rec)
        return rec

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self._label = "unlabeled"

    # -- summaries -------------------------------------------------------
    @property
    def n_calls(self) -> int:
        return len(self.records)

    def totals(self) -> dict[str, int]:
        def _s(key: str) -> int:
            return sum(int(r.get(key) or 0) for r in self.records)

        return {
            "n_calls": self.n_calls,
            "input_tokens": _s("input_tokens"),
            "output_tokens": _s("output_tokens"),
            "reasoning_tokens": _s("reasoning_tokens"),
            "total_tokens": _s("total_tokens"),
        }

    def llm_totals(self) -> dict[str, int]:
        """Return chat-model usage, excluding embedding-model estimates/calls."""

        records = [
            record
            for record in self.records
            if not str(record.get("label") or "").startswith("embedding:")
            and str(record.get("label") or "") != "embedding"
        ]

        def _sum(key: str) -> int:
            return sum(int(record.get(key) or 0) for record in records)

        return {
            "n_calls": len(records),
            "input_tokens": _sum("input_tokens"),
            "output_tokens": _sum("output_tokens"),
            "reasoning_tokens": _sum("reasoning_tokens"),
            "total_tokens": _sum("total_tokens"),
        }

    def by_label(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            lbl = r.get("label") or "unlabeled"
            agg = out.setdefault(
                lbl,
                {
                    "n_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                },
            )
            agg["n_calls"] += 1
            agg["input_tokens"] += int(r.get("input_tokens") or 0)
            agg["output_tokens"] += int(r.get("output_tokens") or 0)
            agg["reasoning_tokens"] += int(r.get("reasoning_tokens") or 0)
            agg["total_tokens"] += int(r.get("total_tokens") or 0)
        return out

    def estimate_cost(self, pricing: dict[str, Any] | None) -> dict[str, Any] | None:
        """pricing = {"input_per_1k": x, "output_per_1k": y} (USD); None -> no cost."""
        if not pricing:
            return None
        t = self.totals()
        in_rate = float(pricing.get("input_per_1k", 0.0))
        out_rate = float(pricing.get("output_per_1k", 0.0))
        input_cost = t["input_tokens"] / 1000.0 * in_rate
        output_cost = t["output_tokens"] / 1000.0 * out_rate
        return {
            "currency": pricing.get("currency", "USD"),
            "input_per_1k": in_rate,
            "output_per_1k": out_rate,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(input_cost + output_cost, 6),
        }

    def summary(self, pricing: dict[str, Any] | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "totals": self.totals(),
            "llm_totals": self.llm_totals(),
            "by_label": self.by_label(),
        }
        cost = self.estimate_cost(pricing)
        if cost is not None:
            d["estimated_cost"] = cost
        return d

    def to_dict(self, pricing: dict[str, Any] | None = None) -> dict[str, Any]:
        d = self.summary(pricing)
        d["calls"] = self.records
        return d

    # -- output ----------------------------------------------------------
    def save_json(self, path: Any, pricing: dict[str, Any] | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(pricing), f, ensure_ascii=False, indent=2)

    def render_text(self, pricing: dict[str, Any] | None = None) -> str:
        t = self.totals()
        bar = "=" * 74
        sub = "-" * 74
        lines = [
            bar,
            " LLM token usage for this input",
            bar,
            f"  total LLM calls : {t['n_calls']}",
            f"  input tokens    : {t['input_tokens']:,}",
            f"  output tokens   : {t['output_tokens']:,}",
            f"  total tokens    : {t['total_tokens']:,}",
        ]
        cost = self.estimate_cost(pricing)
        if cost:
            lines.append(
                f"  estimated cost  : {cost['total_cost']:.6f} {cost['currency']}"
                f"  (in {cost['input_per_1k']}/1k, out {cost['output_per_1k']}/1k)"
            )
        lines += [sub, " by stage:"]
        for lbl, agg in sorted(
            self.by_label().items(), key=lambda kv: kv[1]["total_tokens"], reverse=True
        ):
            lines.append(
                f"   {lbl:<34.34} calls={agg['n_calls']:>3}  "
                f"in={agg['input_tokens']:>8,}  "
                f"out={agg['output_tokens']:>7,}  "
                f"total={agg['total_tokens']:>8,}"
            )
        lines += [
            sub,
            " per-call detail:",
            f"   {'#':>3}  {'label':<34} {'model':<16} "
            f"{'in':>8} {'out':>7} {'total':>8}  est",
        ]
        for r in self.records:
            est = "*" if r.get("estimated") else ""
            lines.append(
                f"   {r['index']:>3}  {str(r.get('label') or ''):<34.34} "
                f"{str(r.get('model') or ''):<16.16} "
                f"{int(r.get('input_tokens') or 0):>8,} "
                f"{int(r.get('output_tokens') or 0):>7,} "
                f"{int(r.get('total_tokens') or 0):>8,}  {est}"
            )
        if any(r.get("estimated") for r in self.records):
            lines += [sub, " * = tokens estimated (API returned no usage block)"]
        lines.append(bar)
        return "\n".join(lines) + "\n"

    def save_text(self, path: Any, pricing: dict[str, Any] | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.render_text(pricing))


# ---------------------------------------------------------------------------
# Context-local engine state (token usage tracker + audit log paths)
# ---------------------------------------------------------------------------
# These used to be bare process-global variables (``USAGE = TokenUsageTracker()``,
# ``PROMPT_LOG_PATH: Optional[Path] = None``). That is only safe when exactly
# one trajectory is ever being built at a time — which is what the single
# server-wide lock in online_server.py used to guarantee.
#
# To let the HTTP server process several problem_ids concurrently, each of
# these now lives in a ``contextvars.ContextVar`` instead: a value set via
# ``.set()`` is only visible within "the current execution context", which in
# practice means the current thread (each new thread starts with a fresh,
# empty context). Since ThreadingHTTPServer hands each incoming request its
# own thread, two concurrently-running sessions never see or clobber each
# other's tracker / log paths, with no extra locking needed on the read side.
#
# run.py / pipeline.py are unaffected by this change: they are single-threaded,
# so the lazy "create on first use" fallback in ``_get_usage()`` behaves
# exactly like the old module-level singleton did — one tracker, created once,
# reused for the rest of that thread's life.
#
# IMPORTANT for callers that hand work to a *new* thread (e.g. merge.py's
# parallel LLM-verify calls): contextvars are NOT automatically inherited by
# threads spawned via threading.Thread / concurrent.futures.ThreadPoolExecutor
# — each worker thread starts with its own empty context. The pattern for
# those callers is: read the current values with current_usage_tracker() /
# current_prompt_log_path() / current_embedding_log_path() in the submitting
# thread, pass them into the worker function as plain arguments, and call
# bind_usage_tracker(...) / bind_prompt_log_path(...) / bind_embedding_log_path(...)
# again at the top of the worker before doing any LLM/embedding call. See
# merge.py's parallel incremental-merge verification loop for a worked example.

_usage_var: contextvars.ContextVar[TokenUsageTracker] = contextvars.ContextVar(
    "usage_tracker"
)
_prompt_log_path_var: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "prompt_log_path", default=None
)
_embedding_log_path_var: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "embedding_log_path", default=None
)

# Small locks around the actual file appends. Two threads (e.g. two parallel
# merge-verify LLM calls within one session, or two different sessions that
# happen to share a log file path) could otherwise interleave partial writes.
_PROMPT_LOG_LOCK = threading.Lock()
_EMBEDDING_LOG_LOCK = threading.Lock()


def _get_usage() -> TokenUsageTracker:
    """Return the tracker bound to the current context, creating one lazily.

    The lazy-create path is what keeps single-threaded callers (run.py /
    pipeline.py) working unchanged: the first access on the main thread
    creates one tracker that then stays bound for the rest of that thread's
    life, exactly like the old ``USAGE = TokenUsageTracker()`` singleton.
    """
    try:
        return _usage_var.get()
    except LookupError:
        tracker = TokenUsageTracker()
        _usage_var.set(tracker)
        return tracker


def current_usage_tracker() -> TokenUsageTracker:
    """Read-only accessor for the current context's tracker (see module notes above)."""
    return _get_usage()


def bind_usage_tracker(tracker: TokenUsageTracker) -> contextvars.Token:
    """Bind ``tracker`` as the active usage tracker for the current context.

    Returns a token; pass it to ``unbind_usage_tracker`` to restore whatever
    was bound before (mirrors ``ContextVar.set()`` / ``.reset()``).
    """
    return _usage_var.set(tracker)


def unbind_usage_tracker(token: contextvars.Token) -> None:
    _usage_var.reset(token)


class _UsageProxy:
    """Drop-in replacement for the old module-level ``USAGE`` singleton.

    Every attribute access is forwarded to the tracker bound to the CURRENT
    context, so existing call sites (``USAGE.set_label(...)``,
    ``USAGE.record(...)``, ``USAGE.reset()``, ``USAGE.summary(...)``, ...) in
    stream.py / merge.py / pipeline.py keep working completely unmodified,
    while different concurrently-running sessions transparently get different
    underlying trackers.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_get_usage(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_get_usage(), name, value)


# Backward-compatible name. Behaves like a process-global singleton for
# single-threaded callers (run.py), and like a per-context tracker for the
# online server (see StreamingTrajectorySession._engine() in online.py).
USAGE = _UsageProxy()


def current_prompt_log_path() -> Path | None:
    return _prompt_log_path_var.get()


def bind_prompt_log_path(path: Any) -> contextvars.Token:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return _prompt_log_path_var.set(p)


def unbind_prompt_log_path(token: contextvars.Token) -> None:
    _prompt_log_path_var.reset(token)


def set_prompt_log_path(path: Any) -> None:
    """Set the JSONL path where prompts (LLM inputs) are appended for auditing.

    The caller should set this to <out_dir>/logs/prompts.jsonl so each item /
    trajectory keeps its own prompt audit file. Kept for backward
    compatibility with call sites that don't need the token-based bind/unbind
    pair (e.g. stream.py's one-shot call at builder construction time).
    """
    bind_prompt_log_path(path)


def _log_prompt(record: dict[str, Any]) -> None:
    path = _prompt_log_path_var.get()
    if path is None:
        return
    try:
        with _PROMPT_LOG_LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must not raise for normal pipeline runs
        return


def current_embedding_log_path() -> Path | None:
    return _embedding_log_path_var.get()


def bind_embedding_log_path(path: Any) -> contextvars.Token:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return _embedding_log_path_var.set(p)


def unbind_embedding_log_path(token: contextvars.Token) -> None:
    _embedding_log_path_var.reset(token)


def _record_usage(resp: Any, *, model: str, prompt: str, label: str | None) -> None:
    """Record token usage from a chat-completions response into USAGE."""
    counts = _coerce_usage(getattr(resp, "usage", None))
    if counts["prompt_tokens"] is not None or counts["completion_tokens"] is not None:
        USAGE.record(
            model=model,
            prompt_tokens=counts["prompt_tokens"],
            completion_tokens=counts["completion_tokens"],
            total_tokens=counts["total_tokens"],
            reasoning_tokens=counts["reasoning_tokens"],
            label=label,
            estimated=False,
        )
        return
    out_text = ""
    try:
        out_text = resp.choices[0].message.content or ""
    except Exception:
        out_text = ""
    pt = _estimate_tokens(prompt)
    ct = _estimate_tokens(out_text)
    USAGE.record(
        model=model,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        label=label,
        estimated=True,
    )


# ---------------------------------------------------------------------------
# Embedding client (shared OpenAI-compatible wrapper)
# ---------------------------------------------------------------------------


class EmbeddingClient:
    """
    OpenAI-compatible embeddings client with an in-memory cache and a JSONL
    audit log.  Every `embed()` call appends one record to the log containing
    the purpose tag, the FULL list of input texts, which of them were served
    from cache, the embedding dimension, and timing — so each step of the
    split / merge process is fully reconstructable.

    The log path is stored in a context-local variable (see
    ``bind_embedding_log_path`` / ``current_embedding_log_path`` above), not
    on the instance. ``SessionManager`` shares ONE ``EmbeddingClient`` across
    every problem_id (so the embedding cache is shared, which is desirable
    for cost/latency) — storing the log path as a plain instance attribute
    would let two concurrently-running sessions clobber each other's log
    destination. The cache dict itself is also lock-protected since it is
    genuinely shared and mutated from multiple threads once problem_ids run
    concurrently.
    """

    def __init__(self, cfg: dict[str, Any], log_path: Any | None = None) -> None:
        self.client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        self.model: str = cfg["model"]
        self.batch_size: int = int(cfg.get("batch_size", 32) or 32)
        self.dimensions: int | None = cfg.get("dimensions")
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = threading.Lock()
        if log_path:
            self.set_log_path(log_path)

    # -- logging ----------------------------------------------------------
    def set_log_path(self, path: Any) -> contextvars.Token:
        """Bind the embedding-audit log path for the CURRENT context (thread).

        Returns a token; pass it to ``unbind_log_path`` to restore the
        previous binding when done (mirrors the llm.py bind/unbind helpers).
        Existing call sites that ignore the return value keep working as
        "set it and forget it" for the lifetime of that thread.
        """
        return bind_embedding_log_path(path)

    def unbind_log_path(self, token: contextvars.Token) -> None:
        unbind_embedding_log_path(token)

    def _log(self, record: dict[str, Any]) -> None:
        path = current_embedding_log_path()
        if path is None:
            return
        with _EMBEDDING_LOG_LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # -- main entry ---------------------------------------------------------
    def embed(self, texts: list[str], purpose: str = "") -> list[list[float]]:
        """Embed a list of texts (order-preserving). Cached texts are reused."""
        t0 = time.time()
        results: list[list[float] | None] = [None] * len(texts)
        cached_flags: list[bool] = [False] * len(texts)
        missing_idx: list[int] = []
        with self._cache_lock:
            for i, t in enumerate(texts):
                v = self._cache.get(t)
                if v is not None:
                    results[i] = v
                    cached_flags[i] = True
                else:
                    missing_idx.append(i)

        for batch_start in range(0, len(missing_idx), self.batch_size):
            batch_ids = missing_idx[batch_start : batch_start + self.batch_size]
            batch = [texts[i] for i in batch_ids]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = int(self.dimensions)
            last_err: Exception | None = None
            resp = None
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(**kwargs)
                    break
                except Exception as e:
                    last_err = e
                    print(
                        f"    [embed retry {attempt + 1}/3] {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    if attempt < 2:
                        time.sleep(2**attempt)
            if resp is None:
                raise RuntimeError(f"Embedding call failed after retries: {last_err}")

            data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
            vecs = [list(d.embedding) for d in data]
            if len(vecs) != len(batch):
                raise RuntimeError(
                    f"Embedding API returned {len(vecs)} vectors for {len(batch)} inputs"
                )
            with self._cache_lock:
                for t, v in zip(batch, vecs, strict=False):
                    self._cache[t] = v
            for i, v in zip(batch_ids, vecs, strict=False):
                results[i] = v

            counts = _coerce_usage(getattr(resp, "usage", None))
            pt = counts["prompt_tokens"]
            estimated = pt is None
            if estimated:
                pt = sum(_estimate_tokens(t) for t in batch)
            USAGE.record(
                model=self.model,
                prompt_tokens=pt,
                completion_tokens=0,
                total_tokens=counts["total_tokens"] or pt,
                label=f"embedding:{purpose}" if purpose else "embedding",
                estimated=estimated,
            )

        dim = len(results[0]) if results and results[0] is not None else 0
        self._log(
            {
                "ts": datetime.now(UTC).isoformat(),
                "purpose": purpose,
                "model": self.model,
                "n_texts": len(texts),
                "n_cached": sum(cached_flags),
                "n_api": len(missing_idx),
                "dimension": dim,
                "elapsed_s": round(time.time() - t0, 3),
                "texts": [
                    {"index": i, "cached": cached_flags[i], "text": t}
                    for i, t in enumerate(texts)
                ],
            }
        )
        return [r if r is not None else [] for r in results]


def cosine_similarity_matrix(vectors: list[list[float]]):
    """Pairwise cosine similarity (numpy array, shape n x n)."""
    import numpy as np

    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = arr / norms
    sim = unit @ unit.T
    return np.clip(sim, -1.0, 1.0)
