"""
llm.py
======
OpenAI-compatible client wrappers used by every stage.

Two clients live here:

  * chat completions  — `load_config` / `make_client` / `call_model`
    (unchanged behaviour: temperature is forced to 0 by callers, retries,
    token accounting via the process-global USAGE tracker).

  * embeddings        — `load_embedding_config` / `EmbeddingClient`
    Used by the split module (global sentence clustering) and the merge
    module (duplicate-belief candidate generation).  Every embedding call
    is appended to a JSONL log file (full input texts + cache hits), so the
    whole embedding-assisted process is auditable, and its token usage is
    recorded into USAGE as well.

`model_config.json` supports a reserved top-level key ``"embedding"`` that
holds the embedding endpoint.  That key is never picked as the default chat
model.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: missing openai. Install with: pip install openai", file=sys.stderr)
    raise


# Top-level config keys that are NOT chat-model entries.
RESERVED_CONFIG_KEYS = {"embedding"}


def _is_reserved_key(key: str) -> bool:
    """Reserved entries are never picked as the default chat model. Any key
    starting with 'embedding' is reserved, so multiple embedding entries
    (e.g. 'embedding', 'embedding_local') can coexist with chat entries."""
    return key in RESERVED_CONFIG_KEYS or key.startswith("embedding")


# ---------------------------------------------------------------------------
# Token-usage tracking
# ---------------------------------------------------------------------------
# Every successful call_model() / EmbeddingClient.embed() appends one record
# to the module-level USAGE tracker.  The pipeline tags each call with the
# current stage/segment via USAGE.set_label(...), and after a run writes the
# accumulated log to outputs/token_usage.{json,txt}.


def _estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) when the API returns no usage block."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _coerce_usage(usage: Any) -> Dict[str, Optional[int]]:
    """Pull prompt/completion/total token counts out of an SDK usage object."""
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    def _get(name: str) -> Optional[int]:
        v = getattr(usage, name, None)
        if v is None and isinstance(usage, dict):
            v = usage.get(name)
        return v

    pt = _get("prompt_tokens")
    ct = _get("completion_tokens")
    tt = _get("total_tokens")

    if pt is None and ct is None and tt is None:
        dump: Optional[Dict[str, Any]] = None
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

    if tt is None and pt is not None and ct is not None:
        tt = pt + ct
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}


class TokenUsageTracker:
    """Accumulates per-call token usage so the cost of one input can be estimated."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._label: str = "unlabeled"

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
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        label: Optional[str] = None,
        estimated: bool = False,
    ) -> Dict[str, Any]:
        rec = {
            "index": len(self.records),
            "label": label if label is not None else self._label,
            "model": model,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated": estimated,
        }
        self.records.append(rec)
        return rec

    def reset(self) -> None:
        self.records.clear()
        self._label = "unlabeled"

    # -- summaries -------------------------------------------------------
    @property
    def n_calls(self) -> int:
        return len(self.records)

    def totals(self) -> Dict[str, int]:
        def _s(key: str) -> int:
            return sum(int(r.get(key) or 0) for r in self.records)
        return {
            "n_calls": self.n_calls,
            "input_tokens": _s("input_tokens"),
            "output_tokens": _s("output_tokens"),
            "total_tokens": _s("total_tokens"),
        }

    def by_label(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.records:
            lbl = r.get("label") or "unlabeled"
            agg = out.setdefault(
                lbl,
                {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            agg["n_calls"] += 1
            agg["input_tokens"] += int(r.get("input_tokens") or 0)
            agg["output_tokens"] += int(r.get("output_tokens") or 0)
            agg["total_tokens"] += int(r.get("total_tokens") or 0)
        return out

    def estimate_cost(self, pricing: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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

    def summary(self, pricing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d: Dict[str, Any] = {"totals": self.totals(), "by_label": self.by_label()}
        cost = self.estimate_cost(pricing)
        if cost is not None:
            d["estimated_cost"] = cost
        return d

    def to_dict(self, pricing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = self.summary(pricing)
        d["calls"] = self.records
        return d

    # -- output ----------------------------------------------------------
    def save_json(self, path: Any, pricing: Optional[Dict[str, Any]] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(pricing), f, ensure_ascii=False, indent=2)

    def render_text(self, pricing: Optional[Dict[str, Any]] = None) -> str:
        t = self.totals()
        bar = "=" * 74
        sub = "-" * 74
        lines = [bar, " LLM token usage for this input", bar,
                 f"  total LLM calls : {t['n_calls']}",
                 f"  input tokens    : {t['input_tokens']:,}",
                 f"  output tokens   : {t['output_tokens']:,}",
                 f"  total tokens    : {t['total_tokens']:,}"]
        cost = self.estimate_cost(pricing)
        if cost:
            lines.append(
                f"  estimated cost  : {cost['total_cost']:.6f} {cost['currency']}"
                f"  (in {cost['input_per_1k']}/1k, out {cost['output_per_1k']}/1k)"
            )
        lines += [sub, " by stage:"]
        for lbl, agg in sorted(self.by_label().items(),
                               key=lambda kv: kv[1]["total_tokens"], reverse=True):
            lines.append(
                f"   {lbl:<34.34} calls={agg['n_calls']:>3}  "
                f"in={agg['input_tokens']:>8,}  "
                f"out={agg['output_tokens']:>7,}  "
                f"total={agg['total_tokens']:>8,}"
            )
        lines += [sub, " per-call detail:",
                  f"   {'#':>3}  {'label':<34} {'model':<16} "
                  f"{'in':>8} {'out':>7} {'total':>8}  est"]
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

    def save_text(self, path: Any, pricing: Optional[Dict[str, Any]] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.render_text(pricing))


# Process-global tracker shared by every stage.
USAGE = TokenUsageTracker()

# Optional prompt audit log path (per-item logs are set from the streaming builder)
PROMPT_LOG_PATH: Optional[Path] = None


def set_prompt_log_path(path: Any) -> None:
    """Set the JSONL path where prompts (LLM inputs) are appended for auditing.
    The caller should set this to <out_dir>/logs/prompts.jsonl so each item
    keeps its own prompt audit file.
    """
    global PROMPT_LOG_PATH
    PROMPT_LOG_PATH = Path(path)
    PROMPT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log_prompt(record: Dict[str, Any]) -> None:
    if PROMPT_LOG_PATH is None:
        return
    try:
        with open(PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must not raise for normal pipeline runs
        return


def _record_usage(resp: Any, *, model: str, prompt: str, label: Optional[str]) -> None:
    """Record token usage from a chat-completions response into USAGE."""
    counts = _coerce_usage(getattr(resp, "usage", None))
    if counts["prompt_tokens"] is not None or counts["completion_tokens"] is not None:
        USAGE.record(
            model=model,
            prompt_tokens=counts["prompt_tokens"],
            completion_tokens=counts["completion_tokens"],
            total_tokens=counts["total_tokens"],
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
# Chat-model config + client
# ---------------------------------------------------------------------------

def load_config(
    path: str = "model_config.json",
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load the chat-model config. Supports two schemas:

    1) Flat:
         { "base_url": "...", "api_key": "...", "model": "gpt-4o" }

    2) Nested by model name (LiteLLM-style):
         { "gpt-5.5":  { "api_key": "...", "base_url": "...",
                         "max_tokens": 16000, ... },
           "deepseek-v4-flash-260425": { ... },
           "embedding": { ... }              <- reserved, never a chat default }

    For the nested form, `model_key` picks which entry to use; if omitted,
    the first NON-RESERVED key is used and its name is the model name.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Create it with base_url / api_key / model "
            f"(flat) or nested by model name."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    is_flat = any(k in raw for k in ("api_key", "base_url"))

    if is_flat:
        cfg = dict(raw)
    else:
        if model_key and model_key in raw:
            chosen = model_key
        elif model_key:
            available = ", ".join(repr(k) for k in raw.keys() if not _is_reserved_key(k))
            raise KeyError(f"Model key {model_key!r} not found in {path}. Available: {available}")
        else:
            chosen = next((k for k in raw.keys() if not _is_reserved_key(k)), None)
            if chosen is None:
                raise ValueError(f"{path} contains no chat-model entries (only reserved keys)")
        inner = raw[chosen]
        if not isinstance(inner, dict):
            raise ValueError(f"Nested entry {chosen!r} must be a JSON object")
        cfg = dict(inner)
        cfg.setdefault("model", chosen)

    for required in ("base_url", "api_key"):
        v = cfg.get(required)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"Config field {required!r} must be a non-empty string. "
                f"Got type={type(v).__name__}, value={v!r}"
            )

    return cfg


def make_client(cfg: Dict[str, Any]) -> OpenAI:
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    backoff: float = 2.0,
    usage_label: Optional[str] = None,
) -> str:
    """Call chat completions and return the response text. Retries on errors.

    Note: the pipeline always calls this with temperature=0 — deterministic
    JSON output is what makes extraction reliable, regardless of what the
    config file says.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    last_err: Optional[Exception] = None
    # Audit the prompt once (do not duplicate for retries). This is intentionally
    # done before making the network call so we retain the exact input text
    # that was sent to the LLM.
    try:
        _log_prompt({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "label": usage_label,
            "max_tokens": max_tokens,
            "prompt_len": len(prompt) if prompt is not None else 0,
            "prompt": prompt,
        })
    except Exception:
        # Never fail the call due to logging issues
        pass
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            _record_usage(resp, model=model, prompt=prompt, label=usage_label)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            print(
                f"    [retry {attempt + 1}/{retries}] {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise RuntimeError(f"All retries failed: {last_err}")


def parse_json_response(text: str) -> Dict[str, Any]:
    """Tolerantly parse a JSON object out of an LLM response."""
    if not text:
        return {"_parse_error": "empty response", "_raw": ""}
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        candidate = s[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            return {"_parse_error": str(e), "_raw": text}
    return {"_parse_error": "no JSON object found", "_raw": text}


# ---------------------------------------------------------------------------
# Embedding config + client
# ---------------------------------------------------------------------------

def load_embedding_config(
    path: str = "model_config.json",
    embedding_key: str = "embedding",
) -> Optional[Dict[str, Any]]:
    """
    Load an embedding entry from model_config.json. Returns None when the
    entry is absent (callers decide whether that's an error).

    Two providers are supported, selected by the entry's "provider" field:

    1) "openai" (default) — any OpenAI-compatible /v1/embeddings endpoint
       (vLLM / SGLang / TEI / OpenAI). Requires base_url, api_key, model:

        "embedding": {
          "api_key":  "EMPTY-or-real-key",
          "base_url": "http://localhost:8000/v1",
          "model":    "Qwen/Qwen3-Embedding-8B",
          "batch_size": 32,
          "dimensions": null
        }

    2) "local" — load the weights IN-PROCESS via sentence-transformers; no
       server needed. "model" is a HF repo id or a LOCAL DIRECTORY containing
       the downloaded weights. Requires only "model":

        "embedding_local": {
          "provider": "local",
          "model": "/data/models/Qwen3-Embedding-8B",
          "device": "auto",
          "dtype": "auto",
          "batch_size": 8,
          "max_length": 8192,
          "model_kwargs": {}
        }
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return None
    entry = raw.get(embedding_key)
    if not isinstance(entry, dict):
        return None
    cfg = dict(entry)
    provider = (cfg.get("provider") or "openai").strip().lower()
    cfg["provider"] = provider

    if provider == "local":
        v = cfg.get("model")
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"Embedding config field 'model' must be a non-empty string "
                f"(HF repo id or local weights directory) for provider='local' "
                f"(in entry {embedding_key!r} of {path})."
            )
        cfg.setdefault("batch_size", 8)
    elif provider == "openai":
        for required in ("base_url", "api_key", "model"):
            v = cfg.get(required)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"Embedding config field {required!r} must be a non-empty string "
                    f"(in entry {embedding_key!r} of {path})."
                )
        cfg.setdefault("batch_size", 32)
    else:
        raise ValueError(
            f"Unknown embedding provider {provider!r} in entry {embedding_key!r} "
            f"of {path}; expected 'openai' or 'local'."
        )
    return cfg


class EmbeddingClient:
    """
    OpenAI-compatible embeddings client with an in-memory cache and a JSONL
    audit log.  Every `embed()` call appends one record to the log containing
    the purpose tag, the FULL list of input texts, which of them were served
    from cache, the embedding dimension, and timing — so each step of the
    split / merge process is fully reconstructable.
    """

    def __init__(self, cfg: Dict[str, Any], log_path: Optional[Any] = None) -> None:
        self.client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        self.model: str = cfg["model"]
        self.batch_size: int = int(cfg.get("batch_size", 32) or 32)
        self.dimensions: Optional[int] = cfg.get("dimensions")
        self._cache: Dict[str, List[float]] = {}
        self._log_path: Optional[Path] = None
        if log_path:
            self.set_log_path(log_path)

    # -- logging ----------------------------------------------------------
    def set_log_path(self, path: Any) -> None:
        self._log_path = Path(path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, record: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- main entry ---------------------------------------------------------
    def embed(self, texts: List[str], purpose: str = "") -> List[List[float]]:
        """Embed a list of texts (order-preserving). Cached texts are reused."""
        t0 = time.time()
        results: List[Optional[List[float]]] = [None] * len(texts)
        cached_flags: List[bool] = [False] * len(texts)
        missing_idx: List[int] = []
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is not None:
                results[i] = v
                cached_flags[i] = True
            else:
                missing_idx.append(i)

        for batch_start in range(0, len(missing_idx), self.batch_size):
            batch_ids = missing_idx[batch_start:batch_start + self.batch_size]
            batch = [texts[i] for i in batch_ids]
            kwargs: Dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = int(self.dimensions)
            last_err: Optional[Exception] = None
            resp = None
            for attempt in range(3):
                try:
                    resp = self.client.embeddings.create(**kwargs)
                    break
                except Exception as e:
                    last_err = e
                    print(f"    [embed retry {attempt + 1}/3] {type(e).__name__}: {e}",
                          file=sys.stderr)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            if resp is None:
                raise RuntimeError(f"Embedding call failed after retries: {last_err}")

            data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
            vecs = [list(d.embedding) for d in data]
            if len(vecs) != len(batch):
                raise RuntimeError(
                    f"Embedding API returned {len(vecs)} vectors for {len(batch)} inputs")
            for i, t, v in zip(batch_ids, batch, vecs):
                self._cache[t] = v
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
        self._log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "model": self.model,
            "n_texts": len(texts),
            "n_cached": sum(cached_flags),
            "n_api": len(missing_idx),
            "dimension": dim,
            "elapsed_s": round(time.time() - t0, 3),
            "texts": [{"index": i, "cached": cached_flags[i], "text": t}
                      for i, t in enumerate(texts)],
        })
        return [r if r is not None else [] for r in results]


def cosine_similarity_matrix(vectors: List[List[float]]):
    """Pairwise cosine similarity (numpy array, shape n x n)."""
    import numpy as np
    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = arr / norms
    sim = unit @ unit.T
    return np.clip(sim, -1.0, 1.0)


class LocalEmbeddingClient:
    """
    In-process embedding client: loads the model weights directly via
    sentence-transformers (no HTTP server). Same interface as
    EmbeddingClient — embed(texts, purpose) / set_log_path / clear_cache /
    .model — so split.py and merge.py work with either, unchanged.

    Config (see load_embedding_config, provider="local"):
        model       : HF repo id ("Qwen/Qwen3-Embedding-8B") or a LOCAL
                      directory with downloaded weights.
        device      : "auto" (default) | "cuda" | "cuda:0" | "cpu" | ...
                      "auto"/None lets sentence-transformers pick.
        dtype       : "auto" (default) | "bfloat16" | "float16" | "float32".
                      Forwarded to from_pretrained as torch_dtype.
        batch_size  : encode batch size (default 8 — the 8B model is large).
        max_length  : optional truncation length (sets max_seq_length).
        model_kwargs: extra from_pretrained kwargs, e.g.
                      {"attn_implementation": "flash_attention_2"}.

    The model loads lazily on the first embed() call, so constructing the
    pipeline stays cheap when split/merge never run. Token usage is recorded
    into USAGE (estimated) and every call is logged to the same
    embedding_calls.jsonl schema with "provider": "local".
    """

    def __init__(self, cfg: Dict[str, Any], log_path: Optional[Any] = None) -> None:
        self.model: str = cfg["model"]
        self.batch_size: int = int(cfg.get("batch_size", 8) or 8)
        self.device: Optional[str] = cfg.get("device")
        self.dtype: Optional[str] = cfg.get("dtype")
        self.max_length: Optional[int] = cfg.get("max_length")
        self.extra_model_kwargs: Dict[str, Any] = dict(cfg.get("model_kwargs") or {})
        self._model = None                       # lazy-loaded
        self._cache: Dict[str, List[float]] = {}
        self._log_path: Optional[Path] = None
        if log_path:
            self.set_log_path(log_path)

    # -- logging (same schema as EmbeddingClient) ---------------------------
    def set_log_path(self, path: Any) -> None:
        self._log_path = Path(path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, record: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- model loading -------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "provider='local' embeddings need sentence-transformers. "
                "Install with:  pip install sentence-transformers torch"
            ) from e
        model_kwargs = dict(self.extra_model_kwargs)
        if self.dtype and self.dtype != "auto":
            model_kwargs.setdefault("torch_dtype", self.dtype)
        device = None if (not self.device or self.device == "auto") else self.device
        print(f"[info] loading local embedding model {self.model!r}"
              + (f" on {device}" if device else "") + " ...", file=sys.stderr)
        t0 = time.time()
        self._model = SentenceTransformer(
            self.model, device=device,
            model_kwargs=model_kwargs or None)
        if self.max_length:
            try:
                self._model.max_seq_length = int(self.max_length)
            except Exception:
                pass
        print(f"[info] local embedding model ready ({time.time() - t0:.1f}s)",
              file=sys.stderr)
        return self._model

    # -- main entry ----------------------------------------------------------
    def embed(self, texts: List[str], purpose: str = "") -> List[List[float]]:
        """Embed a list of texts (order-preserving). Cached texts are reused."""
        t0 = time.time()
        results: List[Optional[List[float]]] = [None] * len(texts)
        cached_flags: List[bool] = [False] * len(texts)
        missing_idx: List[int] = []
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is not None:
                results[i] = v
                cached_flags[i] = True
            else:
                missing_idx.append(i)

        if missing_idx:
            model = self._ensure_model()
            batch = [texts[i] for i in missing_idx]
            vecs = model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vec_lists = [list(map(float, v)) for v in vecs]
            if len(vec_lists) != len(batch):
                raise RuntimeError(
                    f"local embedding returned {len(vec_lists)} vectors "
                    f"for {len(batch)} inputs")
            for i, t, v in zip(missing_idx, batch, vec_lists):
                self._cache[t] = v
                results[i] = v
            pt = sum(_estimate_tokens(t) for t in batch)
            USAGE.record(
                model=self.model,
                prompt_tokens=pt,
                completion_tokens=0,
                total_tokens=pt,
                label=f"embedding:{purpose}" if purpose else "embedding",
                estimated=True,
            )

        dim = len(results[0]) if results and results[0] is not None else 0
        self._log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": "local",
            "purpose": purpose,
            "model": self.model,
            "n_texts": len(texts),
            "n_cached": sum(cached_flags),
            "n_api": len(missing_idx),
            "dimension": dim,
            "elapsed_s": round(time.time() - t0, 3),
            "texts": [{"index": i, "cached": cached_flags[i], "text": t}
                      for i, t in enumerate(texts)],
        })
        return [r if r is not None else [] for r in results]


def make_embedder(cfg: Dict[str, Any], log_path: Optional[Any] = None):
    """Build the right embedding client for a load_embedding_config() entry."""
    provider = (cfg.get("provider") or "openai").strip().lower()
    if provider == "openai":
        return EmbeddingClient(cfg, log_path=log_path)
    if provider == "local":
        return LocalEmbeddingClient(cfg, log_path=log_path)
    raise ValueError(f"Unknown embedding provider {provider!r}")
