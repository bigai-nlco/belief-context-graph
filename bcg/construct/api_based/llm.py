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
    module (duplicate-belief candidate generation). Supports OpenAI-compatible
    embeddings and local sentence-transformers.
    Every embedding call is appended to a JSONL log file (full input texts +
    cache hits), so the whole embedding-assisted process is auditable, and
    its token usage is recorded into USAGE as well.

`model_config.json` supports a reserved top-level key ``"embedding"`` that
holds the embedding endpoint.  That key is never picked as the default chat
model.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
import sys
import threading
import time
from datetime import UTC, datetime
from typing import Any

from bcg.config.runtime import load_construct_config
from bcg.core.env import resolve_config_api_key

from .._shared.llm import (  # noqa: F401 - compat re-exports for legacy imports
    _EMBEDDING_LOG_LOCK,
    _PROMPT_LOG_LOCK,
    USAGE,
    EmbeddingClient,
    TokenUsageTracker,
    _coerce_usage,
    _estimate_tokens,
    _get_usage,
    _log_prompt,
    _record_usage,
    _UsageProxy,
    bind_embedding_log_path,
    bind_prompt_log_path,
    bind_usage_tracker,
    cosine_similarity_matrix,
    current_embedding_log_path,
    current_prompt_log_path,
    current_usage_tracker,
    set_prompt_log_path,
    unbind_embedding_log_path,
    unbind_prompt_log_path,
    unbind_usage_tracker,
)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: missing openai. Install with: pip install openai", file=sys.stderr)
    raise


# Top-level config keys that are NOT chat-model entries.
RESERVED_CONFIG_KEYS = {"embedding", "belief_graph"}


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


def _resolve_config_api_key(
    cfg: dict[str, Any],
    *,
    default_env: str,
    config_path: str,
) -> None:
    """Resolve a runtime key from the root .env without storing it in JSON."""

    resolve_config_api_key(
        cfg,
        default_env=default_env,
        config_path=config_path,
    )


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_belief_graph_config(
    path: str = "model_config.json",
    model_key: str | None = None,
) -> dict[str, Any]:
    """Load optional shared belief-graph settings from YAML or legacy JSON."""
    raw, _ = load_construct_config(path, required=False)
    if raw is None:
        return {}

    top = raw.get("belief_graph")
    cfg: dict[str, Any] = dict(top) if isinstance(top, dict) else {}

    chosen = None
    if model_key and isinstance(raw.get(model_key), dict):
        chosen = model_key
    elif not model_key:
        chosen = next(
            (
                key
                for key in raw
                if not _is_reserved_key(key) and isinstance(raw.get(key), dict)
            ),
            None,
        )

    if chosen and isinstance(raw.get(chosen), dict):
        override = raw[chosen].get("belief_graph")
        if isinstance(override, dict):
            cfg = _deep_merge_dict(cfg, override)
    return cfg


def load_config(
    path: str = "model_config.json",
    model_key: str | None = None,
) -> dict[str, Any]:
    """
    Load the chat-model config. Supports two schemas:

    API keys are resolved from the project-root ``.env``. JSON stores only an
    ``api_key_env`` variable name, never the secret itself.

    1) Flat:
         { "base_url": "...", "api_key_env": "OPENAI_API_KEY",
           "model": "gpt-4o" }

    2) Nested by model name (LiteLLM-style):
         { "gpt-5.5":  { "api_key_env": "OPENAI_API_KEY", "base_url": "...",
                         "max_tokens": 16000, ... },
           "deepseek-v4-flash-260425": { ... },
           "embedding": { ... }              <- reserved, never a chat default }

    For the nested form, `model_key` picks which entry to use; if omitted,
    the first NON-RESERVED key is used and its name is the model name.
    """
    raw, display_path = load_construct_config(path, required=True)
    assert raw is not None

    is_flat = any(k in raw for k in ("api_key", "base_url"))

    if is_flat:
        cfg = dict(raw)
    else:
        if model_key and model_key in raw:
            chosen = model_key
        elif model_key:
            available = ", ".join(repr(k) for k in raw if not _is_reserved_key(k))
            raise KeyError(
                f"Model key {model_key!r} not found in {display_path}. Available: {available}"
            )
        else:
            chosen = next((k for k in raw if not _is_reserved_key(k)), None)
            if chosen is None:
                raise ValueError(
                    f"{path} contains no chat-model entries (only reserved keys)"
                )
        inner = raw[chosen]
        if not isinstance(inner, dict):
            raise ValueError(f"Nested entry {chosen!r} must be a JSON object")
        cfg = dict(inner)
        cfg.setdefault("model", chosen)

    for required in ("base_url",):
        v = cfg.get(required)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"Config field {required!r} must be a non-empty string. "
                f"Got type={type(v).__name__}, value={v!r}"
            )

    _resolve_config_api_key(
        cfg,
        default_env="OPENAI_API_KEY",
        config_path=display_path,
    )
    return cfg


def make_client(cfg: dict[str, Any]) -> OpenAI:
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])


def _normalize_temperature_for_model(
    model: str, temperature: float | None
) -> float | None:
    """Return a chat-completions temperature accepted by known strict models."""
    if temperature is None:
        return None
    model_name = str(model or "").lower()
    if model_name.startswith("gpt-5"):
        # Azure/OpenAI-compatible GPT-5 endpoints reject explicit 0.0 and only
        # accept the default value. The local config already uses 1; normalize
        # here because extraction/linking callers historically pass 0.0.
        return 1
    return temperature


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    retries: int = 3,
    backoff: float = 2.0,
    usage_label: str | None = None,
) -> str:
    """Call chat completions and return the response text. Retries on errors.

    Note: the pipeline always calls this with temperature=0 — deterministic
    JSON output is what makes extraction reliable, regardless of what the
    config file says.
    """
    temperature = _normalize_temperature_for_model(model, temperature)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "medium",
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    last_err: Exception | None = None
    # Audit the prompt once (do not duplicate for retries). This is intentionally
    # done before making the network call so we retain the exact input text
    # that was sent to the LLM.
    with contextlib.suppress(Exception):
        # Never fail the call due to logging issues
        _log_prompt(
            {
                "ts": datetime.now(UTC).isoformat(),
                "model": model,
                "label": usage_label,
                "max_tokens": max_tokens,
                "prompt_len": len(prompt) if prompt is not None else 0,
                "prompt": prompt,
            }
        )
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
                time.sleep(backoff**attempt)
    raise RuntimeError(f"All retries failed: {last_err}")


def parse_json_response(text: str) -> dict[str, Any]:
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
        candidate = s[start : end + 1]
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
) -> dict[str, Any] | None:
    """
    Load an embedding entry from model_config.json. Returns None when the
    entry is absent (callers decide whether that's an error).

    Two providers are supported, selected by the entry's "provider" field:

    1) "openai" (default) — any OpenAI-compatible /v1/embeddings endpoint
       (vLLM / SGLang / TEI / OpenAI). Requires base_url, api_key_env, model:

        "embedding": {
          "provider": "openai",
          "api_key_env": "EMBEDDING_API_KEY",
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
    raw, display_path = load_construct_config(path, required=False)
    if raw is None:
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
        for required in ("base_url", "model"):
            v = cfg.get(required)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"Embedding config field {required!r} must be a non-empty string "
                    f"(in entry {embedding_key!r} of {path})."
                )
        _resolve_config_api_key(
            cfg,
            default_env="EMBEDDING_API_KEY",
            config_path=display_path,
        )
        cfg.setdefault("batch_size", 32)
    else:
        raise ValueError(
            f"Unknown embedding provider {provider!r} in entry {embedding_key!r} "
            f"of {path}; expected 'openai' or 'local'."
        )
    return cfg


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

    def __init__(self, cfg: dict[str, Any], log_path: Any | None = None) -> None:
        self.model: str = cfg["model"]
        self.batch_size: int = int(cfg.get("batch_size", 8) or 8)
        self.device: str | None = cfg.get("device")
        self.dtype: str | None = cfg.get("dtype")
        self.max_length: int | None = cfg.get("max_length")
        self.extra_model_kwargs: dict[str, Any] = dict(cfg.get("model_kwargs") or {})
        self._model = None  # lazy-loaded
        self._model_lock = threading.Lock()  # guards lazy load under concurrency
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = threading.Lock()
        if log_path:
            self.set_log_path(log_path)

    # -- logging (same schema as EmbeddingClient) ---------------------------
    def set_log_path(self, path: Any) -> contextvars.Token:
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

    # -- model loading -------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            # Re-check inside the lock: another thread may have finished
            # loading the (potentially multi-GB) model while we were waiting.
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
            print(
                f"[info] loading local embedding model {self.model!r}"
                + (f" on {device}" if device else "")
                + " ...",
                file=sys.stderr,
            )
            t0 = time.time()
            model = SentenceTransformer(
                self.model, device=device, model_kwargs=model_kwargs or None
            )
            if self.max_length:
                with contextlib.suppress(Exception):
                    model.max_seq_length = int(self.max_length)
            print(
                f"[info] local embedding model ready ({time.time() - t0:.1f}s)",
                file=sys.stderr,
            )
            self._model = model
        return self._model

    # -- main entry ----------------------------------------------------------
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
                    f"for {len(batch)} inputs"
                )
            with self._cache_lock:
                for t, v in zip(batch, vec_lists, strict=False):
                    self._cache[t] = v
            for i, v in zip(missing_idx, vec_lists, strict=False):
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
        self._log(
            {
                "ts": datetime.now(UTC).isoformat(),
                "provider": "local",
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


def make_embedder(cfg: dict[str, Any], log_path: Any | None = None):
    """Build the right embedding client for a load_embedding_config() entry."""
    provider = (cfg.get("provider") or "openai").strip().lower()
    if provider == "openai":
        return EmbeddingClient(cfg, log_path=log_path)
    if provider == "local":
        return LocalEmbeddingClient(cfg, log_path=log_path)
    raise ValueError(f"Unknown embedding provider {provider!r}")
