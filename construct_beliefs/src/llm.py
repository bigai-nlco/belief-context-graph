"""
llm.py
======
Thin OpenAI-compatible client wrapper used by every stage.

Config loading and chat completion calls live here so the rest of the package
does not have to know about openai SDK details.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: missing openai. Install with: pip install openai", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Token-usage tracking
# ---------------------------------------------------------------------------
# Every successful call_model() appends one record to the module-level USAGE
# tracker.  The pipeline tags each call with the current stage/segment via
# USAGE.set_label(...), and after a run writes the accumulated log to
# outputs/token_usage.{json,txt}.  This makes it easy to see how many LLM calls
# one input needed and the input/output token counts per call, for cost
# estimation.  State is process-global and assumes single-threaded use.


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

    # Last resort: some SDKs expose only model_dump()/dict form.
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
                f"   {lbl:<30.30} calls={agg['n_calls']:>3}  "
                f"in={agg['input_tokens']:>8,}  "
                f"out={agg['output_tokens']:>7,}  "
                f"total={agg['total_tokens']:>8,}"
            )
        lines += [sub, " per-call detail:",
                  f"   {'#':>3}  {'label':<30} {'model':<16} "
                  f"{'in':>8} {'out':>7} {'total':>8}  est"]
        for r in self.records:
            est = "*" if r.get("estimated") else ""
            lines.append(
                f"   {r['index']:>3}  {str(r.get('label') or ''):<30.30} "
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
    # Fallback: API gave no usage; estimate from text length.
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


def load_config(
    path: str = "model_config.json",
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load model config. Supports two schemas:

    1) Flat:
         { "base_url": "...", "api_key": "...", "model": "gpt-4o" }

    2) Nested by model name (LiteLLM-style):
         { "gpt-5.5":  { "api_key": "...", "base_url": "...",
                         "max_tokens": 16000, "temperature": 1, ... },
           "claude-3": { ... } }

    For the nested form, `model_key` picks which entry to use; if omitted, the
    first key is used and its name is treated as the model name.
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

    # Flat layouts have api_key / base_url at the top level.
    is_flat = any(k in raw for k in ("api_key", "base_url"))

    if is_flat:
        cfg = dict(raw)
    else:
        # Nested by model name. Pick the requested key, or the first one.
        if model_key and model_key in raw:
            chosen = model_key
        elif model_key:
            available = ", ".join(repr(k) for k in raw.keys())
            raise KeyError(f"Model key {model_key!r} not found in {path}. Available: {available}")
        else:
            chosen = next(iter(raw.keys()))
        inner = raw[chosen]
        if not isinstance(inner, dict):
            raise ValueError(f"Nested entry {chosen!r} must be a JSON object")
        cfg = dict(inner)
        cfg.setdefault("model", chosen)

    # Validate required string fields.
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

    On a successful call the prompt/completion token counts are recorded into
    the module-level ``USAGE`` tracker so the cost of processing one input can
    be estimated.  ``usage_label`` tags the record; if omitted, the tracker's
    current label (set by the pipeline before each stage) is used.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    last_err: Optional[Exception] = None
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
