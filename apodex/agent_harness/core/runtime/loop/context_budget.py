"""Token estimation for context-window accounting.

Uses ``tiktoken cl100k_base`` when available and falls back to a CJK-aware
heuristic. ``estimate_tokens`` is the only export consumed in OSS — the
historical budget-allocation + truncation helpers (``MODEL_CONTEXT_BUDGETS``,
``RESEARCH_PROMPT_ALLOCATION``, ``allocate_prompt_budget``,
``truncate_to_budget``, ``compact_evidence`` etc.) were re-exported from
``core/runtime/loop/__init__.py`` but had no real callers.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


_tokenizer = None
_tokenizer_loaded = False

# Broad CJK regex: Unified Ideographs, Ext-A, radicals, strokes,
# Hiragana, Katakana, CJK compatibility, fullwidth forms.
_CJK_RE = re.compile(
    r"[⺀-⻿　-〿぀-ヿ㐀-䶿"
    r"一-鿿豈-﫿＀-￯]"
)


def _get_tokenizer():
    """Lazily load the tiktoken cl100k_base encoder."""
    global _tokenizer, _tokenizer_loaded  # noqa: PLW0603
    if not _tokenizer_loaded:
        _tokenizer_loaded = True
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            logger.debug(
                "tiktoken unavailable, using heuristic token estimation"
            )
    return _tokenizer


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a plain text string.

    Uses tiktoken cl100k_base when available; falls back to a CJK-aware
    heuristic (CJK character ≈ 1 token, Latin text ≈ chars / 4).
    """
    if not text:
        return 0

    enc = _get_tokenizer()
    if enc is not None:
        return len(enc.encode(text))

    cjk_count = len(_CJK_RE.findall(text))
    other_count = len(text) - cjk_count
    return cjk_count + (other_count // 4)
