"""
split.py
========
Sentence segmentation for "sentence" evidence mode.

split_sentences(content) returns exact, offset-tracked sentence spans
(content[s.start:s.end] == s.text); each sentence becomes an indexed item in
the extraction prompt, and the model's supporting_sentence_indices narrows
each belief's evidence to specific sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .._shared.spans import trim_span


@dataclass
class Sentence:
    index: int
    text: str       # EXACT slice: content[start:end]
    start: int      # offset within the segment content
    end: int        # exclusive


# ---------------------------------------------------------------------------
# Sentence splitting (offset-exact, EN + ZH punctuation, newline-aware)
# ---------------------------------------------------------------------------

# Sentence-final punctuation possibly followed by closing quotes/brackets.
_SENT_END_RE = re.compile(r'[.!?。！？；;…]+[\'"”’\)\]）】」』]*')
_MIN_FRAGMENT_LEN = 4   # fragments shorter than this merge into a neighbour

# Agent / XML-ish tags (e.g. <think>, </think>, <tool_call>, <tool_response>,
# <answer>, or tags with attributes like <tool_call name="x">). These are NOT
# part of any sentence; they act as HARD sentence boundaries and pure-tag spans
# are dropped, so evidence sentences never carry a leading "<think>" or a
# "<tool_response>…</tool_response>" wrapper. A "<" not introducing a tag (e.g.
# "a < b") is left untouched.
_TAG_RE = re.compile(r'</?[A-Za-z][A-Za-z0-9_]*(?:\s+[^<>]*?)?\s*/?>')


def _is_pure_tag(text: str, s: int, e: int) -> bool:
    """True when text[s:e] is nothing but tags and whitespace."""
    return _TAG_RE.sub("", text[s:e]).strip() == ""


def _merge_tiny(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge tiny fragments forward within a contiguous run (offset-exact)."""
    merged: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        s, e = spans[i]
        while (e - s) < _MIN_FRAGMENT_LEN and i + 1 < len(spans):
            i += 1
            e = spans[i][1]
        if (e - s) < _MIN_FRAGMENT_LEN and merged:
            ps, _ = merged.pop()
            s = ps
        merged.append((s, e))
        i += 1
    return merged


def split_sentences(text: str) -> list[Sentence]:
    """
    Split `text` into sentences with EXACT character offsets:
        text[s.start:s.end] == s.text  for every returned Sentence.

    Cut points: (a) after sentence-final punctuation when followed by whitespace
    or end-of-text; (b) at every newline; (c) at every agent/XML tag boundary.
    Pure-tag spans (e.g. "<think>", "</tool_response>") are DROPPED so they never
    contaminate evidence, and tiny fragments are merged only WITHIN a run of
    consecutive non-tag spans (never across a dropped tag), keeping every
    returned span an exact slice of the original content.
    """
    if not text or not text.strip():
        return []

    cuts = set()
    for m in _SENT_END_RE.finditer(text):
        e = m.end()
        if e >= len(text) or text[e].isspace():
            cuts.add(e)
    for m in re.finditer(r"\n", text):
        cuts.add(m.start() + 1)
    for m in _TAG_RE.finditer(text):       # isolate every tag
        cuts.add(m.start())
        cuts.add(m.end())
    cuts.add(len(text))

    spans: list[tuple[int, int]] = []
    prev = 0
    for c in sorted(cuts):
        if c <= prev:
            continue
        s, e = trim_span(text, prev, c)
        if e > s:
            spans.append((s, e))
        prev = c

    # Split into runs separated by pure-tag spans (which are dropped); run the
    # tiny-fragment merge inside each run so tags are never re-absorbed.
    merged: list[tuple[int, int]] = []
    run: list[tuple[int, int]] = []
    for (s, e) in spans:
        if _is_pure_tag(text, s, e):
            if run:
                merged.extend(_merge_tiny(run))
                run = []
            continue
        run.append((s, e))
    if run:
        merged.extend(_merge_tiny(run))

    out: list[Sentence] = []
    for idx, (s, e) in enumerate(merged):
        s, e = trim_span(text, s, e)
        out.append(Sentence(index=idx, text=text[s:e], start=s, end=e))
    return out
