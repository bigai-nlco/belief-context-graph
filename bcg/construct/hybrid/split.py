"""
split.py
========
Sentence segmentation plus SemanticSplitterNodeParser-style breakpoint chunking.

For each turn:
  1. split the content into exact-offset sentences;
  2. combine every sentence with a configurable neighbour buffer;
  3. embed the combined windows;
  4. compute cosine distance between adjacent windows;
  5. break where distance is above the configured percentile threshold;
  6. merge undersized adjacent groups when ``min_chunk_sentences`` requires it.

The result is an ordered list of contiguous semantic chunks. Each chunk is later
stored as one evidence record and summarised into one belief/decision node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .._shared.spans import trim_span


@dataclass
class Sentence:
    index: int
    text: str
    start: int
    end: int


@dataclass
class SemanticChunk:
    chunk_id: int
    sentence_indices: list[int]
    sentences: list[Sentence] = field(default_factory=list)
    start: int = 0
    end: int = 0
    text: str = ""


_SENT_END_RE = re.compile(r'[.!?。！？；;…]+[\'"”’\)\]）】」』]*')
_MIN_FRAGMENT_LEN = 4
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*(?:\s+[^<>]*?)?\s*/?>")


def _is_pure_tag(text: str, s: int, e: int) -> bool:
    return _TAG_RE.sub("", text[s:e]).strip() == ""


def _merge_tiny(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
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
    """Split text into exact-offset sentences, dropping pure XML/agent tags."""
    if not text or not text.strip():
        return []

    cuts = set()
    for match in _SENT_END_RE.finditer(text):
        end = match.end()
        if end >= len(text) or text[end].isspace():
            cuts.add(end)
    for match in re.finditer(r"\n", text):
        cuts.add(match.start() + 1)
    for match in _TAG_RE.finditer(text):
        cuts.add(match.start())
        cuts.add(match.end())
    cuts.add(len(text))

    spans: list[tuple[int, int]] = []
    previous = 0
    for cut in sorted(cuts):
        if cut <= previous:
            continue
        start, end = trim_span(text, previous, cut)
        if end > start:
            spans.append((start, end))
        previous = cut

    merged: list[tuple[int, int]] = []
    run: list[tuple[int, int]] = []
    for start, end in spans:
        if _is_pure_tag(text, start, end):
            if run:
                merged.extend(_merge_tiny(run))
                run = []
            continue
        run.append((start, end))
    if run:
        merged.extend(_merge_tiny(run))

    result: list[Sentence] = []
    for index, (start, end) in enumerate(merged):
        start, end = trim_span(text, start, end)
        result.append(
            Sentence(
                index=index,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
    return result


def _buffered_sentence_texts(
    sentences: list[Sentence],
    buffer_size: int,
) -> list[str]:
    buffer_size = max(0, int(buffer_size or 0))
    texts: list[str] = []
    for index in range(len(sentences)):
        lower = max(0, index - buffer_size)
        upper = min(len(sentences), index + buffer_size + 1)
        texts.append(" ".join(sentence.text for sentence in sentences[lower:upper]))
    return texts


def _groups_from_breakpoints(
    n_sentences: int, breakpoints: list[int]
) -> list[list[int]]:
    groups: list[list[int]] = []
    start = 0
    for breakpoint in sorted(set(breakpoints)):
        end = min(n_sentences, int(breakpoint) + 1)
        if end > start:
            groups.append(list(range(start, end)))
        start = end
    if start < n_sentences:
        groups.append(list(range(start, n_sentences)))
    return groups


def _merge_small_groups(
    groups: list[list[int]],
    adjacent_distances: list[float],
    min_chunk_sentences: int,
) -> list[list[int]]:
    """Merge short groups across the semantically weaker neighbouring boundary."""
    minimum = max(1, int(min_chunk_sentences or 1))
    groups = [list(group) for group in groups if group]
    while len(groups) > 1:
        short_index = next(
            (index for index, group in enumerate(groups) if len(group) < minimum),
            None,
        )
        if short_index is None:
            break

        group = groups[short_index]
        left_distance = None
        right_distance = None
        if short_index > 0:
            boundary = group[0] - 1
            if 0 <= boundary < len(adjacent_distances):
                left_distance = adjacent_distances[boundary]
        if short_index + 1 < len(groups):
            boundary = group[-1]
            if 0 <= boundary < len(adjacent_distances):
                right_distance = adjacent_distances[boundary]

        if left_distance is None:
            merge_left = False
        elif right_distance is None:
            merge_left = True
        else:
            # Smaller distance means the boundary is less semantically justified.
            merge_left = left_distance <= right_distance

        if merge_left:
            groups[short_index - 1].extend(group)
            del groups[short_index]
        else:
            group.extend(groups[short_index + 1])
            groups[short_index] = group
            del groups[short_index + 1]
    return groups


def semantic_breakpoint_chunks(
    sentences: list[Sentence],
    content: str,
    embedder,
    *,
    breakpoint_percentile_threshold: float = 95.0,
    buffer_size: int = 1,
    min_chunk_sentences: int = 1,
    purpose: str = "semantic_chunk",
) -> tuple[list[SemanticChunk], dict[str, Any]]:
    """Create ordered semantic chunks using adjacent-window cosine distances."""
    if not sentences:
        return [], {
            "algorithm": "semantic_breakpoint",
            "n_sentences": 0,
            "n_chunks": 0,
        }

    if len(sentences) == 1:
        sentence = sentences[0]
        chunk = SemanticChunk(
            chunk_id=0,
            sentence_indices=[sentence.index],
            sentences=[sentence],
            start=sentence.start,
            end=sentence.end,
            text=content[sentence.start : sentence.end],
        )
        return [chunk], {
            "algorithm": "semantic_breakpoint",
            "n_sentences": 1,
            "n_chunks": 1,
            "breakpoint_percentile_threshold": float(breakpoint_percentile_threshold),
            "distance_threshold": None,
            "breakpoints": [],
            "buffer_size": max(0, int(buffer_size or 0)),
            "min_chunk_sentences": max(1, int(min_chunk_sentences or 1)),
            "embedding_model": getattr(embedder, "model", None),
        }

    if embedder is None:
        raise RuntimeError("semantic breakpoint chunking requires an embedding client")

    import numpy as np

    from .llm import cosine_similarity_matrix

    window_texts = _buffered_sentence_texts(sentences, buffer_size)
    vectors = embedder.embed(window_texts, purpose=purpose)
    similarity = cosine_similarity_matrix(vectors)
    adjacent_distances = [
        float(1.0 - similarity[index, index + 1]) for index in range(len(sentences) - 1)
    ]

    percentile = min(100.0, max(0.0, float(breakpoint_percentile_threshold)))
    distance_threshold = float(np.percentile(adjacent_distances, percentile))
    raw_breakpoints = [
        index
        for index, distance in enumerate(adjacent_distances)
        if distance > distance_threshold
    ]
    groups = _groups_from_breakpoints(len(sentences), raw_breakpoints)
    groups = _merge_small_groups(groups, adjacent_distances, min_chunk_sentences)

    chunks: list[SemanticChunk] = []
    for chunk_id, indices in enumerate(groups):
        chunk_sentences = [sentences[index] for index in indices]
        start = chunk_sentences[0].start
        end = chunk_sentences[-1].end
        chunks.append(
            SemanticChunk(
                chunk_id=chunk_id,
                sentence_indices=list(indices),
                sentences=chunk_sentences,
                start=start,
                end=end,
                text=content[start:end],
            )
        )

    info: dict[str, Any] = {
        "algorithm": "semantic_breakpoint",
        "n_sentences": len(sentences),
        "n_chunks": len(chunks),
        "breakpoint_percentile_threshold": percentile,
        "distance_threshold": round(distance_threshold, 6),
        "adjacent_distances": [round(value, 6) for value in adjacent_distances],
        "breakpoints": raw_breakpoints,
        "buffer_size": max(0, int(buffer_size or 0)),
        "min_chunk_sentences": max(1, int(min_chunk_sentences or 1)),
        "embedding_model": getattr(embedder, "model", None),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "sentence_indices": chunk.sentence_indices,
                "start": chunk.start,
                "end": chunk.end,
                "preview": (
                    chunk.text[:120] + "…" if len(chunk.text) > 120 else chunk.text
                ),
            }
            for chunk in chunks
        ],
    }
    return chunks, info


def single_fallback_chunk(
    sentences: list[Sentence],
    content: str,
    *,
    reason: str,
) -> tuple[list[SemanticChunk], dict[str, Any]]:
    """Deterministic fallback used when the embedding service is unavailable."""
    if sentences:
        start = sentences[0].start
        end = sentences[-1].end
        indices = [sentence.index for sentence in sentences]
        text = content[start:end]
    else:
        start, end = trim_span(content, 0, len(content))
        indices = []
        text = content[start:end]
    chunks = (
        [
            SemanticChunk(
                chunk_id=0,
                sentence_indices=indices,
                sentences=list(sentences),
                start=start,
                end=end,
                text=text,
            )
        ]
        if text
        else []
    )
    return chunks, {
        "algorithm": "semantic_breakpoint",
        "fallback": "single_chunk",
        "fallback_reason": reason,
        "n_sentences": len(sentences),
        "n_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": 0,
                "sentence_indices": indices,
                "start": start,
                "end": end,
                "preview": text[:120] + ("…" if len(text) > 120 else ""),
            }
        ]
        if chunks
        else [],
    }


# ---------------------------------------------------------------------------
# Tool-call isolation
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def find_tool_call_spans(content: str) -> list[tuple[int, int]]:
    """Absolute [start, end) spans of every <tool_call>...</tool_call> block."""
    return [(m.start(), m.end()) for m in TOOL_CALL_RE.finditer(content or "")]


def _chunk_text_region(
    content: str,
    region_start: int,
    region_end: int,
    embedder,
    *,
    enabled: bool,
    breakpoint_percentile_threshold: float,
    buffer_size: int,
    min_chunk_sentences: int,
    purpose: str,
    sent_base: int,
) -> tuple[list[SemanticChunk], int]:
    """Chunk one non-tool-call text region; offsets stay absolute in ``content``.

    Sentence ``index`` values are shifted by ``sent_base`` so they stay globally
    unique across regions. Returns (chunks, next_sent_base).
    """
    sub = content[region_start:region_end]
    sentences = split_sentences(sub)
    for s in sentences:
        s.start += region_start
        s.end += region_start
        s.index += sent_base
    if not sentences:
        return [], sent_base
    next_base = sent_base + len(sentences)

    if not enabled:
        chunks, _ = single_fallback_chunk(
            sentences, content, reason="chunking disabled"
        )
        return chunks, next_base
    if len(sentences) == 1:
        chunks, _ = single_fallback_chunk(sentences, content, reason="single sentence")
        return chunks, next_base
    try:
        chunks, _ = semantic_breakpoint_chunks(
            sentences,
            content,
            embedder,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size,
            min_chunk_sentences=min_chunk_sentences,
            purpose=purpose,
        )
    except Exception as exc:
        chunks, _ = single_fallback_chunk(sentences, content, reason=str(exc))
    return chunks, next_base


def semantic_chunks_isolating_tool_calls(
    content: str,
    embedder,
    *,
    enabled: bool = True,
    breakpoint_percentile_threshold: float = 95.0,
    buffer_size: int = 1,
    min_chunk_sentences: int = 1,
    purpose: str = "semantic_chunk",
) -> tuple[list[SemanticChunk], dict[str, Any]]:
    """Chunk a turn while forcing every <tool_call>...</tool_call> block to be its
    own standalone chunk. Text between/around tool calls is chunked normally.

    All chunk offsets remain absolute in ``content`` so evidence spans stay valid.
    """
    spans = find_tool_call_spans(content)
    if not spans:
        # No tool call: behave exactly like the normal path.
        sentences = split_sentences(content)
        if not enabled:
            return single_fallback_chunk(sentences, content, reason="chunking disabled")
        if len(sentences) < 2:
            return single_fallback_chunk(
                sentences, content, reason="fewer than 2 sentences"
            )
        try:
            return semantic_breakpoint_chunks(
                sentences,
                content,
                embedder,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                buffer_size=buffer_size,
                min_chunk_sentences=min_chunk_sentences,
                purpose=purpose,
            )
        except Exception as exc:
            return single_fallback_chunk(sentences, content, reason=str(exc))

    # Build ordered regions covering the whole content.
    regions: list[tuple[int, int, str]] = []
    cursor = 0
    for s, e in spans:
        if s > cursor:
            regions.append((cursor, s, "text"))
        regions.append((s, e, "tool_call"))
        cursor = e
    if cursor < len(content):
        regions.append((cursor, len(content), "text"))

    all_chunks: list[SemanticChunk] = []
    sent_base = 0
    n_tool_call_chunks = 0
    for rs, re_, kind in regions:
        if kind == "tool_call":
            ts, te = trim_span(content, rs, re_)
            if te <= ts:
                continue
            all_chunks.append(
                SemanticChunk(
                    chunk_id=0,  # reassigned below
                    sentence_indices=[],  # a tool call is one opaque unit
                    sentences=[],
                    start=ts,
                    end=te,
                    text=content[ts:te],
                )
            )
            n_tool_call_chunks += 1
        else:
            region_chunks, sent_base = _chunk_text_region(
                content,
                rs,
                re_,
                embedder,
                enabled=enabled,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                buffer_size=buffer_size,
                min_chunk_sentences=min_chunk_sentences,
                purpose=purpose,
                sent_base=sent_base,
            )
            all_chunks.extend(region_chunks)

    all_chunks.sort(key=lambda c: c.start)
    for i, c in enumerate(all_chunks):
        c.chunk_id = i

    info = {
        "algorithm": "semantic_breakpoint+tool_call_isolation",
        "n_chunks": len(all_chunks),
        "n_tool_call_chunks": n_tool_call_chunks,
        "tool_call_spans": [{"start": s, "end": e} for s, e in spans],
        "breakpoint_percentile_threshold": float(breakpoint_percentile_threshold),
        "buffer_size": max(0, int(buffer_size or 0)),
        "min_chunk_sentences": max(1, int(min_chunk_sentences or 1)),
        "embedding_model": getattr(embedder, "model", None),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "kind": "tool_call" if not c.sentence_indices else "text",
                "sentence_indices": c.sentence_indices,
                "start": c.start,
                "end": c.end,
                "preview": c.text[:120] + ("…" if len(c.text) > 120 else ""),
            }
            for c in all_chunks
        ],
    }
    return all_chunks, info
