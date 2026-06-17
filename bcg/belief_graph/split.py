"""
split.py
========
Semantic split module: sentence segmentation + GLOBAL clustering.

Unlike llama_index's SemanticSplitterNodeParser — which only looks at the
similarity between ADJACENT sentences and inserts breakpoints (so a topic
revisited later in the text ends up in a different chunk) — this module
clusters sentences GLOBALLY: every sentence is embedded, and agglomerative
average-linkage clustering over the full cosine-similarity matrix groups
same-topic sentences regardless of where they appear in the content.

Pipeline per segment (when --use-split is on and the segment has at least
`min_sentences` sentences):

    1. split_sentences(content)    — exact, offset-tracked sentence spans
                                     (content[s.start:s.end] == s.text);
    2. embed each sentence         — via the shared EmbeddingClient (every
                                     call is logged to embedding_calls.jsonl);
    3. cluster_sentences(...)      — global agglomerative clustering with a
                                     cosine-similarity threshold (clusters
                                     keep merging while the best average
                                     inter-cluster similarity >= threshold);
    4. each cluster goes to the extraction LLM as an indexed sentence list;
       the cluster's sentences become the belief's EVIDENCE (the model may
       narrow this to specific indices via supporting_sentence_indices).

numpy is imported lazily so the base pipeline still runs without it when the
split module is unused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Sentence:
    index: int
    text: str       # EXACT slice: content[start:end]
    start: int      # offset within the segment content
    end: int        # exclusive


@dataclass
class Cluster:
    cluster_id: int
    sentence_indices: List[int]            # ascending
    sentences: List[Sentence] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.sentences)


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


def _trim_span(text: str, s: int, e: int) -> Tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _is_pure_tag(text: str, s: int, e: int) -> bool:
    """True when text[s:e] is nothing but tags and whitespace."""
    return _TAG_RE.sub("", text[s:e]).strip() == ""


def _merge_tiny(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge tiny fragments forward within a contiguous run (offset-exact)."""
    merged: List[Tuple[int, int]] = []
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


def split_sentences(text: str) -> List[Sentence]:
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

    spans: List[Tuple[int, int]] = []
    prev = 0
    for c in sorted(cuts):
        if c <= prev:
            continue
        s, e = _trim_span(text, prev, c)
        if e > s:
            spans.append((s, e))
        prev = c

    # Split into runs separated by pure-tag spans (which are dropped); run the
    # tiny-fragment merge inside each run so tags are never re-absorbed.
    merged: List[Tuple[int, int]] = []
    run: List[Tuple[int, int]] = []
    for (s, e) in spans:
        if _is_pure_tag(text, s, e):
            if run:
                merged.extend(_merge_tiny(run))
                run = []
            continue
        run.append((s, e))
    if run:
        merged.extend(_merge_tiny(run))

    out: List[Sentence] = []
    for idx, (s, e) in enumerate(merged):
        s, e = _trim_span(text, s, e)
        out.append(Sentence(index=idx, text=text[s:e], start=s, end=e))
    return out


# ---------------------------------------------------------------------------
# Global agglomerative clustering (average linkage, cosine similarity)
# ---------------------------------------------------------------------------

def _agglomerative_average_linkage(sim, threshold: float) -> List[List[int]]:
    """
    Plain-numpy agglomerative clustering: start with singletons, repeatedly
    merge the pair of clusters with the highest AVERAGE pairwise cosine
    similarity, stop when that best average drops below `threshold`.
    O(n^3) worst case — fine for the per-segment sentence counts this
    pipeline sees (tens to low hundreds).
    """
    import numpy as np

    n = sim.shape[0]
    clusters: List[List[int]] = [[i] for i in range(n)]
    # cross[a][b] = sum of pairwise sims between cluster a and cluster b
    cross = sim.copy().astype(np.float64)
    np.fill_diagonal(cross, 0.0)
    sizes = np.ones(n, dtype=np.float64)
    alive = [True] * n

    while True:
        best_val, best_a, best_b = -2.0, -1, -1
        for a in range(n):
            if not alive[a]:
                continue
            for b in range(a + 1, n):
                if not alive[b]:
                    continue
                avg = cross[a, b] / (sizes[a] * sizes[b])
                if avg > best_val:
                    best_val, best_a, best_b = avg, a, b
        if best_a < 0 or best_val < threshold:
            break
        a, b = best_a, best_b
        clusters[a].extend(clusters[b])
        clusters[b] = []
        alive[b] = False
        # update cross sums: new cluster a absorbs b
        for c in range(n):
            if c == a or c == b or not alive[c]:
                continue
            cross[a, c] = cross[a, c] + cross[b, c]
            cross[c, a] = cross[a, c]
        sizes[a] += sizes[b]
        sizes[b] = 0.0

    return [sorted(c) for c in clusters if c]


def cluster_sentences(
    sentences: List[Sentence],
    embedder,                               # llm.EmbeddingClient or compatible
    *,
    similarity_threshold: float = 0.6,
    buffer_size: int = 0,
    purpose: str = "split",
) -> Tuple[List[Cluster], Dict[str, Any]]:
    """
    Globally cluster `sentences`. Returns (clusters, info).

    similarity_threshold — cosine-similarity floor for merging clusters
        (average linkage). Higher → more, tighter clusters; lower → fewer,
        broader clusters. Typical useful range for Qwen3-Embedding: 0.5-0.75.
    buffer_size — optional llama_index-style neighbour window: each sentence
        is EMBEDDED together with `buffer_size` neighbours on each side
        (evidence offsets always stay per-sentence). Default 0 (pure global
        clustering; a buffer re-introduces positional smoothing).
    """
    from .llm import cosine_similarity_matrix

    if not sentences:
        return [], {"n_sentences": 0, "n_clusters": 0}
    if len(sentences) == 1:
        cl = Cluster(cluster_id=0, sentence_indices=[0], sentences=list(sentences))
        return [cl], {"n_sentences": 1, "n_clusters": 1, "threshold": similarity_threshold}

    if buffer_size > 0:
        texts = []
        for i in range(len(sentences)):
            lo = max(0, i - buffer_size)
            hi = min(len(sentences), i + buffer_size + 1)
            texts.append(" ".join(s.text for s in sentences[lo:hi]))
    else:
        texts = [s.text for s in sentences]

    vectors = embedder.embed(texts, purpose=purpose)
    sim = cosine_similarity_matrix(vectors)
    groups = _agglomerative_average_linkage(sim, similarity_threshold)
    groups.sort(key=lambda g: g[0])

    clusters: List[Cluster] = []
    for cid, idxs in enumerate(groups):
        clusters.append(Cluster(
            cluster_id=cid,
            sentence_indices=list(idxs),
            sentences=[sentences[i] for i in idxs],
        ))

    import numpy as np
    tri = sim[np.triu_indices(len(sentences), k=1)]
    info: Dict[str, Any] = {
        "n_sentences": len(sentences),
        "n_clusters": len(clusters),
        "threshold": similarity_threshold,
        "buffer_size": buffer_size,
        "embedding_model": getattr(embedder, "model", None),
        "similarity_stats": {
            "min": round(float(tri.min()), 4) if tri.size else None,
            "mean": round(float(tri.mean()), 4) if tri.size else None,
            "max": round(float(tri.max()), 4) if tri.size else None,
        },
        "clusters": [{
            "cluster_id": c.cluster_id,
            "sentence_indices": c.sentence_indices,
            "preview": (c.sentences[0].text[:80] + "…") if c.sentences and len(c.sentences[0].text) > 80
                       else (c.sentences[0].text if c.sentences else ""),
        } for c in clusters],
    }
    return clusters, info