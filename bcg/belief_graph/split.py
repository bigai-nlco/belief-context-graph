"""Sentence splitting and optional semantic clustering for belief extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bcg.belief_graph.constants import MIN_SENTENCE_FRAGMENT_LEN, SENTENCE_END_RE
from bcg.belief_graph.utils import trim_span


@dataclass(frozen=True, slots=True)
class Sentence:
    """Offset-exact sentence inside a segment."""

    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Cluster:
    """A semantic cluster of segment sentences."""

    cluster_id: int
    sentence_indices: list[int]
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(sentence.text for sentence in self.sentences)


def split_sentences(text: str) -> list[Sentence]:
    """Split text into sentences while preserving exact character offsets."""

    if not text or not text.strip():
        return []

    cuts = set()
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        if end >= len(text) or text[end].isspace():
            cuts.add(end)
    for match in re.finditer(r"\n", text):
        cuts.add(match.start() + 1)
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
    index = 0
    while index < len(spans):
        start, end = spans[index]
        while end - start < MIN_SENTENCE_FRAGMENT_LEN and index + 1 < len(spans):
            index += 1
            end = spans[index][1]
        if end - start < MIN_SENTENCE_FRAGMENT_LEN and merged:
            previous_start, _ = merged.pop()
            start = previous_start
        merged.append((start, end))
        index += 1

    output: list[Sentence] = []
    for sentence_index, (start, end) in enumerate(merged):
        start, end = trim_span(text, start, end)
        output.append(
            Sentence(
                index=sentence_index,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
    return output


def cluster_sentences(
    sentences: list[Sentence],
    embedder: Any,
    *,
    similarity_threshold: float = 0.6,
    buffer_size: int = 0,
    purpose: str = "split",
) -> tuple[list[Cluster], dict[str, Any]]:
    """Cluster sentences globally using average-linkage cosine similarity."""

    if not sentences:
        return [], {"n_sentences": 0, "n_clusters": 0}
    if len(sentences) == 1:
        cluster = Cluster(cluster_id=0, sentence_indices=[0], sentences=sentences[:])
        return [cluster], {
            "n_sentences": 1,
            "n_clusters": 1,
            "threshold": similarity_threshold,
            "buffer_size": buffer_size,
        }

    texts = _embedding_texts(sentences, buffer_size)
    vectors = embedder.embed(texts, purpose=purpose)
    sim = cosine_similarity_matrix(vectors)
    groups = _agglomerative_average_linkage(sim, similarity_threshold)
    groups.sort(key=lambda group: group[0])

    clusters = [
        Cluster(
            cluster_id=index,
            sentence_indices=list(group),
            sentences=[sentences[i] for i in group],
        )
        for index, group in enumerate(groups)
    ]
    stats = _similarity_stats(sim)
    info: dict[str, Any] = {
        "n_sentences": len(sentences),
        "n_clusters": len(clusters),
        "threshold": similarity_threshold,
        "buffer_size": buffer_size,
        "embedding_model": getattr(embedder, "model", None),
        "similarity_stats": stats,
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "sentence_indices": cluster.sentence_indices,
                "preview": (
                    cluster.sentences[0].text[:80] + "..."
                    if cluster.sentences and len(cluster.sentences[0].text) > 80
                    else (cluster.sentences[0].text if cluster.sentences else "")
                ),
            }
            for cluster in clusters
        ],
    }
    return clusters, info


def cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Return a cosine similarity matrix without requiring numeric packages."""

    matrix: list[list[float]] = []
    for left in vectors:
        row: list[float] = []
        for right in vectors:
            row.append(_cosine(left, right))
        matrix.append(row)
    return matrix


def _embedding_texts(sentences: list[Sentence], buffer_size: int) -> list[str]:
    if buffer_size <= 0:
        return [sentence.text for sentence in sentences]
    texts: list[str] = []
    for index in range(len(sentences)):
        start = max(0, index - buffer_size)
        end = min(len(sentences), index + buffer_size + 1)
        texts.append(" ".join(sentence.text for sentence in sentences[start:end]))
    return texts


def _agglomerative_average_linkage(
    sim: list[list[float]], threshold: float
) -> list[list[int]]:
    n = len(sim)
    clusters: list[list[int]] = [[i] for i in range(n)]
    cross = [row[:] for row in sim]
    for i in range(n):
        cross[i][i] = 0.0
    sizes = [1.0] * n
    alive = [True] * n

    while True:
        best_value = -2.0
        best_a = -1
        best_b = -1
        for a in range(n):
            if not alive[a]:
                continue
            for b in range(a + 1, n):
                if not alive[b]:
                    continue
                average = cross[a][b] / (sizes[a] * sizes[b])
                if average > best_value:
                    best_value = average
                    best_a = a
                    best_b = b
        if best_a < 0 or best_value < threshold:
            break
        clusters[best_a].extend(clusters[best_b])
        clusters[best_b] = []
        alive[best_b] = False
        for c in range(n):
            if c in (best_a, best_b) or not alive[c]:
                continue
            cross[best_a][c] += cross[best_b][c]
            cross[c][best_a] = cross[best_a][c]
        sizes[best_a] += sizes[best_b]
        sizes[best_b] = 0.0

    return [sorted(cluster) for cluster in clusters if cluster]


def _similarity_stats(sim: list[list[float]]) -> dict[str, float | None]:
    values = [
        sim[row][column]
        for row in range(len(sim))
        for column in range(row + 1, len(sim))
    ]
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 4),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
    }


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(n))
    left_norm = sum(left[i] * left[i] for i in range(n)) ** 0.5
    right_norm = sum(right[i] * right[i] for i in range(n)) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
