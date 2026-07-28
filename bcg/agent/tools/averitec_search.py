"""AVeriTeC per-claim retrieval tool."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import threading
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput

from bcg.agent.tools.averitec_index_types import _Chunk, _ClaimIndex

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_DATA_DIR = Path("/data/user/baijun/datasets/AVeriTeC")
_CONFIG_PATH = Path(__file__).resolve().parent / "averitec_search_config.json"


def _load_config() -> dict[str, Any]:
    """Load tool defaults from the JSON config file."""
    if not _CONFIG_PATH.is_file():
        return {}
    try:
        with _CONFIG_PATH.open() as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _resolve_env_or_config(key: str, env_name: str, fallback: Any) -> Any:
    """Resolve a parameter with priority: env var > config > fallback."""
    env_val = os.environ.get(env_name)
    if env_val is not None:
        return type(fallback)(env_val) if not isinstance(fallback, (str, Path)) else env_val
    return fallback


class AVeriTeCSearchTool(Tool):
    """Search the AVeriTeC knowledge store for the current claim only.

    Parameters are resolved with priority: constructor arg > env var > config file > hardcoded default.
    """

    NAME = "averitec_search"
    # This tool produces new external evidence, so its results should be
    # remembered: archived (two-layer) and pushed to the belief graph.
    FEEDS_MEMORY = True
    DESCRIPTION = (
        "Search the AVeriTeC knowledge store for evidence relevant to the "
        "current claim. The search is automatically restricted to the current "
        "claim's provided knowledge source."
    )
    _CACHE: OrderedDict[tuple[str, str], _ClaimIndex] = OrderedDict()
    _CACHE_SIZE: int | None = None  # resolved once from config
    # Guards _CACHE's read/insert/evict sequence: forward() may run concurrently
    # across threads when a turn issues multiple parallel tool calls.
    _CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        name: str = NAME,
        description: str | None = None,
        dataset_dir: str | Path | None = None,
        max_results: int | None = None,
        max_chunk_chars: int | None = None,
        max_output_chars: int | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
    ) -> None:
        cfg = _load_config()

        # dataset_dir: arg > env > config > default
        self.dataset_dir = _resolve_dataset_dir(
            dataset_dir or cfg.get("dataset_dir")
        )
        self.knowledge_zip = (
            self.dataset_dir
            / "data_store"
            / "knowledge_store"
            / "dev_knowledge_store.zip"
        )
        self.knowledge_dir = (
            self.dataset_dir
            / "data_store"
            / "knowledge_store"
            / "dev_knowledge_store"
        )

        # numerical params: arg > env > config > hardcoded fallback
        self.max_results = (
            max_results
            if max_results is not None
            else int(os.environ.get("AVERITEC_MAX_RESULTS",
                cfg.get("max_results", 3)))
        )
        self.max_chunk_chars = (
            max_chunk_chars
            if max_chunk_chars is not None
            else int(os.environ.get("AVERITEC_MAX_CHUNK_CHARS",
                cfg.get("max_chunk_chars", 800)))
        )
        self.max_output_chars = (
            max_output_chars
            if max_output_chars is not None
            else int(os.environ.get("AVERITEC_MAX_OUTPUT_CHARS",
                cfg.get("max_output_chars", 4000)))
        )
        self.bm25_k1 = (
            bm25_k1
            if bm25_k1 is not None
            else float(os.environ.get("AVERITEC_BM25_K1",
                cfg.get("bm25_k1", 1.5)))
        )
        self.bm25_b = (
            bm25_b
            if bm25_b is not None
            else float(os.environ.get("AVERITEC_BM25_B",
                cfg.get("bm25_b", 0.75)))
        )

        # cache size: env > config > default
        if AVeriTeCSearchTool._CACHE_SIZE is None:
            AVeriTeCSearchTool._CACHE_SIZE = int(os.environ.get(
                "AVERITEC_SEARCH_CACHE_SIZE",
                cfg.get("cache_size", 16),
            ))

        # token pattern: config overrides default
        pattern = cfg.get("token_pattern", r"[A-Za-z0-9]+")
        self._token_re = re.compile(pattern)

        super().__init__(name=name, description=description or self.DESCRIPTION)
        self.claim_id: str | None = None
        self.claim_text: str = ""
        super().__init__(name=name, description=description or self.DESCRIPTION)

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for finding evidence about the current claim.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of evidence chunks to return (default: {self.max_results}).",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def usage_prompt(self, hyde: bool = False, detail: bool = True) -> str:
        """Self-describing usage text for this tool, injected dynamically.

        Each tool returns its own usage block so the harness can assemble the
        instructions only for the tools actually present (no hardcoded blob).

        ``detail`` controls verbosity. The brief variant (``detail=False``) is
        for the system prompt's tool listing: one-line purpose + the tool-call
        JSON shape + argument names, with NO function signature and NO good/bad
        examples. The detailed variant (``detail=True``, the default for all
        legacy callers) keeps the full guidance and goes into the user message.
        """
        mr = self.max_results
        if not detail:
            if hyde:
                return (
                    "- averitec_search: search evidence for the current claim. The "
                    'query MUST be "claim ||| hypo_passage_1 ||| hypo_passage_2 ||| ..." '
                    '(parts joined by " ||| "). Call as:\n'
                    "  <tool_call>\n"
                    '  {"name": "averitec_search", "arguments": {"query": "..."}}\n'
                    "  </tool_call>\n"
                    f"  Optional top_k (integer, default {mr})."
                )
            return (
                "- averitec_search: search evidence for the current claim. Call as:\n"
                "  <tool_call>\n"
                '  {"name": "averitec_search", "arguments": {"query": "..."}}\n'
                "  </tool_call>\n"
                "  The query is a complete natural-language question; optional top_k "
                f"(integer, default {mr})."
            )
        topk_note = (
            f"top_k (optional, default {mr}) is how many evidence chunks to return; "
            f"raise it to cast a wider net or lower it to focus. You usually do not "
            f"need to set it."
        )
        if hyde:
            return (
                f"- averitec_search(query: string, top_k?: integer): search "
                "evidence chunks for the current claim. The query MUST be: "
                'claim ||| hypo_passage_1 ||| hypo_passage_2 ||| ... (separated by " ||| "), '
                "where each hypo_passage is a hypothetical full-sentence passage that "
                "would answer the claim.\n"
                f"  {topk_note}\n"
                "  Example:\n"
                "  <tool_call>\n"
                '  {"name": "averitec_search", "arguments": {"query": "the claim text ||| '
                'hypothetical passage 1 ||| hypothetical passage 2"}}\n'
                "  </tool_call>"
            )
        return (
            f"- averitec_search(query: string, top_k?: integer): search evidence "
            "chunks for the current claim. The query MUST be a complete natural-language "
            "question or full sentence about the claim — NOT a list of keywords. "
            "Write what you want to find as a proper sentence so the evidence can be "
            "matched on meaning, not just keyword overlap.\n"
            "  Good: \"Did France cancel the visas of 183 Pakistani citizens after Imran "
            "Khan criticized Macron?\"\n"
            "  Bad:  \"France visas 183 Pakistani Imran Khan Macron\"\n"
            f"  {topk_note}\n"
            "  Example:\n"
            "  <tool_call>\n"
            '  {"name": "averitec_search", "arguments": {"query": "Did Sean Connery write '
            'a letter to Steve Jobs refusing an Apple commercial?"}}\n'
            "  </tool_call>"
        )

    def set_task(self, task: dict[str, Any]) -> None:
        extra = task.get("extra_info") if isinstance(task, dict) else None
        extra = extra if isinstance(extra, dict) else {}
        claim_id = extra.get("claim_id") or task.get("claim_id") if isinstance(task, dict) else None
        if claim_id is None:
            claim_id = _claim_id_from_question(task.get("question", "") if isinstance(task, dict) else "")
        self.claim_id = str(claim_id) if claim_id is not None else None
        # Capture the full claim text so downstream stages (e.g. the relevance
        # judge) can anchor on the original claim, not just the model's query.
        self.claim_text = str(extra.get("claim") or "") if isinstance(extra, dict) else ""

    @property
    def _has_knowledge_store(self) -> bool:
        return self.knowledge_dir.is_dir() or self.knowledge_zip.is_file()

    def forward(self, query: str, top_k: int | None = None) -> ToolOutput:
        if not self.claim_id:
            return ToolOutput(
                name=self.name,
                error="No current AVeriTeC claim_id is set for this tool.",
            )
        if not query or not str(query).strip():
            return ToolOutput(name=self.name, error="query must be a non-empty string")
        if not self._has_knowledge_store:
            return ToolOutput(
                name=self.name,
                error=f"AVeriTeC knowledge store not found: checked {self.knowledge_dir} and {self.knowledge_zip}",
            )

        try:
            index = self._get_claim_index(self.claim_id)
        except Exception as exc:
            return ToolOutput(name=self.name, error=f"Failed to load claim knowledge: {exc}")

        if not index.chunks:
            return ToolOutput(
                name=self.name,
                output=f"No knowledge-store text found for claim_id={self.claim_id}.",
                metadata={"claim_id": self.claim_id, "query": query, "num_results": 0},
            )

        results, summaries = self._search(index, str(query), top_k or self.max_results)
        output = self._format_results(results)
        # Per-evidence structure for downstream archiving (one entry per chunk).
        # hero4 may return judge summaries keyed by chunk text.
        evidences = [
            {
                "rank": i,
                "score": float(score),
                "text": chunk.text,
                "url": getattr(chunk, "url", ""),
                "summary": summaries.get(chunk.text, ""),
            }
            for i, (score, chunk) in enumerate(results, 1)
        ]
        metadata = {
            "claim_id": self.claim_id,
            "query": query,
            "num_results": len(results),
            "retriever_type": "bm25",
            "knowledge_store": str(self.knowledge_dir if self.knowledge_dir.is_dir() else self.knowledge_zip),
            "evidences": evidences,
        }
        return ToolOutput(name=self.name, output=output, metadata=metadata)

    def _disk_cache_path(self, claim_id: str) -> Path:
        return (
            self.dataset_dir / "data_store" / "index_cache"
            / f"chunk{self.max_chunk_chars}" / f"{claim_id}.pkl"
        )

    def _load_from_disk(self, claim_id: str) -> _ClaimIndex | None:
        pkl_path = self._disk_cache_path(claim_id)
        if not pkl_path.is_file():
            return None
        try:
            with pkl_path.open("rb") as f:
                return pickle.load(f)
        except (
            OSError,
            pickle.UnpicklingError,
            AttributeError,
            EOFError,
            ImportError,
            ModuleNotFoundError,
        ):
            # Cache files are optional and may have been produced by an older
            # script/module path. Ignore incompatible pickles and rebuild from
            # the knowledge-store zip instead of surfacing a tool error.
            return None

    def _get_claim_index(self, claim_id: str) -> _ClaimIndex:
        key = (str(self.knowledge_zip), str(claim_id))
        with self._CACHE_LOCK:
            cached = self._CACHE.get(key)
            if cached is not None:
                self._CACHE.move_to_end(key)
                return cached

        # Load/build outside the lock: disk I/O and index construction are the
        # slow part, and concurrent calls for different claim_ids shouldn't
        # serialize on each other. Only the final cache insert needs the lock.
        index = self._load_from_disk(str(claim_id))
        if index is None:
            index = self._build_claim_index(str(claim_id))

        with self._CACHE_LOCK:
            # Another thread may have inserted the same key meanwhile; keep
            # whichever is already there so identical calls share one object.
            cached = self._CACHE.get(key)
            if cached is not None:
                self._CACHE.move_to_end(key)
                return cached
            self._CACHE[key] = index
            self._CACHE.move_to_end(key)
            while len(self._CACHE) > self._CACHE_SIZE:
                self._CACHE.popitem(last=False)
        return index

    def _build_claim_index(self, claim_id: str) -> _ClaimIndex:
        chunks: list[_Chunk] = []
        claim_file = self.knowledge_dir / "output_dev" / f"{claim_id}.json"
        if claim_file.is_file():
            with claim_file.open("r", encoding="utf-8") as f:
                for line_no, raw_line in enumerate(f):
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    chunks.extend(self._record_chunks(record, line_no=line_no))
        else:
            member = f"output_dev/{claim_id}.json"
            with zipfile.ZipFile(self.knowledge_zip) as zf:
                if member not in zf.namelist():
                    raise FileNotFoundError(
                        f"{claim_id}.json not found in {self.knowledge_dir} or {self.knowledge_zip}"
                    )
                with zf.open(member) as f:
                    for line_no, raw_line in enumerate(f):
                        if not raw_line.strip():
                            continue
                        try:
                            record = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        chunks.extend(self._record_chunks(record, line_no=line_no))

        doc_freq: Counter[str] = Counter()
        total_len = 0
        for chunk in chunks:
            doc_freq.update(set(chunk.tokens))
            total_len += len(chunk.tokens)
        num_docs = max(len(chunks), 1)
        idf = {
            term: math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        avg_doc_len = total_len / num_docs if chunks else 0.0
        return _ClaimIndex(
            claim_id=claim_id,
            chunks=tuple(chunks),
            idf=idf,
            avg_doc_len=avg_doc_len,
        )

    def _record_chunks(self, record: dict[str, Any], *, line_no: int) -> list[_Chunk]:
        paragraphs = record.get("url2text") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        if not isinstance(paragraphs, list):
            return []

        source_type = str(record.get("type") or "unknown")
        source_query = str(record.get("query") or "")
        url = str(record.get("url") or "")
        out: list[_Chunk] = []
        buffer: list[str] = []
        buffer_len = 0
        chunk_idx = 0

        def flush() -> None:
            nonlocal chunk_idx, buffer, buffer_len
            text = "\n".join(part for part in buffer if part).strip()
            buffer = []
            buffer_len = 0
            if not text:
                return
            searchable_text = f"{source_query}\n{url}\n{text}"
            tokens = tuple(_tokenize(searchable_text, self._token_re))
            if not tokens:
                return
            out.append(
                _Chunk(
                    chunk_id=f"{line_no}:{chunk_idx}",
                    source_type=source_type,
                    source_query=source_query,
                    url=url,
                    text=text,
                    tokens=tokens,
                    term_counts=Counter(tokens),
                )
            )
            chunk_idx += 1

        for para in paragraphs:
            para_text = re.sub(r"\s+", " ", str(para or "")).strip()
            if not para_text:
                continue
            while len(para_text) > self.max_chunk_chars:
                piece = para_text[: self.max_chunk_chars]
                split_at = max(piece.rfind(". "), piece.rfind("; "), piece.rfind(", "))
                if split_at < self.max_chunk_chars // 2:
                    split_at = self.max_chunk_chars
                buffer.append(para_text[:split_at].strip())
                flush()
                para_text = para_text[split_at:].strip()
            if buffer_len + len(para_text) > self.max_chunk_chars and buffer:
                flush()
            buffer.append(para_text)
            buffer_len += len(para_text)
        flush()
        return out

    def _tokenize(self, text: Any) -> list[str]:
        return [match.group(0).lower() for match in self._token_re.finditer(str(text or ""))]

    def _search(
        self,
        index: _ClaimIndex,
        query: str,
        top_k: int,
    ) -> tuple[list[tuple[float, _Chunk]], dict[str, str]]:
        """Returns ``(results, evidence_summaries)``.

        ``evidence_summaries`` maps chunk text -> judge summary. Plain BM25 has
        no judge stage, so it's always empty; subclasses (e.g. HerOSearchTool's
        four-stage pipeline) populate it. Returned explicitly rather than
        stashed on ``self`` so concurrent calls on the same tool instance
        (parallel tool-call execution) can't cross-contaminate each other.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return [], {}
        query_counts = Counter(query_terms)
        top_k = max(1, min(int(top_k), 20))
        scored: list[tuple[float, _Chunk]] = []
        for chunk in index.chunks:
            doc_len = len(chunk.tokens) or 1
            score = 0.0
            for term, qf in query_counts.items():
                tf = chunk.term_counts.get(term, 0)
                if tf <= 0:
                    continue
                idf = index.idf.get(term, 0.0)
                denom = tf + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * doc_len / (index.avg_doc_len or 1.0))
                score += idf * (tf * (self.bm25_k1 + 1.0) / denom) * (1.0 + math.log(qf))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k], {}

    def _format_results(self, results: list[tuple[float, _Chunk]]) -> str:
        if not results:
            return "No relevant evidence chunks found."
        parts: list[str] = []
        remaining = self.max_output_chars
        for idx, (score, chunk) in enumerate(results, 1):
            text = chunk.text
            header = (
                f"[Evidence {idx}]\n"
                "text:\n"
            )
            budget = max(300, remaining - len(header))
            if len(text) > budget:
                text = text[:budget].rsplit(" ", 1)[0].strip() + "..."
            block = header + text
            parts.append(block)
            remaining -= len(block)
            if remaining <= 400:
                break
        return "\n\n".join(parts)


def _resolve_dataset_dir(dataset_dir: str | Path | None) -> Path:
    _repo_root = Path(__file__).resolve().parent.parent.parent
    candidates: list[Path] = []
    if dataset_dir:
        candidates.append(Path(dataset_dir))
        if not Path(dataset_dir).is_absolute():
            candidates.append(_repo_root / dataset_dir)
    for env_name in ("AVERITEC_DATA_DIR", "BELIEF_TRACER_AVERITEC_DIR"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.append(_DEFAULT_DATA_DIR)
    for candidate in candidates:
        if (candidate / "data" / "dev.json").is_file():
            return candidate
    return candidates[0]


def _claim_id_from_question(question: str) -> str | None:
    match = re.search(r"(?im)^Claim ID:\s*(\S+)", str(question or ""))
    return match.group(1) if match else None


def _tokenize(text: Any, pattern: re.Pattern[str] = _TOKEN_RE) -> list[str]:
    return [match.group(0).lower() for match in pattern.finditer(str(text or ""))]


__all__ = ["AVeriTeCSearchTool"]
