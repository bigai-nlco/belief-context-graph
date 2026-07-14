#!/usr/bin/env python3
"""Standalone demo: Qwen3-Reranker scoring via /v1/completions.

Why not /v1/rerank? The sglang deployment at :8010 returns a frozen score
(~0.000203) for every input on /v1/rerank, so it cannot rank. Instead we feed
each (query, document) pair through the model's official chat template on
/v1/completions and read P("yes") from the next-token logprobs. This is the
upstream-recommended way to use Qwen3-Reranker.

Run:
    python scripts/demo_rerank_completions.py
"""

from __future__ import annotations

import math
import requests

RERANK_BASE = "http://10.2.152.9:8010"
MODEL = "Qwen3-Reranker-0.6B"

# Official Qwen3-Reranker template pieces.
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


def build_prompt(query: str, document: str) -> str:
    return (
        f"{PREFIX}<Instruct>: {INSTRUCT}\n"
        f"<Query>: {query}\n<Document>: {document}{SUFFIX}"
    )


def relevance_score(query: str, document: str) -> tuple[float, dict]:
    """Return P(yes) for (query, document) plus the raw top-logprobs dict."""
    url = RERANK_BASE.rstrip("/") + "/v1/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    payload = {
        "model": MODEL,
        "prompt": build_prompt(query, document[:4000]),
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 20,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    top = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    yes = math.exp(top["yes"]) if "yes" in top else 0.0
    no = math.exp(top["no"]) if "no" in top else (math.exp(top["No"]) if "No" in top else 0.0)
    return yes / (yes + no + 1e-30), top


def main() -> None:
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "The capital of France is Paris, a major European city.",
        "Bananas are a yellow tropical fruit rich in potassium.",
        "Berlin is the capital and largest city of Germany.",
    ]

    print(f"Query: {query}\n")
    scored = []
    for doc in documents:
        score, top = relevance_score(query, doc)
        scored.append((score, doc))
        # Show the raw yes/no logprobs so you can see how the score is derived.
        yn = {k: round(v, 3) for k, v in top.items() if k.lower() in ("yes", "no")}
        print(f"  P(yes)={score:.4f}  raw_logprobs={yn}  | {doc}")

    print("\n--- ranked (descending P(yes)) ---")
    for score, doc in sorted(scored, key=lambda x: x[0], reverse=True):
        print(f"  {score:.4f} | {doc}")


if __name__ == "__main__":
    main()
