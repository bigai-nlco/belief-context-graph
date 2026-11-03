"""Replay PCST selection on a saved graph (remote embeddings by default).

Run from the checkout: python -m scripts.replay_bcg_pcst --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from workflows.stateful_react_agent.bcg.pcst import PCSTSelector, retrieval_focus
from workflows.stateful_react_agent.bcg.rendering import render_graph
from workflows.stateful_react_agent.bcg.runtime import BCGSettings


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Snapshot JSON or BCG state.json")
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--messages", type=Path, help="Optional JSON array of original model input messages")
    parser.add_argument("--focus", default="", help="Explicit current subgoal; overrides --messages focus")
    parser.add_argument("--strategy", choices=["confidence_pcst", "query_pcst"], nargs="+", default=["query_pcst"])
    parser.add_argument("--budget", type=int, default=6600)
    parser.add_argument("--edge-cost", type=float, default=0.2)
    parser.add_argument("--confidence-floor", type=float, default=0.2)
    parser.add_argument("--embedding-provider", choices=["openai", "local"])
    parser.add_argument("--embedding-url", help="Full OpenAI-compatible embeddings endpoint")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-device")
    parser.add_argument("--embedding-timeout", type=float)
    parser.add_argument("--embedding-chunk-bytes", type=int)
    parser.add_argument("--embedding-store", choices=["memory", "tonggraph"])
    parser.add_argument("--embedding-revision")
    parser.add_argument("--tonggraph-url")
    parser.add_argument("--tonggraph-graph")
    parser.add_argument("--tonggraph-timeout", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    saved = json.loads(raw)
    snapshot = saved.get("snapshot", saved)
    question = args.question_file.read_text()
    focus = args.focus
    if not focus and args.messages:
        focus = retrieval_focus(json.loads(args.messages.read_text()))
    output = {
        "snapshot": str(args.snapshot.resolve()), "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "budget": args.budget, "question": question, "focus": focus, "selections": [],
        "scope": "Saved-graph selection only; may call embedding service. No claim of improved answer quality.",
    }
    for strategy in args.strategy:
        settings = BCGSettings.from_agent({"bcg": {
            "retrieval": strategy, "graph_chars": args.budget,
            "pcst_edge_cost": args.edge_cost, "pcst_confidence_floor": args.confidence_floor,
            **{key: getattr(args, key) for key in (
                "embedding_provider", "embedding_url", "embedding_model", "embedding_device",
                "embedding_timeout", "embedding_chunk_bytes",
                "embedding_store", "embedding_revision", "tonggraph_url", "tonggraph_graph", "tonggraph_timeout",
            ) if getattr(args, key) is not None},
        }})
        started = time.monotonic()
        selection = PCSTSelector(settings).select(snapshot, question, focus, args.budget)
        selection["elapsed_seconds"] = time.monotonic() - started
        selection["rendered_context"] = (
            render_graph(snapshot, selection, args.budget, strict=True)[0]
            if selection["node_ids"] else ""
        )
        output["selections"].append(selection)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "selections": [{k: s[k] for k in ("strategy", "node_ids", "rendered_chars", "elapsed_seconds")} for s in output["selections"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
