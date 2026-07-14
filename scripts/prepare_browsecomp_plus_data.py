"""Prepare BrowseComp-Plus tasks for bcg.agent.benchmark_loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert decrypted BrowseComp-Plus JSONL to belief-context-graph data.json.")
    parser.add_argument(
        "--input",
        default="../BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
        help="Decrypted BrowseComp-Plus JSONL.",
    )
    parser.add_argument(
        "--output",
        default="datasets/browsecomp_plus/data.json",
        help="Output data.json path consumed by load_browsecomp_plus().",
    )
    return parser.parse_args()


def docids(value: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, dict):
            docid = item.get("docid")
        else:
            docid = item
        if docid is not None:
            out.append(str(docid))
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "task_id": str(row.get("query_id", "")),
                    "query_id": str(row.get("query_id", "")),
                    "query": str(row.get("query", "")),
                    "answer": row.get("answer", ""),
                    "gold_docids": docids(row.get("gold_docs")),
                    "evidence_docids": docids(row.get("evidence_docs")),
                    "negative_docids": docids(row.get("negative_docs")),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} BrowseComp-Plus tasks to {output_path}")


if __name__ == "__main__":
    main()
