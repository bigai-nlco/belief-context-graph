"""
Convert BeliefTracer trajectories.jsonl to HerO evaluation format.

This script extracts evidence from tool calls in the trajectory and
formats predictions for HerO's averitec_evaluate.py.
"""

import json
import re
from pathlib import Path
from typing import Any


def extract_evidence_from_trajectory(trajectory: list[dict]) -> list[dict]:
    """
    Extract evidence from tool calls in the trajectory.

    Args:
        trajectory: List of conversation turns

    Returns:
        List of evidence dicts with question, answer, url
    """
    evidence = []

    for turn in trajectory:
        role = turn.get("role")
        content = turn.get("content", "")

        # Extract query from assistant's tool call
        if role == "assistant" and "averitec_search" in content:
            # Parse tool call to get query
            query = extract_query_from_tool_call(content)
            if not query:
                continue

        # Extract evidence from tool response
        if role == "tool":
            evidence_chunks = parse_tool_response(content)
            for chunk in evidence_chunks:
                evidence.append({
                    "question": query if query else "search query",
                    "answer": chunk["text"],
                    "url": chunk.get("url", "")
                })

    return evidence


def extract_query_from_tool_call(content: str) -> str | None:
    """Extract query from tool call JSON."""
    try:
        # Find JSON in tool_call block
        match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', content, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return data.get("arguments", {}).get("query")
    except:
        pass
    return None


def parse_tool_response(content: str) -> list[dict]:
    """Parse evidence chunks from tool response."""
    chunks = []

    # Split by [Evidence N] markers
    pattern = r'\[Evidence \d+\]\s*text:\s*(.*?)(?=\[Evidence \d+\]|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        text = match.strip()
        if text:
            # Try to extract URL if present
            url_match = re.search(r'(https?://[^\s]+)', text)
            url = url_match.group(1) if url_match else ""

            chunks.append({
                "text": text[:500],  # Truncate long texts
                "url": url
            })

    return chunks


def convert_trajectories_to_hero_format(
    trajectories_file: str | Path,
    output_file: str | Path,
    include_evidence: bool = True
) -> None:
    """
    Convert BeliefTracer trajectories to HerO evaluation format.

    Args:
        trajectories_file: Path to trajectories.jsonl
        output_file: Path to output JSON file
        include_evidence: Whether to extract evidence from trajectories
    """
    predictions = []

    with open(trajectories_file) as f:
        for line in f:
            if not line.strip():
                continue

            traj = json.loads(line)

            claim_id = int(traj.get("task_id", 0))
            pred_label = traj.get("sample", {}).get("extracted_answer", "")

            pred = {
                "claim_id": claim_id,
                "pred_label": pred_label
            }

            # Extract evidence if requested
            if include_evidence:
                trajectory = traj.get("sample", {}).get("trajectory", [])
                evidence = extract_evidence_from_trajectory(trajectory)
                if evidence:
                    pred["evidence"] = evidence

            predictions.append(pred)

    # Write as JSON array
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(predictions)} predictions to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BeliefTracer output to HerO evaluation format"
    )
    parser.add_argument(
        "--trajectories",
        required=True,
        help="Path to trajectories.jsonl"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Skip evidence extraction (only convert labels)"
    )

    args = parser.parse_args()

    convert_trajectories_to_hero_format(
        trajectories_file=args.trajectories,
        output_file=args.output,
        include_evidence=not args.no_evidence
    )


if __name__ == "__main__":
    main()
