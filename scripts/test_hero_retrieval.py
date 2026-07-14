#!/usr/bin/env python3
"""Test HerO retrieval tool on a single claim."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bcg.agent.tools.hero_search_tool import HerOSearchTool
from bcg.agent.tools.averitec_search import AVeriTeCSearchTool


def test_single_claim(claim_id: str = "419", query: str = "COVID-19 mortality rate"):
    """Test both BM25 and HerO retrieval on a single claim."""

    print(f"Testing claim_id={claim_id}, query='{query}'")
    print("=" * 80)

    # Test BM25
    print("\n[1/2] BM25 Retrieval:")
    print("-" * 80)
    bm25_tool = AVeriTeCSearchTool()
    bm25_tool.set_task({"claim_id": claim_id})
    bm25_result = bm25_tool.forward(query=query, top_k=3)

    if bm25_result.error:
        print(f"ERROR: {bm25_result.error}")
    else:
        print(bm25_result.output[:1000] + "..." if len(bm25_result.output) > 1000 else bm25_result.output)

    # Test HerO
    print("\n[2/2] HerO Retrieval (BM25 + Embedding Reranking):")
    print("-" * 80)

    hero_tool = HerOSearchTool(
        bm25_top_k=100,
        embedding_model="Salesforce/SFR-Embedding-2_R",
        embedding_device="cpu",  # Use CPU to avoid GPU OOM
        batch_size=4
    )
    hero_tool.set_task({"claim_id": claim_id})
    hero_result = hero_tool.forward(query=query, top_k=3)

    if hero_result.error:
        print(f"ERROR: {hero_result.error}")
    else:
        print(hero_result.output[:1000] + "..." if len(hero_result.output) > 1000 else hero_result.output)

    print("\n" + "=" * 80)
    print("Comparison:")
    print(f"  BM25 results:  {bm25_result.metadata.get('num_results', 0)} chunks")
    print(f"  HerO results:  {hero_result.metadata.get('num_results', 0)} chunks")
    print(f"  Same results?  {bm25_result.output == hero_result.output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test HerO retrieval tool")
    parser.add_argument("--claim_id", default="419", help="Claim ID to test")
    parser.add_argument("--query", default="COVID-19 mortality rate", help="Search query")

    args = parser.parse_args()
    test_single_claim(args.claim_id, args.query)
