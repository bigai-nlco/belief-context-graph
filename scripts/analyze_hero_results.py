#!/usr/bin/env python3
"""
分析 HerO 检索结果：查看 BM25 和 embedding 的中间信息
"""
import json
import sys

def analyze_trajectory(traj_file, sample_idx=0):
    """分析指定 sample 的检索信息"""
    with open(traj_file) as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                data = json.loads(line)
                break
        else:
            print(f"Sample {sample_idx} not found")
            return

    print(f"{'='*80}")
    print(f"Sample {sample_idx} Analysis")
    print(f"{'='*80}\n")

    # 查找所有搜索工具调用
    sample = data.get('sample', {})
    turns = sample.get('turns', [])

    search_count = 0
    for turn_idx, turn in enumerate(turns):
        tool_calls = turn.get('tool_calls', [])

        for call in tool_calls:
            if call.get('tool_name') == 'averitec_search':
                search_count += 1
                query = call.get('tool_input', {}).get('query', 'N/A')
                top_k = call.get('tool_input', {}).get('top_k', 'N/A')
                output = call.get('tool_output', {}).get('output', '')

                print(f"[Search {search_count}] Turn {turn_idx}")
                print(f"  Query: {query}")
                print(f"  Top-k: {top_k}")
                print(f"  Output length: {len(output)} chars")
                print(f"  Output preview:")
                print(f"    {output[:200]}...")
                print()

    if search_count == 0:
        print("No averitec_search calls found.")
        print("\nTurn structure:")
        for i, turn in enumerate(turns[:3]):
            print(f"  Turn {i}: keys = {list(turn.keys())}")

    print(f"\nTotal searches: {search_count}")

if __name__ == '__main__':
    traj_file = sys.argv[1] if len(sys.argv) > 1 else 'output/hero_10samples_cpu/Qwen3-8B_thinking/averitec/trajectories.jsonl'
    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    analyze_trajectory(traj_file, sample_idx)
