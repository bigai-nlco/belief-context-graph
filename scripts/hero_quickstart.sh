#!/bin/bash
# Quick start guide for HerO retrieval integration

set -e

echo "=== HerO Retrieval Integration Quick Start ==="
echo ""

# Step 1: Install dependencies
echo "[Step 1/4] Installing dependencies..."
pip install -q sentence-transformers torch scikit-learn

# Step 2: Test single-claim retrieval
echo ""
echo "[Step 2/4] Testing HerO retrieval on a single claim..."
python scripts/test_hero_retrieval.py --claim_id 0 --query "iMac Sean Connery"

# Step 3: Run on sub_AVeriTeC (50 samples)
echo ""
echo "[Step 3/4] Running experiment on sub_AVeriTeC (50 samples)..."
echo "This will take ~10-15 minutes with HerO retrieval..."
echo ""
echo "Command to run:"
echo "  bcg agent run averitec \\"
echo "    --retrieval-method hero \\"
echo "    --max-problems 50 \\"
echo "    --output-dir output/hero_test/"
echo ""
read -p "Run now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bcg agent run averitec \
        --retrieval-method hero \
        --max-problems 50 \
        --output-dir output/hero_test/
fi

# Step 4: Evaluate
echo ""
echo "[Step 4/4] Evaluating results..."
if [ -f output/hero_test/averitec/trajectories.jsonl ]; then
    bash scripts/evaluate_with_hero.sh \
        output/hero_test/averitec/trajectories.jsonl \
        datasets/sub_AVeriTeC/data/dev.json
else
    echo "Trajectories file not found. Please run the experiment first."
fi

echo ""
echo "=== Quick Start Complete ==="
echo ""
echo "Next steps:"
echo "  1. Compare with BM25 baseline:"
echo "     bcg agent run averitec --retrieval-method bm25 --output-dir output/bm25_test/"
echo ""
echo "  2. Run on full dev set (500 samples):"
echo "     bcg agent run averitec --retrieval-method hero --output-dir output/hero_full/"
echo ""
echo "  3. See docs/hero_search_tool_integration.md for more details"
