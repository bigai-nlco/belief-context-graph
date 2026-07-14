#!/bin/bash
# 端到端评测流程：BeliefTracer 输出 → HerO 格式 → 评测

set -e

TRAJECTORIES_FILE="$1"
LABEL_FILE="${2:-datasets/AVeriTeC/data/dev.json}"
OUTPUT_DIR="${3:-output/hero_eval}"

if [ -z "$TRAJECTORIES_FILE" ]; then
    echo "Usage: $0 <trajectories.jsonl> [label_file] [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 output/experiment/averitec/trajectories.jsonl"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

HERO_FORMAT="$OUTPUT_DIR/predictions_hero_format.json"

echo "=== BeliefTracer → HerO Format Conversion ==="
python bcg/agent/evaluators/convert_to_hero_format.py \
    --trajectories "$TRAJECTORIES_FILE" \
    --output "$HERO_FORMAT"

echo ""
echo "=== Running HerO Evaluation ==="
python bcg/agent/evaluators/hero_evaluator.py \
    --prediction_file "$HERO_FORMAT" \
    --label_file "$LABEL_FILE"

echo ""
echo "=== Evaluation Complete ==="
echo "Predictions saved to: $HERO_FORMAT"
