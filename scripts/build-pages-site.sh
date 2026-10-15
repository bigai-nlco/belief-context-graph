#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=${1:-"$ROOT_DIR/_site"}

mkdir -p "$OUTPUT_DIR/assets"

cp "$ROOT_DIR/website/index.html" "$OUTPUT_DIR/index.html"
cp "$ROOT_DIR/website/styles.css" "$OUTPUT_DIR/styles.css"
cp "$ROOT_DIR/website/responsive.css" "$OUTPUT_DIR/responsive.css"
cp "$ROOT_DIR/website/main.js" "$OUTPUT_DIR/main.js"
cp "$ROOT_DIR/website/favicon.svg" "$OUTPUT_DIR/favicon.svg"
cp "$ROOT_DIR/website/site.webmanifest" "$OUTPUT_DIR/site.webmanifest"
cp "$ROOT_DIR/website/index.html" "$OUTPUT_DIR/404.html"

cp "$ROOT_DIR/assets/benchmark_overview.svg" "$OUTPUT_DIR/assets/benchmark-overview.svg"
cp "$ROOT_DIR/assets/architecture.svg" "$OUTPUT_DIR/assets/architecture.svg"
cp "$ROOT_DIR/assets/case_study.svg" "$OUTPUT_DIR/assets/case-study.svg"

: > "$OUTPUT_DIR/.nojekyll"

printf 'Built GitHub Pages site at %s\n' "$OUTPUT_DIR"
