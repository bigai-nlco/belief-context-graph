#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

tracked_ignored="$(git ls-files -ci --exclude-standard)"
if [[ -n "$tracked_ignored" ]]; then
  printf 'Tracked files must not also be ignored:\n%s\n' "$tracked_ignored" >&2
  exit 1
fi

forbidden_paths=()
while IFS= read -r path; do
  case "$path" in
    .env|*/.env|node_modules/*|*/node_modules/*|dist/*|*/dist/*|outputs*/*|*/outputs*/*)
      forbidden_paths+=("$path")
      ;;
  esac
done < <(git ls-files)

if ((${#forbidden_paths[@]} > 0)); then
  printf 'Private or generated paths must not be tracked:\n' >&2
  printf '  %s\n' "${forbidden_paths[@]}" >&2
  exit 1
fi

printf 'Repository hygiene checks passed.\n'
