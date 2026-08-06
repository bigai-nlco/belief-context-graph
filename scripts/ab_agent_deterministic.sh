#!/usr/bin/env bash
# Deterministic Agent A/B comparison (no LLM required): runs the same fixture
# through the old branch and the current branch and diffs the observable
# outputs (graph markdown, context-manager transform/injection).
#
# Usage: bash scripts/ab_agent_deterministic.sh [--old-branch main]
set -euo pipefail

OLD_BRANCH="${1:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OLD_ROOT="/tmp/bcg-ab-old"
OLD_AGENT="$OLD_ROOT/agent-cli"
NEW_AGENT="$NEW_ROOT/agent-cli"
OUT_DIR="/tmp/bcg-ab-out"
mkdir -p "$OUT_DIR"

if [[ ! -d "$OLD_AGENT" ]]; then
  git -C "$NEW_ROOT" worktree add "$OLD_ROOT" "$OLD_BRANCH"
fi
if [[ ! -d "$OLD_AGENT/node_modules" ]]; then
  (cd "$OLD_AGENT" && npm ci --silent)
fi

cat > "$OLD_AGENT/test/ab-dump.test.ts" <<'TEST'
import { writeFileSync } from "node:fs";
import { describe, it } from "vitest";
const SNAPSHOT = {
	generated_at: "2026-08-06T00:00:00+00:00",
	problem_id: "ab-session",
	stage: "turn",
	finalized: false,
	n_nodes: 2,
	n_beliefs: 2,
	n_decisions: 0,
	nodes: [
		{ id: 1, node_type: "belief", belief: "The user asked for a summary of key beliefs.", confidence: 0.88, stance: "asserted", role: "user", layer: "io", evidence_ids: [1] },
		{ id: 2, node_type: "belief", belief: "The assistant speculated a follow-up belief.", confidence: 0.5, stance: "speculated", role: "assistant", layer: "io", evidence_ids: [1] },
	],
	beliefs: [
		{ id: 1, node_type: "belief", belief: "The user asked for a summary of key beliefs.", confidence: 0.88, stance: "asserted", role: "user", layer: "io", evidence_ids: [1] },
		{ id: 2, node_type: "belief", belief: "The assistant speculated a follow-up belief.", confidence: 0.5, stance: "speculated", role: "assistant", layer: "io", evidence_ids: [1] },
	],
	decisions: [],
	relations: [
		{ id: 1, from_id: 1, to_id: 2, type: "depends_on", note: "because", weight: 0.5 },
		{ id: 2, from_id: 2, to_id: 1, type: "contradicts", note: "disagrees", weight: 0.3 },
	],
	evidence: {},
	merges: [],
	sessions: [],
};
import { formatBcgMarkdown } from "../src/core/context/bcg-context.ts";
describe("ab-dump", () => {
	it("dumps markdown", () => {
		writeFileSync(process.env.AB_OUT!, JSON.stringify(formatBcgMarkdown(SNAPSHOT as never)));
	});
});
TEST
cp "$OLD_AGENT/test/ab-dump.test.ts" "$NEW_AGENT/test/ab-dump.test.ts"

cat > "$OLD_AGENT/test/ab-manager.test.ts" <<'TEST'
import { writeFileSync } from "node:fs";
import { describe, it } from "vitest";
const RESPONSE = {
	pushed: 2,
	finalized: [],
	latest: {
		"ab-session": {
			generated_at: "2026-08-06T00:00:00+00:00",
			problem_id: "ab-session",
			stage: "turn",
			finalized: false,
			n_nodes: 2,
			n_beliefs: 2,
			n_decisions: 0,
			nodes: [
				{ id: 1, node_type: "belief", belief: "The user asked for a summary of key beliefs.", confidence: 0.88, stance: "asserted", role: "user", layer: "io", evidence_ids: [1] },
				{ id: 2, node_type: "belief", belief: "The assistant speculated a follow-up belief.", confidence: 0.5, stance: "speculated", role: "assistant", layer: "io", evidence_ids: [1] },
			],
			beliefs: [
				{ id: 1, node_type: "belief", belief: "The user asked for a summary of key beliefs.", confidence: 0.88, stance: "asserted", role: "user", layer: "io", evidence_ids: [1] },
				{ id: 2, node_type: "belief", belief: "The assistant speculated a follow-up belief.", confidence: 0.5, stance: "speculated", role: "assistant", layer: "io", evidence_ids: [1] },
			],
			decisions: [],
			relations: [
				{ id: 1, from_id: 1, to_id: 2, type: "depends_on", note: "because", weight: 0.5 },
				{ id: 2, from_id: 2, to_id: 1, type: "contradicts", note: "disagrees", weight: 0.3 },
			],
			evidence: {},
			merges: [],
			sessions: [],
		},
	},
};
import { BcgContextManager } from "../src/core/context/bcg-context.ts";
describe("ab-manager", () => {
	it("dumps manager behavior", async () => {
		const fetchMock = (async () =>
			new Response(JSON.stringify(RESPONSE), { status: 200, headers: { "content-type": "application/json" } })) as typeof globalThis.fetch;
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "ab-session",
			recentTurns: 2,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "base system",
			fetch: fetchMock,
		});
		const turned = await manager.transform([
			{ role: "user", content: "hello", timestamp: 1 },
			{ role: "assistant", content: [{ type: "text", text: "hi" }], timestamp: 2, api: "t", provider: "t", model: "m" },
		]);
		const augmented = manager.augmentSystemPrompt("base system");
		writeFileSync(process.env.AB_OUT!, JSON.stringify({ turnedRoles: turned.map((m) => m.role), turnedCount: turned.length, augmented }));
	});
});
TEST
cp "$OLD_AGENT/test/ab-manager.test.ts" "$NEW_AGENT/test/ab-manager.test.ts"

failures=0
for name in dump manager; do
  (cd "$OLD_AGENT" && AB_OUT="$OUT_DIR/old-$name.txt" npx vitest --run "test/ab-$name.test.ts" >/dev/null 2>&1)
  (cd "$NEW_AGENT" && AB_OUT="$OUT_DIR/new-$name.txt" npx vitest --run "test/ab-$name.test.ts" >/dev/null 2>&1)
  if diff -q "$OUT_DIR/old-$name.txt" "$OUT_DIR/new-$name.txt" >/dev/null; then
    printf 'ok   - %s: old == new (%s bytes)\n' "$name" "$(wc -c < "$OUT_DIR/old-$name.txt")"
  else
    printf 'FAIL - %s differs; diff:\n' "$name"
    diff "$OUT_DIR/old-$name.txt" "$OUT_DIR/new-$name.txt" | head -20
    failures=$((failures + 1))
  fi
done

rm -f "$OLD_AGENT/test/ab-dump.test.ts" "$OLD_AGENT/test/ab-manager.test.ts" "$NEW_AGENT/test/ab-dump.test.ts" "$NEW_AGENT/test/ab-manager.test.ts"

if ((failures > 0)); then
  printf '%d deterministic A/B check(s) failed.\n' "$failures" >&2
  exit 1
fi
printf 'All deterministic Agent A/B checks passed against %s.\n' "$OLD_BRANCH"
