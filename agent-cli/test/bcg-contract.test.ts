import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
	BcgContextManager,
	formatBcgDialogueContext,
	formatBcgMarkdown,
} from "../src/core/context/bcg-context.ts";
import type { BcgTurnsResponse } from "../src/core/context/bcg-contract.types.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS = path.resolve(HERE, "../../contracts");

function fixture(name: string): unknown {
	return JSON.parse(readFileSync(path.join(CONTRACTS, "fixtures", name), "utf8"));
}

const turnsResponse = fixture("turns-response.json") as BcgTurnsResponse;
const problemId = "fixture-session:seed";

describe("cross-language contract fixtures (step 10)", () => {
	it("formats the fixture snapshot into belief graph markdown", () => {
		const snapshot = turnsResponse.latest[problemId];
		const markdown = formatBcgMarkdown(snapshot);

		expect(markdown).toContain("## Belief Graph");
		expect(markdown).toContain("The user asked for a summary of key beliefs.");
		expect(markdown).toContain("[1]");
	});

	it("parseSnapshot resolves the fixture envelope through latest[problemId]", async () => {
		const requests: Array<Record<string, unknown>[]> = [];
		const fetchMock = (async (_input, init) => {
			const body = JSON.parse(String(init?.body)) as Array<Record<string, unknown>>;
			requests.push(body);
			return new Response(JSON.stringify(turnsResponse), {
				status: 200,
				headers: { "content-type": "application/json" },
			});
		}) as typeof globalThis.fetch;

		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId,
			recentTurns: 2,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "base system",
			fetch: fetchMock,
		});

		const effective = (async () => {
			// transform triggers the /turns call that populates the snapshot
			await manager.transform([{ role: "user", content: "hi", timestamp: 1 }]);
			return manager.augmentSystemPrompt("base system");
		})();
		expect(await effective).toContain("The user asked for a summary of key beliefs.");

		// the fixture request turns must carry the wire contract shape
		const fixtureRequest = fixture("turns-request.json") as Array<Record<string, unknown>>;
		expect(fixtureRequest).toHaveLength(2);
		for (const turn of fixtureRequest) {
			expect(typeof turn.problem_id).toBe("string");
			expect(typeof turn.role).toBe("string");
			expect(typeof turn.content).toBe("string");
		}
	});

	it("formatBcgMarkdown emits relations from the relations array (no forward_relations)", () => {
		const snapshot = turnsResponse.latest[problemId];
		const withRelations = {
			...snapshot,
			relations: [{ id: 1, from_id: 1, to_id: 2, type: "depends_on", note: "because" }],
		};
		const markdown = formatBcgMarkdown(withRelations);
		expect(markdown).toContain("### Relations");
		expect(markdown).toContain("[1] → [2] (depends_on) — because");
	});

	it("formatBcgDialogueContext preserves fixture roles and confidence", () => {
		const snapshot = turnsResponse.latest[problemId];
		const encoded = formatBcgDialogueContext(snapshot);

		expect(encoded).toContain("<｜begin▁of▁sentence｜><｜User｜>");
		expect(encoded).toContain("### Belief 1");
		expect(encoded).toContain("**Content:** The user asked for a summary of key beliefs.");
		expect(encoded).toContain("**Confidence:** 0.88");
		expect(encoded).not.toContain("<belief_graph");
	});
});
