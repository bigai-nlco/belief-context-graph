import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_BCG_GRAPH_URL } from "../src/core/settings-manager.ts";

// Step 11.5: the agent's default BCG graph URL must agree with the
// cross-language contract (contracts/defaults.json) — one source for the
// server host/port defaults.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS = path.resolve(HERE, "../../contracts");

describe("BCG settings contract (step 11)", () => {
	it("default graph URL matches contracts/defaults.json", () => {
		const defaults = JSON.parse(
			readFileSync(path.join(CONTRACTS, "defaults.json"), "utf8"),
		) as { server: { host: string; port: number } };
		const expected = `http://${defaults.server.host}:${defaults.server.port}`;
		expect(DEFAULT_BCG_GRAPH_URL).toBe(expected);
	});
});
