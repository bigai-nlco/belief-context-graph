import { describe, expect, it, vi } from "vitest";
import { BcgClient } from "../src/core/context/bcg-client.ts";
import type { BcgTurnsResponse } from "../src/core/context/bcg-contract.types.ts";

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "content-type": "application/json" },
	});
}

const turn = {
	problem_id: "p",
	role: "user",
	content: "hi",
	is_message_end: true,
	is_trajectory_end: false,
} as const;

describe("BcgClient (contract-backed HTTP client, step 11)", () => {
	it("resolves the snapshot from the latest[problemId] envelope", async () => {
		const envelope: BcgTurnsResponse = {
			pushed: 1,
			finalized: [],
			latest: {
				p: {
					generated_at: "2026-08-06T00:00:00+00:00",
					n_nodes: 1,
					n_beliefs: 1,
					n_decisions: 0,
					nodes: [],
					beliefs: [{ id: 7, node_type: "belief", belief: "hello" }],
					decisions: [],
					relations: [],
				},
			},
		};
		const fetchMock = vi.fn(async () => jsonResponse(envelope)) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		const snapshot = await client.postTurns([turn]);
		expect(snapshot.beliefs?.[0]?.id).toBe(7);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it("surfaces the server error envelope text on non-2xx", async () => {
		const fetchMock = vi.fn(async () =>
			jsonResponse({ error: "trajectory already finalized" }, 409),
		) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		await expect(client.postTurns([turn])).rejects.toThrow(
			"trajectory already finalized",
		);
	});

	it("requests a connected context selection for the current Agent state", async () => {
		const fetchMock = vi.fn(async () =>
			jsonResponse({
				problem_id: "p",
				strategy: "connected",
				retrieval: "embedding",
				node_ids: [8, 5],
				relation_ids: [3],
				node_chars: 420,
			}),
		) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		const selection = await client.selectContext("question plus recent state");

		expect(selection.node_ids).toEqual([8, 5]);
		expect(fetchMock).toHaveBeenCalledWith(
			"http://127.0.0.1:8848/context-selection",
			expect.objectContaining({
				method: "POST",
				body: JSON.stringify({
					problem_id: "p",
					query: "question plus recent state",
					strategy: "connected",
					node_char_budget: 6000,
					max_depth: 4,
				}),
			}),
		);
	});

	it("sends focused selection intent separately from raw Agent state", async () => {
		const fetchMock = vi.fn(async () =>
			jsonResponse({
				problem_id: "p",
				strategy: "focused",
				retrieval: "embedding",
				node_ids: [8],
				relation_ids: [],
				node_chars: 120,
			}),
		) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		await client.selectContext("question plus raw result", {
			strategy: "focused",
			focusQuery: "question plus reasoning intent",
			question: "question",
		});

		expect(fetchMock).toHaveBeenCalledWith(
			"http://127.0.0.1:8848/context-selection",
			expect.objectContaining({
				body: JSON.stringify({
					problem_id: "p",
					query: "question plus raw result",
					strategy: "focused",
					focus_query: "question plus reasoning intent",
					question: "question",
					node_char_budget: 6000,
					max_depth: 4,
				}),
			}),
		);
	});

	it("treats release 404 as idempotent success and reports released", async () => {
		const fetchMock = vi.fn(async () =>
			jsonResponse({ error: "no trajectory for problem_id 'p'" }, 404),
		) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		const result = await client.release();
		expect(result).toEqual({ problem_id: "p", released: true });
	});

	it("finalizes and returns the persisted graph snapshot", async () => {
		const fetchMock = vi.fn(async () =>
			jsonResponse({
				problem_id: "p",
				stage: "final",
				finalized: true,
				generated_at: "2026-08-09T00:00:00+00:00",
				n_nodes: 1,
				n_beliefs: 1,
				n_decisions: 0,
				nodes: [{ id: 1, node_type: "belief", belief: "done" }],
				beliefs: [{ id: 1, node_type: "belief", belief: "done" }],
				decisions: [],
				relations: [],
			}),
		) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		const result = await client.finalize();
		expect(result.finalized).toBe(true);
		expect(result.stage).toBe("final");
		expect(fetchMock).toHaveBeenCalledWith(
			"http://127.0.0.1:8848/finalize",
			expect.objectContaining({
				method: "POST",
				body: JSON.stringify({ problem_id: "p" }),
			}),
		);
	});

	it("reads the released flag from a successful release body", async () => {
		const fetchMock = vi.fn(async () =>
			jsonResponse({ problem_id: "p", released: false }),
		) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		expect(await client.release()).toEqual({ problem_id: "p", released: false });
	});

	it("honours caller AbortSignal", async () => {
		const controller = new AbortController();
		const fetchMock = vi.fn(
			async (_input: unknown, init: { signal?: AbortSignal }) => {
				controller.abort();
				if (init.signal?.aborted) {
					throw new DOMException("aborted", "AbortError");
				}
				return jsonResponse({});
			},
		) as typeof globalThis.fetch;
		const client = new BcgClient({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "p",
			timeoutMs: 1000,
			fetch: fetchMock,
		});

		await expect(
			client.postTurns([turn], controller.signal),
		).rejects.toThrow();
	});
});
