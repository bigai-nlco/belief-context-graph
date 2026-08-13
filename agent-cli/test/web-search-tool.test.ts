import { describe, expect, it, vi } from "vitest";
import { createAllToolDefinitions, createWebSearchTool, isSerperSearchConfigured } from "../src/core/tools/index.ts";

describe("Serper web_search tool", () => {
	it("sends the configured search request and returns structured evidence", async () => {
		const apiKey = "test-serper-key";
		let capturedRequest: { input: string | URL | Request; init?: RequestInit } | undefined;
		const fetchMock: typeof globalThis.fetch = vi.fn(async (input, init) => {
			capturedRequest = { input, init };
			return Response.json({
				answerBox: {
					title: "Direct answer",
					answer: "42",
					link: "https://example.com/answer",
				},
				knowledgeGraph: {
					title: "Example topic",
					description: "A concise description.",
					website: "https://example.com/topic",
					attributes: { Founded: "2024" },
				},
				organic: [
					{
						title: "Organic result",
						link: "https://example.com/organic",
						snippet: "An organic result snippet.",
					},
					{
						title: "Duplicate organic result",
						link: "https://example.com/organic",
						snippet: "An organic result snippet.",
					},
				],
			});
		});
		const tool = createWebSearchTool({
			apiKey,
			endpoint: "https://search.example.test/search",
			country: "gb",
			language: "en",
			fetch: fetchMock,
		});

		const result = await tool.execute("search-1", { query: "  example query  ", top_k: 5 });

		expect(fetchMock).toHaveBeenCalledOnce();
		expect(String(capturedRequest?.input)).toBe("https://search.example.test/search");
		const headers = new Headers(capturedRequest?.init?.headers);
		expect(headers.get("X-API-KEY")).toBe(apiKey);
		expect(headers.get("Content-Type")).toBe("application/json");
		expect(JSON.parse(String(capturedRequest?.init?.body))).toEqual({
			q: "example query",
			num: 5,
			gl: "gb",
			hl: "en",
		});
		expect(result.content).toEqual([
			{
				type: "text",
				text: [
					"[1] Direct answer",
					"URL: https://example.com/answer",
					"Snippet: 42",
					"",
					"[2] Example topic",
					"URL: https://example.com/topic",
					"Snippet: A concise description.; Founded: 2024",
					"",
					"[3] Organic result",
					"URL: https://example.com/organic",
					"Snippet: An organic result snippet.",
				].join("\n"),
			},
		]);
		expect(result.details).toMatchObject({
			provider: "serper",
			query: "example query",
			numResults: 3,
			searchParameters: { country: "gb", language: "en", topK: 5 },
			truncated: false,
		});
		expect(JSON.stringify(result.details)).not.toContain(apiKey);
	});

	it("requests five results by default and bounds each rendered snippet", async () => {
		let requestBody: Record<string, unknown> | undefined;
		const fetchMock: typeof globalThis.fetch = vi.fn(async (_input, init) => {
			requestBody = JSON.parse(String(init?.body));
			return Response.json({
				organic: [
					{
						title: "Long result",
						link: "https://example.com/long",
						snippet: "x".repeat(500),
					},
				],
			});
		});
		const tool = createWebSearchTool({ apiKey: "test-key", fetch: fetchMock });

		const result = await tool.execute("search-default", { query: "example" });
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";

		expect(requestBody?.num).toBe(5);
		expect(result.details.searchParameters.topK).toBe(5);
		expect(text).toContain(`Snippet: ${"x".repeat(199)}…`);
		expect(text).not.toContain("Source type:");
	});

	it("requires a real API key without making a request", async () => {
		const fetchMock: typeof globalThis.fetch = vi.fn();
		const tool = createWebSearchTool({
			env: { SERPER_API_KEY: "your-api-key" },
			fetch: fetchMock,
		});

		await expect(tool.execute("search-1", { query: "example" })).rejects.toThrow("SERPER_API_KEY is not configured");
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("enforces the per-session call budget across parallel executions", async () => {
		const fetchMock: typeof globalThis.fetch = vi.fn(async () =>
			Response.json({
				organic: [
					{
						title: "Evidence",
						link: "https://example.com/evidence",
						snippet: "Useful evidence.",
					},
				],
			}),
		);
		const tool = createWebSearchTool({ apiKey: "test-key", maxCalls: 2, fetch: fetchMock });

		const [first, second, blocked] = await Promise.all([
			tool.execute("search-1", { query: "first" }),
			tool.execute("search-2", { query: "second" }),
			tool.execute("search-3", { query: "third" }),
		]);

		expect(fetchMock).toHaveBeenCalledTimes(2);
		expect(first.details.budget).toEqual({ callsUsed: 1, maxCalls: 2, exhausted: false, blocked: false });
		expect(second.details.budget).toEqual({ callsUsed: 2, maxCalls: 2, exhausted: true, blocked: false });
		expect(blocked.details).toMatchObject({
			query: "third",
			numResults: 0,
			budget: { callsUsed: 2, maxCalls: 2, exhausted: true, blocked: true },
		});
		const blockedText = blocked.content[0]?.type === "text" ? blocked.content[0].text : "";
		expect(blockedText).toContain("Search budget exhausted after 2 calls");
		expect(blockedText).toContain("Do not call web_search again");
	});

	it("reads the session call budget from SERPER_MAX_CALLS", async () => {
		const fetchMock: typeof globalThis.fetch = vi.fn(async () => Response.json({ organic: [] }));
		const tool = createWebSearchTool({
			env: { SERPER_API_KEY: "test-key", SERPER_MAX_CALLS: "1" },
			fetch: fetchMock,
		});

		const first = await tool.execute("search-1", { query: "first" });
		const blocked = await tool.execute("search-2", { query: "second" });

		expect(fetchMock).toHaveBeenCalledOnce();
		expect(first.details.budget.maxCalls).toBe(1);
		expect(blocked.details.budget.exhausted).toBe(true);
	});

	it("redacts the API key from HTTP errors", async () => {
		const apiKey = "secret-value";
		const fetchMock: typeof globalThis.fetch = vi.fn(
			async () => new Response(`invalid key: ${apiKey}`, { status: 401 }),
		);
		const tool = createWebSearchTool({ apiKey, fetch: fetchMock });

		const error = await tool.execute("search-1", { query: "example" }).catch((caught: unknown) => caught);

		expect(error).toBeInstanceOf(Error);
		expect((error as Error).message).toContain("[REDACTED]");
		expect((error as Error).message).not.toContain(apiKey);
	});

	it("marks formatted output when it reaches the output limit", async () => {
		const fetchMock: typeof globalThis.fetch = vi.fn(async () =>
			Response.json({
				organic: [
					{
						title: "Long result",
						link: "https://example.com/long",
						snippet: "x".repeat(500),
					},
				],
			}),
		);
		const tool = createWebSearchTool({
			apiKey: "test-key",
			maxOutputChars: 100,
			fetch: fetchMock,
		});

		const result = await tool.execute("search-1", { query: "example" });
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";

		expect(text.length).toBe(100);
		expect(text).toContain("[Search results truncated]");
		expect(result.details.truncated).toBe(true);
	});

	it("reads the output limit from SERPER_MAX_OUTPUT_CHARS", async () => {
		const fetchMock: typeof globalThis.fetch = vi.fn(async () =>
			Response.json({
				organic: [
					{
						title: "Long result",
						link: "https://example.com/long",
						snippet: "x".repeat(500),
					},
				],
			}),
		);
		const tool = createWebSearchTool({
			env: { SERPER_API_KEY: "test-key", SERPER_MAX_OUTPUT_CHARS: "120" },
			fetch: fetchMock,
		});

		const result = await tool.execute("search-1", { query: "example" });
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";

		expect(text.length).toBe(120);
		expect(text).toContain("[Search results truncated]");
		expect(result.details.truncated).toBe(true);
	});

	it("detects configuration and registers web_search with all built-in tools", () => {
		expect(isSerperSearchConfigured({ env: {} })).toBe(false);
		expect(isSerperSearchConfigured({ env: { SERPER_API_KEY: "replace-me" } })).toBe(false);
		expect(isSerperSearchConfigured({ env: { SERPER_API_KEY: "configured-key" } })).toBe(true);
		expect(createAllToolDefinitions("/workspace").web_search.name).toBe("web_search");
	});
});
