import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import type { AssistantMessage, ToolResultMessage, Usage } from "@bigai-nlco/bcg-ai/compat";
import { describe, expect, it, vi } from "vitest";
import {
	BcgContextManager,
	BcgTurnLimitError,
	formatBcgDialogueContext,
	formatCompactBcgDialogueContext,
	formatBcgMarkdown,
	splitBcgTurns,
} from "../src/core/context/bcg-context.ts";
import {
	ensureSessionContextMode,
	getSessionContextMode,
	hasSessionConversationStarted,
	setSessionContextMode,
} from "../src/core/context/context-mode.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";

const EMPTY_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function user(text: string, timestamp: number): AgentMessage {
	return { role: "user", content: text, timestamp };
}

function assistant(text: string, timestamp: number): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "test-api",
		provider: "test-provider",
		model: "test-model",
		usage: EMPTY_USAGE,
		stopReason: "toolUse",
		timestamp,
	};
}

function assistantToolCalls(timestamp: number): AssistantMessage {
	return {
		role: "assistant",
		content: [
			{
				type: "toolCall",
				id: "call-one",
				name: "web_search",
				arguments: { query: "first query" },
			},
			{
				type: "toolCall",
				id: "call-two",
				name: "read",
				arguments: { path: "notes.txt", offset: 10 },
			},
		],
		api: "test-api",
		provider: "test-provider",
		model: "test-model",
		usage: EMPTY_USAGE,
		stopReason: "toolUse",
		timestamp,
	};
}

function assistantThinkingToolCalls(timestamp: number): AssistantMessage {
	return {
		...assistantToolCalls(timestamp),
		content: [
			{ type: "thinking", thinking: "Compare both searches before deciding." },
			...assistantToolCalls(timestamp).content,
		],
	};
}

function tool(text: string, timestamp: number): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: `call-${timestamp}`,
		toolName: "search",
		content: [{ type: "text", text }],
		isError: false,
		timestamp,
	};
}

function toolForCall(
	text: string,
	timestamp: number,
	toolCallId: string,
	toolName: string,
): ToolResultMessage {
	return { ...tool(text, timestamp), toolCallId, toolName };
}

function createFetch(requests: Array<Record<string, unknown>[]>): typeof globalThis.fetch {
	let graphVersion = 0;
	return (async (_input, init) => {
		const body = JSON.parse(String(init?.body)) as Array<Record<string, unknown>>;
		requests.push(body);
		graphVersion += 1;
		return new Response(
			JSON.stringify({
				latest: {
					problem: {
						beliefs: [{ id: graphVersion, belief: `graph version ${graphVersion}` }],
						relations: [],
					},
				},
			}),
			{ status: 200, headers: { "content-type": "application/json" } },
		);
	}) as typeof globalThis.fetch;
}

describe("BCG context management", () => {
	it("keeps the initial user input and only sends evicted complete turns to BCG", async () => {
		const requests: Array<Record<string, unknown>[]> = [];
		const initial = user("initial question", 1);
		const first = [assistant("first answer", 2), tool("first evidence", 3)];
		const second = [assistant("second answer", 4), tool("second evidence", 5)];
		const third = [assistant("third answer", 6), tool("third evidence", 7)];
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 2,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "base system",
			fetch: createFetch(requests),
		});

		expect(await manager.transform([initial])).toEqual([initial]);
		const transformed = await manager.transform([initial, ...first, ...second, ...third]);

		expect(transformed).toEqual([initial, ...second, ...third]);
		expect(requests).toHaveLength(2);
		expect(requests[0].map((turn) => turn.role)).toEqual(["system", "user"]);
		expect(requests[1].map((turn) => turn.role)).toEqual(["assistant", "tool"]);
		expect(requests[1].map((turn) => turn.content)).toEqual([
			"first answer",
			'<tool_result>\n{"tool_call_id": "call-3", "name": "search", "is_error": false, "content": "first evidence"}\n</tool_result>',
		]);

		const effectiveSystem = manager.augmentSystemPrompt("base system");
		expect(effectiveSystem).toContain("<context_blocks_guide>");
		expect(effectiveSystem).toContain("<｜begin▁of▁sentence｜>");
		expect(effectiveSystem).toContain("<｜Assistant｜>");
		expect(effectiveSystem).not.toContain("<belief_graph");
		expect(effectiveSystem).toContain("earlier turns have been omitted from the raw conversation context");
		expect(effectiveSystem).toContain("A belief is a self-contained claim or reasoning unit");
		expect(effectiveSystem).toContain("confidence to judge how trustworthy its content is");
		expect(effectiveSystem).toContain("`A depends_on B` means A requires B");
		expect(effectiveSystem).toContain("`A supplements B` means A adds compatible detail or evidence");
		expect(effectiveSystem).toContain("`A contradicts B` means A conflicts with");
		expect(effectiveSystem).toContain("avoid repeating searches that were already performed");
		expect(effectiveSystem).toContain("graph version 2");
		expect(effectiveSystem?.match(/base system/g)).toHaveLength(1);
	});

	it("supports graph-only intermediate context while retaining the initial user input", async () => {
		const requests: Array<Record<string, unknown>[]> = [];
		const initial = user("permanent input", 1);
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 0,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: createFetch(requests),
		});

		await manager.transform([initial]);
		const transformed = await manager.transform([initial, assistant("old reasoning", 2), tool("old evidence", 3)]);

		expect(transformed).toEqual([initial]);
	});

	it("defines beliefs and all three relation types in compact context guidance", async () => {
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 2,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			graphView: "compact",
			graphSelection: "ranked",
			getSystemPrompt: () => "base system",
			fetch: createFetch([]),
		});

		await manager.transform([user("initial question", 1)]);
		const effectiveSystem = manager.augmentSystemPrompt("base system");

		expect(effectiveSystem).toContain(
			"A belief is a self-contained fact, hypothesis, intermediate conclusion, or decision",
		);
		expect(effectiveSystem).toContain("`A depends_on B` means A requires B");
		expect(effectiveSystem).toContain("`A supplements B` adds compatible detail or evidence");
		expect(effectiveSystem).toContain("`A contradicts B` conflicts with");
		expect(effectiveSystem).toContain("Confidence estimates reliability, not answer relevance");
		expect(effectiveSystem).toContain("Relations record reasoning or provenance, not truth");
		expect(effectiveSystem).toContain("rationale for the link, not independent evidence");
		expect(effectiveSystem).toContain("follow outgoing relations to its premises");
		expect(effectiveSystem).toContain("incoming relations to later checks or results");
		expect(effectiveSystem).toContain("generic high-confidence fact is not an answer");
		expect(effectiveSystem).toContain("answer without re-verifying every clue");
		expect(effectiveSystem).toContain("one parallel tool-call batch");
	});

	it("uses the connected selector result to render compact graph context", async () => {
		const requests: Array<{ url: string; body: unknown }> = [];
		const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
			const url = String(input);
			const body = JSON.parse(String(init?.body));
			requests.push({ url, body });
			if (url.endsWith("/context-selection")) {
				return new Response(
					JSON.stringify({
						problem_id: "problem",
						strategy: "connected",
						retrieval: "embedding",
						node_ids: [2],
						relation_ids: [],
						node_chars: 40,
					}),
					{ status: 200 },
				);
			}
			return new Response(
				JSON.stringify({
					latest: {
						problem: {
							beliefs: [
								{ id: 1, belief: "hidden distractor", source: { turn_id: 3 } },
								{ id: 2, belief: "selected evidence", source: { turn_id: 3 } },
							],
							relations: [],
						},
					},
				}),
				{ status: 200 },
			);
		}) as typeof globalThis.fetch;
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 2,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			graphView: "compact",
			graphSelection: "connected",
			getSystemPrompt: () => "base system",
			fetch: fetchMock,
		});

		await manager.transform([user("initial question", 1)]);
		const effectiveSystem = manager.augmentSystemPrompt("base system");

		expect(requests.map((request) => request.url)).toEqual([
			"http://127.0.0.1:8848/turns",
			"http://127.0.0.1:8848/context-selection",
		]);
		expect(requests[1].body).toMatchObject({ query: "initial question" });
		expect(effectiveSystem).toContain("selected evidence");
		expect(effectiveSystem).not.toContain("hidden distractor");
	});

	it("uses a Tool-Result-free focus query for focused compact selection", async () => {
		const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
		const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
			const url = String(input);
			const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
			requests.push({ url, body });
			if (url.endsWith("/context-selection")) {
				return new Response(
					JSON.stringify({
						problem_id: "problem",
						strategy: "focused",
						retrieval: "embedding",
						node_ids: [2],
						relation_ids: [],
						node_chars: 40,
					}),
					{ status: 200 },
				);
			}
			return new Response(
				JSON.stringify({
					latest: {
						problem: {
							beliefs: [
								{ id: 1, belief: "hidden distractor", source: { turn_id: 3 } },
								{ id: 2, belief: "selected evidence", source: { turn_id: 3 } },
							],
							relations: [],
						},
					},
				}),
				{ status: 200 },
			);
		}) as typeof globalThis.fetch;
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 2,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			graphView: "compact",
			graphSelection: "focused",
			getSystemPrompt: () => "base system",
			fetch: fetchMock,
		});

		await manager.transform([
			user("permanent question", 1),
			assistant("compare the leading candidates", 2),
			tool("raw result that must not become retrieval intent", 3),
		]);

		const selection = requests.find((request) => request.url.endsWith("/context-selection"));
		expect(selection?.body.strategy).toBe("focused");
		expect(selection?.body.question).toBe("permanent question");
		expect(selection?.body.query).toContain("raw result that must not become retrieval intent");
		expect(selection?.body.focus_query).toContain("compare the leading candidates");
		expect(selection?.body.focus_query).not.toContain(
			"raw result that must not become retrieval intent",
		);
	});

	it("serializes tool calls with XML tags while preserving Agent tool names", async () => {
		const requests: Array<Record<string, unknown>[]> = [];
		const initial = user("initial", 1);
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 0,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: createFetch(requests),
		});

		await manager.transform([initial]);
		await manager.transform([initial, assistantToolCalls(2)]);

		expect(requests[1]).toHaveLength(1);
		expect(requests[1][0].content).toBe(
			'<tool_call>\n{"id": "call-one", "name": "web_search", "arguments": {"query": "first query"}}\n</tool_call>\n\n' +
				'<tool_call>\n{"id": "call-two", "name": "read", "arguments": {"path": "notes.txt", "offset": 10}}\n</tool_call>',
		);
	});

	it("keeps thinking and tool calls in one Assistant graph turn", async () => {
		const requests: Array<Record<string, unknown>[]> = [];
		const initial = user("initial", 1);
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 0,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: createFetch(requests),
		});

		await manager.transform([initial]);
		await manager.transform([initial, assistantThinkingToolCalls(2)]);

		expect(requests[1]).toHaveLength(1);
		expect(requests[1][0].role).toBe("assistant");
		expect(requests[1][0].content).toContain(
			"<thinking>\nCompare both searches before deciding.\n</thinking>",
		);
		expect(requests[1][0].content).toContain("<tool_call>");
	});

	it("groups parallel tool results into one ID-bearing graph turn", async () => {
		const requests: Array<Record<string, unknown>[]> = [];
		const initial = user("initial", 1);
		const assistantMessage = assistantToolCalls(2);
		const firstResult = toolForCall("first evidence", 3, "call-one", "web_search");
		const secondResult = toolForCall("second evidence", 4, "call-two", "read");
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 0,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: createFetch(requests),
		});

		await manager.transform([initial]);
		await manager.transform([initial, assistantMessage, firstResult, secondResult]);

		expect(requests[1].map((turn) => turn.role)).toEqual(["assistant", "tool"]);
		const grouped = String(requests[1][1].content);
		expect(grouped.match(/<tool_result>/g)).toHaveLength(2);
		expect(grouped).toContain('"tool_call_id": "call-one"');
		expect(grouped).toContain('"tool_call_id": "call-two"');
	});

	it("preserves a later user input until its assistant turn exists", async () => {
		const initial = user("initial", 1);
		const followUp = user("follow up", 4);
		const turns = splitBcgTurns([
			assistant("first", 2),
			tool("first result", 3),
			followUp,
			assistant("second", 5),
			tool("second result", 6),
			assistant("third", 7),
			tool("third result", 8),
		]);

		expect(turns).toEqual([
			[assistant("first", 2), tool("first result", 3)],
			[followUp, assistant("second", 5), tool("second result", 6)],
			[assistant("third", 7), tool("third result", 8)],
		]);
		expect(initial.role).toBe("user");
	});

	it("falls back to the complete context and original system prompt when BCG fails", async () => {
		const warning = vi.fn();
		const initial = user("initial", 1);
		const messages = [initial, assistant("answer", 2), tool("result", 3)];
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 0,
			maxTurns: 100,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: vi.fn(async () => {
				throw new Error("connection refused");
			}),
			onWarning: warning,
		});

		expect(await manager.transform(messages)).toBe(messages);
		expect(manager.augmentSystemPrompt("system")).toBe("system");
		expect(warning).toHaveBeenCalledOnce();
	});

	it("terminates instead of falling back when the Graph message limit would be exceeded", async () => {
		const requests: Array<Record<string, unknown>[]> = [];
		const warning = vi.fn();
		const initial = user("initial", 1);
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 0,
			maxTurns: 3,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: createFetch(requests),
			onWarning: warning,
		});

		await manager.transform([initial]);
		await expect(
			manager.transform([initial, assistant("reasoning", 2), tool("evidence", 3)]),
		).rejects.toBeInstanceOf(BcgTurnLimitError);
		expect(requests).toHaveLength(1);
		expect(warning).not.toHaveBeenCalled();
	});

	it("releases a seeded Graph session exactly once", async () => {
		const paths: string[] = [];
		const initial = user("initial", 1);
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 2,
			maxTurns: 300,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: (async (input) => {
				const path = new URL(String(input)).pathname;
				paths.push(path);
				if (path === "/release") {
					return new Response(JSON.stringify({ problem_id: "problem", released: true }), { status: 200 });
				}
				return new Response(
					JSON.stringify({
						latest: {
							problem: {
								beliefs: [],
								relations: [],
								token_usage: {
									llm_totals: {
										input_tokens: 20,
										output_tokens: 7,
										reasoning_tokens: 3,
									},
								},
							},
						},
					}),
					{ status: 200 },
				);
			}) as typeof globalThis.fetch,
		});

		await manager.release();
		await manager.transform([initial]);
		const usage = await manager.release();
		await manager.release();

		expect(paths).toEqual(["/turns", "/finalize", "/release"]);
		expect(usage).toEqual({
			llm_totals: { input_tokens: 20, output_tokens: 7, reasoning_tokens: 3 },
		});
	});

	it("ingests every final unsent message before finalizing the Graph session", async () => {
		const requests: Array<{ path: string; body: unknown }> = [];
		let graphRequestCount = 0;
		const initial = user("initial", 1);
		const firstAssistant = assistant("searching", 2);
		const evidence = tool("evidence", 3);
		const finalAssistant = assistant("FINAL ANSWER: result", 4);
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 2,
			maxTurns: 300,
			timeoutMs: 1000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			fetch: (async (input, init) => {
				const path = new URL(String(input)).pathname;
				requests.push({
					path,
					body: init?.body ? JSON.parse(String(init.body)) : undefined,
				});
				if (path === "/release") {
					return new Response(JSON.stringify({ problem_id: "problem", released: true }), { status: 200 });
				}
				graphRequestCount += 1;
				const inputTokens = graphRequestCount === 1 ? 10 : graphRequestCount === 2 ? 90 : 100;
				return new Response(
					JSON.stringify({
						latest: {
							problem: {
								beliefs: [],
								relations: [],
								token_usage: {
									llm_totals: {
										input_tokens: inputTokens,
										output_tokens: inputTokens / 10,
										total_tokens: inputTokens + inputTokens / 10,
									},
								},
							},
						},
					}),
					{ status: 200 },
				);
			}) as typeof globalThis.fetch,
		});

		await manager.transform([initial]);
		const usage = await manager.release([initial, firstAssistant, evidence, finalAssistant]);

		expect(requests.map((request) => request.path)).toEqual([
			"/turns",
			"/turns",
			"/finalize",
			"/release",
		]);
		expect(requests[1]?.body).toEqual([
			expect.objectContaining({ role: "assistant", content: "searching" }),
			expect.objectContaining({
				role: "tool",
				content:
					'<tool_result>\n{"tool_call_id": "call-3", "name": "search", "is_error": false, "content": "evidence"}\n</tool_result>',
			}),
			expect.objectContaining({ role: "assistant", content: "FINAL ANSWER: result" }),
		]);
		expect(usage).toEqual({
			llm_totals: { input_tokens: 10, output_tokens: 1, total_tokens: 11 },
		});
	});

	it("uses the finalization timeout and does not label final supplement failure as runtime fallback", async () => {
		const warnings: string[] = [];
		const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
		let turns = 0;
		const initial = user("initial", 1);
		const manager = new BcgContextManager({
			baseUrl: "http://127.0.0.1:8848",
			problemId: "problem",
			recentTurns: 2,
			maxTurns: 300,
			timeoutMs: 1000,
			finalizationTimeoutMs: 9000,
			includeRelations: true,
			getSystemPrompt: () => "system",
			onWarning: (warning) => warnings.push(warning),
			fetch: (async (input) => {
				const path = new URL(String(input)).pathname;
				if (path === "/turns" && turns++ > 0) throw new Error("supplement timeout");
				if (path === "/release") {
					return new Response(JSON.stringify({ problem_id: "problem", released: true }), { status: 200 });
				}
				return new Response(
					JSON.stringify({ latest: { problem: { beliefs: [], relations: [] } } }),
					{ status: 200 },
				);
			}) as typeof globalThis.fetch,
		});

		await manager.transform([initial]);
		await manager.release([initial, assistant("FINAL ANSWER: result", 2)]);

		expect(warnings).toEqual([
			expect.stringContaining("[BCG finalization] failed to ingest final unsent messages"),
		]);
		expect(warnings.join("\n")).not.toContain("using the complete raw context");
		expect(timeoutSpy).toHaveBeenCalledWith(9000);
		timeoutSpy.mockRestore();
	});

	it("formats relations as Markdown", () => {
		expect(
			formatBcgMarkdown({
				beliefs: [
					{ id: 1, belief: "first" },
					{ id: 2, belief: "second" },
				],
				relations: [{ from_id: 1, to_id: 2, type: "supports", note: "evidence" }],
			}),
		).toContain("- [1] → [2] (supports) — evidence");
	});

	it("encodes graph beliefs as Markdown with the generic dialogue context template", () => {
		const encoded = formatBcgDialogueContext({
			beliefs: [
				{ id: 1, belief: "user belief", role: "user", confidence: 0.9 },
				{ id: 2, belief: "assistant belief", role: "assistant", confidence: 0.8 },
				{ id: 3, belief: "tool belief", role: "tool" },
			],
			relations: [{ from_id: 1, to_id: 2, type: "depends_on", note: "because" }],
		});

		expect(encoded).toBe(
			"<｜begin▁of▁sentence｜><｜User｜>### Belief 1\n" +
				"**Content:** user belief\n" +
				"**Relations:**\n" +
				"- direction=outgoing; to=2; type=depends_on; reason=because\n" +
				"**Confidence:** 0.9" +
				"<｜Assistant｜>### Belief 2\n" +
				"**Content:** assistant belief\n" +
				"**Relations:**\n" +
				"- None\n" +
				"**Confidence:** 0.8<｜end▁of▁sentence｜>" +
				"<｜User｜>### Belief 3\n" +
				"**Content:** tool belief\n" +
				"**Relations:**\n" +
				"- None\n" +
				"**Confidence:** ",
		);
	});

	it("records each relation only once on its source belief", () => {
		const encoded = formatBcgDialogueContext({
			beliefs: [
				{ id: 1, belief: "source", role: "user" },
				{ id: 2, belief: "target", role: "assistant" },
			],
			relations: [{ from_id: 1, to_id: 2, type: "supplements", note: "one edge" }],
		});

		expect(encoded.match(/type=supplements/g)).toHaveLength(1);
		expect(encoded).toContain("direction=outgoing; to=2; type=supplements; reason=one edge");
		expect(encoded).not.toContain("direction=incoming");
	});

	it("projects original graph beliefs and their retained relations into compact chat-marked context", () => {
		const encoded = formatCompactBcgDialogueContext({
			beliefs: [
				{ id: 1, belief: "duplicated initial request", source: { turn_id: 1 } },
				{
					id: 10,
					belief: 'The assistant is using web_search to search for "first historical query".',
					extraction_method: "rule_tool_call",
					tool_name: "web_search",
					query: "first historical query",
				},
				{
					id: 11,
					belief: "The source establishes the answer-relevant date as 1912.",
					extraction_method: "compact_llm_tool_result",
					tool_result_items: [
						{ title: "Source", url: "https://secret.example/full", snippet: "long raw result" },
					],
					confidence: 0.8,
				},
				{
					id: 12,
					belief: "large raw result",
					extraction_method: "rule_tool_result",
					tool_result_items: [
						{
							title: "Useful title",
							url: "https://secret.example/result",
							snippet: "A bounded useful snippet.",
						},
					],
				},
				{
					id: 13,
					belief: 'The assistant is using web_search to search for "second historical query".',
					extraction_method: "rule_tool_call",
					tool_name: "web_search",
					query: "second historical query",
				},
			],
			relations: [
				{ from_id: 11, to_id: 10, type: "depends_on", note: "provenance prose" },
				{ from_id: 12, to_id: 13, type: "depends_on", note: "provenance prose" },
			],
		});

		expect(encoded).toContain("<｜begin▁of▁sentence｜><｜User｜>");
		expect(encoded).toContain("<｜Assistant｜>");
		expect(encoded).not.toContain("<belief_context>");
		expect(encoded).toContain(
			'[B10] The assistant is using web_search to search for "first historical query".',
		);
		expect(encoded).toContain(
			'[B13] The assistant is using web_search to search for "second historical query".',
		);
		expect(encoded).not.toMatch(/\[Q\d+\]/);
		expect(encoded).toContain("[B11] The source establishes the answer-relevant date as 1912.");
		expect(encoded).toContain("[B12] large raw result");
		expect(encoded).not.toContain("→ evidence");
		expect(encoded).not.toContain("Useful title");
		expect(encoded).not.toContain("Continuation policy");
		expect(encoded).not.toContain("I will continue from this earlier investigation state");
		expect(encoded).not.toContain("secret.example");
		expect(encoded).not.toContain("provenance prose");
		expect(encoded).not.toContain("duplicated initial request");
		expect(encoded).toContain("#### Retained relations");
		expect(encoded).toContain("[B11] depends_on [B10]");
		expect(encoded).toContain("[B12] depends_on [B13]");
	});

	it("can omit compact relations without changing retained belief text", () => {
		const snapshot = {
			beliefs: [
				{ id: 2, belief: "candidate answer", confidence: 0.9 },
				{ id: 3, belief: "supporting fact", confidence: 0.7 },
			],
			relations: [{ id: 1, from_id: 2, to_id: 3, type: "depends_on" as const }],
		};

		const encoded = formatCompactBcgDialogueContext(snapshot, false);

		expect(encoded).toContain("[B2] candidate answer (confidence 0.90)");
		expect(encoded).toContain("[B3] supporting fact (confidence 0.70)");
		expect(encoded).not.toContain("Retained relations");
		expect(encoded).not.toContain("depends_on");
	});

	it("renders focused selections as evidence first and contiguous relation paths", () => {
		const encoded = formatCompactBcgDialogueContext(
			{
				beliefs: [
					{ id: 2, belief: "specific candidate", confidence: 0.61, source: { turn_id: 3 } },
					{ id: 3, belief: "candidate premise", confidence: 0.82, source: { turn_id: 4 } },
					{
						id: 4,
						belief: 'The assistant is using web_search to search for "candidate evidence".',
						extraction_method: "rule_tool_call",
						query: "candidate evidence",
						source: { turn_id: 5 },
					},
					{ id: 6, belief: "later search evidence", confidence: 0.88, source: { turn_id: 6 } },
				],
				relations: [
					{
						id: 20,
						from_id: 2,
						to_id: 3,
						type: "depends_on",
						note: "The candidate requires this premise.",
						weight: 0.5,
					},
					{
						id: 21,
						from_id: 4,
						to_id: 2,
						type: "depends_on",
						note: "The tool result was produced by the preceding tool call.",
						weight: 0,
					},
					{ id: 22, from_id: 6, to_id: 4, type: "depends_on" },
				],
			},
			true,
			new Set([2, 3, 4, 6]),
			new Set([20, 21, 22]),
			true,
		);

		expect(encoded).toContain("#### Candidate evidence");
		expect(encoded).toContain("#### Search history");
		expect(encoded).toContain("#### Relation paths");
		expect(encoded).not.toContain("#### Factual beliefs");
		expect(encoded).not.toContain("#### Search-history beliefs");
		expect(encoded).not.toContain("#### Retained relations");
		const evidence = encoded.indexOf("[B6] later search evidence");
		const premise = encoded.indexOf("[B3] candidate premise");
		const candidate = encoded.indexOf("[B2] specific candidate");
		const search = encoded.indexOf('[B4] The assistant is using web_search');
		const relationHeading = encoded.indexOf("#### Relation paths");
		const evidenceEdge = encoded.indexOf("[B6] depends_on [B4]");
		const incoming = encoded.indexOf("[B4] depends_on [B2]");
		const premiseEdge = encoded.indexOf("[B2] depends_on [B3]");
		expect(evidence).toBeLessThan(premise);
		expect(premise).toBeLessThan(candidate);
		expect(candidate).toBeLessThan(search);
		expect(search).toBeLessThan(relationHeading);
		expect(relationHeading).toBeLessThan(evidenceEdge);
		expect(evidenceEdge).toBeLessThan(incoming);
		expect(incoming).toBeLessThan(premiseEdge);
		expect(encoded).toContain(
			"[B2] depends_on [B3] — The candidate requires this premise.",
		);
		expect(encoded).not.toContain(
			"[B4] depends_on [B2] — The tool result was produced",
		);
	});

	it("keeps candidate ordering independent of a long relation chain", () => {
		const beliefs = Array.from({ length: 8 }, (_, index) => ({
			id: index + 1,
			belief: `belief ${index + 1}`,
			confidence: 0.8,
			source: { turn_id: index + 2 },
		}));
		const relations = Array.from({ length: 6 }, (_, index) => ({
			id: index + 1,
			from_id: index + 1,
			to_id: index + 2,
			type: "depends_on" as const,
		}));
		const encoded = formatCompactBcgDialogueContext(
			{ beliefs, relations },
			true,
			new Set([1, 8, 2, 3, 4, 5, 6, 7]),
			new Set(relations.map((relation) => relation.id)),
			true,
		);
		const nodeOrder = Array.from(encoded.matchAll(/^- \[B(\d+)\] belief/gm), (match) =>
			Number(match[1]),
		);

		expect(nodeOrder).toEqual([8, 7, 6, 5, 4, 3, 2, 1]);
		expect(new Set(nodeOrder)).toEqual(new Set([1, 2, 3, 4, 5, 6, 7, 8]));
		expect(encoded.indexOf("[B1] belief 1")).toBeLessThan(encoded.indexOf("#### Relation paths"));
	});

	it("ranks factual confidence before recency and treats queries as history", () => {
		const encoded = formatCompactBcgDialogueContext({
			beliefs: [
				{ id: 2, belief: "strong exact answer", confidence: 0.91, source: { turn_id: 3 } },
				{ id: 8, belief: "newer weak distractor", confidence: 0.78, source: { turn_id: 8 } },
				{
					id: 9,
					belief: 'The assistant is using web_search to search for "already tried".',
					confidence: 0.99,
					query: "already tried",
					extraction_method: "rule_tool_call",
					source: { turn_id: 9 },
				},
			],
			relations: [],
		});

		expect(encoded).toContain("#### Factual beliefs");
		expect(encoded).toContain("#### Search-history beliefs");
		expect(encoded.indexOf("[B2] strong exact answer")).toBeLessThan(
			encoded.indexOf("[B8] newer weak distractor"),
		);
		expect(encoded).toContain('[B9] The assistant is using web_search to search for "already tried".');
		expect(encoded).not.toContain("[B9] The assistant is using web_search to search for \"already tried\". (confidence");
	});

	it("omits empty search-result beliefs and relations connected to them from compact context", () => {
		const encoded = formatCompactBcgDialogueContext({
			beliefs: [
				{
					id: 20,
					belief: 'The assistant is using web_search to search for "rare source".',
					extraction_method: "rule_tool_call",
					query: "rare source",
					source: { turn_id: 4 },
				},
				{
					id: 21,
					belief: "The web_search tool returned no results.",
					extraction_method: "rule_tool_result",
					source: { turn_id: 5 },
				},
			],
			relations: [{ id: 1, from_id: 21, to_id: 20, type: "depends_on" }],
		});

		expect(encoded).toContain('[B20] The assistant is using web_search to search for "rare source".');
		expect(encoded).not.toContain("[B21]");
		expect(encoded).not.toContain("Retained relations");
		expect(encoded).not.toContain("depends_on");
	});

	it("keeps compact graph injection within its bounded rendering budget", () => {
		const beliefs = Array.from({ length: 80 }, (_, index) => ({
			id: index + 2,
			belief: `fact ${index} ${"x".repeat(300)}`,
			confidence: 0.78,
			source: { turn_id: index + 2 },
		}));

		const encoded = formatCompactBcgDialogueContext({ beliefs, relations: [] });

		expect(encoded.length).toBeLessThanOrEqual(8_100);
	});

	it("renders exact tool and query metadata for query-derived beliefs", () => {
		const encoded = formatBcgDialogueContext({
			beliefs: [
				{
					id: 7,
					belief: "The assistant searches for a World Bank statistic.",
					role: "assistant",
					tool_name: "web_search",
					query: "World Bank gross savings 2001",
					tool_arguments: { query: "World Bank gross savings 2001", num: 10 },
				},
			],
			relations: [],
		});

		expect(encoded).toContain("**Tool:** web_search");
		expect(encoded).toContain("**Query:** World Bank gross savings 2001");
		expect(encoded).toContain(
			'**Arguments:** {"query": "World Bank gross savings 2001", "num": 10}',
		);
	});

	it("resolves BCG settings and supports -1 as an unbounded raw context", () => {
		const settings = SettingsManager.inMemory({
			contextManagement: {
				provider: "bcg",
				bcg: {
					url: "http://bcg.example",
					recentTurns: -1,
					maxTurns: 100,
					timeoutMs: 1234,
					finalizationTimeoutMs: 5678,
					includeRelations: false,
					graphView: "compact",
					graphSelection: "ranked",
				},
			},
		});

		expect(settings.getContextManagementSettings()).toEqual({
			provider: "bcg",
			bcg: {
				url: "http://bcg.example",
				recentTurns: -1,
				maxTurns: 100,
				timeoutMs: 1234,
				finalizationTimeoutMs: 5678,
				includeRelations: false,
				graphView: "compact",
				graphSelection: "ranked",
			},
			summary: {
				provider: "summary",
				model: "",
				recentTurns: -1,
				timeoutMs: 300000,
				maxTokens: 2048,
				thinkingLevel: "off",
			},
			recentOnly: {
				recentTurns: -1,
			},
			rag: {
				recentTurns: -1,
				databasePath: "",
				topK: 6,
				maxChars: 12000,
			},
		});
	});

	it("pins context mode to the session and detects the first user message", () => {
		const session = SessionManager.inMemory();

		expect(ensureSessionContextMode(session, "default")).toBe("default");
		expect(getSessionContextMode(session)).toBe("default");
		expect(hasSessionConversationStarted(session)).toBe(false);

		setSessionContextMode(session, "bcg");
		expect(getSessionContextMode(session)).toBe("bcg");
		setSessionContextMode(session, "summary");
		expect(getSessionContextMode(session)).toBe("summary");
		setSessionContextMode(session, "recent-only");
		expect(getSessionContextMode(session)).toBe("recent-only");
		setSessionContextMode(session, "rag");
		expect(getSessionContextMode(session)).toBe("rag");

		session.appendMessage(user("first message", 1));
		expect(hasSessionConversationStarted(session)).toBe(true);
	});

	it("migrates legacy conversations to their original BCG mode", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(user("legacy message", 1));

		expect(ensureSessionContextMode(session, "default")).toBe("bcg");
		expect(getSessionContextMode(session)).toBe("bcg");
	});

	it("updates the configured default for future sessions", () => {
		const settings = SettingsManager.inMemory({
			contextManagement: { provider: "bcg" },
		});

		settings.setContextManagementProvider("default");
		expect(settings.getContextManagementSettings().provider).toBe("default");
	});
});
