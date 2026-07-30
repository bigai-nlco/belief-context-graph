import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import type { AssistantMessage, ToolResultMessage, Usage } from "@bigai-nlco/bcg-ai/compat";
import { describe, expect, it, vi } from "vitest";
import {
	BcgContextManager,
	BcgTurnLimitError,
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
			"[Tool result: search]\nfirst evidence",
		]);

		const effectiveSystem = manager.augmentSystemPrompt("base system");
		expect(effectiveSystem).toContain('<belief_graph format="markdown">');
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
							problem: { beliefs: [], relations: [] },
						},
					}),
					{ status: 200 },
				);
			}) as typeof globalThis.fetch,
		});

		await manager.release();
		await manager.transform([initial]);
		await manager.release();
		await manager.release();

		expect(paths).toEqual(["/turns", "/release"]);
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

	it("resolves BCG settings and supports -1 as an unbounded raw context", () => {
		const settings = SettingsManager.inMemory({
			contextManagement: {
				provider: "bcg",
				bcg: {
					url: "http://bcg.example",
					recentTurns: -1,
					maxTurns: 100,
					timeoutMs: 1234,
					includeRelations: false,
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
				includeRelations: false,
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
