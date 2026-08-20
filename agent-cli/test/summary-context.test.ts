import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import type { AssistantMessage, ToolResultMessage, Usage } from "@bigai-nlco/bcg-ai/compat";
import { describe, expect, it, vi } from "vitest";
import { SummaryContextManager } from "../src/core/context/summary-context.ts";

const USAGE: Usage = {
	input: 10,
	output: 4,
	cacheRead: 2,
	cacheWrite: 1,
	reasoning: 1,
	totalTokens: 17,
	cost: { input: 0.01, output: 0.02, cacheRead: 0.001, cacheWrite: 0.002, total: 0.033 },
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
		usage: USAGE,
		stopReason: "toolUse",
		timestamp,
	};
}

function tool(text: string, timestamp: number, id = `call-${timestamp}`): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: "web_search",
		content: [{ type: "text", text }],
		isError: false,
		timestamp,
	};
}

function summaryResponse(text: string, timestamp: number): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "test-api",
		provider: "summary-provider",
		model: "summary-model",
		usage: USAGE,
		stopReason: "stop",
		timestamp,
	};
}

describe("rolling summary context management", () => {
	it("pins the initial input and summarizes only newly evicted turns", async () => {
		const prompts: string[] = [];
		let revision = 0;
		const initial = user("original task", 1);
		const first = [assistant("first search", 2), tool("first evidence", 3)];
		const second = [assistant("second search", 4), tool("second evidence", 5)];
		const third = [assistant("third search", 6), tool("third evidence", 7)];
		const manager = new SummaryContextManager({
			recentTurns: 2,
			getInitialUserMessage: () => initial,
			complete: async (context) => {
				prompts.push(String(context.messages[0]?.content));
				revision += 1;
				return summaryResponse(`summary ${revision}`, 10 + revision);
			},
		});

		expect(await manager.transform([initial])).toEqual([initial]);
		const transformed = await manager.transform([initial, ...first, ...second, ...third]);

		expect(transformed).toEqual([initial, ...second, ...third]);
		expect(prompts).toHaveLength(2);
		expect(prompts[0]).toContain("(none — initialize the summary)");
		expect(prompts[0]).toContain("original task");
		expect(prompts[1]).toContain("summary 1");
		expect(prompts[1]).toContain("first search");
		expect(prompts[1]).toContain("first evidence");
		expect(prompts[1]).not.toContain("second search");

		const system = manager.augmentSystemPrompt("base system");
		expect(system).toContain("<context_summary_guide>");
		expect(system).toContain("<context_summary>\nsummary 2\n</context_summary>");
		expect(system?.match(/base system/g)).toHaveLength(1);
	});

	it("summarizes one tool-call batch with multiple results in one update", async () => {
		const prompts: string[] = [];
		const initial = user("task", 1);
		const batchedTurn = [
			assistant("search two sources", 2),
			tool("result one", 3, "one"),
			tool("result two", 4, "two"),
		];
		const latest = [assistant("continue", 5), tool("latest result", 6)];
		const manager = new SummaryContextManager({
			recentTurns: 1,
			complete: async (context) => {
				prompts.push(String(context.messages[0]?.content));
				return summaryResponse(`summary ${prompts.length}`, 10 + prompts.length);
			},
		});

		await manager.transform([initial]);
		await manager.transform([initial, ...batchedTurn, ...latest]);

		expect(prompts).toHaveLength(2);
		expect(prompts[1]).toContain("result one");
		expect(prompts[1]).toContain("result two");
	});

	it("falls back to full raw context when the summary model fails", async () => {
		const warning = vi.fn();
		const messages = [user("task", 1), assistant("work", 2), tool("evidence", 3)];
		const manager = new SummaryContextManager({
			recentTurns: 0,
			complete: async () => {
				throw new Error("provider unavailable");
			},
			onWarning: warning,
		});

		expect(await manager.transform(messages)).toBe(messages);
		expect(manager.augmentSystemPrompt("base")).toBe("base");
		expect(warning).toHaveBeenCalledOnce();
	});

	it("reports summary-model tokens, cost, updates, and latency once", async () => {
		const manager = new SummaryContextManager({
			recentTurns: 2,
			complete: async () => summaryResponse("summary", 2),
		});

		await manager.transform([user("task", 1)]);
		const usage = manager.release();

		expect(usage).toMatchObject({
			llm_totals: {
				input_tokens: 10,
				output_tokens: 4,
				cache_read_tokens: 2,
				cache_write_tokens: 1,
				reasoning_tokens: 1,
				total_tokens: 17,
			},
			cost: { input: 0.01, output: 0.02, cache_read: 0.001, cache_write: 0.002, total: 0.033 },
			updates: 1,
		});
		expect(manager.release()).toBeUndefined();
	});
});
