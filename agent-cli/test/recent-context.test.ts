import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import type { AssistantMessage, ToolResultMessage, Usage } from "@bigai-nlco/bcg-ai/compat";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RagContextManager, RecentOnlyContextManager } from "../src/core/context/recent-context.ts";

const USAGE: Usage = {
	input: 1,
	output: 1,
	cacheRead: 0,
	cacheWrite: 0,
	reasoning: 0,
	totalTokens: 2,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const temporaryDirectories: string[] = [];

afterEach(() => {
	for (const directory of temporaryDirectories.splice(0)) {
		rmSync(directory, { recursive: true, force: true });
	}
});

function temporaryDatabase(): string {
	const directory = mkdtempSync(join(tmpdir(), "bcg-rag-context-"));
	temporaryDirectories.push(directory);
	return join(directory, "history.sqlite");
}

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

function tool(text: string, timestamp: number): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: `call-${timestamp}`,
		toolName: "web_search",
		content: [{ type: "text", text }],
		isError: false,
		timestamp,
	};
}

describe("recent-only context management", () => {
	it("permanently pins the initial user input and keeps two recent completed turns", async () => {
		const initial = user("original task that must remain visible", 1);
		const first = [assistant("first action", 2), tool("first evidence", 3)];
		const second = [assistant("second action", 4), tool("second evidence", 5)];
		const third = [assistant("third action", 6), tool("third evidence", 7)];
		const manager = new RecentOnlyContextManager({
			recentTurns: 2,
			getInitialUserMessage: () => initial,
		});

		const transformed = await manager.transform([initial, ...first, ...second, ...third]);

		expect(transformed).toEqual([initial, ...second, ...third]);
		expect(transformed).not.toContain(first[0]);
		expect(manager.augmentSystemPrompt("unchanged system prompt")).toBe("unchanged system prompt");
	});
});

describe("RAG context management", () => {
	it("stores evicted turns, retrieves them from SQLite, and injects them into the system prompt", async () => {
		const traces: Array<{ storedTurns: number; retrievedTurns: number }> = [];
		const databasePath = temporaryDatabase();
		const initial = user("original task that must remain visible", 1);
		const earlier = [
			assistant("Investigate Saturn ring composition", 2),
			tool("Saturn's rings are predominantly water ice", 3),
		];
		const recent = [
			assistant("Use the Saturn evidence to prepare the answer", 4),
			tool("The question asks specifically about Saturn", 5),
		];
		const manager = new RagContextManager({
			recentTurns: 1,
			databasePath,
			topK: 3,
			getInitialUserMessage: () => initial,
			onRagContext: (trace) => traces.push(trace),
		});

		const transformed = await manager.transform([initial, ...earlier, ...recent]);
		const augmented = manager.augmentSystemPrompt("base system prompt");

		expect(transformed).toEqual([initial, ...recent]);
		expect(augmented).toContain("base system prompt");
		expect(augmented).toContain("<retrieved_history_guide>");
		expect(augmented).toContain("Saturn's rings are predominantly water ice");
		expect(augmented).not.toContain("original task that must remain visible");
		expect(augmented?.match(/base system prompt/g)).toHaveLength(1);
		expect(traces.at(-1)).toMatchObject({ storedTurns: 1, retrievedTurns: 1 });

		await manager.transform([initial, ...earlier, ...recent]);
		expect(traces.at(-1)).toMatchObject({ storedTurns: 1, retrievedTurns: 1 });
		manager.release();
	});

	it("persists history across manager instances without duplicating the pinned user input", async () => {
		const databasePath = temporaryDatabase();
		const initial = user("persistent original task", 1);
		const earlier = [assistant("Find the Zephyr codename", 2), tool("Zephyr maps to project Aurora", 3)];
		const recent = [assistant("Recall the Zephyr mapping", 4), tool("Need the Aurora answer", 5)];
		const firstManager = new RagContextManager({ recentTurns: 1, databasePath });
		await firstManager.transform([initial, ...earlier, ...recent]);
		firstManager.release();

		const traces: Array<{ storedTurns: number }> = [];
		const resumedManager = new RagContextManager({
			recentTurns: 1,
			databasePath,
			onRagContext: (trace) => traces.push(trace),
		});
		await resumedManager.transform([initial, ...earlier, ...recent]);

		expect(resumedManager.augmentSystemPrompt("system")).toContain("Zephyr maps to project Aurora");
		expect(traces.at(-1)?.storedTurns).toBe(1);
		resumedManager.release();
	});

	it("falls back to the pinned recent-only window when the database cannot be opened", async () => {
		const warning = vi.fn();
		const directoryPath = temporaryDatabase().replace(/\/history\.sqlite$/, "");
		const initial = user("task", 1);
		const first = [assistant("old work", 2), tool("old result", 3)];
		const recent = [assistant("new work", 4), tool("new result", 5)];
		const manager = new RagContextManager({
			recentTurns: 1,
			databasePath: directoryPath,
			onWarning: warning,
		});

		const transformed = await manager.transform([initial, ...first, ...recent]);

		expect(transformed).toEqual([initial, ...recent]);
		expect(manager.augmentSystemPrompt("system")).toBe("system");
		expect(warning).toHaveBeenCalledOnce();
		manager.release();
	});
});
