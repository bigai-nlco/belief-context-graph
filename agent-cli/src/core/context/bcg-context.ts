import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import { BcgClient, type BcgTurnPayload } from "./bcg-client.ts";
import type { BcgSnapshot, BcgTurn } from "./bcg-contract.types.ts";
import { bashExecutionToText } from "../messages.ts";

const GRAPH_PREFIX =
	"The following belief graph captures the reasoning trajectory so far. " +
	"These are preliminary beliefs derived from prior turns and may contain errors or incomplete information. " +
	"Use them to guide the next action, but do not treat them as verified evidence.";

export const BCG_TURN_LIMIT_MARKER = "BCG_TURN_LIMIT_EXCEEDED";

export class BcgTurnLimitError extends Error {
	readonly submittedTurns: number;
	readonly maxTurns: number;

	constructor(submittedTurns: number, maxTurns: number) {
		super(
			`${BCG_TURN_LIMIT_MARKER}: Graph message limit ${maxTurns} reached ` +
				`after ${submittedTurns} submitted messages; this task is a failure.`,
		);
		this.name = "BcgTurnLimitError";
		this.submittedTurns = submittedTurns;
		this.maxTurns = maxTurns;
	}
}


export interface BcgContextManagerOptions {
	baseUrl: string;
	problemId: string;
	recentTurns: number;
	maxTurns: number;
	timeoutMs: number;
	includeRelations: boolean;
	getSystemPrompt: () => string;
	getInitialUserMessage?: () => AgentMessage | undefined;
	fetch?: typeof globalThis.fetch;
	onWarning?: (message: string) => void;
}

interface SerializedMessage {
	message: AgentMessage;
	payload: BcgTurnPayload | undefined;
}


function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function compactJson(value: unknown): string {
	if (value === null) {
		return "null";
	}
	if (Array.isArray(value)) {
		return `[${value.map((item) => compactJson(item)).join(", ")}]`;
	}
	if (isRecord(value)) {
		const fields = Object.entries(value)
			.filter(([, item]) => item !== undefined)
			.map(([key, item]) => `${JSON.stringify(key)}: ${compactJson(item)}`);
		return `{${fields.join(", ")}}`;
	}
	try {
		return JSON.stringify(value) ?? "null";
	} catch {
		return JSON.stringify(String(value));
	}
}

function contentToText(content: unknown): string {
	if (typeof content === "string") {
		return content;
	}
	if (!Array.isArray(content)) {
		return "";
	}

	const parts: string[] = [];
	for (const block of content) {
		if (!isRecord(block) || typeof block.type !== "string") {
			continue;
		}
		switch (block.type) {
			case "text":
				if (typeof block.text === "string") {
					parts.push(block.text);
				}
				break;
			case "image":
				parts.push("[Image omitted from BCG text context]");
				break;
			case "toolCall": {
				const name = typeof block.name === "string" ? block.name : "unknown";
				const args = "arguments" in block ? block.arguments : {};
				parts.push(
					`<tool_call>\n${compactJson({ name, arguments: args })}\n</tool_call>`,
				);
				break;
			}
		}
	}
	return parts.join("\n\n").trim();
}

function messageText(message: AgentMessage): string {
	switch (message.role) {
		case "user":
		case "assistant":
			return contentToText(message.content);
		case "toolResult": {
			const result = contentToText(message.content);
			return result ? `[Tool result: ${message.toolName}]\n${result}` : `[Tool result: ${message.toolName}]`;
		}
		case "bashExecution":
			return bashExecutionToText(message);
		case "custom":
			return contentToText(message.content);
		case "branchSummary":
			return message.summary;
		case "compactionSummary":
			return message.summary;
	}
}

function graphRole(message: AgentMessage): BcgTurnPayload["role"] | undefined {
	switch (message.role) {
		case "user":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return "user";
		case "assistant":
			return "assistant";
		case "toolResult":
		case "bashExecution":
			return "tool";
	}
}

function messageKey(message: AgentMessage): string {
	const timestamp = "timestamp" in message ? String(message.timestamp) : "";
	return `${message.role}\u0000${timestamp}\u0000${messageText(message)}`;
}

function hasAssistant(messages: AgentMessage[]): boolean {
	return messages.some((message) => message.role === "assistant");
}

/**
 * Split the transcript after the permanent initial user input into model turns.
 * A turn starts with an assistant message (or a later user input) and owns all
 * following tool results until the next assistant/user boundary.
 */
export function splitBcgTurns(messages: AgentMessage[]): AgentMessage[][] {
	const turns: AgentMessage[][] = [];
	let current: AgentMessage[] = [];

	for (const message of messages) {
		if (message.role === "user") {
			if (current.length > 0) {
				turns.push(current);
			}
			current = [message];
			continue;
		}

		if (message.role === "assistant" && hasAssistant(current)) {
			turns.push(current);
			current = [message];
			continue;
		}

		current.push(message);
	}

	if (current.length > 0) {
		turns.push(current);
	}
	return turns;
}

function partitionTurns(
	turns: AgentMessage[][],
	recentTurns: number,
): {
	evicted: AgentMessage[][];
	retained: AgentMessage[][];
} {
	if (recentTurns < 0) {
		return { evicted: [], retained: turns };
	}

	let completedTurnsToKeep = recentTurns;
	let cut = turns.length;
	while (cut > 0) {
		const turn = turns[cut - 1];
		if (!hasAssistant(turn)) {
			cut -= 1;
			continue;
		}
		if (completedTurnsToKeep <= 0) {
			break;
		}
		completedTurnsToKeep -= 1;
		cut -= 1;
	}
	return {
		evicted: turns.slice(0, cut),
		retained: turns.slice(cut),
	};
}

export function formatBcgMarkdown(snapshot: BcgSnapshot, includeRelations = true): string {
	const beliefs = Array.isArray(snapshot.beliefs) ? snapshot.beliefs : [];
	if (beliefs.length === 0) {
		return "";
	}

	const lines = ["## Belief Graph", "", GRAPH_PREFIX, ""];
	for (const belief of beliefs) {
		lines.push(`- **[${belief.id ?? "?"}]** ${belief.belief ?? ""}`);
	}

	const relations = includeRelations ? (snapshot.relations ?? []) : [];
	if (relations.length > 0) {
		lines.push("", "### Relations");
		for (const relation of relations) {
			const note = relation.note ? ` — ${relation.note}` : "";
			lines.push(
				`- [${relation.from_id ?? "?"}] → [${relation.to_id ?? "?"}] ` + `(${relation.type ?? "informs"})${note}`,
			);
		}
	}
	return lines.join("\n");
}

export class BcgContextManager {
	private readonly baseUrl: string;
	private readonly problemId: string;
	private readonly recentTurns: number;
	private readonly maxTurns: number;
	private readonly timeoutMs: number;
	private readonly includeRelations: boolean;
	private readonly getSystemPrompt: () => string;
	private readonly getInitialUserMessage?: () => AgentMessage | undefined;
	private readonly client: BcgClient;
	private readonly onWarning: (message: string) => void;
	private readonly sentMessages = new WeakSet<object>();
	private seeded = false;
	private released = false;
	private submittedTurns = 0;
	private warned = false;
	private graphText = "";
	private requestReady = false;
	private initialUserMessage: AgentMessage | undefined;

	constructor(options: BcgContextManagerOptions) {
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.problemId = options.problemId;
		this.recentTurns = Math.max(-1, Math.trunc(options.recentTurns));
		this.maxTurns = Math.max(1, Math.trunc(options.maxTurns));
		this.timeoutMs = Math.max(1, Math.trunc(options.timeoutMs));
		this.includeRelations = options.includeRelations;
		this.getSystemPrompt = options.getSystemPrompt;
		this.getInitialUserMessage = options.getInitialUserMessage;
		this.client = new BcgClient({ baseUrl: options.baseUrl, problemId: options.problemId, timeoutMs: options.timeoutMs, fetch: options.fetch });
		this.onWarning = options.onWarning ?? ((message) => console.warn(message));
	}

	async transform(messages: AgentMessage[], signal?: AbortSignal): Promise<AgentMessage[]> {
		this.requestReady = false;
		try {
			const initialUser = this.resolveInitialUser(messages);
			if (!initialUser) {
				throw new Error("the session has no initial user input");
			}

			if (!this.seeded) {
				const snapshot = await this.postTurns(
					[this.turnPayload("system", this.getSystemPrompt()), this.turnPayload("user", messageText(initialUser))],
					signal,
				);
				this.updateSnapshot(snapshot);
				this.seeded = true;
				this.sentMessages.add(initialUser);
			}

			const initialKey = messageKey(initialUser);
			let removedInitial = false;
			const rest = messages.filter((message) => {
				if (!removedInitial && messageKey(message) === initialKey) {
					removedInitial = true;
					return false;
				}
				return true;
			});
			const { evicted, retained } = partitionTurns(splitBcgTurns(rest), this.recentTurns);
			const serialized = evicted.flatMap((turn) => turn.map((message) => this.serialize(message)));
			const unsent = serialized.filter((message) => !this.sentMessages.has(message.message));
			const payloads = unsent.flatMap((message) => (message.payload ? [message.payload] : []));

			if (payloads.length > 0) {
				const snapshot = await this.postTurns(payloads, signal);
				this.updateSnapshot(snapshot);
			}
			for (const message of unsent) {
				this.sentMessages.add(message.message);
			}

			this.warned = false;
			this.requestReady = true;
			return [initialUser, ...retained.flat()];
		} catch (error) {
			if (error instanceof BcgTurnLimitError) {
				throw error;
			}
			if (!this.warned) {
				const detail = error instanceof Error ? error.message : String(error);
				this.onWarning(`[BCG context] ${detail}; using the complete raw context for this request.`);
				this.warned = true;
			}
			return messages;
		}
	}

	augmentSystemPrompt(systemPrompt: string | undefined): string | undefined {
		if (!this.requestReady || !this.graphText) {
			return systemPrompt;
		}
		const prefix = systemPrompt ? `${systemPrompt}\n\n` : "";
		return `${prefix}<belief_graph format="markdown">\n${this.graphText}\n</belief_graph>`;
	}

	async release(): Promise<Record<string, unknown> | undefined> {
		if (!this.seeded || this.released) {
			return undefined;
		}
		this.released = true;
		let tokenUsage: Record<string, unknown> | undefined;
		try {
			const snapshot = await this.client.finalize();
			this.updateSnapshot(snapshot);
			tokenUsage = snapshot.token_usage;
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			this.onWarning(`[BCG context] failed to finalize Graph session: ${detail}`);
		}
		try {
			await this.client.release();
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			this.onWarning(`[BCG context] failed to release Graph session: ${detail}`);
		}
		return tokenUsage;
	}

	private resolveInitialUser(messages: AgentMessage[]): AgentMessage | undefined {
		if (this.initialUserMessage) {
			return this.initialUserMessage;
		}
		const fromSession = this.getInitialUserMessage?.();
		if (fromSession?.role === "user") {
			this.initialUserMessage = fromSession;
			return fromSession;
		}
		const fromContext = messages.find((message) => message.role === "user");
		if (fromContext) {
			this.initialUserMessage = fromContext;
		}
		return this.initialUserMessage;
	}

	private serialize(message: AgentMessage): SerializedMessage {
		const role = graphRole(message);
		const content = messageText(message);
		return {
			message,
			payload: role && content ? this.turnPayload(role, content) : undefined,
		};
	}

	private turnPayload(role: BcgTurnPayload["role"], content: string): BcgTurnPayload {
		return {
			problem_id: this.problemId,
			role,
			content,
			is_message_end: true,
			is_trajectory_end: false,
		};
	}

	private async postTurns(payloads: BcgTurnPayload[], signal?: AbortSignal): Promise<BcgSnapshot> {
		if (this.submittedTurns + payloads.length > this.maxTurns) {
			throw new BcgTurnLimitError(this.submittedTurns, this.maxTurns);
		}
		const snapshot = await this.client.postTurns(payloads, signal);
		this.submittedTurns += payloads.length;
		return snapshot;
	}

	private updateSnapshot(snapshot: BcgSnapshot): void {
		this.graphText = formatBcgMarkdown(snapshot, this.includeRelations);
	}
}
