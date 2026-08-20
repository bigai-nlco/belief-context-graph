import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import type { AssistantMessage, Context, Usage } from "@bigai-nlco/bcg-ai/compat";
import {
	contextMessageKey,
	contextMessageText,
	partitionContextTurns,
	splitBcgTurns,
} from "./bcg-context.ts";

const SUMMARY_SYSTEM_PROMPT = `You maintain one rolling summary of an agent's investigation.

Update the previous summary using only the newly evicted messages. Preserve the original task objective, established facts, tentative hypotheses, tool names and search queries, useful tool-result evidence, failed searches, contradictions, unresolved questions, and the latest intended next step. Clearly distinguish verified evidence from tentative claims. Remove repetition and obsolete procedural detail, but never invent facts or silently strengthen uncertainty. Do not solve the task or address the user. Output only the updated summary in concise Markdown.`;

const SUMMARY_CONTEXT_GUIDE = `<context_summary_guide>
Earlier completed turns have been omitted from the raw conversation and compressed into the rolling summary below. The summary may be incomplete or mistaken and is not verified evidence. Use it to preserve progress, avoid repeating searches and actions already attempted, and identify the most useful next step. Prefer current raw messages when they conflict with the summary.
</context_summary_guide>`;

export interface SummaryContextTrace {
	revision: number;
	evictedMessages: number;
	chars: number;
	text: string;
}

export interface SummaryContextUsage {
	llm_totals: {
		input_tokens: number;
		output_tokens: number;
		cache_read_tokens: number;
		cache_write_tokens: number;
		reasoning_tokens: number;
		total_tokens: number;
	};
	cost: {
		input: number;
		output: number;
		cache_read: number;
		cache_write: number;
		total: number;
	};
	wall_time_seconds: number;
	updates: number;
}

export interface SummaryContextManagerOptions {
	recentTurns: number;
	complete: (context: Context, signal?: AbortSignal) => Promise<AssistantMessage>;
	getInitialUserMessage?: () => AgentMessage | undefined;
	onWarning?: (message: string) => void;
	onSummaryContext?: (trace: SummaryContextTrace) => void;
}

interface MutableUsage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	reasoning: number;
	total: number;
	costInput: number;
	costOutput: number;
	costCacheRead: number;
	costCacheWrite: number;
	costTotal: number;
}

function emptyUsage(): MutableUsage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		reasoning: 0,
		total: 0,
		costInput: 0,
		costOutput: 0,
		costCacheRead: 0,
		costCacheWrite: 0,
		costTotal: 0,
	};
}

function addUsage(target: MutableUsage, usage: Usage): void {
	target.input += usage.input;
	target.output += usage.output;
	target.cacheRead += usage.cacheRead;
	target.cacheWrite += usage.cacheWrite;
	target.reasoning += usage.reasoning ?? 0;
	target.total +=
		usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
	target.costInput += usage.cost.input;
	target.costOutput += usage.cost.output;
	target.costCacheRead += usage.cost.cacheRead;
	target.costCacheWrite += usage.cost.cacheWrite;
	target.costTotal += usage.cost.total;
}

function summaryRole(message: AgentMessage): string {
	switch (message.role) {
		case "toolResult":
		case "bashExecution":
			return "tool";
		case "branchSummary":
		case "compactionSummary":
		case "custom":
			return "user";
		default:
			return message.role;
	}
}

function renderEvictedMessages(messages: AgentMessage[]): string {
	return messages
		.map((message, index) => {
			const content = contextMessageText(message);
			return `### Message ${index + 1} (${summaryRole(message)})\n${content || "(empty)"}`;
		})
		.join("\n\n");
}

function updatePrompt(previousSummary: string, evicted: AgentMessage[]): string {
	return `## Previous rolling summary
${previousSummary || "(none — initialize the summary)"}

## Newly evicted messages
${renderEvictedMessages(evicted)}`;
}

function assistantText(message: AssistantMessage): string {
	return message.content
		.filter((block) => block.type === "text")
		.map((block) => block.text)
		.join("\n")
		.trim();
}

/**
 * Rolling-summary context with exactly the same pinned-input and turn eviction
 * semantics as BCG. Only the memory backend differs.
 */
export class SummaryContextManager {
	private readonly recentTurns: number;
	private readonly complete: SummaryContextManagerOptions["complete"];
	private readonly getInitialUserMessage?: () => AgentMessage | undefined;
	private readonly onWarning: (message: string) => void;
	private readonly onSummaryContext?: (trace: SummaryContextTrace) => void;
	private readonly sentMessages = new WeakSet<object>();
	private readonly usage = emptyUsage();
	private initialUserMessage: AgentMessage | undefined;
	private summary = "";
	private seeded = false;
	private requestReady = false;
	private warned = false;
	private released = false;
	private revision = 0;
	private wallTimeMs = 0;

	constructor(options: SummaryContextManagerOptions) {
		this.recentTurns = Math.max(-1, Math.trunc(options.recentTurns));
		this.complete = options.complete;
		this.getInitialUserMessage = options.getInitialUserMessage;
		this.onWarning = options.onWarning ?? ((message) => console.warn(message));
		this.onSummaryContext = options.onSummaryContext;
	}

	async transform(messages: AgentMessage[], signal?: AbortSignal): Promise<AgentMessage[]> {
		this.requestReady = false;
		try {
			const initialUser = this.resolveInitialUser(messages);
			if (!initialUser) {
				throw new Error("the session has no initial user input");
			}

			if (!this.seeded) {
				await this.updateSummary([initialUser], signal);
				this.seeded = true;
				this.sentMessages.add(initialUser);
			}

			const initialKey = contextMessageKey(initialUser);
			let removedInitial = false;
			const rest = messages.filter((message) => {
				if (!removedInitial && contextMessageKey(message) === initialKey) {
					removedInitial = true;
					return false;
				}
				return true;
			});
			const { evicted, retained } = partitionContextTurns(
				splitBcgTurns(rest),
				this.recentTurns,
			);
			const unsent = evicted
				.flat()
				.filter((message) => !this.sentMessages.has(message));
			if (unsent.length > 0) {
				// One update per eviction batch, even when it contains multiple tool
				// results. This is the summary-mode analogue of BCG same-batch ingest.
				await this.updateSummary(unsent, signal);
				for (const message of unsent) {
					this.sentMessages.add(message);
				}
			}

			this.warned = false;
			this.requestReady = true;
			return [initialUser, ...retained.flat()];
		} catch (error) {
			if (!this.warned) {
				const detail = error instanceof Error ? error.message : String(error);
				this.onWarning(
					`[Summary context] ${detail}; using the complete raw context for this request.`,
				);
				this.warned = true;
			}
			return messages;
		}
	}

	augmentSystemPrompt(systemPrompt: string | undefined): string | undefined {
		if (!this.requestReady || !this.summary) {
			return systemPrompt;
		}
		const block = `<context_summary>\n${this.summary}\n</context_summary>`;
		return [systemPrompt, SUMMARY_CONTEXT_GUIDE, block].filter(Boolean).join("\n\n");
	}

	release(): SummaryContextUsage | undefined {
		if (!this.seeded || this.released) {
			return undefined;
		}
		this.released = true;
		return {
			llm_totals: {
				input_tokens: this.usage.input,
				output_tokens: this.usage.output,
				cache_read_tokens: this.usage.cacheRead,
				cache_write_tokens: this.usage.cacheWrite,
				reasoning_tokens: this.usage.reasoning,
				total_tokens: this.usage.total,
			},
			cost: {
				input: this.usage.costInput,
				output: this.usage.costOutput,
				cache_read: this.usage.costCacheRead,
				cache_write: this.usage.costCacheWrite,
				total: this.usage.costTotal,
			},
			wall_time_seconds: this.wallTimeMs / 1000,
			updates: this.revision,
		};
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

	private async updateSummary(evicted: AgentMessage[], signal?: AbortSignal): Promise<void> {
		const started = Date.now();
		let response: AssistantMessage;
		try {
			response = await this.complete(
				{
					systemPrompt: SUMMARY_SYSTEM_PROMPT,
					messages: [
						{
							role: "user",
							content: updatePrompt(this.summary, evicted),
							timestamp: Date.now(),
						},
					],
					tools: [],
				},
				signal,
			);
		} finally {
			this.wallTimeMs += Date.now() - started;
		}
		if (response.stopReason === "error" || response.stopReason === "aborted") {
			throw new Error(response.errorMessage || `summary model stopped with ${response.stopReason}`);
		}
		const text = assistantText(response);
		if (!text) {
			throw new Error("summary model returned no text");
		}
		addUsage(this.usage, response.usage);
		this.summary = text;
		this.revision += 1;
		this.onSummaryContext?.({
			revision: this.revision,
			evictedMessages: evicted.length,
			chars: text.length,
			text,
		});
	}
}

export { SUMMARY_CONTEXT_GUIDE, SUMMARY_SYSTEM_PROMPT };
