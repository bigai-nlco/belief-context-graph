import type { AssistantMessage, Model, Usage } from "@bigai-nlco/bcg-ai/compat";
import type { ResponseStreamEvent } from "openai/resources/responses/responses.js";
import { describe, expect, it } from "vitest";
import {
	processResponsesStream,
	resolveResponsesReasoningContext,
} from "../packages/ai/src/api/openai-responses-shared.ts";
import { AssistantMessageEventStream } from "../packages/ai/src/utils/event-stream.ts";

const EMPTY_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistant(thinkingSignature?: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "thinking", thinking: "readable summary", thinkingSignature }],
		api: "openai-responses",
		provider: "openrouter",
		model: "gpt-5.6-luna",
		usage: EMPTY_USAGE,
		stopReason: "stop",
		timestamp: 1,
	};
}

describe("OpenAI Responses stateless reasoning replay", () => {
	it("uses current_turn before a persisted reasoning item exists", () => {
		expect(resolveResponsesReasoningContext({ messages: [] })).toBe("current_turn");
		expect(resolveResponsesReasoningContext({ messages: [assistant("opaque-chat-signature")] })).toBe(
			"current_turn",
		);
	});

	it("uses all_turns once an exact Responses reasoning item is replayed", () => {
		const item = {
			id: "rs_123",
			type: "reasoning",
			summary: [{ type: "summary_text", text: "Readable summary" }],
			encrypted_content: "opaque-encrypted-content",
		};
		expect(
			resolveResponsesReasoningContext({ messages: [assistant(JSON.stringify(item))] }),
		).toBe("all_turns");
	});

	it("takes the complete summary and encrypted item from terminal output while preserving tool-call fields", async () => {
		const earlyReasoning = {
			id: "rs_123",
			type: "reasoning",
			summary: [],
		};
		const completeReasoning = {
			...earlyReasoning,
			summary: [{ type: "summary_text", text: "Verify the primary source before answering." }],
			encrypted_content: "terminal-encrypted-content",
		};
		const functionCall = {
			type: "function_call",
			id: "fc_123",
			call_id: "call_123",
			name: "web_search",
			arguments: '{"query":"primary source","page":2}',
		};
		const rawEvents = [
			{ type: "response.output_item.added", output_index: 0, item: earlyReasoning },
			{ type: "response.output_item.done", output_index: 0, item: earlyReasoning },
			{ type: "response.output_item.added", output_index: 1, item: functionCall },
			{ type: "response.output_item.done", output_index: 1, item: functionCall },
			{
				type: "response.completed",
				response: {
					id: "resp_123",
					status: "completed",
					output: [completeReasoning, functionCall],
					usage: {
						input_tokens: 10,
						output_tokens: 5,
						total_tokens: 15,
						output_tokens_details: { reasoning_tokens: 3 },
					},
				},
			},
		];
		async function* events(): AsyncGenerator<Record<string, unknown>> {
			for (const event of rawEvents) yield event;
		}

		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: "openai-responses",
			provider: "openrouter",
			model: "gpt-5.6-luna",
			usage: EMPTY_USAGE,
			stopReason: "pending",
			timestamp: 1,
		};
		const model: Model<"openai-responses"> = {
			id: "gpt-5.6-luna",
			name: "gpt-5.6-luna",
			api: "openai-responses",
			provider: "openrouter",
			baseUrl: "https://openrouter.ai/api/v1",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 400_000,
			maxTokens: 128_000,
		};

		await processResponsesStream(
			events() as unknown as AsyncIterable<ResponseStreamEvent>,
			output,
			new AssistantMessageEventStream(),
			model,
		);

		const thinking = output.content.find((block) => block.type === "thinking");
		const toolCall = output.content.find((block) => block.type === "toolCall");
		expect(thinking).toMatchObject({
			type: "thinking",
			thinking: "Verify the primary source before answering.",
			thinkingSignature: JSON.stringify(completeReasoning),
		});
		expect(toolCall).toMatchObject({
			type: "toolCall",
			id: "call_123|fc_123",
			name: "web_search",
			arguments: { query: "primary source", page: 2 },
		});
	});
});
