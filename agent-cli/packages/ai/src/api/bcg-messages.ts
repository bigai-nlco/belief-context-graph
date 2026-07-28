/**
 * bcg-messages API implementation.
 *
 * Streams bcg's own message protocol directly to a backend: the request is a
 * single POST of `{ model, context, options }` to `<baseUrl>/messages`, the
 * response is an SSE stream of serialized assistant-message events plus a
 * terminal `done`/`error` event. This is the wire protocol spoken by the
 * Radius gateway, but any backend implementing it can be used, e.g. via a
 * models.json custom provider with `"api": "bcg-messages"`.
 */

import type {
	AssistantMessage,
	AssistantMessageEvent,
	CacheRetention,
	Context,
	Model,
	ProviderEnv,
	SimpleStreamOptions,
	StreamFunction,
	StreamOptions,
	ThinkingLevel,
	ToolCall,
} from "../types.ts";
import { appendAssistantMessageDiagnostic, createAssistantMessageDiagnostic } from "../utils/diagnostics.ts";
import { AssistantMessageEventStream } from "../utils/event-stream.ts";
import { headersToRecord, providerHeadersToRecord } from "../utils/headers.ts";
import { parseStreamingJson } from "../utils/json-parse.ts";
import { getProviderEnvValue } from "../utils/provider-env.ts";

export interface BcgMessagesOptions extends StreamOptions {
	reasoning?: ThinkingLevel;
	toolChoice?: "auto" | "none" | "required" | { type: "function"; function: { name: string } };
	/** Ask the backend for debug metadata (e.g. routing response headers). */
	debug?: boolean;
}

type BcgMessagesUsage = AssistantMessage["usage"];
type BcgMessagesStopReason = AssistantMessage["stopReason"];

/** Impact summary of a server-side message rewrite (e.g. a gateway policy). */
export type BcgMessagesRewriteImpact = {
	policyId: string;
	policyVersion: number;
	changed: boolean;
	tokenCountChange: number;
	messageCountChange: number;
	systemPromptChanged: boolean;
};

/** Serialized assistant-message event as sent by a bcg-messages backend. */
export type BcgMessagesEvent =
	| { type: "start" }
	| { type: "text_start"; contentIndex: number }
	| { type: "text_delta"; contentIndex: number; delta: string }
	| { type: "text_end"; contentIndex: number; content: string; contentSignature?: string }
	| { type: "thinking_start"; contentIndex: number }
	| { type: "thinking_delta"; contentIndex: number; delta: string }
	| {
			type: "thinking_end";
			contentIndex: number;
			content: string;
			contentSignature?: string;
			redacted?: boolean;
	  }
	| { type: "toolcall_start"; contentIndex: number; id: string; toolName: string }
	| { type: "toolcall_delta"; contentIndex: number; delta: string }
	| { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall }
	| {
			type: "done";
			reason: Extract<BcgMessagesStopReason, "stop" | "length" | "toolUse">;
			usage: BcgMessagesUsage;
			responseId?: string;
			rewrite?: BcgMessagesRewriteImpact;
	  }
	| {
			type: "error";
			reason: Extract<BcgMessagesStopReason, "aborted" | "error">;
			usage: BcgMessagesUsage;
			errorMessage?: string;
			responseId?: string;
			rewrite?: BcgMessagesRewriteImpact;
	  };

type BcgMessagesErrorBody = {
	error?: {
		message?: unknown;
		code?: unknown;
		details?: unknown;
		[key: string]: unknown;
	};
};

export class BcgMessagesResponseError extends Error {
	code?: string;
	readonly diagnosticDetails: Record<string, unknown>;

	constructor(message: string, code: string | undefined, diagnosticDetails: Record<string, unknown>) {
		super(message);
		this.name = "BcgMessagesResponseError";
		this.code = code;
		this.diagnosticDetails = diagnosticDetails;
	}
}

function parseBcgMessagesErrorBody(body: string): BcgMessagesErrorBody | undefined {
	try {
		const parsed = JSON.parse(body) as BcgMessagesErrorBody | null;
		const error = parsed?.error;
		return parsed && typeof error === "object" && error !== null && !Array.isArray(error) ? parsed : undefined;
	} catch {
		return undefined;
	}
}

function truncateDiagnosticString(value: string): string {
	const maxLength = 8192;
	return value.length > maxLength ? `${value.slice(0, maxLength)}…` : value;
}

function formatBcgMessagesResponseError(
	response: Response,
	body: string,
	errorBody: BcgMessagesErrorBody | undefined,
): string {
	const message = typeof errorBody?.error?.message === "string" ? errorBody.error.message : undefined;
	const code = typeof errorBody?.error?.code === "string" ? errorBody.error.code : undefined;
	const suffix = message ?? body;
	const codeSuffix = code ? ` (${code})` : "";
	return `${response.status} ${response.statusText}: ${suffix}${codeSuffix}`;
}

function createBcgMessagesResponseError(
	model: Model<"bcg-messages">,
	url: URL,
	response: Response,
	body: string,
): BcgMessagesResponseError {
	const errorBody = parseBcgMessagesErrorBody(body);
	const code = typeof errorBody?.error?.code === "string" ? errorBody.error.code : undefined;
	return new BcgMessagesResponseError(formatBcgMessagesResponseError(response, body, errorBody), code, {
		version: 1,
		provider: model.provider,
		model: model.id,
		url: url.toString(),
		status: response.status,
		statusText: response.statusText,
		error: errorBody?.error,
		body: errorBody ? undefined : truncateDiagnosticString(body),
		timestampMs: Date.now(),
	});
}

function createEmptyUsage(): BcgMessagesUsage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function appendRewriteDiagnostic(message: AssistantMessage, rewrite: BcgMessagesRewriteImpact | undefined): void {
	if (!rewrite) {
		return;
	}
	appendAssistantMessageDiagnostic(message, {
		type: "bcg_messages_rewrite",
		timestamp: Date.now(),
		details: { ...rewrite },
	});
}

function createEventConverter(model: Model<"bcg-messages">) {
	const partial: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: createEmptyUsage(),
		stopReason: "pending",
		timestamp: Date.now(),
	};
	const toolJson = new Map<number, string>();

	return (event: BcgMessagesEvent): AssistantMessageEvent => {
		switch (event.type) {
			case "done":
				Object.assign(partial, {
					stopReason: event.reason,
					usage: event.usage,
					responseId: event.responseId,
				});
				appendRewriteDiagnostic(partial, event.rewrite);
				return { type: "done", reason: event.reason, message: partial };
			case "error":
				Object.assign(partial, {
					stopReason: event.reason,
					usage: event.usage,
					errorMessage: event.errorMessage,
					responseId: event.responseId,
				});
				appendRewriteDiagnostic(partial, event.rewrite);
				return { type: "error", reason: event.reason, error: partial };
			case "start":
				break;
			case "text_start":
				partial.content[event.contentIndex] = { type: "text", text: "" };
				break;
			case "text_delta":
				(partial.content[event.contentIndex] as { text: string }).text += event.delta;
				break;
			case "text_end":
				Object.assign(partial.content[event.contentIndex]!, {
					text: event.content,
					textSignature: event.contentSignature,
				});
				break;
			case "thinking_start":
				partial.content[event.contentIndex] = { type: "thinking", thinking: "" };
				break;
			case "thinking_delta":
				(partial.content[event.contentIndex] as { thinking: string }).thinking += event.delta;
				break;
			case "thinking_end":
				Object.assign(partial.content[event.contentIndex]!, {
					thinking: event.content,
					thinkingSignature: event.contentSignature,
					redacted: event.redacted,
				});
				break;
			case "toolcall_start":
				partial.content[event.contentIndex] = {
					type: "toolCall",
					id: event.id,
					name: event.toolName,
					arguments: {},
				};
				toolJson.set(event.contentIndex, "");
				break;
			case "toolcall_delta": {
				const json = `${toolJson.get(event.contentIndex) ?? ""}${event.delta}`;
				toolJson.set(event.contentIndex, json);
				(partial.content[event.contentIndex] as ToolCall).arguments =
					parseStreamingJson<ToolCall["arguments"]>(json);
				break;
			}
			case "toolcall_end":
				Object.assign(partial.content[event.contentIndex]!, event.toolCall);
				toolJson.delete(event.contentIndex);
				return {
					type: "toolcall_end",
					contentIndex: event.contentIndex,
					toolCall: partial.content[event.contentIndex] as ToolCall,
					partial,
				};
		}

		return { ...event, partial } as AssistantMessageEvent;
	};
}

async function* readBcgMessagesEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<BcgMessagesEvent> {
	const decoder = new TextDecoder();
	const reader = stream.getReader();
	let buffer = "";

	try {
		while (true) {
			const { done, value } = await reader.read();
			buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
			buffer = buffer.replace(/\r\n/g, "\n");

			let split = buffer.indexOf("\n\n");
			while (split !== -1) {
				const event = parseBcgMessagesEvent(buffer.slice(0, split));
				if (event) {
					yield event;
				}
				buffer = buffer.slice(split + 2);
				split = buffer.indexOf("\n\n");
			}

			if (done) {
				break;
			}
		}

		if (buffer.trim()) {
			const event = parseBcgMessagesEvent(buffer);
			if (event) {
				yield event;
			}
		}
	} finally {
		reader.releaseLock();
	}
}

function parseBcgMessagesEvent(raw: string): BcgMessagesEvent | undefined {
	const data = raw
		.split("\n")
		.find((line) => line.startsWith("data:"))
		?.slice(5)
		.trim();

	return data && data !== "[DONE]" ? (JSON.parse(data) as BcgMessagesEvent) : undefined;
}

function createErrorEvent(model: Model<"bcg-messages">, error: unknown, aborted: boolean): AssistantMessageEvent {
	const reason = aborted ? "aborted" : "error";
	const assistantMessage: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: createEmptyUsage(),
		stopReason: reason,
		errorMessage: error instanceof Error ? error.message : String(error),
		timestamp: Date.now(),
	};

	if (!aborted && error instanceof BcgMessagesResponseError) {
		appendAssistantMessageDiagnostic(
			assistantMessage,
			createAssistantMessageDiagnostic("bcg_messages_response_failure", error, error.diagnosticDetails),
		);
	}

	return { type: "error", reason, error: assistantMessage };
}

function resolveCacheRetention(cacheRetention?: CacheRetention, env?: ProviderEnv): CacheRetention | undefined {
	if (cacheRetention) {
		return cacheRetention;
	}
	// Backend defaults apply when unset; only the legacy env opt-in is mapped.
	return getProviderEnvValue("BCG_CACHE_RETENTION", env) === "long" ? "long" : undefined;
}

export const stream: StreamFunction<"bcg-messages", BcgMessagesOptions> = (
	model: Model<"bcg-messages">,
	context: Context,
	options?: BcgMessagesOptions,
): AssistantMessageEventStream => {
	const eventStream = new AssistantMessageEventStream();
	const convertEvent = createEventConverter(model);

	void (async () => {
		try {
			const apiKey = options?.apiKey;
			if (!apiKey) {
				throw new Error(`No API key provided for provider "${model.provider}"`);
			}

			const url = new URL(`${model.baseUrl.replace(/\/+$/u, "")}/messages`);
			if (options?.debug) {
				url.searchParams.set("debug", "1");
			}

			let payload: unknown = {
				model: model.id,
				context,
				options: {
					temperature: options?.temperature,
					maxTokens: options?.maxTokens,
					reasoning: options?.reasoning,
					cacheRetention: resolveCacheRetention(options?.cacheRetention, options?.env),
					sessionId: options?.sessionId,
					toolChoice: options?.toolChoice,
				},
			};
			const nextPayload = await options?.onPayload?.(payload, model);
			if (nextPayload !== undefined) {
				payload = nextPayload;
			}

			const response = await fetch(url, {
				method: "POST",
				headers: {
					authorization: `Bearer ${apiKey}`,
					accept: "text/event-stream",
					"content-type": "application/json",
					...providerHeadersToRecord(options?.headers),
				},
				body: JSON.stringify(payload),
				signal: options?.signal,
			});

			await options?.onResponse?.({ status: response.status, headers: headersToRecord(response.headers) }, model);

			if (!response.ok) {
				const body = await response.text();
				throw createBcgMessagesResponseError(model, url, response, body);
			}
			if (!response.body) {
				throw new Error(`${model.provider} response has no body`);
			}

			for await (const piEvent of readBcgMessagesEvents(response.body)) {
				const event = convertEvent(piEvent);
				eventStream.push(event);
				if (event.type === "done" || event.type === "error") {
					return;
				}
			}

			throw new Error(`${model.provider} stream ended without a terminal event`);
		} catch (error) {
			eventStream.push(createErrorEvent(model, error, options?.signal?.aborted ?? false));
		}
	})();

	return eventStream;
};

export const streamSimple: StreamFunction<"bcg-messages", SimpleStreamOptions> = (
	model: Model<"bcg-messages">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	const extra = options as BcgMessagesOptions | undefined;
	return stream(model, context, {
		...options,
		reasoning: options?.reasoning,
		toolChoice: extra?.toolChoice,
		debug: extra?.debug,
	});
};
