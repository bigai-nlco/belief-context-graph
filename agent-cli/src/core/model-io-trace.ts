import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

const MODEL_IO_SCHEMA = "bcg.model_io.v1";

type TraceModel = {
	api?: string;
	id?: string;
	provider?: string;
};

function errorText(error: unknown): string {
	return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

/**
 * Append-only audit log at the model provider boundary.
 *
 * Request records contain the final provider payload after every payload hook;
 * response records contain the finalized assistant message returned by the
 * provider stream. Trace failures are deliberately non-fatal so observability
 * can never break an Agent request.
 */
export class ModelIoTraceRecorder {
	private nextCallId = 1;
	private readonly path: string;

	constructor(path: string) {
		this.path = path;
	}

	beginCall(): number {
		return this.nextCallId++;
	}

	recordRequest(callId: number, model: TraceModel, payload: unknown): void {
		this.append({
			schema: MODEL_IO_SCHEMA,
			type: "request",
			call_id: callId,
			timestamp: new Date().toISOString(),
			model: {
				provider: model.provider,
				id: model.id,
				api: model.api,
			},
			payload,
		});
	}

	recordResponse(callId: number, message: unknown): void {
		this.append({
			schema: MODEL_IO_SCHEMA,
			type: "response",
			call_id: callId,
			timestamp: new Date().toISOString(),
			message,
		});
	}

	recordError(callId: number, error: unknown): void {
		this.append({
			schema: MODEL_IO_SCHEMA,
			type: "error",
			call_id: callId,
			timestamp: new Date().toISOString(),
			error: errorText(error),
		});
	}

	private append(value: Record<string, unknown>): void {
		try {
			mkdirSync(dirname(this.path), { recursive: true });
			appendFileSync(this.path, `${JSON.stringify(value)}\n`, "utf8");
		} catch {
			// Model tracing is diagnostic only. Never fail a model request because
			// its local audit file cannot be written.
		}
	}
}

export { MODEL_IO_SCHEMA };
