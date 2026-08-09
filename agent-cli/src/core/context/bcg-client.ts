import type {
	BcgReleaseResponse,
	BcgSnapshot,
	BcgTurn,
} from "./bcg-contract.types.ts";

// Payloads this client always sends with the seed/eviction flags pinned.
export type BcgTurnPayload = BcgTurn & { is_message_end: true; is_trajectory_end: false };

/**
 * Contract-backed HTTP client for the BCG construct server (step 11).
 *
 * Owns the wire protocol: URL assembly, timeout/AbortSignal composition,
 * response envelope decoding (including the uniform `{"error": ...}` body)
 * and snapshot resolution through `latest[problemId]`. Context-window policy
 * (which turns to send, limits, fallbacks) stays in BcgContextManager.
 */
export interface BcgClientOptions {
	baseUrl: string;
	problemId: string;
	timeoutMs: number;
	fetch?: typeof globalThis.fetch;
}

export class BcgClient {
	readonly problemId: string;

	private readonly baseUrl: string;
	private readonly timeoutMs: number;
	private readonly fetchImpl: typeof globalThis.fetch;

	constructor(options: BcgClientOptions) {
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.problemId = options.problemId;
		this.timeoutMs = Math.max(1, Math.trunc(options.timeoutMs));
		this.fetchImpl = options.fetch ?? globalThis.fetch;
	}

	/** POST /turns and resolve the caller's snapshot from the envelope. */
	async postTurns(payloads: BcgTurnPayload[], signal?: AbortSignal): Promise<BcgSnapshot> {
		const response = await this.fetchImpl(`${this.baseUrl}/turns`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(payloads),
			signal: this.composeSignal(signal),
		});
		if (!response.ok) {
			throw await this.httpError(response);
		}
		const snapshot = parseSnapshot(await response.json(), this.problemId);
		if (!snapshot) {
			throw new Error("BCG server returned an invalid graph snapshot");
		}
		return snapshot;
	}

	/** POST /finalize and return the final persisted graph snapshot. */
	async finalize(): Promise<BcgSnapshot> {
		const response = await this.fetchImpl(`${this.baseUrl}/finalize`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ problem_id: this.problemId }),
			signal: AbortSignal.timeout(this.timeoutMs),
		});
		if (!response.ok) {
			throw await this.httpError(response);
		}
		const snapshot = parseSnapshot(await response.json(), this.problemId);
		if (!snapshot) {
			throw new Error("BCG server returned an invalid final graph snapshot");
		}
		return snapshot;
	}

	/**
	 * POST /release. Idempotent per contract: 404 (already released or unknown)
	 * is tolerated, and the response body reports the released flag.
	 */
	async release(): Promise<BcgReleaseResponse> {
		const response = await this.fetchImpl(`${this.baseUrl}/release`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ problem_id: this.problemId }),
			signal: AbortSignal.timeout(this.timeoutMs),
		});
		if (!response.ok && response.status !== 404) {
			throw await this.httpError(response);
		}
		const body = (await response.json().catch(() => ({}))) as Partial<BcgReleaseResponse>;
		return {
			problem_id: this.problemId,
			released: body.released ?? true,
		};
	}

	private composeSignal(signal?: AbortSignal): AbortSignal {
		const signals = [AbortSignal.timeout(this.timeoutMs)];
		if (signal) {
			signals.push(signal);
		}
		return AbortSignal.any(signals);
	}

	private async httpError(response: Response): Promise<Error> {
		let detail = "";
		try {
			const body = (await response.json()) as { error?: unknown };
			if (typeof body?.error === "string" && body.error) {
				detail = `: ${body.error}`;
			}
		} catch {
			// non-JSON error body; keep the status-only message
		}
		return new Error(`BCG server error ${response.status}${detail}`);
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Resolve a snapshot from a /turns response envelope (or a bare snapshot).
 * Tolerant by design: the server may return either shape.
 */
function parseSnapshot(value: unknown, problemId: string): BcgSnapshot | undefined {
	if (!isRecord(value)) {
		return undefined;
	}
	if (Array.isArray(value.beliefs)) {
		return value as unknown as BcgSnapshot;
	}
	const latest = value.latest;
	if (!isRecord(latest)) {
		return undefined;
	}
	const matching = latest[problemId];
	if (isRecord(matching)) {
		return matching as unknown as BcgSnapshot;
	}
	for (const snapshot of Object.values(latest)) {
		if (isRecord(snapshot)) {
			return snapshot as unknown as BcgSnapshot;
		}
	}
	return undefined;
}
