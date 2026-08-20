import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MODEL_IO_SCHEMA, ModelIoTraceRecorder } from "../src/core/model-io-trace.ts";

const temporaryDirectories: string[] = [];

afterEach(() => {
	for (const directory of temporaryDirectories.splice(0)) {
		rmSync(directory, { recursive: true, force: true });
	}
});

describe("ModelIoTraceRecorder", () => {
	it("pairs the exact provider payload with the finalized assistant response", () => {
		const directory = mkdtempSync(join(tmpdir(), "bcg-model-io-"));
		temporaryDirectories.push(directory);
		const path = join(directory, "nested", "trace.jsonl");
		const recorder = new ModelIoTraceRecorder(path);
		const callId = recorder.beginCall();

		recorder.recordRequest(
			callId,
			{ provider: "benchmark", id: "model", api: "openai-completions" },
			{ model: "model", messages: [{ role: "user", content: "hello" }] },
		);
		recorder.recordResponse(callId, {
			role: "assistant",
			content: [{ type: "text", text: "world" }],
			stopReason: "stop",
		});

		const records = readFileSync(path, "utf8")
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line));
		expect(records).toHaveLength(2);
		expect(records[0]).toMatchObject({
			schema: MODEL_IO_SCHEMA,
			type: "request",
			call_id: 1,
			payload: { messages: [{ role: "user", content: "hello" }] },
		});
		expect(records[1]).toMatchObject({
			type: "response",
			call_id: 1,
			message: { stopReason: "stop" },
		});
	});
});
