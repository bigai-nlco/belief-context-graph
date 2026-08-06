import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	SessionManager,
	migrateSessionEntries,
	type FileEntry,
} from "../src/core/session-manager.ts";
import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";

// Session schema golden + migration tests (step 11).
// The session file format is versioned (CURRENT_SESSION_VERSION = 3); reading
// old fixtures must migrate them to the current schema before use, and the
// golden fixture must stay readable unchanged.

const SESSION_V3_GOLDEN = [
	{
		type: "session",
		version: 3,
		id: "golden-session",
		timestamp: "2026-08-06T00:00:00.000Z",
		cwd: "/tmp",
	},
	{
		type: "message",
		id: "e1",
		parentId: null,
		timestamp: "2026-08-06T00:00:00.100Z",
		message: {
			role: "user",
			content: "hello",
			timestamp: 100,
		},
	},
	{
		type: "message",
		id: "e2",
		parentId: "e1",
		timestamp: "2026-08-06T00:00:00.200Z",
		message: {
			role: "assistant",
			content: [{ type: "text", text: "hi" }],
			timestamp: 200,
			api: "test-api",
			provider: "test-provider",
			model: "test-model",
		} as AgentMessage,
	},
];

const SESSION_V2_FIXTURE: FileEntry[] = [
	{
		type: "session",
		version: 2,
		id: "v2-session",
		timestamp: "2026-08-05T00:00:00.000Z",
		cwd: "/tmp",
	},
	{
		type: "message",
		id: "a",
		parentId: null,
		timestamp: "2026-08-05T00:00:00.100Z",
		message: { role: "user", content: "one", timestamp: 100 },
	},
	{
		type: "compaction",
		id: "b",
		parentId: "a",
		timestamp: "2026-08-05T00:00:00.200Z",
		firstKeptEntryId: "a",
		summary: "compacted",
	},
	{
		type: "message",
		id: "c",
		parentId: "b",
		timestamp: "2026-08-05T00:00:00.300Z",
		message: { role: "assistant", content: "two", timestamp: 300 },
	},
] as unknown as FileEntry[];

// v1: no version, no ids, no parentId, hookMessage role
const SESSION_V1_FIXTURE: Array<Record<string, unknown>> = [
	{ type: "session", timestamp: "2026-08-04T00:00:00.000Z", cwd: "/tmp" },
	{
		type: "message",
		timestamp: "2026-08-04T00:00:00.100Z",
		message: { role: "user", content: "old", timestamp: 100 },
	},
	{
		type: "compaction",
		timestamp: "2026-08-04T00:00:00.200Z",
		firstKeptEntryIndex: 1,
		summary: "compacted",
	},
	{
		type: "message",
		timestamp: "2026-08-04T00:00:00.300Z",
		message: {
			role: "hookMessage",
			content: [{ type: "text", text: "hook" }],
			timestamp: 300,
		},
	},
];

function writeSessionFile(entries: unknown[]): string {
	const dir = mkdtempSync(path.join(tmpdir(), "bcg-session-test-"));
	const file = path.join(dir, "session.jsonl");
	writeFileSync(file, entries.map((e) => JSON.stringify(e)).join("\n") + "\n", "utf8");
	return file;
}

const cleanups: string[] = [];
afterEach(() => {
	for (const dir of cleanups.splice(0)) {
		rmSync(dir, { recursive: true, force: true });
	}
});

describe("session schema golden and migration (step 11)", () => {
	it("reads the v3 golden fixture unchanged", () => {
		const file = writeSessionFile(SESSION_V3_GOLDEN);
		cleanups.push(path.dirname(file));
		const manager = SessionManager.open(file, path.dirname(file), "/tmp");

		expect(manager.getHeader()?.version).toBe(3);
		expect(manager.getHeader()?.id).toBe("golden-session");
		const entries = manager.getEntries();
		expect(entries).toHaveLength(2);
		expect(entries[0].id).toBe("e1");
		expect(entries[1].parentId).toBe("e1");
	});

	it("migrates v2 fixtures: firstKeptEntryIndex becomes firstKeptEntryId", () => {
		const file = writeSessionFile(SESSION_V2_FIXTURE);
		cleanups.push(path.dirname(file));
		const manager = SessionManager.open(file, path.dirname(file), "/tmp");

		const comp = manager
			.getEntries()
			.find((e) => e.type === "compaction") as { firstKeptEntryId?: string };
		expect(comp?.firstKeptEntryId).toBe("a");
		expect(manager.getHeader()?.version).toBe(3);
	});

	it("migrates v1 fixtures: adds ids, parentId chain and renames hookMessage", () => {
		const file = writeSessionFile(SESSION_V1_FIXTURE);
		cleanups.push(path.dirname(file));
		const manager = SessionManager.open(file, path.dirname(file), "/tmp");

		const entries = manager.getEntries();
		expect(manager.getHeader()?.version).toBe(3);
		expect(entries).toHaveLength(3);
		for (const entry of entries) {
			expect(entry.id).toBeTruthy();
		}
		// v1 migration builds the parentId chain: first entry is a root
		expect(entries[0].parentId).toBeNull();
		expect(entries[1].parentId).toBe(entries[0].id);
		expect(entries[2].parentId).toBe(entries[1].id);
		const comp = entries[1] as { firstKeptEntryId?: string; firstKeptEntryIndex?: number };
		expect(comp.firstKeptEntryId).toBe(entries[0].id);
		expect("firstKeptEntryIndex" in comp).toBe(false);
		const messages = entries.map((e) => (e as { message?: { role?: string } }).message?.role);
		expect(messages).not.toContain("hookMessage");
	});

	it("migrateSessionEntries is idempotent", () => {
		const fixture = structuredClone(SESSION_V2_FIXTURE);
		migrateSessionEntries(fixture);
		const once = structuredClone(fixture);
		migrateSessionEntries(fixture);
		expect(fixture).toEqual(once);
	});

	it("round-trips a session through the golden shape", () => {
		const file = writeSessionFile(SESSION_V3_GOLDEN);
		cleanups.push(path.dirname(file));
		const manager = SessionManager.open(file, path.dirname(file), "/tmp");
		const onDisk = readFileSync(file, "utf8");
		const reloaded = SessionManager.open(file, path.dirname(file), "/tmp");

		expect(reloaded.getEntries()).toEqual(manager.getEntries());
		expect(onDisk).toContain('"version":3');
	});
});
