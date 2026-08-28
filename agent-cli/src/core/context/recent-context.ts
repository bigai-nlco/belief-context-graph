import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { DatabaseSync } from "node:sqlite";
import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import {
	contextMessageKey,
	contextMessageText,
	partitionContextTurns,
	splitBcgTurns,
} from "./bcg-context.ts";

const RAG_CONTEXT_GUIDE = `<retrieved_history_guide>
Earlier completed turns have been omitted from the raw conversation. The excerpts below were retrieved from the session's local history database using the recent raw turns as the query. They are prior working context, not verified evidence. Use relevant excerpts to continue prior work and avoid repeating completed actions; ignore irrelevant excerpts and prefer the current raw conversation when they conflict.
</retrieved_history_guide>`;

const DEFAULT_TOP_K = 6;
const DEFAULT_MAX_CHARS = 12_000;
const MAX_QUERY_TERMS = 48;

const QUERY_STOP_WORDS = new Set([
	"about",
	"after",
	"again",
	"also",
	"assistant",
	"before",
	"could",
	"from",
	"have",
	"into",
	"just",
	"more",
	"result",
	"search",
	"should",
	"that",
	"their",
	"then",
	"there",
	"these",
	"they",
	"this",
	"tool",
	"using",
	"what",
	"when",
	"where",
	"which",
	"with",
	"would",
]);

export interface RecentContextManagerOptions {
	recentTurns: number;
	getInitialUserMessage?: () => AgentMessage | undefined;
}

export interface RagContextManagerOptions extends RecentContextManagerOptions {
	databasePath: string;
	topK?: number;
	maxChars?: number;
	onWarning?: (message: string) => void;
	onRagContext?: (trace: RagContextTrace) => void;
}

export interface RagContextTrace {
	query: string;
	storedTurns: number;
	retrievedTurns: number;
	retrievedTurnIndices: number[];
	chars: number;
	text: string;
}

interface PartitionedContext {
	initialUser: AgentMessage;
	evicted: AgentMessage[][];
	retained: AgentMessage[][];
}

interface RetrievedTurn {
	id: number;
	turnIndex: number;
	content: string;
	rank: number;
}

interface SqliteModule {
	DatabaseSync: typeof import("node:sqlite").DatabaseSync;
}

let sqliteModulePromise: Promise<SqliteModule> | undefined;

function loadSqlite(): Promise<SqliteModule> {
	if (!sqliteModulePromise) {
		// Node 22.19 satisfies this package's engine floor and ships node:sqlite,
		// but still labels it experimental. Hide only that one runtime warning so
		// entering RAG mode does not pollute the terminal or benchmark stderr.
		const originalEmitWarning = process.emitWarning;
		process.emitWarning = ((warning: string | Error, ...args: unknown[]) => {
			const warningType = typeof args[0] === "string" ? args[0] : undefined;
			if (warningType === "ExperimentalWarning" && String(warning).includes("SQLite")) {
				return;
			}
			(originalEmitWarning as (...values: unknown[]) => void)(warning, ...args);
		}) as typeof process.emitWarning;
		sqliteModulePromise = (import("node:sqlite") as Promise<SqliteModule>).finally(() => {
			process.emitWarning = originalEmitWarning;
		});
	}
	return sqliteModulePromise;
}

function resolveInitialUser(
	messages: AgentMessage[],
	configured?: () => AgentMessage | undefined,
): AgentMessage | undefined {
	const explicit = configured?.();
	if (explicit?.role === "user") return explicit;
	return messages.find((message) => message.role === "user");
}

function partitionMessages(
	messages: AgentMessage[],
	recentTurns: number,
	configuredInitial?: () => AgentMessage | undefined,
): PartitionedContext {
	const initialUser = resolveInitialUser(messages, configuredInitial);
	if (!initialUser) {
		throw new Error("the session has no initial user input");
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
	const { evicted, retained } = partitionContextTurns(splitBcgTurns(rest), recentTurns);
	return { initialUser, evicted, retained };
}

function messageRole(message: AgentMessage): string {
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

function renderTurn(messages: AgentMessage[]): string {
	return messages
		.map((message) => {
			const content = contextMessageText(message).trim();
			return `[${messageRole(message)}]\n${content || "(empty)"}`;
		})
		.join("\n\n");
}

function turnKey(messages: AgentMessage[]): string {
	return createHash("sha256")
		.update(messages.map((message) => contextMessageKey(message)).join("\u0001"))
		.digest("hex");
}

function queryTerms(text: string): string[] {
	const terms = text
		.normalize("NFKC")
		.toLocaleLowerCase()
		.match(/[\p{L}\p{N}_-]{2,}/gu);
	if (!terms) return [];

	const unique: string[] = [];
	const seen = new Set<string>();
	for (let index = terms.length - 1; index >= 0 && unique.length < MAX_QUERY_TERMS; index -= 1) {
		const term = terms[index];
		if (QUERY_STOP_WORDS.has(term) || seen.has(term)) continue;
		seen.add(term);
		unique.push(term);
	}
	return unique.reverse();
}

function ftsQuery(text: string): string {
	return queryTerms(text)
		.map((term) => `"${term.replaceAll('"', '""')}"`)
		.join(" OR ");
}

function renderRetrievedHistory(turns: RetrievedTurn[], maxChars: number): string {
	const sections: string[] = [];
	let used = 0;
	for (const turn of turns) {
		const section = `### Earlier turn ${turn.turnIndex}\n${turn.content}`;
		const separator = sections.length > 0 ? 2 : 0;
		if (used + separator + section.length <= maxChars) {
			sections.push(section);
			used += separator + section.length;
			continue;
		}
		const remaining = maxChars - used - separator;
		if (remaining > 80) {
			sections.push(`${section.slice(0, remaining - 1).trimEnd()}…`);
		}
		break;
	}
	return sections.join("\n\n");
}

/** Keep the initial request plus the configured number of recent completed turns. */
export class RecentOnlyContextManager {
	private readonly recentTurns: number;
	private readonly getInitialUserMessage?: () => AgentMessage | undefined;

	constructor(options: RecentContextManagerOptions) {
		this.recentTurns = Math.max(-1, Math.trunc(options.recentTurns));
		this.getInitialUserMessage = options.getInitialUserMessage;
	}

	async transform(messages: AgentMessage[]): Promise<AgentMessage[]> {
		const { initialUser, retained } = partitionMessages(
			messages,
			this.recentTurns,
			this.getInitialUserMessage,
		);
		return [initialUser, ...retained.flat()];
	}

	augmentSystemPrompt(systemPrompt: string | undefined): string | undefined {
		return systemPrompt;
	}
}

/** SQLite FTS-backed retrieval over turns evicted by the recent-context window. */
export class RagContextManager {
	private readonly recentTurns: number;
	private readonly databasePath: string;
	private readonly topK: number;
	private readonly maxChars: number;
	private readonly getInitialUserMessage?: () => AgentMessage | undefined;
	private readonly onWarning: (message: string) => void;
	private readonly onRagContext?: (trace: RagContextTrace) => void;
	private database: DatabaseSync | undefined;
	private databasePromise: Promise<DatabaseSync> | undefined;
	private retrievedText = "";
	private requestReady = false;
	private warned = false;
	private closed = false;

	constructor(options: RagContextManagerOptions) {
		this.recentTurns = Math.max(-1, Math.trunc(options.recentTurns));
		this.databasePath = options.databasePath;
		this.topK = Math.max(1, Math.trunc(options.topK ?? DEFAULT_TOP_K));
		this.maxChars = Math.max(256, Math.trunc(options.maxChars ?? DEFAULT_MAX_CHARS));
		this.getInitialUserMessage = options.getInitialUserMessage;
		this.onWarning = options.onWarning ?? ((message) => console.warn(message));
		this.onRagContext = options.onRagContext;
	}

	async transform(messages: AgentMessage[]): Promise<AgentMessage[]> {
		this.requestReady = false;
		const partitioned = partitionMessages(messages, this.recentTurns, this.getInitialUserMessage);
		const bounded = [partitioned.initialUser, ...partitioned.retained.flat()];
		if (this.recentTurns < 0) {
			this.retrievedText = "";
			this.requestReady = true;
			return bounded;
		}
		try {
			const database = await this.getDatabase();
			this.storeTurns(database, partitioned.evicted);
			const query = partitioned.retained.flat().map(contextMessageText).filter(Boolean).join("\n\n");
			const retrieved = this.retrieve(database, query);
			this.retrievedText = renderRetrievedHistory(retrieved, this.maxChars);
			this.requestReady = true;
			this.warned = false;
			this.onRagContext?.({
				query,
				storedTurns: this.storedTurnCount(database),
				retrievedTurns: retrieved.length,
				retrievedTurnIndices: retrieved.map((turn) => turn.turnIndex),
				chars: this.retrievedText.length,
				text: this.retrievedText,
			});
		} catch (error) {
			this.retrievedText = "";
			if (!this.warned) {
				const detail = error instanceof Error ? error.message : String(error);
				this.onWarning(`[RAG context] ${detail}; using recent-only context for this request.`);
				this.warned = true;
			}
		}
		return bounded;
	}

	augmentSystemPrompt(systemPrompt: string | undefined): string | undefined {
		if (!this.requestReady || !this.retrievedText) return systemPrompt;
		const block = `<retrieved_history>\n${this.retrievedText}\n</retrieved_history>`;
		return [systemPrompt, RAG_CONTEXT_GUIDE, block].filter(Boolean).join("\n\n");
	}

	release(): void {
		if (this.closed) return;
		this.closed = true;
		this.database?.close();
		this.database = undefined;
	}

	private async getDatabase(): Promise<DatabaseSync> {
		if (this.closed) throw new Error("the RAG history database is already closed");
		if (this.database) return this.database;
		this.databasePromise ??= this.openDatabase();
		return this.databasePromise;
	}

	private async openDatabase(): Promise<DatabaseSync> {
		mkdirSync(dirname(this.databasePath), { recursive: true });
		const { DatabaseSync } = await loadSqlite();
		const database = new DatabaseSync(this.databasePath);
		database.exec(`
			PRAGMA journal_mode = WAL;
			CREATE TABLE IF NOT EXISTS memory (
				id INTEGER PRIMARY KEY,
				turn_key TEXT NOT NULL UNIQUE,
				turn_index INTEGER NOT NULL,
				content TEXT NOT NULL,
				created_at TEXT NOT NULL
			);
			CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
				content,
				content='memory',
				content_rowid='id',
				tokenize='unicode61 remove_diacritics 2'
			);
			CREATE TRIGGER IF NOT EXISTS memory_insert AFTER INSERT ON memory BEGIN
				INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
			END;
			CREATE TRIGGER IF NOT EXISTS memory_delete AFTER DELETE ON memory BEGIN
				INSERT INTO memory_fts(memory_fts, rowid, content) VALUES ('delete', old.id, old.content);
			END;
			CREATE TRIGGER IF NOT EXISTS memory_update AFTER UPDATE ON memory BEGIN
				INSERT INTO memory_fts(memory_fts, rowid, content) VALUES ('delete', old.id, old.content);
				INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
			END;
		`);
		this.database = database;
		return database;
	}

	private storeTurns(database: DatabaseSync, turns: AgentMessage[][]): void {
		if (turns.length === 0) return;
		const nextIndexRow = database.prepare("SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_index FROM memory").get() as
			| { next_index?: number }
			| undefined;
		let nextIndex = Number(nextIndexRow?.next_index ?? 1);
		const insert = database.prepare(
			"INSERT OR IGNORE INTO memory(turn_key, turn_index, content, created_at) VALUES (?, ?, ?, ?)",
		);
		for (const turn of turns) {
			if (turn.length === 0) continue;
			const result = insert.run(turnKey(turn), nextIndex, renderTurn(turn), new Date().toISOString());
			if (Number(result.changes) > 0) nextIndex += 1;
		}
	}

	private retrieve(database: DatabaseSync, query: string): RetrievedTurn[] {
		const match = ftsQuery(query);
		if (!match) return [];
		const rows = database
			.prepare(
				`SELECT memory.id AS id, memory.turn_index AS turn_index, memory.content AS content,
				        bm25(memory_fts) AS rank
				 FROM memory_fts
				 JOIN memory ON memory.id = memory_fts.rowid
				 WHERE memory_fts MATCH ?
				 ORDER BY rank ASC, memory.turn_index DESC
				 LIMIT ?`,
			)
			.all(match, this.topK) as Array<Record<string, unknown>>;
		return rows
			.map((row) => ({
				id: Number(row.id),
				turnIndex: Number(row.turn_index),
				content: String(row.content ?? ""),
				rank: Number(row.rank ?? 0),
			}))
			.sort((left, right) => left.turnIndex - right.turnIndex);
	}

	private storedTurnCount(database: DatabaseSync): number {
		const row = database.prepare("SELECT COUNT(*) AS count FROM memory").get() as { count?: number } | undefined;
		return Number(row?.count ?? 0);
	}
}
