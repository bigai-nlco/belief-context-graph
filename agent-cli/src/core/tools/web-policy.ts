import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Per-session web policy shared by web_search and web_fetch.
 *
 * - BCG_WEB_BLOCKLIST: path to a JSON file `{ url_keys, dois, titles }` naming
 *   sources the Agent must not use (DeepResearch Bench II blocks each task's
 *   expert source article). Matching mirrors `bcg.apps.benchmark.drb2`.
 * - BCG_WEB_CACHE_DIR: directory caching raw responses, so repeated queries
 *   and pages are identical across context modes and runs. The blocklist is
 *   applied after the cache, so one cache can serve every task.
 */

export interface WebBlocklist {
	urlKeys: string[];
	dois: string[];
	titles: string[];
}

const MIN_TITLE_CHARS = 24;
const URL_PATH_SUFFIXES = ["/pdf", "/full", "/htm", "/html", "/abstract"];
const blocklistCache = new Map<string, WebBlocklist | undefined>();

function stringList(value: unknown): string[] {
	return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && !!item.trim()) : [];
}

export function loadWebBlocklist(env: NodeJS.ProcessEnv = process.env): WebBlocklist | undefined {
	const path = env.BCG_WEB_BLOCKLIST?.trim();
	if (!path) return undefined;
	if (blocklistCache.has(path)) return blocklistCache.get(path);
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(path, "utf8"));
	} catch (error: unknown) {
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(`BCG_WEB_BLOCKLIST could not be read (${path}): ${message}`);
	}
	const record = typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : {};
	const urls = stringList(record.urls);
	const blocklist: WebBlocklist = {
		urlKeys: [...new Set([...stringList(record.url_keys), ...urls.map(normalizeUrlKey)].filter(Boolean))],
		dois: [...new Set(stringList(record.dois).map((doi) => doi.toLowerCase()))],
		titles: stringList(record.titles)
			.map(normalizeTitle)
			.filter((title) => title.length >= MIN_TITLE_CHARS),
	};
	blocklistCache.set(path, blocklist);
	return blocklist;
}

/** Scheme/`www`/query/fragment-insensitive key; mirrors drb2.normalize_url. */
export function normalizeUrlKey(url: string): string {
	let text = String(url ?? "").trim();
	if (!text) return "";
	if (!text.includes("://")) text = `https://${text}`;
	let parsed: URL;
	try {
		parsed = new URL(text);
	} catch {
		return "";
	}
	let host = parsed.hostname.toLowerCase();
	if (host.startsWith("www.")) host = host.slice(4);
	let path: string;
	try {
		path = decodeURIComponent(parsed.pathname || "");
	} catch {
		path = parsed.pathname || "";
	}
	path = path.replace(/\/+$/, "");
	for (const suffix of URL_PATH_SUFFIXES) {
		if (path.toLowerCase().endsWith(suffix)) {
			path = path.slice(0, -suffix.length);
			break;
		}
	}
	return host ? `${host}${path}`.toLowerCase() : "";
}

export function normalizeTitle(title: string): string {
	return String(title ?? "")
		.toLowerCase()
		.replace(/[^\p{L}\p{N}\p{M}]+/gu, " ")
		.trim();
}

export function isBlockedUrl(url: string, blocklist: WebBlocklist | undefined): boolean {
	if (!blocklist || !url) return false;
	const key = normalizeUrlKey(url);
	if (key && blocklist.urlKeys.some((blocked) => key === blocked || key.startsWith(`${blocked}/`))) return true;
	const folded = url.toLowerCase();
	return blocklist.dois.some((doi) => folded.includes(doi));
}

export function isBlockedTitle(title: string, blocklist: WebBlocklist | undefined): boolean {
	if (!blocklist || !title) return false;
	const normalized = normalizeTitle(title.replace(/(\.\.\.|…)\s*$/u, ""));
	if (normalized.length < MIN_TITLE_CHARS) return false;
	return blocklist.titles.some((blocked) => blocked.includes(normalized) || normalized.includes(blocked));
}

export function webCacheKey(kind: string, value: unknown): string {
	return createHash("sha256").update(`${kind}\n${JSON.stringify(value)}`).digest("hex");
}

function cachePath(kind: string, key: string, env: NodeJS.ProcessEnv): string | undefined {
	const directory = env.BCG_WEB_CACHE_DIR?.trim();
	if (!directory) return undefined;
	return join(directory, kind, key.slice(0, 2), `${key}.json`);
}

export function readWebCache<T>(kind: string, key: string, env: NodeJS.ProcessEnv = process.env): T | undefined {
	const path = cachePath(kind, key, env);
	if (!path) return undefined;
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8")) as { value?: T };
		return parsed.value;
	} catch {
		return undefined;
	}
}

export function writeWebCache(kind: string, key: string, value: unknown, env: NodeJS.ProcessEnv = process.env): void {
	const path = cachePath(kind, key, env);
	if (!path) return;
	try {
		mkdirSync(join(path, ".."), { recursive: true });
		const temporary = `${path}.${process.pid}.tmp`;
		writeFileSync(temporary, JSON.stringify({ savedAt: new Date().toISOString(), value }));
		renameSync(temporary, path);
	} catch {
		// The cache is an optimization; never fail a tool call because of it.
	}
}
