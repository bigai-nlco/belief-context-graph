import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { assertInsideToolRoot, resolveToCwd } from "../src/core/tools/path-utils.ts";
import {
	createAllToolDefinitions,
	createWebFetchTool,
	createWebSearchTool,
	htmlToText,
	isBlockedTitle,
	isBlockedUrl,
	loadWebBlocklist,
	normalizeUrlKey,
	resolveScrapeEndpoint,
} from "../src/core/tools/index.ts";

const LONG_TEXT = `Land finance report. ${"Revenue reached 8.71 trillion yuan in 2021. ".repeat(20)}`;

let directory: string;

beforeEach(() => {
	directory = mkdtempSync(join(tmpdir(), "bcg-web-fetch-"));
});

afterEach(() => {
	rmSync(directory, { recursive: true, force: true });
	delete process.env.BCG_TOOL_ROOT;
});

function blocklistEnv(extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
	const path = join(directory, `blocklist-${Math.random()}.json`);
	writeFileSync(
		path,
		JSON.stringify({
			urls: ["https://www.mdpi.com/2073-445X/11/9/1529"],
			dois: ["10.3390/land11091529"],
			titles: ["The Local Land Finance Transformation with the Synergy of Increment and Inventory"],
		}),
	);
	return { BCG_WEB_BLOCKLIST: path, ...extra };
}

describe("web policy", () => {
	it("normalizes URLs like the Python benchmark adapter", () => {
		expect(normalizeUrlKey("https://www.MDPI.com/2073-445X/11/9/1529/pdf?version=2#top")).toBe(
			"mdpi.com/2073-445x/11/9/1529",
		);
		expect(normalizeUrlKey("not a url")).toBe("");
	});

	it("blocks exact sources, sub-paths, DOIs, and truncated titles", () => {
		const blocklist = loadWebBlocklist(blocklistEnv());
		expect(isBlockedUrl("http://mdpi.com/2073-445X/11/9/1529/htm", blocklist)).toBe(true);
		expect(isBlockedUrl("https://www.mdpi.com/2073-445X/11/9/1529/s1", blocklist)).toBe(true);
		expect(isBlockedUrl("https://doi.org/10.3390/land11091529", blocklist)).toBe(true);
		expect(isBlockedUrl("https://www.mdpi.com/2073-445X/11/9/1530", blocklist)).toBe(false);
		expect(isBlockedTitle("The Local Land Finance Transformation with the Synergy of …", blocklist)).toBe(true);
		expect(isBlockedTitle("Land finance in China", blocklist)).toBe(false);
	});

	it("keeps CJK titles when normalizing", () => {
		const path = join(directory, "zh.json");
		writeFileSync(path, JSON.stringify({ titles: ["中国地方土地财政转型的增量与存量协同研究：基于案例的分析"] }));
		const blocklist = loadWebBlocklist({ BCG_WEB_BLOCKLIST: path });
		expect(isBlockedTitle("中国地方土地财政转型的增量与存量协同研究：基于案例的分析 - 知网", blocklist)).toBe(true);
	});
});

describe("web_fetch tool", () => {
	it("is registered as a built-in tool", () => {
		expect(Object.keys(createAllToolDefinitions(directory))).toContain("web_fetch");
	});

	it("derives the scrape endpoint from a self-hosted search proxy", () => {
		expect(resolveScrapeEndpoint({ SERPER_ENDPOINT: "http://10.0.0.1:9999/search" })).toBe(
			"http://10.0.0.1:9999/scrape",
		);
		expect(resolveScrapeEndpoint({ SERPER_ENDPOINT: "https://google.serper.dev/search" })).toBe(
			"https://scrape.serper.dev",
		);
		expect(resolveScrapeEndpoint({ BCG_FETCH_ENDPOINT: "http://x/fetch", SERPER_ENDPOINT: "http://y/search" })).toBe(
			"http://x/fetch",
		);
	});

	it("prefers scraped markdown and pages long content with an offset", async () => {
		const fetchMock = vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
			expect(JSON.parse(String(init?.body))).toEqual({ url: "https://example.org/report", includeMarkdown: true });
			expect(new Headers(init?.headers).get("X-API-KEY")).toBe("key");
			return Response.json({ text: "plain", markdown: `# Report\n\n${LONG_TEXT}`, metadata: { title: "Report" } });
		}) as unknown as typeof globalThis.fetch;
		const tool = createWebFetchTool({
			env: { SERPER_API_KEY: "key", SERPER_ENDPOINT: "http://proxy/search" },
			fetch: fetchMock,
			maxChars: 100,
		});
		const first = await tool.execute("f1", { url: "https://example.org/report" });
		const text = (first.content[0] as { text: string }).text;
		expect(text).toContain("Title: Report");
		expect(text).toContain("# Report");
		expect(text).toContain("offset=100");
		expect(first.details.provider).toBe("scrape");
		expect(first.details.truncated).toBe(true);
		const second = await tool.execute("f2", { url: "https://example.org/report", offset: 100 });
		expect((second.content[0] as { text: string }).text).toContain("Characters 100-200");
		expect(String(vi.mocked(fetchMock).mock.calls[0][0])).toBe("http://proxy/scrape");
	});

	it("falls back to a direct fetch when scraping returns too little text", async () => {
		const fetchMock = vi.fn(async (input: string | URL | Request) => {
			if (String(input).endsWith("/scrape")) return Response.json({ text: "short" });
			return new Response(
				`<html><head><title>Direct page</title><style>.x{}</style></head><body><h2>Section</h2><p>${LONG_TEXT}</p><script>alert(1)</script></body></html>`,
				{ headers: { "content-type": "text/html; charset=utf-8" } },
			);
		}) as unknown as typeof globalThis.fetch;
		const tool = createWebFetchTool({ env: { SERPER_API_KEY: "key", SERPER_ENDPOINT: "http://p/search" }, fetch: fetchMock });
		const result = await tool.execute("f1", { url: "https://example.org/page" });
		const text = (result.content[0] as { text: string }).text;
		expect(result.details.provider).toBe("direct");
		expect(text).toContain("Title: Direct page");
		expect(text).toContain("## Section");
		expect(text).not.toContain("alert(1)");
	});

	it("refuses blocked sources without spending budget or making requests", async () => {
		const fetchMock = vi.fn() as unknown as typeof globalThis.fetch;
		const tool = createWebFetchTool({ env: blocklistEnv({ SERPER_API_KEY: "key" }), fetch: fetchMock, maxCalls: 1 });
		const refused = await tool.execute("f1", { url: "https://mdpi.com/2073-445X/11/9/1529" });
		expect(refused.details.policy.blocked).toBe(true);
		expect(refused.details.budget.callsUsed).toBe(0);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("enforces the call budget", async () => {
		const fetchMock = vi.fn(async () => Response.json({ markdown: LONG_TEXT })) as unknown as typeof globalThis.fetch;
		const tool = createWebFetchTool({ env: { SERPER_API_KEY: "key" }, fetch: fetchMock, maxCalls: 1 });
		await tool.execute("f1", { url: "https://example.org/a" });
		const blocked = await tool.execute("f2", { url: "https://example.org/b" });
		expect(blocked.details.budget.blocked).toBe(true);
		expect(fetchMock).toHaveBeenCalledOnce();
	});

	it("serves repeated pages from BCG_WEB_CACHE_DIR", async () => {
		const env = { SERPER_API_KEY: "key", BCG_WEB_CACHE_DIR: join(directory, "cache") };
		const fetchMock = vi.fn(async () => Response.json({ markdown: LONG_TEXT })) as unknown as typeof globalThis.fetch;
		await createWebFetchTool({ env, fetch: fetchMock }).execute("f1", { url: "https://example.org/a" });
		const again = await createWebFetchTool({ env, fetch: fetchMock }).execute("f2", { url: "https://example.org/a" });
		expect(again.details.cached).toBe(true);
		expect(fetchMock).toHaveBeenCalledOnce();
	});

	it("converts HTML tables and entities to text", () => {
		const { text } = htmlToText("<table><tr><th>Year</th><th>Value</th></tr><tr><td>2021</td><td>&#36;5 &amp; up</td></tr></table>");
		expect(text).toContain("| Year | Value |");
		expect(text).toContain("| 2021 | $5 & up |");
	});
});

describe("web_search source policy", () => {
	it("hides blocked results and reports how many were filtered", async () => {
		const fetchMock = vi.fn(async () =>
			Response.json({
				organic: [
					{ title: "Blocked article", link: "https://www.mdpi.com/2073-445X/11/9/1529", snippet: "a" },
					{
						title: "The Local Land Finance Transformation with the Synergy of Increment and Inventory: A Case",
						link: "https://mirror.example/paper",
						snippet: "b",
					},
					{ title: "Allowed source", link: "https://stats.gov.cn/land", snippet: "c" },
				],
			}),
		) as unknown as typeof globalThis.fetch;
		const tool = createWebSearchTool({ apiKey: "key", fetch: fetchMock, env: blocklistEnv() });
		const result = await tool.execute("s1", { query: "land finance" });
		const text = (result.content[0] as { text: string }).text;
		expect(result.details.evidences.map((item) => item.url)).toEqual(["https://stats.gov.cn/land"]);
		expect(result.details.evidences[0].rank).toBe(1);
		expect(result.details.policy).toEqual({ filtered: 2 });
		expect(text).toContain("2 result(s) hidden");
	});
});

describe("BCG_TOOL_ROOT sandbox", () => {
	it("allows paths inside the root and rejects escapes, including symlinks", () => {
		const root = join(directory, "workspace");
		mkdirSync(join(root, "shared"), { recursive: true });
		writeFileSync(join(directory, "secret.json"), "{}");
		symlinkSync(join(directory, "secret.json"), join(root, "shared", "link.json"));
		process.env.BCG_TOOL_ROOT = root;
		expect(resolveToCwd("shared/new-file.md", root)).toBe(join(root, "shared", "new-file.md"));
		expect(() => resolveToCwd("../secret.json", root)).toThrow(/outside the task workspace/);
		expect(() => resolveToCwd(join(directory, "secret.json"), root)).toThrow(/outside the task workspace/);
		expect(() => resolveToCwd("shared/link.json", root)).toThrow(/outside the task workspace/);
		expect(() => assertInsideToolRoot("/etc/passwd")).toThrow();
	});

	it("is a no-op when unset", () => {
		expect(resolveToCwd("../x", directory)).toBe(join(directory, "..", "x"));
	});
});
