import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { loadArtifactReplay, loadLiveGraph, sampleMemory } from "./data-sources.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SNAPSHOT = JSON.parse(
  readFileSync(path.resolve(HERE, "../../contracts/fixtures/turns-response.json"), "utf8"),
) as { latest: Record<string, unknown> };

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("data-sources (step 12)", () => {
  it("resolves the active problem via /health and loads /graph", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ status: "ok", active: ["sess-1"], all: ["sess-1"], schema_version: 1 }))
      .mockResolvedValueOnce(jsonResponse(SNAPSHOT.latest["fixture-session:seed"]));
    const memory = await loadLiveGraph(
      { baseUrl: "http://127.0.0.1:8848/", problemId: undefined, fetchImpl: fetchMock },
    );
    expect(memory.nodes).toHaveLength(1);
    expect(fetchMock).toHaveBeenNthCalledWith(1, "http://127.0.0.1:8848/health");
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:8848/graph?problem_id=sess-1",
    );
  });

  it("uses the explicit problem id without calling /health", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse(SNAPSHOT.latest["fixture-session:seed"]));
    await loadLiveGraph({ baseUrl: "http://x", problemId: "p-1", fetchImpl: fetchMock });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("http://x/graph?problem_id=p-1");
  });

  it("throws when no active session exists (no sample masking)", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({ status: "ok", active: [], all: [] }));
    await expect(
      loadLiveGraph({ baseUrl: "http://x", fetchImpl: fetchMock }),
    ).rejects.toThrow(/No active graph session/);
  });

  it("throws on server errors instead of silently falling back", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({ error: "boom" }, 500));
    await expect(
      loadLiveGraph({ baseUrl: "http://x", problemId: "p", fetchImpl: fetchMock }),
    ).rejects.toThrow(/HTTP 500/);
  });

  it("replays artifact payloads (string JSON or object)", () => {
    const artifact = JSON.stringify(SNAPSHOT.latest["fixture-session:seed"]);
    expect(loadArtifactReplay(artifact).nodes).toHaveLength(1);
    expect(loadArtifactReplay(SNAPSHOT.latest["fixture-session:seed"]).nodes).toHaveLength(1);
    expect(() => loadArtifactReplay("{not json")).toThrow(/not valid JSON/);
  });

  it("sample always yields a graph", () => {
    const memory = sampleMemory();
    expect(memory.nodes.length).toBeGreaterThan(0);
    expect(memory.summary).toBeDefined();
  });
});
