import { beforeAll, describe, expect, it } from "vitest";
import { buildSessionInfoText } from "../src/modes/interactive/session-info.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";
import type { SessionStats } from "../src/core/agent-session.ts";

const STATS: SessionStats = {
  sessionFile: "/sessions/abc.jsonl",
  sessionId: "sess-1",
  userMessages: 3,
  assistantMessages: 2,
  toolCalls: 4,
  toolResults: 4,
  totalMessages: 5,
  tokens: { input: 1000, cacheRead: 0, cacheWrite: 0, output: 500, total: 1500 },
  cost: 0,
} as SessionStats;

function plain(text: string): string {
  return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function deps(overrides: Partial<Parameters<typeof buildSessionInfoText>[0]> = {}) {
  return {
    getSessionStats: () => STATS,
    getSessionName: () => undefined,
    getEntries: () => [],
    modelRuntime: {} as never,
    ...overrides,
  };
}

beforeAll(() => {
  initTheme("default", false);
});

describe("session-info (step 11.3)", () => {
  it("renders header, message and token sections", () => {
    const text = plain(buildSessionInfoText(deps()));
    expect(text).toContain("Session Info");
    expect(text).toContain("sess-1");
    expect(text).toContain("Total: 5");
    expect(text).toContain("User: 3");
    expect(text).toContain("Tools: 4 calls, 4 results");
    expect(text).toContain("Total: 1,500");
  });

  it("shows the session name when set", () => {
    const text = plain(buildSessionInfoText(deps({ getSessionName: () => "my-session" })));
    expect(text).toContain("my-session");
  });

  it("omits the cost section when there is no cost or cache waste", () => {
    const text = plain(buildSessionInfoText(deps()));
    expect(text).not.toContain("Cost");
  });

  it("includes a cache hit-rate breakdown when cache activity exists", () => {
    const text = plain(
      buildSessionInfoText(
        deps({
          getSessionStats: () => ({
            ...STATS,
            tokens: { input: 800, cacheRead: 200, cacheWrite: 0, output: 500, total: 1500 },
          }),
        }),
      ),
    );
    expect(text).toContain("Cached: 200");
    expect(text).toContain("(20.0%)");
  });
});
