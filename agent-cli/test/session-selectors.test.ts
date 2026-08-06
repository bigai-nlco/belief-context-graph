import { beforeAll, describe, expect, it, vi } from "vitest";

const constructed = vi.hoisted(() => [] as Array<{ name: string; args: unknown[] }>);

vi.mock("../src/modes/interactive/components/session-selector.ts", () => ({
	SessionSelectorComponent: class {
		constructor(...args: unknown[]) {
			constructed.push({ name: "session", args });
		}
	},
}));

import { showSessionSelector, type SessionSelectorDeps } from "../src/modes/interactive/session-selectors.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

function fakeDeps(overrides: Partial<SessionSelectorDeps> = {}): SessionSelectorDeps & {
	resumed: string[];
	renders: number;
	renamed: Array<[string, string]>;
} {
	const result = {
		resumed: [] as string[],
		renders: 0,
		renamed: [] as [string, string],
		showSelector: (create: (done: () => void) => unknown) => create(() => {}),
		requestRender: () => {
			result.renders += 1;
		},
		resumeSession: async (sessionPath: string) => {
			result.resumed.push(sessionPath);
		},
		shutdown: () => {},
		keybindings: {} as SessionSelectorDeps["keybindings"],
		listSessions: async () => [],
		listAllSessions: async () => [],
		usesDefaultSessionDir: () => true,
		getCwd: () => "/cwd",
		getSessionDir: () => "/sessions",
		getSessionFile: () => "/sessions/current.jsonl",
		renameSession: (sessionFilePath: string, nextName: string) => {
			result.renamed.push([sessionFilePath, nextName]);
		},
		...overrides,
	};
	return result;
}

beforeAll(() => {
	initTheme("default", false);
});

describe("session-selectors (step 11.3)", () => {
	it("builds the session selector with list callbacks and current file", () => {
		const deps = fakeDeps();
		showSessionSelector(deps);

		const last = constructed.at(-1)!;
		expect(last.name).toBe("session");
		expect(last.args[0]).toBeTypeOf("function"); // list cwd sessions
		expect(last.args[1]).toBeTypeOf("function"); // list all sessions
		expect(last.args.at(-1)).toBe("/sessions/current.jsonl");
	});

	it("dispatches resume through the injected handler", async () => {
		const deps = fakeDeps();
		showSessionSelector(deps);
		const last = constructed.at(-1)!;
		const onSelect = last.args[2] as (sessionPath: string) => Promise<void>;
		await onSelect("/sessions/other.jsonl");
		expect(deps.resumed).toEqual(["/sessions/other.jsonl"]);
	});

	it("dispatches shutdown through the injected handler", () => {
		const deps = fakeDeps();
		const shutdown = vi.fn();
		deps.shutdown = shutdown;
		showSessionSelector(deps);
		const last = constructed.at(-1)!;
		const onShutdown = last.args[4] as () => void;
		onShutdown();
		expect(shutdown).toHaveBeenCalledOnce();
	});

	it("trims empty rename names before calling the backend", async () => {
		const deps = fakeDeps();
		showSessionSelector(deps);
		const last = constructed.at(-1)!;
		const options = last.args[6] as { renameSession: (p: string, n: string | undefined) => Promise<void> };

		await options.renameSession("/sessions/a.jsonl", "   ");
		expect(deps.renamed).toHaveLength(0);

		await options.renameSession("/sessions/a.jsonl", "  new name  ");
		expect(deps.renamed).toEqual([["/sessions/a.jsonl", "new name"]]);
	});
});
