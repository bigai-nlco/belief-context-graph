import { beforeAll, describe, expect, it, vi } from "vitest";

const constructed = vi.hoisted(() => [] as Array<{ name: string; args: unknown[] }>);

vi.mock("../src/modes/interactive/components/trust-selector.ts", () => ({
	TrustSelectorComponent: class {
		constructor(...args: unknown[]) {
			constructed.push({ name: "trust", args });
		}
	},
}));

vi.mock("../src/modes/interactive/components/user-message-selector.ts", () => ({
	UserMessageSelectorComponent: class {
		getMessageList() {
			return {};
		}
		constructor(...args: unknown[]) {
			constructed.push({ name: "user-message", args });
		}
	},
}));

import {
	showTrustSelector,
	showUserMessageSelector,
	type TrustSelectorDeps,
	type UserMessageSelectorDeps,
} from "../src/modes/interactive/trust-selectors.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

function trustDeps(overrides: Partial<TrustSelectorDeps> = {}): TrustSelectorDeps & {
	status: string[];
	saved: Array<{ path: string; decision: boolean | null }[]>;
} {
	const result = {
		status: [] as string[],
		saved: [] as Array<{ path: string; decision: boolean | null }[]>,
		showSelector: (create: (done: () => void) => unknown) => create(() => {}),
		requestRender: () => {},
		showStatus: (m: string) => result.status.push(m),
		getCwd: () => "/proj",
		getSavedDecision: () => null,
		isProjectTrusted: () => false,
		saveTrust: (updates) => result.saved.push(updates),
		...overrides,
	};
	return result;
}

function messageDeps(overrides: Partial<UserMessageSelectorDeps> = {}): UserMessageSelectorDeps & {
	status: string[];
	errors: string[];
	forks: string[];
	setText: string[];
} {
	const result = {
		status: [] as string[],
		errors: [] as string[],
		forks: [] as string[],
		setText: [] as string[],
		showSelector: (create: (done: () => void) => unknown) => create(() => {}),
		requestRender: () => {},
		showStatus: (m: string) => result.status.push(m),
		showError: (m: string) => result.errors.push(m),
		getUserMessagesForForking: () => [
			{ entryId: "m1", text: "first" },
			{ entryId: "m2", text: "second" },
		],
		fork: async (entryId: string) => {
			result.forks.push(entryId);
			return { cancelled: false, selectedText: "sel" };
		},
		setEditorText: (text: string) => result.setText.push(text),
		...overrides,
	};
	return result;
}

beforeAll(() => {
	initTheme("default", false);
});

describe("trust-selectors (step 11.3)", () => {
	it("builds the trust selector with cwd and saved decision wiring", () => {
		const deps = trustDeps();
		showTrustSelector(deps);
		const last = constructed.at(-1)!;
		expect(last.name).toBe("trust");
		const options = last.args[0] as { cwd: string; projectTrusted: boolean };
		expect(options.cwd).toBe("/proj");
		expect(options.projectTrusted).toBe(false);
	});

	it("saves the trust decision and reports status on selection", () => {
		const deps = trustDeps();
		showTrustSelector(deps);
		const last = constructed.at(-1)!;
		const options = last.args[0] as {
			onSelect: (selection: { trusted: boolean; updates: Array<{ path: string; decision: boolean | null }> }) => void;
		};
		options.onSelect({ trusted: true, updates: [{ path: "/proj", decision: true }] });
		expect(deps.saved).toEqual([[{ path: "/proj", decision: true }]]);
		expect(deps.status[0]).toContain("Saved trust decision: trusted");
	});
});

describe("user-message-selector (step 11.3)", () => {
	it("reports when there is nothing to fork from", () => {
		const deps = messageDeps({ getUserMessagesForForking: () => [] });
		showUserMessageSelector(deps);
		expect(deps.status).toEqual(["No messages to fork from"]);
	});

	it("builds the fork picker and forks on selection", async () => {
		const deps = messageDeps();
		showUserMessageSelector(deps);
		const last = constructed.at(-1)!;
		expect(last.name).toBe("user-message");
		const messages = last.args[0] as Array<{ id: string; text: string }>;
		expect(messages).toHaveLength(2);
		expect(last.args[3]).toBe("m2"); // most recent preselected

		const onSelect = last.args[1] as (entryId: string) => Promise<void>;
		await onSelect("m1");
		expect(deps.forks).toEqual(["m1"]);
		expect(deps.setText).toEqual(["sel"]);
		expect(deps.status).toContain("Forked to new session");
	});

	it("surfaces fork errors", async () => {
		const deps = messageDeps({
			fork: async () => {
				throw new Error("boom");
			},
		});
		showUserMessageSelector(deps);
		const last = constructed.at(-1)!;
		const onSelect = last.args[1] as (entryId: string) => Promise<void>;
		await onSelect("m1");
		expect(deps.errors[0]).toContain("boom");
	});
});
