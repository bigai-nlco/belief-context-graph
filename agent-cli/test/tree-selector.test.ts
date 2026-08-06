import { beforeAll, describe, expect, it, vi } from "vitest";

const constructed = vi.hoisted(() => [] as Array<{ name: string; args: unknown[] }>);
const instances = vi.hoisted(
	() => [] as Array<{ onCopy?: (text: string) => Promise<void> }>,
);

vi.mock("../src/modes/interactive/components/tree-selector.ts", () => ({
	TreeSelectorComponent: class {
		onCopy: ((text: string) => Promise<void>) | undefined;
		constructor(...args: unknown[]) {
			constructed.push({ name: "tree", args });
			instances.push(this);
		}
	},
}));

import { showTreeSelector, type TreeSelectorDeps } from "../src/modes/interactive/tree-selector.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

function fakeDeps(overrides: Partial<TreeSelectorDeps> = {}): TreeSelectorDeps & {
	status: string[];
	errors: string[];
	navigated: Array<{ id: string; options: { summarize: boolean; customInstructions?: string } }>;
	copied: string[];
	escapeInstalls: number;
} {
	const result = {
		status: [] as string[],
		errors: [] as string[],
		navigated: [] as Array<{ id: string; options: { summarize: boolean; customInstructions?: string } }>,
		copied: [] as string[],
		escapeInstalls: 0,
		showSelector: (create: (done: () => void) => unknown) => create(() => {}),
		requestRender: () => {},
		showStatus: (m: string) => result.status.push(m),
		showError: (m: string) => result.errors.push(m),
		getTree: () => [{ id: "n1", label: "one" }, { id: "n2", label: "two" }],
		getLeafId: () => "n2",
		getTerminalRows: () => 40,
		getTreeFilterMode: () => "default" as const,
		getBranchSummarySkipPrompt: () => true,
		appendLabelChange: () => {},
		promptForSummaryChoice: async () => "No summary",
		promptForCustomInstructions: async () => undefined,
		abortBranchSummary: () => {},
		navigateTree: async (id, options) => {
			result.navigated.push({ id, options });
			return { cancelled: false, editorText: "text" };
		},
		setEditorTextIfEmpty: () => {},
		flushCompactionQueue: () => {},
		withSummaryEscapeHandler: (handler) => {
			result.escapeInstalls += 1;
			return () => {};
		},
		showSummaryIndicator: () => {},
		clearSummaryIndicator: () => {},
		renderInitialMessages: () => {},
		clearChat: () => {},
		copyToClipboard: async (text) => {
			result.copied.push(text);
		},
		...overrides,
	};
	return result;
}

beforeAll(() => {
	initTheme("default", false);
});

describe("tree-selector (step 11.3)", () => {
	it("reports an empty session tree", () => {
		const deps = fakeDeps({ getTree: () => [] });
		showTreeSelector(deps);
		expect(deps.status).toEqual(["No entries in session"]);
	});

	it("is a no-op when selecting the current leaf", async () => {
		const deps = fakeDeps();
		showTreeSelector(deps);
		const last = constructed.at(-1)!;
		expect(last.name).toBe("tree");
		expect(last.args[0]).toHaveLength(2);

		const onSelect = last.args[3] as (entryId: string) => Promise<void>;
		await onSelect("n2");
		expect(deps.navigated).toHaveLength(0);
		expect(deps.status).toContain("Already at this point");
	});

	it("navigates without summary when the prompt is skipped", async () => {
		const deps = fakeDeps();
		showTreeSelector(deps);
		const last = constructed.at(-1)!;
		const onSelect = last.args[3] as (entryId: string) => Promise<void>;
		await onSelect("n1");
		expect(deps.navigated).toEqual([{ id: "n1", options: { summarize: false } }]);
		expect(deps.status).toContain("Navigated to selected point");
		expect(deps.escapeInstalls).toBe(0);
	});

	it("installs the summary escape handler and reports aborts", async () => {
		const deps = fakeDeps({
			getBranchSummarySkipPrompt: () => false,
			promptForSummaryChoice: async () => "Summarize",
			navigateTree: async (id) => {
				deps.navigated.push({ id, options: { summarize: true } });
				return { aborted: true };
			},
		});
		showTreeSelector(deps);
		const last = constructed.at(-1)!;
		const onSelect = last.args[3] as (entryId: string) => Promise<void>;
		await onSelect("n1");
		expect(deps.escapeInstalls).toBe(1);
		expect(deps.status).toContain("Branch summarization cancelled");
	});

	it("copies selected text through the injected clipboard", async () => {
		const deps = fakeDeps();
		showTreeSelector(deps);
		const instance = instances.at(-1)!;
		await instance.onCopy?.("selected text");
		expect(deps.copied).toEqual(["selected text"]);
		expect(deps.status).toContain("Copied selected message to clipboard");
	});
});
