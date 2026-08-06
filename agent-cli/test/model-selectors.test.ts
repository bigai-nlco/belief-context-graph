import { beforeAll, describe, expect, it, vi } from "vitest";

const constructed = vi.hoisted(() => [] as Array<{ name: string; args: unknown[] }>);

vi.mock("../src/modes/interactive/components/model-selector.ts", () => ({
	ModelSelectorComponent: class {
		constructor(...args: unknown[]) {
			constructed.push({ name: "model", args });
		}
	},
}));

vi.mock("../src/modes/interactive/components/scoped-models-selector.ts", () => ({
	ScopedModelsSelectorComponent: class {
		constructor(...args: unknown[]) {
			constructed.push({ name: "scoped", args });
		}
	},
}));

import {
	showModelSelector,
	showModelsSelector,
	type ModelSelectorDeps,
} from "../src/modes/interactive/model-selectors.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

const MODEL = {
	id: "qwen",
	provider: "local",
	api: "test",
} as never;

function fakeDeps(overrides: Partial<ModelSelectorDeps> = {}): ModelSelectorDeps & {
	status: string[];
	errors: string[];
	set: unknown[];
	scoped: unknown[];
	renders: number;
} {
	const result = {
		status: [] as string[],
		errors: [] as string[],
		set: [] as unknown[],
		scoped: [] as unknown[],
		renders: 0,
		ui: {} as ModelSelectorDeps["ui"],
		settingsManager: {
			getEnabledModels: () => undefined,
			setEnabledModels: () => {},
		} as unknown as ModelSelectorDeps["settingsManager"],
		modelRuntime: {
			refresh: async () => {},
			getAvailable: async () => [MODEL],
		} as unknown as ModelSelectorDeps["modelRuntime"],
		getCurrentModel: () => MODEL,
		scopedModels: [],
		showSelector: (create: (done: () => void) => unknown) => create(() => {}),
		showStatus: (m: string) => result.status.push(m),
		showError: (m: string) => result.errors.push(m),
		requestRender: () => {
			result.renders += 1;
		},
		setModel: async (m) => {
			result.set.push(m);
		},
		onModelSetSuccess: () => {},
		refreshProviderCount: async () => {},
		setScopedModels: (models) => {
			result.scoped.push(models);
		},
		...overrides,
	};
	return result;
}

beforeAll(() => {
	initTheme("default", false);
});

describe("model-selectors (step 11.3)", () => {
	it("builds the single-model selector and applies the selection", async () => {
		const deps = fakeDeps();
		showModelSelector(deps);

		const last = constructed.at(-1)!;
		expect(last.name).toBe("model");
		const onSelect = last.args[5] as (model: unknown) => Promise<void>;
		await onSelect(MODEL);
		expect(deps.set).toEqual([MODEL]);
		expect(deps.status).toEqual(["Model: qwen"]);
	});

	it("surfaces model-set errors", async () => {
		const deps = fakeDeps({
			setModel: async () => {
				throw new Error("boom");
			},
		});
		showModelSelector(deps);
		const last = constructed.at(-1)!;
		const onSelect = last.args[5] as (model: unknown) => Promise<void>;
		await onSelect(MODEL);
		expect(deps.errors[0]).toContain("boom");
		expect(deps.status).toHaveLength(0);
	});

	it("reports when no models are available", async () => {
		const deps = fakeDeps({
			modelRuntime: {
				refresh: async () => {},
				getAvailable: async () => [],
			} as unknown as ModelSelectorDeps["modelRuntime"],
		});
		await showModelsSelector(deps);
		expect(deps.status).toEqual(["No models available"]);
	});

	it("builds the scoped selector and persists patterns", async () => {
		const deps = fakeDeps({
			modelRuntime: {
				refresh: async () => {},
				getAvailable: async () => [MODEL, { id: "qwen2", provider: "local", api: "test" }],
			} as unknown as ModelSelectorDeps["modelRuntime"],
		});
		await showModelsSelector(deps);

		const last = constructed.at(-1)!;
		expect(last.name).toBe("scoped");
		const callbacks = last.args[1] as {
			onPersist: (enabledIds: string[] | null) => void;
		};
		const setEnabledPatterns = vi.spyOn(deps.settingsManager, "setEnabledModels");
		callbacks.onPersist(["local/qwen"]);
		expect(setEnabledPatterns).toHaveBeenCalledWith(["local/qwen"]);
		expect(deps.status).toContain("Model selection saved to settings");
	});

	it("clears scoped models when all are enabled", async () => {
		const deps = fakeDeps();
		await showModelsSelector(deps);
		const last = constructed.at(-1)!;
		const callbacks = last.args[1] as { onChange: (ids: string[] | null) => Promise<void> };
		// selecting every available model means "no filter"
		await callbacks.onChange(["local/qwen"]);
		expect(deps.scoped).toEqual([[]]);
	});
});
